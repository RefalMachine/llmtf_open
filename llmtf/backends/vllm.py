VLLM_IMPORT_ERROR = None
try:
    from vllm import LLM as vLLM
    from vllm import SamplingParams
    from vllm.lora.request import LoRARequest
except Exception as exc:
    VLLM_IMPORT_ERROR = exc
    vLLM = None
    SamplingParams = None
    LoRARequest = None

try:
    from vllm.attention.backends.flash_attn import FlashAttentionBackend
except Exception:
    FlashAttentionBackend = None

import codecs
import copy
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from peft import PeftConfig
from transformers import AutoTokenizer, GenerationConfig

from llmtf.backends.base import Backend, normalize_stop_token_ids
from llmtf.config import DEFAULT_VLLM_GPU_MEMORY_UTILIZATION
from llmtf.continuation import render_local_chat_prompt, single_token_candidate_ids
from llmtf.utils import json_to_jinja


class VLLMBackend(Backend):
    """vLLM engine backend (in-process).

    Supports generate / calculate_tokens_proba. PPL not supported.
    Reasoning orchestration lives on the LLM level.
    """

    def __init__(
        self,
        conversation_template_path="auto",
        is_foundational=False,
        use_fast_tokenizer=True,
        device_map='auto',
        max_seq_len_to_capture=None,
        gpu_memory_utilization=DEFAULT_VLLM_GPU_MEMORY_UTILIZATION,
        disable_sliding_window=True,
        enable_prefix_caching=True,
        trust_remote_code=False,
        calculate_tokens_proba_logprobs_count=50,
        tensor_parallel_size=1,
        limit_mm_per_prompt=None,
        model_context_len=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.conversation_template_path = conversation_template_path
        self.is_foundational = is_foundational
        self.use_fast_tokenizer = use_fast_tokenizer
        self.device_map = device_map
        # model_context_len wins over the legacy max_seq_len_to_capture kwarg;
        # fall back to 8192 (8K) when neither is provided.
        if model_context_len is not None:
            self.max_seq_len_to_capture = model_context_len
        elif max_seq_len_to_capture is not None:
            self.max_seq_len_to_capture = max_seq_len_to_capture
        else:
            self.max_seq_len_to_capture = 4096 * 2
        self.gpu_memory_utilization = gpu_memory_utilization
        self.disable_sliding_window = disable_sliding_window
        self.enable_prefix_caching = enable_prefix_caching
        self.trust_remote_code = trust_remote_code
        self.calculate_tokens_proba_logprobs_count = calculate_tokens_proba_logprobs_count
        self.tensor_parallel_size = tensor_parallel_size
        self.limit_mm_per_prompt = {"image": 0, "video": 0} if limit_mm_per_prompt is None else limit_mm_per_prompt

        self._ensure_vllm_available()
        assert 'CUDA_VISIBLE_DEVICES' in os.environ
        self.logger.info('CUDA_VISIBLE_DEVICES=' + os.environ['CUDA_VISIBLE_DEVICES'])
        self.logger.info('device_map=' + self.device_map)

    def _ensure_vllm_available(self):
        if vLLM is None:
            raise ImportError(
                "Failed to import vLLM. Install a compatible vllm package and its "
                "dependencies in the active environment before using VLLMBackend."
            ) from VLLM_IMPORT_ERROR

    def support_method(self, method):
        return method in ['generate', 'calculate_tokens_proba']

    # --- model loading ---

    def from_pretrained(
        self,
        model_dir,
        *,
        conversation_template_path="auto",
        is_foundational=False,
        **kwargs
    ):
        self.conversation_template_path = conversation_template_path
        self.is_foundational = bool(is_foundational)
        self._load_model(
            model_dir,
            conversation_template_path=conversation_template_path,
            is_foundational=is_foundational
        )
        self.reset_stop_strings()
        self.logger.info(f'Leading space: {self.leading_space}')

        tokenizer = self.model.get_tokenizer()
        tokenizer.pad_token_id = self.tokenizer.pad_token_id
        tokenizer.truncation_side = self.tokenizer.truncation_side

        self.attn_backend = self._get_attn_backend()
        self.special_attn_warning_complete = False

    def _get_attn_backend(self):
        try:
            return self.model.llm_engine.model_executor.driver_worker.model_runner.attn_backend
        except AttributeError:
            self.logger.warning("Could not read vLLM attention backend from this vLLM version")
            return None

    def _load_model(
        self,
        model_dir,
        conversation_template_path,
        is_foundational
    ):
        self.model_name_or_path = model_dir
        if self._check_if_lora(model_dir):
            self._load_lora(model_dir)
        else:
            self._load_plain_model(model_dir)

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_dir, use_fast=self.use_fast_tokenizer, trust_remote_code=self.trust_remote_code)
        except:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_dir, use_fast=not self.use_fast_tokenizer, trust_remote_code=self.trust_remote_code)

        self.tokenizer.truncation_side = 'left'
        self.tokenizer.padding_side = 'left'
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        try:
            self.generation_config = GenerationConfig.from_pretrained(
                model_dir, trust_remote_code=self.trust_remote_code)
        except:
            self.generation_config = GenerationConfig.from_dict({})

        self.conv_template_eos_string = None
        self._update_chat_template(is_foundational, conversation_template_path)
        self._init_default_gen_params()
        self._check_if_leading_space()
        self._override_eos_token_conv_template()

        self.logger.info(f"Model id: {self.model_name_or_path}")

    def _check_if_lora(self, model_dir):
        self.if_lora = False
        if os.path.exists(model_dir):
            adapter_config_exists = os.path.exists(
                os.path.join(model_dir, 'adapter_config.json'))
            adapter_model_exists = os.path.exists(os.path.join(model_dir, 'adapter_model.bin')) or os.path.exists(
                os.path.join(model_dir, 'adapter_model.safetensors'))
            self.if_lora = adapter_config_exists and adapter_model_exists
            return self.if_lora
        try:
            PeftConfig.from_pretrained(model_dir)
            self.if_lora = True
        except:
            pass
        return self.if_lora

    def _check_if_leading_space(self):
        self.leading_space = False
        self.space_token = None
        char = '1'
        tokens = self.tokenizer(char, add_special_tokens=False)['input_ids']
        if len(tokens) > 1:
            self.logger.info(f'_check_if_leading_space: \"{tokens}\"')
            self.space_token = tokens[0]
            self.leading_space = True
        else:
            token_str = self.tokenizer.convert_ids_to_tokens(tokens)[0]
            if len(token_str) != 1:
                self.logger.info(f'_check_if_leading_space: \"{token_str}\"')
                self.space_token = token_str[0]
                self.leading_space = True

    def _load_plain_model(self, model_dir):
        self._ensure_vllm_available()
        self.model = vLLM(
            model=model_dir,
            max_model_len=self.max_seq_len_to_capture,
            gpu_memory_utilization=self.gpu_memory_utilization, max_logprobs=1000000,
            trust_remote_code=self.trust_remote_code, tensor_parallel_size=self.tensor_parallel_size,
            limit_mm_per_prompt=self.limit_mm_per_prompt,
            disable_sliding_window=self.disable_sliding_window,
            enable_prefix_caching=self.enable_prefix_caching,
        )

    def _load_lora(self, model_dir):
        self._ensure_vllm_available()
        config = PeftConfig.from_pretrained(model_dir)
        self.model = vLLM(
            model=config.base_model_name_or_path,
            max_model_len=self.max_seq_len_to_capture,
            gpu_memory_utilization=self.gpu_memory_utilization, max_logprobs=1000000,
            enable_lora=True, trust_remote_code=self.trust_remote_code, tensor_parallel_size=self.tensor_parallel_size,
            limit_mm_per_prompt=self.limit_mm_per_prompt,
            max_lora_rank=self._get_max_lora_rank(config),
            disable_sliding_window=self.disable_sliding_window,
            enable_prefix_caching=self.enable_prefix_caching,
        )

    def _get_lora_request(self):
        if not self.if_lora:
            return None
        return LoRARequest("lora", 1, self.model_name_or_path)

    def _get_max_lora_rank(self, lora_config):
        return lora_config.r

    # --- chat template / gen config ---

    def _update_chat_template(self, is_foundational, conversation_template_path):
        if not is_foundational and conversation_template_path == "auto":
            return
        if conversation_template_path == "auto":
            conversation_template_path = str(Path(__file__).parent.parent / 'conversation_configs' / 'default_foundational.json')
        with codecs.open(conversation_template_path, "r", "utf-8") as file:
            template = json.load(file)
        chat_template, eos_token = json_to_jinja(template)

        self.tokenizer.chat_template = chat_template
        if eos_token:
            self.conv_template_eos_string = eos_token
            eos_token_tokens = self.tokenizer.tokenize(eos_token)
            if len(eos_token_tokens) > 1:
                self.logger.warning("eos token from chat template consists out of several tokens. First one will be used")
            self.tokenizer.eos_token_id = self.tokenizer.convert_tokens_to_ids(eos_token_tokens)[0]

    def _init_default_gen_params(self):
        self.generation_config.bos_token_id = self.tokenizer.bos_token_id
        self.generation_config.eos_token_id = self.tokenizer.eos_token_id
        self.generation_config.pad_token_id = self.tokenizer.pad_token_id
        self.generation_config.do_sample = True
        self.generation_config.max_new_tokens = 64
        self.generation_config.max_length = self.get_model_context_len()
        self.generation_config.repetition_penalty = 1.0
        self.generation_config.temperature = 0.1
        self.generation_config.top_k = 40
        self.generation_config.top_p = 0.9
        self.generation_config.num_beams = 1
        self.generation_config.stop_strings = []

    def _override_eos_token_conv_template(self):
        if isinstance(self.generation_config.eos_token_id, int):
            self.generation_config.eos_token_id = [self.generation_config.eos_token_id]

        eos_token_from_conv = self.tokenizer.eos_token
        if eos_token_from_conv:
            self._add_stop_string(eos_token_from_conv)

        if self.conv_template_eos_string:
            self._add_stop_string(self.conv_template_eos_string)

        if type(self.generation_config.eos_token_id) == int:
            self.generation_config.eos_token_id = [self.generation_config.eos_token_id]

        self.eos_token_ids_base = copy.deepcopy(self.generation_config.eos_token_id)
        self.stop_strings_base = copy.deepcopy(self.generation_config.stop_strings)

    # --- stop strings ---

    def add_stop_strings(self, stop_strings):
        for stop_string in stop_strings:
            self._add_stop_string(stop_string)

        self.logger.info(f'Updated generation_config.eos_token_id: {self.generation_config.eos_token_id}')
        self.logger.info(f'Updated generation_config.stop_strings: {self.generation_config.stop_strings}')

    def _add_stop_string(self, stop_string):
        if stop_string in self.generation_config.stop_strings:
            return
        is_token, stop_string_ids = self._check_word_is_token(stop_string)
        if is_token:
            self.add_stop_token(stop_string_ids)
        self.generation_config.stop_strings.append(stop_string)

    def reset_stop_strings(self):
        self.generation_config.eos_token_id = copy.deepcopy(self.eos_token_ids_base)
        self.generation_config.stop_strings = copy.deepcopy(self.stop_strings_base)

    def _check_word_is_token(self, word):
        tokens = self.tokenizer.encode(word, add_special_tokens=False)
        if self.leading_space and tokens[0] == self.space_token:
            tokens = tokens[1:]

        return len(tokens) == 1, tokens

    def add_stop_token(self, stop_token):
        if type(stop_token) == str:
            is_token, stop_token_id = self._check_word_is_token(stop_token)
        else:
            assert type(stop_token) == list
            stop_token_id = stop_token

        if len(stop_token_id) > 1:
            self.logger.warning(f'Can\'t stop on sequence {stop_token_id} with HF model. Try --vvlm for correct behaviour. Ignoring this stop_token')
        elif len(stop_token_id) == 1:
            if stop_token_id[0] not in self.generation_config.eos_token_id:
                self.generation_config.eos_token_id.append(stop_token_id[0])
        else:
            self.logger.warning(f'len(stop_token_id) == 1 in add_stop_token with {stop_token}')

    # --- rendering / token counting ---

    def _render_model_prompt(self, messages, continue_last_assistant_message=True,
                             add_think_token=False):
        return render_local_chat_prompt(
            self.tokenizer,
            messages,
            config=getattr(self, 'continuation_config', None),
            continue_last_assistant_message=continue_last_assistant_message,
            enable_thinking=add_think_token
        )

    def apply_model_prompt(self, messages, continue_last_assistant_message=True,
                           add_think_token=False):
        return self._render_model_prompt(
            messages,
            continue_last_assistant_message=continue_last_assistant_message,
            add_think_token=add_think_token,
        )[0]

    def count_tokens_for_prompt(self, prompt):
        return len(self.tokenizer(prompt, add_special_tokens=False)['input_ids'])

    def count_tokens_for_messages(self, messages, *, continue_last_assistant_message=True, add_think_token=False):
        prompt = self.apply_model_prompt(
            messages,
            continue_last_assistant_message=continue_last_assistant_message,
            add_think_token=add_think_token
        )
        return self.count_tokens_for_prompt(prompt)

    # --- primitives ---

    def generate(
        self,
        messages,
        generation_config=None,
        continue_last_assistant_message=True,
        return_tokens=False,
        include_stop_str_in_output=False,
        skip_special_tokens=True,
        enable_thinking=False,
        **kwargs
    ):
        prompts, outputs, infos = self.generate_batch(
            [messages],
            generation_config=generation_config,
            continue_last_assistant_message=continue_last_assistant_message,
            return_tokens=return_tokens,
            include_stop_str_in_output=include_stop_str_in_output,
            skip_special_tokens=skip_special_tokens,
            enable_thinking=enable_thinking,
            **kwargs
        )
        return prompts[0], outputs[0], infos[0]

    def generate_batch(
        self,
        messages,
        *,
        generation_config=None,
        continue_last_assistant_message=True,
        return_tokens=False,
        include_stop_str_in_output=False,
        skip_special_tokens=True,
        enable_thinking=False,
        **kwargs
    ):
        messages_batch = messages
        generation_config = self.generation_config if generation_config is None else generation_config
        prompts_tokens_batch = []
        prefill_decisions = []
        for i, messages in enumerate(messages_batch):
            prompt, decision = self._render_model_prompt(
                messages,
                continue_last_assistant_message=continue_last_assistant_message,
                add_think_token=enable_thinking
            )
            prefill_decisions.append(decision)
            prompts_tokens_batch.append(
                self.tokenizer(
                    prompt,
                    add_special_tokens=False,
                    truncation=True,
                    max_length=self.generation_config.max_length
                )['input_ids']
            )

        sampling_params = SamplingParams(
            temperature=generation_config.temperature,
            top_p=generation_config.top_p,
            top_k=generation_config.top_k,
            max_tokens=generation_config.max_new_tokens,
            repetition_penalty=generation_config.repetition_penalty,
            presence_penalty=getattr(generation_config, 'presence_penalty', 0.0) or 0.0,
            stop=generation_config.stop_strings,
            stop_token_ids=normalize_stop_token_ids(
                getattr(generation_config, 'eos_token_id', None)
            ),
            n=generation_config.num_return_sequences,
            include_stop_str_in_output=enable_thinking or include_stop_str_in_output
        )

        prompts_vllm = []
        outputs = []
        infos = []

        vllm_responses = self.model.generate(
            prompts=[{"prompt_token_ids": prompt_tokens} for prompt_tokens in prompts_tokens_batch],
            sampling_params=sampling_params,
            use_tqdm=False,
            lora_request=self._get_lora_request(),
        )

        for response_index, response in enumerate(vllm_responses):
            infos.append(
                {
                    'prompt_len': len(response.prompt_token_ids),
                    'generated_len': [len(out.token_ids) for out in response.outputs],
                    'generated_cumulative_logprob': [out.cumulative_logprob for out in response.outputs],
                    'assistant_prefill': prefill_decisions[response_index].to_dict(),
                }
            )
            prompts_vllm.append(self.tokenizer.decode(response.prompt_token_ids))

            generated = [{'tokens': out.token_ids, 'text': out.text} if return_tokens else out.text for out in response.outputs]
            if len(generated) == 1:
                outputs.append(generated[0])
            else:
                outputs.append(generated)

        return prompts_vllm, outputs, infos

    def calculate_tokens_proba(self, messages, tokens_of_interest, continue_last_assistant_message=True, **kwargs):
        prompts, probs, infos = self.calculate_tokens_proba_batch(
            [messages], [tokens_of_interest],
            continue_last_assistant_message=continue_last_assistant_message,
            **kwargs
        )
        return prompts[0], probs[0], infos[0]

    def calculate_tokens_proba_batch(self, messages, tokens_of_interest, *, continue_last_assistant_message=True, **kwargs):
        if FlashAttentionBackend is not None and len(messages) > 1 and self.attn_backend == FlashAttentionBackend:
            if not self.special_attn_warning_complete:
                self.logger.warning(
                    'Flash Attention 2 most probably can work incorrectly with logproba and batch size > 1 '
                    '(because of padding).\nHighly recommended to use Xformer backend in this case '
                    '(set VLLM_ATTENTION_BACKEND env var to XFORMERS) or batch size=1.'
                )
                self.special_attn_warning_complete = True

        prompts_tokens_batch = []
        prefill_decisions = []
        tokens_of_interest_ids_batch = []
        for _messages, _tokens_of_interest in zip(messages, tokens_of_interest):
            prompt, decision = self._render_model_prompt(
                _messages,
                continue_last_assistant_message=continue_last_assistant_message,
            )
            prompts_tokens_batch.append(self.tokenizer(prompt, add_special_tokens=False, truncation=True, max_length=self.generation_config.max_length)['input_ids'])
            prefill_decisions.append(decision)
            tokens_of_interest_ids_batch.append(
                single_token_candidate_ids(self.tokenizer, _tokens_of_interest)
            )

        sampling_params = SamplingParams(
            temperature=0,
            logprobs=self.calculate_tokens_proba_logprobs_count,
            max_tokens=1,
            repetition_penalty=1.0
        )

        prompts_vllm = []
        probs_batch = []
        infos = []

        vllm_responses = self.model.generate(
            prompts=[{"prompt_token_ids": prompt_tokens} for prompt_tokens in prompts_tokens_batch],
            sampling_params=sampling_params,
            use_tqdm=False,
            lora_request=self._get_lora_request()
        )

        for i, response in enumerate(vllm_responses):
            logprobs = response.outputs[0].logprobs[-1]

            token2prob_by_id = {}
            for token_rep, lp in logprobs.items():
                token_id = token_rep
                token2prob_by_id[token_id] = np.exp(lp.logprob)

            result_probs = {}
            for token_str, variant_ids in zip(tokens_of_interest[i], tokens_of_interest_ids_batch[i]):
                max_prob = 0.0
                for var_id in variant_ids:
                    if var_id in token2prob_by_id:
                        prob = token2prob_by_id[var_id]
                        if prob > max_prob:
                            max_prob = prob
                result_probs[token_str] = max_prob

            probs_batch.append(result_probs)
            prompts_vllm.append(self.tokenizer.decode(response.prompt_token_ids))
            infos.append({
                'prompt_len': len(response.prompt_token_ids),
                'generated_len': len(response.outputs[0].token_ids),
                'generated_cumulative_logprob': response.outputs[0].cumulative_logprob,
                'generated_token': response.outputs[0].text,
                'assistant_prefill': prefill_decisions[i].to_dict(),
                'candidate_surface_form_aggregation': 'max',
            })

        return prompts_vllm, probs_batch, infos

    # --- introspection ---

    def get_params(self):
        return {
            'model_name_or_path': self.model_name_or_path,
            'backend_class': type(self).__name__,
            'generation_config': json.loads(self.generation_config.to_json_string(use_diff=True)),
            'device_map': self.device_map,
            'use_fast_tokenizer': self.use_fast_tokenizer,
            'leading_space': self.leading_space,
            'space_token': self.space_token,
            'max_model_len': self.get_model_context_len(),
            'max_seq_len_to_capture': self.max_seq_len_to_capture,
            'gpu_memory_utilization': self.gpu_memory_utilization,
            'disable_sliding_window': self.disable_sliding_window,
            'enable_prefix_caching': self.enable_prefix_caching,
            'trust_remote_code': self.trust_remote_code,
            'calculate_tokens_proba_logprobs_count': self.calculate_tokens_proba_logprobs_count,
            'limit_mm_per_prompt': self.limit_mm_per_prompt,
            'conversation_template_path': self.conversation_template_path,
            'conversation_template_hash': hashlib.sha256(
                (getattr(self.tokenizer, 'chat_template', None) or '').encode('utf-8')
            ).hexdigest(),
            'is_foundational': self.is_foundational,
            'tensor_parallel_size': self.tensor_parallel_size,
            'vllm': True
        }

    def get_model_context_len(self):
        return min(self.model.llm_engine.model_config.max_model_len, self.max_seq_len_to_capture)
