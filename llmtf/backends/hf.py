import codecs
import copy
import hashlib
import json
import os
from pathlib import Path

import torch
import tqdm
from peft import PeftConfig, PeftModel
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from llmtf.backends.base import Backend
from llmtf.continuation import render_local_chat_prompt, single_token_candidate_ids
from llmtf.utils import add_tokens_with_logsoftmax_messages, json_to_jinja


class HFBackend(Backend):
    """HuggingFace transformers backend (in-process weights).

    Supports generate / calculate_tokens_proba / calculate_logsoftmax (PPL).
    Reasoning orchestration lives on the LLM level; this backend only exposes
    primitives and a render/token-count contract.
    """

    def __init__(
        self,
        conversation_template_path="auto",
        is_foundational=False,
        load_in_8bit=False,
        torch_dtype='auto',
        device_map='auto',
        attn_implementation="flash_attention_2",
        model_context_len=None,
        use_fast_tokenizer=True,
        trust_remote_code=False,
        alpha_scale=1.0,
        not_scale_lm_head=False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.conversation_template_path = conversation_template_path
        self.is_foundational = is_foundational
        self.load_in_8bit = load_in_8bit
        self.torch_dtype = torch_dtype
        self.attn_implementation = attn_implementation
        self.device_map = device_map
        self.use_fast_tokenizer = use_fast_tokenizer
        self._model_context_len_override = model_context_len
        self.trust_remote_code = trust_remote_code
        self.alpha_scale = alpha_scale
        self.not_scale_lm_head = not_scale_lm_head
        self._presence_penalty_warned = False

    def support_method(self, method):
        return method in ['generate', 'calculate_tokens_proba', 'calculate_logsoftmax']

    def _warn_presence_penalty_once(self, generation_config):
        """HF generate does not support presence_penalty natively; warn once."""
        pp = getattr(generation_config, 'presence_penalty', 0.0)
        if pp is not None and pp != 0.0 and not self._presence_penalty_warned:
            self.logger.warning(
                "presence_penalty=%s is set but the HF backend does not support it; "
                "it will be ignored. Use repetition_penalty instead.", pp)
            self._presence_penalty_warned = True

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
        self._check_if_leading_space()
        self.logger.info(f'Leading space: {self.leading_space}')

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

    def _resolve_model_class(self, config):
        architectures = getattr(config, "architectures", None) or []
        if "Qwen3_5ForConditionalGeneration" in architectures:
            from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration
            self.logger.warning("Using Qwen3_5ForConditionalGeneration based on config.architectures")
            return Qwen3_5ForConditionalGeneration

        self.logger.warning("Using AutoModelForCausalLM")
        return AutoModelForCausalLM

    def _load_plain_model(self, model_dir):
        base_model_config = AutoConfig.from_pretrained(model_dir, trust_remote_code=self.trust_remote_code)
        torch_dtype = base_model_config.torch_dtype if self.torch_dtype == 'auto' else self.torch_dtype
        model_class = self._resolve_model_class(base_model_config)
        model_kwargs = {
            'torch_dtype': torch_dtype,
            'device_map': self.device_map,
            'attn_implementation': self.attn_implementation,
            'trust_remote_code': self.trust_remote_code,
        }
        if self.load_in_8bit:
            model_kwargs['load_in_8bit'] = self.load_in_8bit
        self.model = model_class.from_pretrained(model_dir, **model_kwargs)
        self.model.eval()

    def _load_lora(self, model_dir):
        config = PeftConfig.from_pretrained(model_dir)
        lm_head_alpha = config.alpha_pattern.get("lm_head", config.lora_alpha)

        config.lora_alpha /= self.alpha_scale
        for name in config.alpha_pattern:
            config.alpha_pattern[name] /= self.alpha_scale

        if self.not_scale_lm_head:
            config.alpha_pattern["lm_head"] = lm_head_alpha

        base_model_config = AutoConfig.from_pretrained(config.base_model_name_or_path, trust_remote_code=self.trust_remote_code)
        torch_dtype = base_model_config.torch_dtype if self.torch_dtype == 'auto' else self.torch_dtype

        model_class = self._resolve_model_class(base_model_config)
        model_kwargs = {
            'torch_dtype': torch_dtype,
            'device_map': self.device_map,
            'attn_implementation': self.attn_implementation,
            'trust_remote_code': self.trust_remote_code,
        }
        if self.load_in_8bit:
            model_kwargs['load_in_8bit'] = self.load_in_8bit
        self.model = model_class.from_pretrained(config.base_model_name_or_path, **model_kwargs)
        self.model = PeftModel.from_pretrained(
            self.model,
            model_dir,
            torch_dtype=torch_dtype,
            config=config
        )

        self.model = self.model.merge_and_unload()
        self.model.train(False)

        if base_model_config.tie_word_embeddings and config.modules_to_save is not None and 'lm_head' in config.modules_to_save:
            assert 'embed_tokens' not in config.modules_to_save
            self.model.model.embed_tokens.weight = self.model.lm_head.weight

        self.model.eval()

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
        vocab = self.tokenizer.vocab
        for t in tqdm.tqdm(vocab):
            token = self.tokenizer.convert_tokens_to_string([t])
            if token.endswith(stop_string):
                self.add_stop_token([vocab[t]])
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
        self._warn_presence_penalty_once(generation_config)

        prompts = []
        prefill_decisions = []
        for messages in messages_batch:
            prompt, decision = self._render_model_prompt(
                messages,
                continue_last_assistant_message=continue_last_assistant_message,
                add_think_token=enable_thinking,
            )
            prompts.append(prompt)
            prefill_decisions.append(decision)
        tokens = self.tokenizer(
            prompts,
            return_tensors="pt",
            truncation=True,
            add_special_tokens=False,
            max_length=generation_config.max_length,
            padding=True
        )
        data = {k: v.to(self.model.device) for k, v in tokens.items()}

        stop_strings = generation_config.stop_strings if generation_config.stop_strings else []
        with torch.no_grad():
            output_ids = self.model.generate(
                **data,
                generation_config=generation_config,
                tokenizer=self.tokenizer
            )

        output_ids = output_ids.view(len(messages_batch), -1, output_ids.shape[-1])

        outputs = []
        infos = []
        for batch_idx, (sample_output_ids_all, sample_input_ids) in enumerate(zip(output_ids, data["input_ids"])):
            sample_output_all = []
            generated_len = []
            prompt_len = int(data["attention_mask"][batch_idx].cpu().detach().sum())
            for sample_output_ids in sample_output_ids_all:
                sample_output_ids = sample_output_ids[len(sample_input_ids):]

                if return_tokens:
                    generated_ids = sample_output_ids.cpu().detach().tolist()

                    generated_tokens = [self.tokenizer.convert_tokens_to_string([t]) for t in self.tokenizer.convert_ids_to_tokens(generated_ids)]
                    for stop_string in stop_strings:
                        if stop_string in ''.join(generated_tokens):
                            for token_i, token in enumerate(generated_tokens):
                                if stop_string in token:
                                    generated_tokens = generated_tokens[:token_i + include_stop_str_in_output]
                                    break
                    if len(generated_tokens) != len(generated_ids):
                        generated_ids = generated_ids[:len(generated_tokens)]

                    sample_output_all.append({'tokens': generated_ids, 'text': self.tokenizer.decode(generated_ids, skip_special_tokens=skip_special_tokens)})
                else:
                    sample_output = self.tokenizer.decode(sample_output_ids, skip_special_tokens=skip_special_tokens)
                    for stop_string in stop_strings:
                        if stop_string in sample_output:
                            sample_output = sample_output[:sample_output.find(stop_string) + include_stop_str_in_output * len(stop_string)]
                    sample_output_all.append(sample_output)
                generated_len.append(len(sample_output_ids))

            if len(sample_output_all) == 1:
                sample_output_all = sample_output_all[0]

            outputs.append(sample_output_all)
            infos.append(
                {
                    'prompt_len': prompt_len,
                    'generated_len': generated_len,
                    'generated_cumulative_logprob': None,
                    'assistant_prefill': prefill_decisions[batch_idx].to_dict(),
                }
            )

        return prompts, outputs, infos

    def calculate_tokens_proba(self, messages, tokens_of_interest, continue_last_assistant_message=True, **kwargs):
        prompts, probs, infos = self.calculate_tokens_proba_batch([messages], [tokens_of_interest], continue_last_assistant_message=continue_last_assistant_message, **kwargs)
        return prompts[0], probs[0], infos[0]

    def calculate_tokens_proba_batch(self, messages, tokens_of_interest, *, continue_last_assistant_message=True, **kwargs):
        self._warn_presence_penalty_once(self.generation_config)
        prompts_batch = []
        prefill_decisions = []
        tokens_of_interest_ids_batch = []
        for _messages, _tokens_of_interest in zip(messages, tokens_of_interest):
            prompt, decision = self._render_model_prompt(
                _messages,
                continue_last_assistant_message=continue_last_assistant_message,
            )
            prompts_batch.append(prompt)
            prefill_decisions.append(decision)
            token_variants = single_token_candidate_ids(
                self.tokenizer, _tokens_of_interest
            )
            tokens_of_interest_ids_batch.append(token_variants)

        data = self.tokenizer(
            prompts_batch, return_tensors="pt", truncation=True, padding=True,
            add_special_tokens=False,
            max_length=self.generation_config.max_length
        )
        data = {k: v.to(self.model.device) for k, v in data.items()}

        with torch.no_grad():
            outputs = self.model(**data)
        logits = outputs.logits
        next_token_logits_batch = logits[:, -1, :]

        probs_batch = []
        infos = []
        for batch_idx in range(next_token_logits_batch.shape[0]):
            next_token_logits = next_token_logits_batch[batch_idx].flatten()
            next_token_probs = torch.nn.functional.softmax(next_token_logits, dim=-1).cpu()

            token_probs = {}
            for token_str, variants in zip(tokens_of_interest[batch_idx], tokens_of_interest_ids_batch[batch_idx]):
                max_prob = 0.0
                for var_id in variants:
                    prob = next_token_probs[var_id].item()
                    if prob > max_prob:
                        max_prob = prob
                token_probs[token_str] = max_prob

            probs_batch.append(token_probs)

            infos.append(
                {
                    'prompt_len': int(data["attention_mask"][batch_idx].cpu().detach().sum()),
                    'generated_len': 1,
                    'generated_cumulative_logprob': None,
                    'generated_token': self.tokenizer.decode([next_token_probs.argmax()]),
                    'assistant_prefill': prefill_decisions[batch_idx].to_dict(),
                    'candidate_surface_form_aggregation': 'max',
                }
            )

        return prompts_batch, probs_batch, infos

    # --- PPL (HF-only) ---

    def calculate_logsoftmax(self, messages, continue_last_assistant_message=True, log_only_last=True):
        prompts, messages, infos = self.calculate_logsoftmax_batch(
            [messages],
            continue_last_assistant_message=continue_last_assistant_message,
            log_only_last=log_only_last
        )
        return prompts[0], messages[0], infos[0]

    def calculate_logsoftmax_batch(self, messages, *, continue_last_assistant_message=True, log_only_last=True):
        prompts = []
        for _messages in messages:
            prompts.append(self.apply_model_prompt(_messages, continue_last_assistant_message=continue_last_assistant_message))

        data = self.tokenizer(
            prompts, return_tensors="pt", truncation=True, padding=True,
            add_special_tokens=False,
            max_length=self.generation_config.max_length, return_offsets_mapping=True
        )
        offset_mapping = data.pop('offset_mapping').tolist()

        model_input = {k: v.clone().to(self.model.device)
                       for k, v in data.items()}
        with torch.no_grad():
            outputs = self.model(**model_input).logits
            logsoftmax_batch = torch.nn.LogSoftmax(dim=-1)(outputs)

        labels = model_input['input_ids'][:, 1:]
        tokens_with_logsoftmax = []
        labels_len = labels.shape[1]
        seq_pos_list = list(range(labels_len))
        infos = []
        for batch_idx in range(labels.shape[0]):
            shift = labels_len - int(data['attention_mask'][batch_idx].sum()) + 1
            scores = logsoftmax_batch[batch_idx, seq_pos_list[shift:], labels[batch_idx][shift:]]
            scores = [0.0] + scores.tolist()
            tokens = data['input_ids'][batch_idx][shift:].tolist()
            positions = offset_mapping[batch_idx][shift:]
            tokens_with_logsoftmax.append([[tokens[i], scores[i], positions[i]] for i in range(len(scores))])

            infos.append(
                {
                    'prompt_len': int(data["attention_mask"][batch_idx].sum()),
                    'generated_len': 1,
                    'generated_cumulative_logprob': None,
                }
            )

        add_tokens_with_logsoftmax_messages(
            messages, prompts, tokens_with_logsoftmax, log_only_last)

        return prompts, messages, infos

    # --- introspection ---

    def get_params(self):
        return {
            'model_name_or_path': self.model_name_or_path,
            'backend_class': type(self).__name__,
            'generation_config': json.loads(self.generation_config.to_json_string(use_diff=True)),
            'load_in_8bit': self.load_in_8bit,
            'torch_dtype': str(self.torch_dtype),
            'attn_implementation': self.attn_implementation,
            'device_map': self.device_map,
            'use_fast_tokenizer': self.use_fast_tokenizer,
            'leading_space': self.leading_space,
            'space_token': self.space_token,
            'trust_remote_code': self.trust_remote_code,
            'max_model_len': self.get_model_context_len(),
            'model_context_len_override': self._model_context_len_override,
            'conversation_template_path': self.conversation_template_path,
            'conversation_template_hash': hashlib.sha256(
                (getattr(self.tokenizer, 'chat_template', None) or '').encode('utf-8')
            ).hexdigest(),
            'is_foundational': self.is_foundational,
            'alpha_scale': self.alpha_scale,
            'not_scale_lm_head': self.not_scale_lm_head,
        }

    def get_model_context_len(self):
        config = self.model.config
        max_position_embeddings = getattr(config, 'max_position_embeddings', None)
        if max_position_embeddings is None:
            model_type = getattr(config, 'model_type', None)
            architectures = getattr(config, 'architectures', None) or []
            is_qwen35 = model_type == 'qwen3_5' or any('Qwen3_5' in arch for arch in architectures)
            if is_qwen35:
                text_config = getattr(config, 'text_config', None)
                if isinstance(text_config, dict):
                    max_position_embeddings = text_config.get('max_position_embeddings')
                elif text_config is not None:
                    max_position_embeddings = getattr(text_config, 'max_position_embeddings', None)

        if max_position_embeddings is None:
            raise AttributeError(
                f"{config.__class__.__name__} does not define max_position_embeddings"
            )

        # Apply optional override (--model_context_len). Cap at the model's
        # architectural maximum; raise-only if user requests more, warn + clamp.
        if self._model_context_len_override is not None:
            if self._model_context_len_override > max_position_embeddings:
                self.logger.warning(
                    "model_context_len override (%d) exceeds the model's "
                    "max_position_embeddings (%d); clamping to %d.",
                    self._model_context_len_override, max_position_embeddings, max_position_embeddings
                )
                return max_position_embeddings
            return self._model_context_len_override
        return max_position_embeddings
