from abc import abstractmethod
import abc
from peft import PeftConfig, PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, GenerationConfig
import logging
import torch
import codecs
import json
from typing import List
import os
import requests
import numpy as np
from llmtf.base import LLM
try:
    from vllm import LLM as vLLM
    from vllm import SamplingParams
    from vllm.lora.request import LoRARequest
    from vllm.attention.backends.flash_attn import FlashAttentionBackend
except:
    pass
DEFAULT_MESSAGE_TEMPLATE = "{content}\n"
DEFAULT_SYSTEM_PROMPT = ""


class Conversation:
    def __init__(
        self,
        system_message_template: str = DEFAULT_MESSAGE_TEMPLATE,
        user_message_template: str = DEFAULT_MESSAGE_TEMPLATE,
        bot_message_template: str = DEFAULT_MESSAGE_TEMPLATE,
        bot_message_template_incomplete: str = DEFAULT_MESSAGE_TEMPLATE,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        system_role: str = "system",
        user_role: str = "user",
        bot_role: str = "bot",
        global_prefix: str = '',
        suffix: str = "<s>bot",
        add_special_tokens: bool = False,
        eos_token: str =''
    ):
        self.system_message_template = system_message_template
        self.user_message_template = user_message_template
        self.bot_message_template = bot_message_template
        self.system_role = system_role
        self.user_role = user_role
        self.bot_role = bot_role
        self.global_prefix = global_prefix
        self.suffix = suffix
        self.bot_message_template_incomplete = bot_message_template_incomplete
        self.add_special_tokens = add_special_tokens
        self.messages = []

        if system_prompt is not None and len(system_prompt) > 0:
            self.messages.append({
                "role": self.system_role,
                "content": system_prompt
            })

    def add_system_message(self, message):
        if len(self.messages) == 0:
            self.messages.append({
                "role": self.system_role,
                "content": message
            })
        else:
            if self.messages[0]["role"] == self.system_role:
                self.messages[0]["content"] = message
            else:
                self.messages = [{
                    "role": self.system_role,
                    "content": message
                }] + self.messages

    def add_user_message(self, message):
        self.messages.append({
            "role": self.user_role,
            "content": message
        })

    def add_bot_message(self, message):
        self.messages.append({
            "role": self.bot_role,
            "content": message
        })

    def count_tokens(self, tokenizer, current_messages):
        final_text = ""
        for message in current_messages:
            final_text += self.format_message(message)
        tokens = tokenizer([final_text])["input_ids"][0]
        return len(tokens)

    def shrink(self, tokenizer, messages, max_tokens):
        system_message = messages[0]
        other_messages = messages[1:]
        while self.count_tokens(tokenizer, [system_message] + other_messages) > max_tokens:
            other_messages = other_messages[2:]
        return [system_message] + other_messages

    def format_message(self, message, incomplete_last_bot_message=False):
        if message["role"] == self.system_role:
            return self.system_message_template.format(**message)
        if message["role"] == self.user_role:
            return self.user_message_template.format(**message)
        if message["role"] == self.bot_role:
            if incomplete_last_bot_message:
                return self.bot_message_template_incomplete.format(**message)
            return self.bot_message_template.format(**message)

        raise Exception('Unknown role')

    def get_prompt(self, tokenizer=None, max_tokens: int = None, add_suffix: bool = True, incomplete_last_bot_message: bool = False):
        messages = self.messages
        if max_tokens is not None:
            assert tokenizer is not None
            messages = self.shrink(tokenizer, messages, max_tokens)

        final_text = self.global_prefix
        for i, message in enumerate(messages):
            if i == len(messages) - 1 and incomplete_last_bot_message and message['role'] == self.bot_role:
                final_text += self.format_message(message, incomplete_last_bot_message=True)
            else:
                final_text += self.format_message(message)

        if add_suffix and (not incomplete_last_bot_message or messages[-1]['role'] != self.bot_role):
            final_text += self.suffix
            return final_text

        return final_text

    def iter_messages(self):
        for message in self.messages:
            yield self.format_message(message), message["role"]

    @classmethod
    def from_template(cls, file_name):
        with open(file_name, encoding="utf-8") as r:
            template = json.load(r)
        return Conversation(
            **template
        )

    def expand(self, messages, role_mapping = None):
        if not role_mapping:
            role_mapping = dict()

        if messages[0]["role"] == "system":
            self.messages = []

        for message in messages:
            self.messages.append({
                "role": role_mapping.get(message["role"], message["role"]),
                "content": message["content"]
            })
        self.messages[-1]['content'] = self.messages[-1]['content'].rstrip()

class LocalHostedLLM(LLM):
    def support_method(self, method):
        return method in ['generate', 'calculate_tokens_proba']

    def from_pretrained(self, model_dir):
        self._load_model(model_dir)
        self._check_if_leading_space()
        self.logger.info(f'Leading space: {self.leading_space}')

    def _load_model(self, model_dir):
        self.model_name_or_path = model_dir
        if self._check_if_lora(model_dir):
            self._load_lora(model_dir)
        else:
            self._load_plain_model(model_dir)

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=self.use_fast_tokenizer, trust_remote_code=self.trust_remote_code)
        except:
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=not self.use_fast_tokenizer, trust_remote_code=self.trust_remote_code)

        self.tokenizer.truncation_side = 'left'
        self.tokenizer.padding_side = 'left'
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        try:
            self.generation_config = GenerationConfig.from_pretrained(model_dir, trust_remote_code=self.trust_remote_code)
        except:
            self.generation_config = GenerationConfig.from_dict({})
        self._init_default_gen_params()
        self.logger.info(f"Model id: {self.model_name_or_path}")

    def _check_if_lora(self, model_dir):
        self.if_lora = False
        if os.path.exists(model_dir):
            adapter_config_exists = os.path.exists(os.path.join(model_dir, 'adapter_config.json'))
            adapter_model_exists = os.path.exists(os.path.join(model_dir, 'adapter_model.bin')) or os.path.exists(os.path.join(model_dir, 'adapter_model.safetensors'))
            self.if_lora = adapter_config_exists and adapter_model_exists
            return self.if_lora
        try:
            PeftConfig.from_pretrained(model_dir)
            self.if_lora  = True
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

    def _get_tokens_of_interest_ids_modify_prompt(self, messages, tokens_of_interest, incomplete_last_bot_message):
        prompt = self.apply_model_prompt(messages, incomplete_last_bot_message=incomplete_last_bot_message)
        tokens_of_interest_ids, add_space = self._calculate_tokens_of_interest_ids_and_addition_spaces(prompt, tokens_of_interest)
        if add_space:
            prompt += ' '
        return prompt, tokens_of_interest_ids

    def apply_model_prompt(self, messages, incomplete_last_bot_message=True):
        conv = Conversation(**self.conversation_template)
        for m in messages:
            if m['role'] == 'user':
                conv.add_user_message(m['content'])
            elif m['role'] == 'bot':
                conv.add_bot_message(m['content'])
            else:
                role = m['role']
                raise Exception(f'Unknown role {role}')
        return conv.get_prompt(self.tokenizer, incomplete_last_bot_message=incomplete_last_bot_message)
    
    def count_tokens_for_prompt(self, prompt):
        return len(self.tokenizer(
            prompt,
            add_special_tokens=self.conversation_template['add_special_tokens']
        )['input_ids'])

    def _calculate_tokens_of_interest_ids_and_addition_spaces(self, prompt, tokens_of_interest):
        assert prompt[-1] != ' '

        shift = len(self.tokenizer(prompt, add_special_tokens=self.conversation_template['add_special_tokens']).input_ids)
        tokens_of_interest_ids = []
        add_spaces = []
        for token_str in tokens_of_interest:
            if prompt.endswith('\n'):
                prompt_check = prompt + token_str
            else:
                prompt_check = prompt + ' ' + token_str
            tokens_rest = self.tokenizer(prompt_check, add_special_tokens=self.conversation_template['add_special_tokens']).input_ids[shift:]
            skip = 0
            is_ok = False
            for token in tokens_rest:
                if len(self.tokenizer.decode([token]).strip()) > 0 and token_str.startswith(self.tokenizer.decode([token]).strip()):
                    is_ok = True
                    break
                skip += 1

            assert skip <= 1 and is_ok
            tokens_of_interest_ids.append(token)
            add_spaces.append(skip > 0)
        assert sum(add_spaces) == 0 or sum(add_spaces) == len(add_spaces)
        return tokens_of_interest_ids, add_spaces[0]

    def _init_default_gen_params(self):
        self.generation_config.bos_token_id = self.tokenizer.bos_token_id
        self.generation_config.eos_token_id = self.tokenizer.eos_token_id
        self.generation_config.pad_token_id = self.tokenizer.pad_token_id
        self.generation_config.do_sample = True
        self.generation_config.max_new_tokens = 64
        self.generation_config.max_length = self.get_max_model_len()
        self.generation_config.repetition_penalty = 1.0
        self.generation_config.temperature = 0.1
        self.generation_config.top_k = 40
        self.generation_config.top_p = 0.9
        #self.generation_config.stop_strings = [] #does not work 

        self._override_eos_token_conv_template()

    def _override_eos_token_conv_template(self):
        eos_token = self.conversation_template.get('eos_token', None)
        if eos_token is not None and len(eos_token) > 0:
            # TODO: encode + учесть первый пробел, если есть.
            eos_token_id = self.tokenizer.convert_tokens_to_ids([eos_token])
            assert len(eos_token_id) == 1 and eos_token_id[0] != None
            eos_token_id = eos_token_id[0]
            eos_token_id_old = self.generation_config.eos_token_id
            self.generation_config.eos_token_id = eos_token_id
            self.logger.info(f'Override eos_token_id in generation_config from {eos_token_id_old} to {eos_token_id}')

        global_prefix = self.conversation_template.get('global_prefix', None)
        if global_prefix is None:
            global_prefix = self.tokenizer.decode([self.tokenizer.bos_token_id])
            self.conversation_template['global_prefix'] = global_prefix
            self.logger.info(f'Set global prefix {global_prefix} to conv config')

        if len(global_prefix) == 0:
            self.logger.warning(f'Global prefix is equal to empty string!')

    def add_stop_token(self, stop_token):
        # TODO: переделать всю логику по поводу стоп токенов. Позволить генерировать пока не будет достигнут stop criteria, но потом просто обрезать результат до stop_string.
        if type(self.generation_config.eos_token_id) != list:
            self.generation_config.eos_token_id = [self.generation_config.eos_token_id]

        # есть способ получше обработать принудительный пробел вначале в некоторых токенайзерах?
        stop_token_id = self.tokenizer.encode(stop_token, add_special_tokens=False)
        if self.leading_space and stop_token_id[0] == self.space_token:
            stop_token_id = stop_token_id[1:]
        
        if len(stop_token_id) > 1:
            self.logger.warning(f'Can\'t stop on sequence {stop_token_id} with HF model. Try --vvlm for correct behaviour. Will stop on {stop_token_id[0]}')
            stop_token_id = stop_token_id[:1]
        self.logger.info(f'Updating generation_config.eos_token_id: add {stop_token_id}')
        assert len(stop_token_id) == 1
        self.generation_config.eos_token_id.append(stop_token_id[0])
        self.logger.info(f'generation_config.eos_token_id: {self.generation_config.eos_token_id}')

    def reset_stop_tokens(self):
        self.generation_config.eos_token_id = self.generation_config.eos_token_id[0] if type(self.generation_config.eos_token_id) == list else self.generation_config.eos_token_id

class HFModel(LocalHostedLLM):
    def __init__(
            self, conversation_template_path, 
            load_in_8bit=False, 
            torch_dtype='auto', device_map='auto', 
            use_flash_attention_2=True, use_fast_tokenizer=True, 
            trust_remote_code=False,
            **kwargs
        ):
        super().__init__()
        self.load_in_8bit = load_in_8bit
        self.torch_dtype = torch_dtype
        self.use_flash_attention_2 = use_flash_attention_2
        self.device_map = device_map
        self.use_fast_tokenizer = use_fast_tokenizer
        self.trust_remote_code=trust_remote_code,
        with codecs.open(conversation_template_path, 'r', 'utf-8') as file:
            template = json.load(file)
        self.conversation_template = template

    def get_params(self):
        return {
            'model_name_or_path': self.model_name_or_path,
            'generation_config': json.loads(self.generation_config.to_json_string(use_diff=True)),
            'conversation_template': self.conversation_template,
            'load_in_8bit': self.load_in_8bit,
            'torch_dtype': self.torch_dtype,
            'use_flash_attention_2': self.use_flash_attention_2,
            'device_map': self.device_map,
            'use_fast_tokenizer': self.use_fast_tokenizer,
            'leading_space': self.leading_space,
            'space_token': self.space_token,
            'trust_remote_code': self.trust_remote_code,
            'max_model_len': self.get_max_model_len()
        }
        
    def generate(self, messages, generation_config=None, incomplete_last_bot_message=True):
        prompts, outputs, infos = self.generate_batch([messages], generation_config=generation_config, incomplete_last_bot_message=incomplete_last_bot_message)
        return prompts[0], outputs[0], infos[0]

    def generate_batch(self, messages, generation_config=None, incomplete_last_bot_message=True):
        prompts = []
        for _messages in messages:
            prompts.append(self.apply_model_prompt(_messages, incomplete_last_bot_message=incomplete_last_bot_message))
        data = self.tokenizer(
            prompts,
            return_tensors="pt",
            truncation=True,
            add_special_tokens=self.conversation_template['add_special_tokens'], 
            max_length=self.generation_config.max_length,
            padding=True
        )
        data = {k: v.to(self.model.device) for k, v in data.items()}
        with torch.no_grad():
            output_ids = self.model.generate(
                **data,
                generation_config=self.generation_config if generation_config is None else generation_config
            )
        outputs = []
        infos = []
        for sample_output_ids, sample_input_ids in zip(output_ids, data["input_ids"]):
            sample_output_ids = sample_output_ids[len(sample_input_ids):]
            sample_output = self.tokenizer.decode(sample_output_ids, skip_special_tokens=True)
            outputs.append(sample_output)

            infos.append(
                {
                    'prompt_len': len(data['input_ids']), 
                    'generated_len': len(sample_output_ids), 
                    'generated_cumulative_logprob': 'TODO: calculate for hf model'
                }
            )

        return prompts, outputs, infos
    
    def calculate_tokens_proba(self, messages, tokens_of_interest, incomplete_last_bot_message=True):
        prompts, probs, infos = self.calculate_tokens_proba_batch([messages], [tokens_of_interest], incomplete_last_bot_message=incomplete_last_bot_message)
        return prompts[0], probs[0], infos[0]
    
    def calculate_tokens_proba_batch(self, messages, tokens_of_interest, incomplete_last_bot_message=True):
        prompts_batch = []
        tokens_of_interest_ids_batch = []
        for _messages, _tokens_of_interest in zip(*[messages, tokens_of_interest]):
            prompt, tokens_of_interest_ids = self._get_tokens_of_interest_ids_modify_prompt(_messages, _tokens_of_interest, incomplete_last_bot_message)
            prompts_batch.append(prompt)
            tokens_of_interest_ids_batch.append(tokens_of_interest_ids)

        data = self.tokenizer(
            prompts_batch, return_tensors="pt", truncation=True, padding=True,
            add_special_tokens=self.conversation_template['add_special_tokens'], 
            max_length=self.generation_config.max_length
        )
        data = {k: v.to(self.model.device) for k, v in data.items()}
    
        with torch.no_grad():
            outputs = self.model(**data)
        logits = outputs.logits  # shape (batch_size, sequence_length, vocab_size)
        next_token_logits_batch = logits[:, -1, :]  # shape (batch_size, vocab_size)

        probs_batch = []
        infos = []
        for i in range(next_token_logits_batch.shape[0]):
            next_token_logits = next_token_logits_batch[i]
            next_token_logits = next_token_logits.flatten()
            assert next_token_logits.shape == torch.Size((self.model.config.vocab_size, ))

            next_token_probs = torch.nn.functional.softmax(next_token_logits, dim=-1).cpu()
            assert torch.isclose(next_token_probs.sum(), torch.tensor(1.0).to(next_token_probs.dtype), atol=1e-03)
        
            probs = next_token_probs[tokens_of_interest_ids_batch[i]].tolist()
            probs = dict(zip(tokens_of_interest[i], probs))
            probs_batch.append(probs)

            infos.append(
                {
                    'prompt_len': len(data['input_ids']), 
                    'generated_len': 1, 
                    'generated_cumulative_logprob': 'TODO: calculate for hf model', 
                    'generated_token': self.tokenizer.decode([next_token_probs.argmax()])
                }
            )

        return prompts_batch, probs_batch, infos

    def _load_plain_model(self, model_dir):
        base_model_config = AutoConfig.from_pretrained(model_dir, trust_remote_code=self.trust_remote_code)
        torch_dtype = base_model_config.torch_dtype if self.torch_dtype == 'auto' else self.torch_dtype
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=torch_dtype,
            load_in_8bit=self.load_in_8bit,
            device_map=self.device_map,
            use_flash_attention_2=self.use_flash_attention_2, 
            trust_remote_code=self.trust_remote_code
        )
        self.model.eval()

    def _load_lora(self, model_dir):
        config = PeftConfig.from_pretrained(model_dir)
        base_model_config = AutoConfig.from_pretrained(config.base_model_name_or_path, trust_remote_code=self.trust_remote_code)
        torch_dtype = base_model_config.torch_dtype if self.torch_dtype == 'auto' else self.torch_dtype

        self.model = AutoModelForCausalLM.from_pretrained(
            config.base_model_name_or_path,
            load_in_8bit=self.load_in_8bit,
            torch_dtype=torch_dtype,
            device_map=self.device_map,
            use_flash_attention_2=self.use_flash_attention_2, 
            trust_remote_code=self.trust_remote_code
        )
        self.model = PeftModel.from_pretrained(
            self.model,
            model_dir,
            torch_dtype=torch_dtype
        )

        self.model = self.model.merge_and_unload()
        self.model.train(False)

        self.model.eval()

    def get_max_model_len(self):
        return self.model.config.max_position_embeddings

class VLLMModel(LocalHostedLLM):
    def __init__(
            self, 
            conversation_template_path, 
            use_fast_tokenizer=True, 
            device_map='auto',
            max_seq_len_to_capture=8192,
            gpu_memory_utilization=0.9,
            disable_sliding_window=False,
            enable_prefix_caching=True,
            trust_remote_code=False,
            calculate_tokens_proba_logprobs_count=100,
            **kwargs
        ):
        super().__init__()
        with codecs.open(conversation_template_path, 'r', 'utf-8') as file:
            template = json.load(file)
        self.conversation_template = template
        self.use_fast_tokenizer = use_fast_tokenizer
        self.device_map = device_map
        self.max_seq_len_to_capture = max_seq_len_to_capture 
        self.gpu_memory_utilization = gpu_memory_utilization
        self.disable_sliding_window = disable_sliding_window
        self.enable_prefix_caching = enable_prefix_caching
        self.trust_remote_code = trust_remote_code
        self.calculate_tokens_proba_logprobs_count = calculate_tokens_proba_logprobs_count

    def from_pretrained(self, model_dir):
        self._load_model(model_dir)
        self._check_if_leading_space()
        self._conv_template_bos_vllm_test()
        self.reset_stop_tokens()
        self.logger.info(f'Leading space: {self.leading_space}')
        #self.logger.info(f'For calculate_tokens_proba batch always will be 1 because of possible errors in logprobs') # TODO: verify

        tokenizer = self.model.get_tokenizer()
        tokenizer.pad_token_id = self.tokenizer.pad_token_id
        tokenizer.padding_side = self.tokenizer.padding_side
        tokenizer.truncation_side = self.tokenizer.truncation_side

        self.attn_backend = self.model.llm_engine.model_executor.driver_worker.model_runner.attn_backend
        self.special_attn_warning_complete = False

    def generate(self, messages, generation_config=None, incomplete_last_bot_message=True):
        prompts, outputs, infos = self.generate_batch([messages], generation_config=None, incomplete_last_bot_message=True)
        return prompts[0], outputs[0], infos[0]

    def generate_batch(self, messages, generation_config=None, incomplete_last_bot_message=True):
        prompts_tokens_batch = []
        for _messages in messages:
            prompt = self.apply_model_prompt(_messages, incomplete_last_bot_message=incomplete_last_bot_message)
            prompts_tokens_batch.append(self.tokenizer(prompt, add_special_tokens=False)['input_ids'])

        generation_config = self.generation_config if generation_config is None else generation_config
        sampling_params = SamplingParams(
            temperature=generation_config.temperature,
            top_p=generation_config.top_p,
            top_k=generation_config.top_k,
            max_tokens=generation_config.max_new_tokens,
            repetition_penalty=generation_config.repetition_penalty,
            stop=generation_config.stop_strings
        )

        prompts_vllm = []
        outputs = []
        infos = []
        vllm_responses = self.model.generate(prompt_token_ids=prompts_tokens_batch, sampling_params=sampling_params, use_tqdm=False, lora_request=self._get_lora_request())
        for response in vllm_responses:
            infos.append(
                {
                    'prompt_len': len(response.prompt_token_ids), 
                    'generated_len': len(response.outputs[0].token_ids), 
                    'generated_cumulative_logprob': response.outputs[0].cumulative_logprob
                }
            )
            prompts_vllm.append(self.tokenizer.decode(response.prompt_token_ids))
            outputs.append(response.outputs[0].text)

        return prompts_vllm, outputs, infos

    def calculate_tokens_proba(self, messages, tokens_of_interest, incomplete_last_bot_message=True):
        prompts, probs, infos = self.calculate_tokens_proba_batch([messages], [tokens_of_interest], incomplete_last_bot_message)
        return prompts[0], probs[0], infos[0]

    def calculate_tokens_proba_batch(self, messages, tokens_of_interest, incomplete_last_bot_message=True):
        if len(messages) > 1 and self.attn_backend == FlashAttentionBackend:
            if not self.special_attn_warning_complete:
                self.logger.warning('Flash Attention 2 most probably can work incorrectly with logproba and batch size > 1 (because of padding).\nHighly recommended to use Xformer backend in this case (set VLLM_ATTENTION_BACKEND env var to XFORMERS) or batch size=1.')
                self.special_attn_warning_complete = True

        prompts_tokens_batch = []
        tokens_of_interest_ids_batch = []
        for _messages, _tokens_of_interest in zip(*[messages, tokens_of_interest]):
            prompt, tokens_of_interest_ids = self._get_tokens_of_interest_ids_modify_prompt(_messages, _tokens_of_interest, incomplete_last_bot_message)
            prompts_tokens_batch.append(self.tokenizer(prompt, add_special_tokens=False)['input_ids'])
            tokens_of_interest_ids_batch.append(tokens_of_interest_ids)

        sampling_params = SamplingParams(
            temperature=0,
            logprobs=self.calculate_tokens_proba_logprobs_count,
            max_tokens=1,
            repetition_penalty=1.0
        )

        prompts_vllm = []
        probs_batch = []
        infos = []
        vllm_responses = self.model.generate(prompt_token_ids=prompts_tokens_batch, sampling_params=sampling_params, use_tqdm=False, lora_request=self._get_lora_request())
        for i, response in enumerate(vllm_responses):
            infos.append(
                {
                    'prompt_len': len(response.prompt_token_ids), 
                    'generated_len': len(response.outputs[0].token_ids), 
                    'generated_cumulative_logprob': response.outputs[0].cumulative_logprob, 
                    'generated_token': response.outputs[0].text
                }
            )
            prompts_vllm.append(self.tokenizer.decode(response.prompt_token_ids))

            logprobs = response.outputs[0].logprobs[-1]
            tokens = [lp for lp in logprobs]
            probs = np.exp([logprobs[lp].logprob for lp in logprobs])
            token2prob = {tokens[i]: probs[i] for i in range(len(tokens))}
            token2prob = {tokens_of_interest[i][j]: token2prob[token] if token in token2prob else 0.0 for j, token in enumerate(tokens_of_interest_ids_batch[i])}
            probs_batch.append(token2prob)

        return prompts_vllm, probs_batch, infos

    def get_params(self):
        return {
            'model_name_or_path': self.model_name_or_path,
            'generation_config': json.loads(self.generation_config.to_json_string(use_diff=True)),
            'conversation_template': self.conversation_template,
            'device_map': self.device_map,
            'use_fast_tokenizer': self.use_fast_tokenizer,
            'leading_space': self.leading_space,
            'space_token': self.space_token,
            'max_model_len': self.get_max_model_len(),
            'max_seq_len_to_capture': self.max_seq_len_to_capture,
            'gpu_memory_utilization': self.gpu_memory_utilization,
            'disable_sliding_window': self.disable_sliding_window,
            'enable_prefix_caching': self.enable_prefix_caching,
            'trust_remote_code': self.trust_remote_code,
            'calculate_tokens_proba_logprobs_count': self.calculate_tokens_proba_logprobs_count,
            'vllm': True
        }

    def add_stop_token(self, stop_token):
        self.generation_config.stop_strings.append(stop_token)
        self.logger.info(f'Updating generation_config.stop_strings: {self.generation_config.stop_strings}')

    def reset_stop_tokens(self):
        self.generation_config.stop_strings = []#[self.generation_config.eos_token_id] if type(self.generation_config.eos_token_id) == int else self.generation_config.eos_token_id
        self.logger.info(f'Resetting generation_config.stop_strings to {self.generation_config.stop_strings}')

    def _conv_template_bos_vllm_test(self):
        self.global_prefix = self.conversation_template['global_prefix']
        global_prefix_check = None if self.global_prefix  == '' else self.global_prefix 
        #assert global_prefix_check == self.tokenizer.bos_token

        sampling_params = SamplingParams(
            temperature=0,
            max_tokens=1,
            repetition_penalty=1.0
        )
        vllm_response = self.model.generate(['Тестовый запрос'], sampling_params, use_tqdm=False, lora_request=self._get_lora_request())[0]
        self.vllm_adds_bos = vllm_response.prompt_token_ids[0] == self.tokenizer.bos_token_id
        self.logger.info(f'global_prefix = {self.global_prefix}')
        self.logger.info(f'vllm_adds_bos = {self.vllm_adds_bos}')

    def _load_plain_model(self, model_dir):
        self.model = vLLM(
            model=model_dir, device=self.device_map, #max_model_len=self.max_model_len,
            max_seq_len_to_capture=self.max_seq_len_to_capture, 
            gpu_memory_utilization=self.gpu_memory_utilization, max_logprobs=1000000,
            disable_sliding_window=self.disable_sliding_window, enable_prefix_caching=self.enable_prefix_caching, 
            trust_remote_code=self.trust_remote_code
        )

    def _load_lora(self, model_dir):
        # TODO: не работает с modules_to_save, и вообще пока не тестил
        config = PeftConfig.from_pretrained(model_dir)
        self.model = vLLM(
            model=config.base_model_name_or_path, device=self.device_map,
            max_model_len=self.max_model_len, max_seq_len_to_capture=self.max_seq_len_to_capture, 
            gpu_memory_utilization=self.gpu_memory_utilization, max_logprobs=1000000,
            disable_sliding_window=self.disable_sliding_window, enable_prefix_caching=self.enable_prefix_caching,
            enable_lora=True, trust_remote_code=self.trust_remote_code, max_lora_rank=self._get_max_lora_rank(config))

    def _get_lora_request(self):
        if not self.if_lora:
            return None
        return LoRARequest("lora", 1, self.model_name_or_path)

    def _get_max_lora_rank(self, lora_config):
        # TODO: случай с наличием не дефолтного ранга
        return lora_config.r

    def get_max_model_len(self):
        return min(self.model.llm_engine.model_config.max_model_len, self.max_seq_len_to_capture)