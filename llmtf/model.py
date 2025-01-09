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
import copy
import tqdm
import functools
from concurrent.futures import ThreadPoolExecutor
from llmtf.base import LLM
from llmtf.conversation import Conversation
from llmtf.utils import calculate_offset_mapping_llama3_workaround, add_tokens_with_logsoftmax_messages, convert_chat_template_to_conv_template
import re
try:
    from vllm import LLM as vLLM
    from vllm import SamplingParams
    from vllm.lora.request import LoRARequest
    from vllm.attention.backends.flash_attn import FlashAttentionBackend
except:
    pass

class ApiVLLMModel(LLM):
    def __init__(self, api_base, **kwargs):
        super().__init__(**kwargs)
        #requests.packages.urllib3.util.connection.HAS_IPV6 = False
        self.logger.info('ATTENTION! Hosting vLLM server must have vllm 0.6.3+')
        self.api_base = api_base
        self.num_procs = os.getenv('OPENAI_MAX_CONCURRENCY', 20)
        self.api_key = os.getenv('OPENAI_API_KEY', '123')
        self.model_name = None
        self.max_model_len = None
        self.generation_config = None

    def support_method(self, method):
        return method in ['generate']
    
    def from_pretrained(self, model_dir=None):
        url = self.api_base + '/v1/models'
        r = requests.get(url, headers={'Authorization': 'Bearer ' + self.api_key})
        if r.status_code != 200:
            print(r.text)
        assert r.status_code == 200

        data = r.json()
        if len(data['data']) == 1:
            self.model_name = data['data'][0]['id']
            self.max_model_len = data['data'][0]['max_model_len']
        else:
            data = [d for d in data['data'] if d['id'] == model_dir]
            assert len(data) == 1
            self.model_name = data[0]['id']
            self.max_model_len = data[0]['max_model_len']

        self.generation_config = GenerationConfig.from_dict({
            'repetition_penalty':  1.0,
            'temperature': 0.1,
            'top_p':  0.9,
            'top_k': 40,
            'max_new_tokens': 256,
            'do_sample': True
        })
        self.eos_token_ids_base = copy.deepcopy(self.generation_config.eos_token_id)
        self.stop_strings_base = copy.deepcopy(self.generation_config.stop_strings)

    def generate(self, messages, generation_config=None, incomplete_last_bot_message=True, return_tokens=False):
        if return_tokens:
            return NotImplementedError
        
        messages = self._preprocess_messages(messages)
        last_role = messages[-1]['role']

        generation_config = self.generation_config if generation_config is None else generation_config
        if generation_config.num_return_sequences > 1:
            return NotImplementedError
        
        r = requests.post(
            f'{self.api_base}/v1/chat/completions', 
            json={
                'messages': messages, 
                'model': self.model_name, 

                'max_tokens': generation_config.max_new_tokens,
                'temperature': generation_config.temperature if generation_config.do_sample else 0.0,
                'top_p': generation_config.top_p,
                'top_k': generation_config.top_k,
                'repetition_penalty': generation_config.repetition_penalty,
                'stop': generation_config.stop_strings,
                'n': generation_config.num_return_sequences if generation_config.num_return_sequences is not None else 1,

                'add_generation_prompt': last_role == 'user', 
                'skip_special_tokens': False, 
                'continue_final_message': incomplete_last_bot_message and last_role == 'assistant'
            },
            headers={'Authorization': 'Bearer ' + self.api_key}
        )
        if r.status_code != 200:
            print(r.text)
        assert r.status_code == 200

        data = r.json()
        outputs = data['choices'][0]['message']['content']
        if last_role == 'assistant':
            outputs = outputs[len(messages[-1]['content']):]
        
        info = {
            'prompt_len': data['usage']['prompt_tokens'], 
            'generated_len': [data['usage']['completion_tokens']], 
            'generated_cumulative_logprob': 'TODO: implement'
        }
        return messages, outputs, info

    def generate_batch(self, messages, generation_config=None, incomplete_last_bot_message=True, return_tokens=False):
        prompts, outputs, infos = [], [], []
        kwargs = {'generation_config': generation_config, 'incomplete_last_bot_message': incomplete_last_bot_message, 'return_tokens': return_tokens}
        with ThreadPoolExecutor(max_workers=self.num_procs) as p:
            partial_completion_helper = functools.partial(self.generate, **kwargs)
            res = list(
                tqdm.tqdm(
                    p.map(partial_completion_helper, messages),
                    desc="prompt_batches",
                    total=len(messages),
                    disable=True
                )
            )
            prompts, outputs, infos = list(zip(*res))
        return prompts, outputs, infos


    def add_stop_strings(self, stop_strings):
        for stop_string in stop_strings:
            self._add_stop_string(stop_string)

        self.logger.info(f'Updated generation_config.eos_token_id: {self.generation_config.eos_token_id}')
        self.logger.info(f'Updated generation_config.stop_strings: {self.generation_config.stop_strings}') 

    def _add_stop_string(self, stop_string):
        self.generation_config.stop_strings.append(stop_string)

    def reset_stop_strings(self):
        self.generation_config.eos_token_id = copy.deepcopy(self.eos_token_ids_base)
        self.generation_config.stop_strings = copy.deepcopy(self.stop_strings_base)
    
    def calculate_tokens_proba(self, messages, tokens_of_interest, incomplete_last_bot_message=True):
        messages = self._preprocess_messages(messages)
        last_role = messages[-1]['role']
        
        r = requests.post(
            f'{self.api_base}/v1/chat/completions', 
            json={
                'messages': messages, 
                'model': self.model_name, 
                'max_tokens': 1,
                'temperature': 0.0,
                'add_generation_prompt': last_role == 'user', 
                'skip_special_tokens': False, 
                'continue_final_message': incomplete_last_bot_message and last_role == 'assistant',
                'logprobs': True,
                'top_logprobs': 20
            },
            headers={'Authorization': 'Bearer ' + self.api_key}
        )
        if r.status_code != 200:
            print(r.text)
        assert r.status_code == 200

        data = r.json()
        logprobs = data['choices'][0]['logprobs']['content'][0]['top_logprobs']
        probs = {lp['token']: np.exp(lp['logprob']) for lp in logprobs}
        probs = {token: probs.get(token, 0.0) for token in tokens_of_interest}
            

        info = {
            'generated_len': 1, 
            'generated_token': data['choices'][0]['logprobs']['content'][0]['token']
        }
        return messages, probs, info
            
    
    def calculate_tokens_proba_batch(self, messages, tokens_of_interest, incomplete_last_bot_message=True):

        '''
        prompts, outputs, infos = [], [], []
        for _messages, _tokens_of_interest in zip(messages, tokens_of_interest):
            prompt, output, info = self.calculate_tokens_proba(_messages, _tokens_of_interest, incomplete_last_bot_message)
            prompts.append(prompt)
            outputs.append(output)
            infos.append(info)

        return prompts, outputs, infos
        '''
        prompts, outputs, infos = [], [], []
        kwargs = { 'incomplete_last_bot_message': incomplete_last_bot_message}
        with ThreadPoolExecutor(max_workers=self.num_procs) as p:
            partial_completion_helper = functools.partial(self.calculate_tokens_proba, **kwargs)
            res = list(
                tqdm.tqdm(
                    p.map(partial_completion_helper, messages, tokens_of_interest),
                    desc="prompt_batches",
                    total=len(messages),
                    disable=True
                )
            )
            prompts, outputs, infos = list(zip(*res))
        return prompts, outputs, infos

    def _preprocess_messages(self, messages):
        _messages = []
        for m in messages:
            if m['role'] == 'user':
                _messages.append({'role': m['role'], 'content': m['content']})
            elif m['role'] == 'bot':
                _messages.append({'role': 'assistant', 'content': m['content']})
            elif m['role'] == 'system':
                _messages.append({'role': m['role'], 'content': m['content']})
            elif m['role'] == 'assistant':
                _messages.append({'role': m['role'], 'content': m['content']})
            else:
                role = m['role']
                raise Exception(f'Unknown role {role}')
            
        assert _messages[-1]['role'] in ['assistant', 'user']
        return _messages
    
    def apply_model_prompt(self, messages, incomplete_last_bot_message=True):
        # Only for count_tokens_for_prompt method, not for any other use
        # TODO fix
        return 0
        _messages = self._preprocess_messages(messages)
        last_role = _messages[-1]['role']
        r = requests.post(
            f'{self.api_base}/v1/chat/completions', 
            json={
                'messages': _messages, 
                'model': self.model_name, 
                'max_tokens': 1, 
                'add_generation_prompt': last_role == 'user', 
                'skip_special_tokens': False, 
                'continue_final_message': incomplete_last_bot_message and last_role == 'assistant', 
            }
        )
        if r.status_code != 200:
            tokens =  None
            if 'Please reduce the length of the messages or completion.' in r.text:
                try:
                    tokens = int(re.findall('However, you requested (.*?) tokens', '''{"object":"error","message":"This model's maximum context length is 1600 tokens. However, you requested 1914 tokens (1913 in the messages, 1 in the completion). Please reduce the length of the messages or completion.","type":"BadRequestError","param":null,"code":400}''')[0]) - 1
                except:
                    pass
            if tokens is not None:
                return tokens
        assert r.status_code == 200
        data = r.json()
        return data['usage']['prompt_tokens']

    def count_tokens_for_prompt(self, prompt_tokens):
        assert type(prompt_tokens) == int
        return prompt_tokens

    def get_params(self):
        return {
            'model_name_or_path': self.model_name,
            'api_base': self.api_base,
            'generation_config': json.loads(self.generation_config.to_json_string(use_diff=True)),
            'max_model_len': self.get_max_model_len()
        }

    def get_max_model_len(self):
        return self.max_model_len

class LocalHostedLLM(LLM):
    def support_method(self, method):
        return method in ['generate', 'calculate_tokens_proba', 'calculate_logsoftmax']

    def from_pretrained(self, model_dir):
        self._load_model(model_dir)
        #self._check_if_leading_space()
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

        if self.conversation_template == 'auto':
            self.conversation_template = convert_chat_template_to_conv_template(self.tokenizer)

        self.tokenizer.truncation_side = 'left'
        self.tokenizer.padding_side = 'left' #TODO: а нужно ли это вообще? нужно перепроверить имплементации.
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        try:
            self.generation_config = GenerationConfig.from_pretrained(model_dir, trust_remote_code=self.trust_remote_code)
        except:
            self.generation_config = GenerationConfig.from_dict({})
        
        self._init_default_gen_params()
        self._check_if_leading_space()
        self._override_eos_token_conv_template()

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
            elif m['role'] == 'system':
                conv.add_system_message(m['content'])
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
            #print(token_str)
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
        assert len(tokens_of_interest_ids) == len(set(tokens_of_interest_ids))
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
        self.generation_config.num_beams = 1
        self.generation_config.stop_strings = []

    def _override_eos_token_conv_template(self):
        eos_token_from_conv = self.conversation_template.get('eos_token', [])

        assert type(eos_token_from_conv) in [str, list]
        if type(eos_token_from_conv) == str:
            eos_token_from_conv = [eos_token_from_conv]
        
        if type(self.generation_config.eos_token_id) == int:
            self.generation_config.eos_token_id = [self.generation_config.eos_token_id]

        #gen_config_eos_token_id = copy.deepcopy(self.generation_config.eos_token_id)
        for eos_token in eos_token_from_conv:
            if eos_token is not None and len(eos_token) > 0:
                self._add_stop_string(eos_token)
                '''
                is_token, eos_token_id = self._check_word_is_token(eos_token)
                if not is_token:
                    self.logger.warning(f'Provided eos_token in conv template is not token, but sequence of tokens {eos_token_id}. It cannot be added to eos_token_id, but results will be truncated using {eos_token}')
                    self._add_stop_string(eos_token)
                else:
                    assert len(eos_token_id) == 1 and eos_token_id[0] != None
                    eos_token_id = eos_token_id[0]
                    
                    if eos_token_id not in self.generation_config.eos_token_id:
                        self.generation_config.eos_token_id.append(eos_token_id)
                '''
        self.logger.info(f'Set eos_token_id in generation_config to {self.generation_config.eos_token_id}')

        global_prefix = self.conversation_template.get('global_prefix', None)
        if global_prefix is None:
            global_prefix = ''
            if self.tokenizer.bos_token_id is not None:
                global_prefix = self.tokenizer.decode([self.tokenizer.bos_token_id])
            self.conversation_template['global_prefix'] = global_prefix
            self.logger.info(f'Set global prefix {global_prefix} to conv config')

        if len(global_prefix) == 0:
            self.logger.warning(f'Global prefix is equal to empty string!')

        self.eos_token_ids_base = copy.deepcopy(self.generation_config.eos_token_id)
        self.stop_strings_base = copy.deepcopy(self.generation_config.stop_strings)

    def add_stop_strings(self, stop_strings):
        for stop_string in stop_strings:
            self._add_stop_string(stop_string)

        self.logger.info(f'Updated generation_config.eos_token_id: {self.generation_config.eos_token_id}')
        self.logger.info(f'Updated generation_config.stop_strings: {self.generation_config.stop_strings}') 

    def _add_stop_string(self, stop_string):
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

class HFModel(LocalHostedLLM):
    def __init__(
            self, conversation_template_path='auto', 
            load_in_8bit=False, 
            torch_dtype='auto', device_map='auto', 
            attn_implementation="flash_attention_2", use_fast_tokenizer=True, 
            trust_remote_code=False, alpha_scale=1.0, not_scale_lm_head=False,
            **kwargs
        ):
        super().__init__(**kwargs)
        self.load_in_8bit = load_in_8bit
        self.torch_dtype = torch_dtype
        self.attn_implementation = attn_implementation
        self.device_map = device_map
        self.use_fast_tokenizer = use_fast_tokenizer
        self.trust_remote_code=trust_remote_code
        self.alpha_scale = alpha_scale
        self.not_scale_lm_head = not_scale_lm_head

        if conversation_template_path != 'auto':
            with codecs.open(conversation_template_path, 'r', 'utf-8') as file:
                template = json.load(file)
            self.conversation_template = template
        else:
            self.conversation_template = 'auto'

    def get_params(self):
        return {
            'model_name_or_path': self.model_name_or_path,
            'generation_config': json.loads(self.generation_config.to_json_string(use_diff=True)),
            'conversation_template': self.conversation_template,
            'load_in_8bit': self.load_in_8bit,
            'torch_dtype': str(self.torch_dtype),
            'attn_implementation': self.attn_implementation,
            'device_map': self.device_map,
            'use_fast_tokenizer': self.use_fast_tokenizer,
            'leading_space': self.leading_space,
            'space_token': self.space_token,
            'trust_remote_code': self.trust_remote_code,
            'max_model_len': self.get_max_model_len()
        }
        
    def generate(self, messages, generation_config=None, incomplete_last_bot_message=True, return_tokens=False):
        prompts, outputs, infos = self.generate_batch([messages], generation_config=generation_config, incomplete_last_bot_message=incomplete_last_bot_message, return_tokens=return_tokens)
        return prompts[0], outputs[0], infos[0]

    def generate_batch(self, messages, generation_config=None, incomplete_last_bot_message=True, return_tokens=False):
        generation_config = self.generation_config if generation_config is None else generation_config
        prompts = []
        for _messages in messages:
            prompts.append(self.apply_model_prompt(_messages, incomplete_last_bot_message=incomplete_last_bot_message))
        data = self.tokenizer(
            prompts,
            return_tensors="pt",
            truncation=True,
            add_special_tokens=self.conversation_template['add_special_tokens'], 
            max_length=generation_config.max_length,
            padding=True
        )
        data = {k: v.to(self.model.device) for k, v in data.items()}

        #TODO: upgrade to 4.40+ version with propper testing
        stop_strings = generation_config.stop_strings
        generation_config.stop_strings = None
        with torch.no_grad():
            output_ids = self.model.generate(
                **data,
                generation_config=generation_config
            )
        generation_config.stop_strings = stop_strings
        output_ids = output_ids.view(len(messages), -1, output_ids.shape[-1])
        
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
                    eos_tokens = [self.tokenizer.eos_token_id]
                    if generation_config.eos_token_id is not None:
                        eos_tokens += generation_config.eos_token_id if type(generation_config.eos_token_id) == list else [generation_config.eos_token_id]
                    eos_tokens = list(set(eos_tokens))
                    for eos_token in eos_tokens:
                        if eos_token in generated_ids:
                            generated_ids = generated_ids[:generated_ids.index(eos_token)]

                    #TODO: better stop strings tructation. 
                    generated_tokens = [self.tokenizer.convert_tokens_to_string([t]) for t in self.tokenizer.convert_ids_to_tokens(generated_ids)]
                    for stop_string in generation_config.stop_strings:
                        if stop_string in ''.join(generated_tokens):
                            for token_i, token in enumerate(generated_tokens):
                                if stop_string in token:
                                    generated_tokens = generated_tokens[:token_i]
                                    break
                    if len(generated_tokens) != len(generated_ids):
                        generated_ids = generated_ids[:len(generated_tokens)]
                        
                    sample_output_all.append({'tokens': generated_ids, 'text': self.tokenizer.decode(generated_ids, skip_special_tokens=True)})
                else:
                    sample_output = self.tokenizer.decode(sample_output_ids, skip_special_tokens=True)
                    for stop_string in generation_config.stop_strings:
                        if stop_string in sample_output:
                            sample_output = sample_output[:sample_output.find(stop_string)]
                    sample_output_all.append(sample_output)
                generated_len.append(len(sample_output_ids))

            if len(sample_output_all) == 1:
                sample_output_all = sample_output_all[0]

            outputs.append(sample_output_all)
            infos.append(
                {
                    'prompt_len': prompt_len, 
                    'generated_len': generated_len, 
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
        for batch_idx in range(next_token_logits_batch.shape[0]):
            next_token_logits = next_token_logits_batch[batch_idx]
            next_token_logits = next_token_logits.flatten()
            assert next_token_logits.shape == torch.Size((self.model.config.vocab_size, ))

            next_token_probs = torch.nn.functional.softmax(next_token_logits, dim=-1).cpu()
            assert torch.isclose(next_token_probs.sum(), torch.tensor(1.0).to(next_token_probs.dtype), atol=1e-03)
        
            probs = next_token_probs[tokens_of_interest_ids_batch[batch_idx]].tolist()
            probs = dict(zip(tokens_of_interest[batch_idx], probs))
            probs_batch.append(probs)

            infos.append(
                {
                    'prompt_len': int(data["attention_mask"][batch_idx].cpu().detach().sum()), 
                    'generated_len': 1, 
                    'generated_cumulative_logprob': 'TODO: calculate for hf model', 
                    'generated_token': self.tokenizer.decode([next_token_probs.argmax()])
                }
            )

        return prompts_batch, probs_batch, infos
    
    def calculate_logsoftmax_batch(self, messages, incomplete_last_bot_message=True, log_only_last=True):
        ## TODO: transformers 4.38.2 will be ok for llama3
        prompts = []
        for _messages in messages:
            prompts.append(self.apply_model_prompt(_messages, incomplete_last_bot_message=incomplete_last_bot_message))

        data = self.tokenizer(
            prompts, return_tensors="pt", truncation=True, padding=True,
            add_special_tokens=self.conversation_template['add_special_tokens'], 
            max_length=self.generation_config.max_length, return_offsets_mapping=True
        )
        #print(data['input_ids'].shape)
        offset_mapping = data.pop('offset_mapping').tolist()
        #if 'llama3' in self.model.config._name_or_path.lower() or 'llama-3' in self.model.config._name_or_path.lower() or os.environ.get('FORCE_CALCULATE_OFFSET_MAPPING_CUSTOM', False):
        #    offset_mapping = calculate_offset_mapping_llama3_workaround(prompts, data['input_ids'], self.tokenizer)

        model_input = {k: v.clone().to(self.model.device) for k, v in data.items()}
        with torch.no_grad():
            outputs = self.model(**model_input).logits
            logsoftmax_batch = torch.nn.LogSoftmax(dim=-1)(outputs)
        
        labels = model_input['input_ids'][:,1:]
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
                    'generated_cumulative_logprob': 'TODO: calculate for hf model', 
                }
            )

        add_tokens_with_logsoftmax_messages(messages, prompts, tokens_with_logsoftmax, log_only_last)
        return prompts, messages, infos

    def _load_plain_model(self, model_dir):
        base_model_config = AutoConfig.from_pretrained(model_dir, trust_remote_code=self.trust_remote_code)
        torch_dtype = base_model_config.torch_dtype if self.torch_dtype == 'auto' else self.torch_dtype
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=torch_dtype,
            load_in_8bit=self.load_in_8bit,
            device_map=self.device_map,
            attn_implementation=self.attn_implementation, 
            trust_remote_code=self.trust_remote_code
        )
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

        self.model = AutoModelForCausalLM.from_pretrained(
            config.base_model_name_or_path,
            load_in_8bit=self.load_in_8bit,
            torch_dtype=torch_dtype,
            device_map=self.device_map,
            attn_implementation=self.attn_implementation, 
            trust_remote_code=self.trust_remote_code
        )
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

    def get_max_model_len(self):
        return self.model.config.max_position_embeddings

class VLLMModel(LocalHostedLLM):
    def __init__(
            self, 
            conversation_template_path='auto', 
            use_fast_tokenizer=True, 
            device_map='auto',
            max_seq_len_to_capture=4096,
            gpu_memory_utilization=0.9,
            disable_sliding_window=True,
            enable_prefix_caching=True,
            trust_remote_code=False,
            calculate_tokens_proba_logprobs_count=50,
            **kwargs
        ):
        super().__init__(**kwargs)
        self.use_fast_tokenizer = use_fast_tokenizer
        self.device_map = device_map
        self.max_seq_len_to_capture = max_seq_len_to_capture 
        self.gpu_memory_utilization = gpu_memory_utilization
        self.disable_sliding_window = disable_sliding_window
        self.enable_prefix_caching = enable_prefix_caching
        self.trust_remote_code = trust_remote_code
        self.calculate_tokens_proba_logprobs_count = calculate_tokens_proba_logprobs_count

        assert 'CUDA_VISIBLE_DEVICES' in os.environ
        self.logger.info('CUDA_VISIBLE_DEVICES=' + os.environ['CUDA_VISIBLE_DEVICES'])
        self.logger.info('device_map=' + self.device_map)

        if conversation_template_path != 'auto':
            with codecs.open(conversation_template_path, 'r', 'utf-8') as file:
                template = json.load(file)
            self.conversation_template = template
        else:
            self.conversation_template = 'auto'

    def support_method(self, method):
        return method in ['generate', 'calculate_tokens_proba']
    
    def from_pretrained(self, model_dir):
        self._load_model(model_dir)
        #self._check_if_leading_space()
        self._conv_template_bos_vllm_test()
        self.reset_stop_strings()
        self.logger.info(f'Leading space: {self.leading_space}')
        #self.logger.info(f'For calculate_tokens_proba batch always will be 1 because of possible errors in logprobs') # TODO: verify

        tokenizer = self.model.get_tokenizer()
        tokenizer.pad_token_id = self.tokenizer.pad_token_id
        #tokenizer.padding_side = self.tokenizer.padding_side ?????
        tokenizer.truncation_side = self.tokenizer.truncation_side

        self.attn_backend = self.model.llm_engine.model_executor.driver_worker.model_runner.attn_backend
        self.special_attn_warning_complete = False

    def generate(self, messages, generation_config=None, incomplete_last_bot_message=True, return_tokens=False):
        prompts, outputs, infos = self.generate_batch([messages], generation_config=generation_config, incomplete_last_bot_message=incomplete_last_bot_message, return_tokens=return_tokens)
        return prompts[0], outputs[0], infos[0]

    def generate_batch(self, messages, generation_config=None, incomplete_last_bot_message=True, return_tokens=False):
        prompts_tokens_batch = []
        for _messages in messages:
            prompt = self.apply_model_prompt(_messages, incomplete_last_bot_message=incomplete_last_bot_message)
            prompts_tokens_batch.append(self.tokenizer(prompt, add_special_tokens=self.conversation_template['add_special_tokens'], truncation=True, max_length=self.generation_config.max_length)['input_ids'])

        generation_config = self.generation_config if generation_config is None else generation_config
        sampling_params = SamplingParams(
            temperature=generation_config.temperature,
            top_p=generation_config.top_p,
            top_k=generation_config.top_k,
            max_tokens=generation_config.max_new_tokens,
            repetition_penalty=generation_config.repetition_penalty,
            stop=generation_config.stop_strings,
            n=generation_config.num_return_sequences
        )

        prompts_vllm = []
        outputs = []
        infos = []
        
        vllm_responses = self.model.generate(prompt_token_ids=prompts_tokens_batch, sampling_params=sampling_params, use_tqdm=False, lora_request=self._get_lora_request())
        for response in vllm_responses:
            infos.append(
                {
                    'prompt_len': len(response.prompt_token_ids), 
                    'generated_len': [len(out.token_ids) for out in response.outputs], 
                    'generated_cumulative_logprob': [out.cumulative_logprob for out in response.outputs]
                }
            )
            prompts_vllm.append(self.tokenizer.decode(response.prompt_token_ids))

            generated = [{'tokens': out.token_ids, 'text': out.text} if return_tokens else out.text for out in response.outputs]
            if len(generated) == 1:
                outputs.append(generated[0])
            else:
                outputs.append(generated)

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
            prompts_tokens_batch.append(self.tokenizer(prompt, add_special_tokens=self.conversation_template['add_special_tokens'], truncation=True, max_length=self.generation_config.max_length)['input_ids'])
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

    def calculate_logsoftmax_batch(self, messages, incomplete_last_bot_message=True, log_only_last=True):
        # BUGGED https://github.com/vllm-project/vllm/pull/5355
        prompts_tokens_batch = []
        prompts = []
        offset_mapping = []
        for _messages in messages:
            prompt = self.apply_model_prompt(_messages, incomplete_last_bot_message=incomplete_last_bot_message)
            data = self.tokenizer(prompt, add_special_tokens=self.conversation_template['add_special_tokens'], truncation=True, max_length=self.generation_config.max_length, return_offsets_mapping=True)

            prompts.append(prompt)
            prompts_tokens_batch.append(data['input_ids'])
            offset_mapping.append(data['offset_mapping'])

        #config = self.model.llm_engine.get_model_config().hf_config
        #if 'llama3' in config._name_or_path.lower() or 'llama-3' in config._name_or_path.lower() or os.environ.get('FORCE_CALCULATE_OFFSET_MAPPING_CUSTOM', False):
        #    offset_mapping = calculate_offset_mapping_llama3_workaround(prompts, prompts_tokens_batch, self.tokenizer)

        sampling_params = SamplingParams(
            temperature=0,
            prompt_logprobs=self.calculate_tokens_proba_logprobs_count,
            logprobs=self.calculate_tokens_proba_logprobs_count,
            max_tokens=1,
            repetition_penalty=1.0
        )

        tokens_with_logsoftmax = []
        infos = []
        vllm_responses = self.model.generate(prompt_token_ids=prompts_tokens_batch, sampling_params=sampling_params, use_tqdm=False, lora_request=self._get_lora_request())
        for i, response in enumerate(vllm_responses):
            infos.append(
                {
                    'prompt_len': len(response.prompt_token_ids), 
                    'generated_len': len(response.outputs[0].token_ids), 
                    'generated_cumulative_logprob': response.outputs[0].cumulative_logprob, 
                    'generated_token': None
                }
            )

            prompt_logsoftmax = []
            for j, token in enumerate(response.prompt_token_ids):
                if j == 0:
                    logsoftmax = 0.0
                else:
                    logsoftmax = response.prompt_logprobs[j][token].logprob
                prompt_logsoftmax.append([token, logsoftmax, offset_mapping[i][j]])
            tokens_with_logsoftmax.append(prompt_logsoftmax)
        add_tokens_with_logsoftmax_messages(messages, prompts, tokens_with_logsoftmax, log_only_last)
        return prompts, messages, infos

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
            max_model_len=self.max_seq_len_to_capture, max_seq_len_to_capture=self.max_seq_len_to_capture,
            gpu_memory_utilization=self.gpu_memory_utilization, max_logprobs=1000000,
            disable_sliding_window=self.disable_sliding_window, enable_prefix_caching=self.enable_prefix_caching, 
            trust_remote_code=self.trust_remote_code#, tensor_parallel_size=2,
            #rope_scaling='{"type": "extended", "factor": 8.0}'
        )

    def _load_lora(self, model_dir):
        # TODO: не работает с modules_to_save, и вообще пока не тестил
        config = PeftConfig.from_pretrained(model_dir)
        self.model = vLLM(
            model=config.base_model_name_or_path, device=self.device_map,
            max_model_len=self.max_seq_len_to_capture, max_seq_len_to_capture=self.max_seq_len_to_capture,
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