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


DEFAULT_MESSAGE_TEMPLATE = "<s>{role}\n{content}</s>\n"
DEFAULT_SYSTEM_PROMPT = "Ты — Сайга, русскоязычный автоматический ассистент. Ты разговариваешь с людьми и помогаешь им."


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
        self.messages.append({
            "role": self.system_role,
            "content": message
        })

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

    def get_prompt(self, tokenizer, max_tokens: int = None, add_suffix: bool = True, incomplete_last_bot_message: bool = False):
        messages = self.messages
        if max_tokens is not None:
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


class LLM(abc.ABC):
    def __init__(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    @abstractmethod
    def generate(self, **kwargs):
        pass

    @abstractmethod
    def generate_batch(self, **kwargs):
        pass

    @abstractmethod
    def calculate_token_interest_probs(self, **kwargs):
        pass

    @abstractmethod
    def calculate_token_interest_probs_batch(self, **kwargs):
        pass

    @abstractmethod
    def apply_model_prompt(self, **kwargs):
        pass

    @abstractmethod
    def support_method(self, **kwargs):
        pass
    
    @abstractmethod
    def count_tokens_for_prompt(self, **kwargs):
        pass

class HuggingFaceLLM(LLM):
    def __init__(self, conversation_template_path, load_in_8bit=False, load_in_4bit=False, torch_dtype='auto', device_map='auto', use_flash_attention_2=True, use_fast_tokenizer=True):
        super().__init__()
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit and not load_in_8bit
        self.torch_dtype = torch_dtype
        self.use_flash_attention_2 = use_flash_attention_2
        self.device_map = device_map
        self.use_fast_tokenizer = use_fast_tokenizer

        with codecs.open(conversation_template_path, 'r', 'utf-8') as file:
            template = json.load(file)
        self.conversation_template = template

    def support_method(self, method):
        return method in ['generate', 'calculate_token_interest_probs']

    def from_pretrained(self, model_dir):
        self._load_model(model_dir)
        self._check_if_leading_space()
        print(f'Leading space: {self.leading_space}')

        self.tokenizer.truncation_side = 'left'
        self.tokenizer.padding_side = 'left'
        
        eos_token = self.conversation_template.get('eos_token', self.tokenizer.eos_token_id)
        eos_token_id = eos_token
        if type(eos_token) == str:
            eos_token_id = self.tokenizer.encode(eos_token, add_special_tokens=False)
            if len(eos_token_id) == 2 and self.leading_space and eos_token_id[0] == self.tokenizer.convert_tokens_to_ids([' '])[0]:
                eos_token_ids = eos_token_ids[1:]
            assert len(eos_token_id) == 1
            eos_token_id = eos_token_id[0]

        #eos_token_id = self.tokenizer.convert_tokens_to_ids([eos_token])[0] if type(eos_token) == str else eos_token
        #print(self.tokenizer.convert_tokens_to_ids([eos_token]))
        #print(self.tokenizer.encode(eos_token, add_special_tokens=False))
        eos_token_id = self.tokenizer.encode(eos_token, add_special_tokens=False)[0]


        #self.tokenizer.pad_token = eos_token
        self.tokenizer.pad_token_id = eos_token_id

        self.generation_config.bos_token_id = self.tokenizer.bos_token_id
        self.generation_config.eos_token_id = [eos_token_id, self.tokenizer.encode('\n', add_special_tokens=False)[-1]]
        #print(self.generation_config.eos_token_id)
        self.generation_config.pad_token_id = eos_token_id
        self.generation_config.do_sample = True
        self.generation_config.max_new_tokens = 256
        self.generation_config.max_length = 2048
        self.generation_config.repetition_penalty = 1.0
        self.generation_config.temperature = 0.1
        self.generation_config.top_k = 40
        self.generation_config.top_p = 0.9

        print(self.generation_config)

        
        #print(self.tokenizer.convert_ids_to_tokens([270, 276]))
        #print(self.tokenizer.convert_ids_to_tokens([28740, 28750]))

    def load_from_mem(self, model, tokenizer, generation_config):
        self.model = model
        self.model.eval()

        self.tokenizer = tokenizer
        self.tokenizer.truncation_side = 'left'
        self.tokenizer.padding_side = 'left'

        self.generation_config = generation_config

    def _check_if_leading_space(self):
        self.leading_space = False
        char = '1'
        tokens = self.tokenizer(char, add_special_tokens=False)['input_ids']
        if len(tokens) > 1:
            self.leading_space = True
        else:
            if len(model.tokenizer.convert_ids_to_tokens(tokens)[0]) != 1:
                self.leading_space = True
        


    def generate(self, messages, generation_config=None, incomplete_last_bot_message=True):
        prompt = self.apply_model_prompt(messages, incomplete_last_bot_message=incomplete_last_bot_message)
        #print(f'INPUT: {prompt}')
        data = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            add_special_tokens=self.conversation_template['add_special_tokens'], 
            max_length=self.generation_config.max_length
        )
        data = {k: v.to(self.model.device) for k, v in data.items()}
        with torch.no_grad():
            output_ids = self.model.generate(
                **data,
                generation_config=self.generation_config if generation_config is None else generation_config
            )
        outputs = []
        for sample_output_ids, sample_input_ids in zip(output_ids, data["input_ids"]):
            sample_output_ids = sample_output_ids[len(sample_input_ids):]
            sample_output = self.tokenizer.decode(sample_output_ids, skip_special_tokens=True)
            outputs.append(sample_output)
        
        if len(outputs) == 1:
            outputs = outputs[0]

        return prompt, outputs

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
        for sample_output_ids, sample_input_ids in zip(output_ids, data["input_ids"]):
            sample_output_ids = sample_output_ids[len(sample_input_ids):]
            sample_output = self.tokenizer.decode(sample_output_ids, skip_special_tokens=True)
            outputs.append(sample_output)
        
        if len(outputs) == 1:
            outputs = outputs[0]

        return prompts, outputs
    
    def calculate_token_interest_probs(self, messages, tokens_of_interest, incomplete_last_bot_message=True):
        prompt, tokens_of_interest_ids = self._get_tokens_of_interest_ids_modify_prompt(messages, tokens_of_interest, incomplete_last_bot_message)
        data = self.tokenizer(prompt, return_tensors="pt", truncation=True, add_special_tokens=self.conversation_template['add_special_tokens'], max_length=self.generation_config.max_length)
        data = {k: v.to(self.model.device) for k, v in data.items()}
    
        with torch.no_grad():
            outputs = self.model(**data)
        logits = outputs.logits  # shape (batch_size, sequence_length, vocab_size)
        next_token_logits = logits[:, -1, :]  # shape (batch_size, vocab_size)

        next_token_logits = next_token_logits.flatten()
        assert next_token_logits.shape == torch.Size((self.model.config.vocab_size, ))

        next_token_probs = torch.nn.functional.softmax(next_token_logits, dim=-1).cpu()  # all probs over vocab
        assert torch.isclose(next_token_probs.sum(), torch.tensor(1.0).to(next_token_probs.dtype), atol=1e-03)  # dtype for half/nothalf, -03 for float16
        probs = next_token_probs[tokens_of_interest_ids].tolist()
        probs = dict(zip(tokens_of_interest, probs))
        return prompt, probs
    
    def calculate_token_interest_probs_batch(self, messages, tokens_of_interest, incomplete_last_bot_message=True):
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

        #print(next_token_logits.shape)
        probs_batch = []
        for i in range(next_token_logits_batch.shape[0]):
            next_token_logits = next_token_logits_batch[i]
            next_token_logits = next_token_logits.flatten()
            assert next_token_logits.shape == torch.Size((self.model.config.vocab_size, ))

            next_token_probs = torch.nn.functional.softmax(next_token_logits, dim=-1).cpu()  # all probs over vocab
            assert torch.isclose(next_token_probs.sum(), torch.tensor(1.0).to(next_token_probs.dtype), atol=1e-03)  # dtype for half/nothalf, -03 for float16
        
            probs = next_token_probs[tokens_of_interest_ids_batch[i]].tolist()
            probs = dict(zip(tokens_of_interest[i], probs))
            probs_batch.append(probs)
        return prompts_batch, probs_batch
    
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
    
    def _load_model(self, model_dir):
        if self._check_if_lora(model_dir):
            self._load_lora(model_dir)
            self.model = self.model.merge_and_unload()
            self.model.train(False)
        else:
            self._load_plain_model(model_dir)

        self.logger.info(f"Model id: {model_dir}, params: {self.model.num_parameters()}, dtype: {self.model.dtype}")

    def _check_if_lora(self, model_dir):
        if os.path.exists(model_dir):
            adapter_config_exists = os.path.exists(os.path.join(model_dir, 'adapter_config.json'))
            adapter_model_exists = os.path.exists(os.path.join(model_dir, 'adapter_model.bin')) or os.path.exists(os.path.join(model_dir, 'adapter_model.safetensors'))

            return adapter_config_exists and adapter_model_exists

        if_lora = False
        try:
            PeftConfig.from_pretrained(model_dir)
            if_lora = True
        except:
            pass
        return if_lora 

    def _load_plain_model(self, model_dir):
        base_model_config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
        torch_dtype = base_model_config.torch_dtype if self.torch_dtype == 'auto' else self.torch_dtype
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=torch_dtype,
            load_in_8bit=self.load_in_8bit,
            device_map=self.device_map,
            use_flash_attention_2=self.use_flash_attention_2
        )
        self.model.eval()
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=self.use_fast_tokenizer)
        except:
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=not self.use_fast_tokenizer)

        try:
            self.generation_config = GenerationConfig.from_pretrained(model_dir)
        except:
            self.generation_config = GenerationConfig.from_dict({})

        #self.logger.info(f"Model id: {model_dir}, params: {self.model.num_parameters()}, dtype: {self.model.dtype}")

    def _load_lora(self, model_dir):
        config = PeftConfig.from_pretrained(model_dir)
        base_model_config = AutoConfig.from_pretrained(config.base_model_name_or_path)
        torch_dtype = base_model_config.torch_dtype if self.torch_dtype == 'auto' else self.torch_dtype

        self.model = AutoModelForCausalLM.from_pretrained(
            config.base_model_name_or_path,
            load_in_8bit=self.load_in_8bit,
            torch_dtype=torch_dtype,
            device_map=self.device_map,
            use_flash_attention_2=self.use_flash_attention_2
        )
        self.model = PeftModel.from_pretrained(
            self.model,
            model_dir,
            torch_dtype=torch_dtype
        )
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=self.use_fast_tokenizer)
        self.generation_config = GenerationConfig.from_pretrained(model_dir)

        #self.logger.info(f"Model id: {model_dir}, params: {self.model.num_parameters()}, dtype: {self.model.dtype}")
