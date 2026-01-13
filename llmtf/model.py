from abc import abstractmethod
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
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from llmtf.base import LLM
from llmtf.utils import calculate_offset_mapping_llama3_workaround, add_tokens_with_logsoftmax_messages, json_to_jinja
import re
try:
    from vllm import LLM as vLLM
    from vllm import SamplingParams
    from vllm.lora.request import LoRARequest
    from vllm.attention.backends.flash_attn import FlashAttentionBackend
except:
    pass


class ReasoningModel():
    def reasoning_from_pretrained(
        self,
        max_new_tokens_reasoning=None,
        reasoning_truncing_prompt="\n…the rest of the reasoning chain is hidden due to limit on the length of the thoughts.\n",
        end_thinking_token_id=None
    ):
        if max_new_tokens_reasoning:
            self.generation_config.max_new_tokens_reasoning = max_new_tokens_reasoning
        else:
            self.generation_config.max_new_tokens_reasoning = self.generation_config.max_new_tokens
        self.generation_config.reasoning_truncing_prompt = reasoning_truncing_prompt
        if not hasattr(self.generation_config, "stop_strings") or self.generation_config.stop_strings == None:
            self.generation_config.stop_strings = []
        if end_thinking_token_id:
            self.generation_config.end_thinking_token_id=end_thinking_token_id
        else:
            self.get_end_thinking_token_id()
            

    @abstractmethod
    def get_end_thinking_token_id(self):
        pass

    def prepare_generation_config(self, old_params=None):
        if old_params:
            if "eos_token_id" in old_params.keys():
                self.generation_config.eos_token_id = old_params["eos_token_id"]
            self.generation_config.max_new_tokens = old_params["max_new_tokens"]
            self.generation_config.stop_strings = old_params["stop_strings"]
        else:
            old_params = {}
            if hasattr(self.generation_config, "stop_token_ids"):
                old_params["stop_token_ids"] = self.generation_config.stop_token_ids
            if hasattr(self.generation_config, "eos_token_id"):
                old_params["eos_token_id"] = self.generation_config.eos_token_id
            old_params["max_new_tokens"] = self.generation_config.max_new_tokens
            old_params["stop_strings"] = self.generation_config.stop_strings
            
            if "eos_token_id" in old_params.keys():
                self.generation_config.eos_token_id = [self.generation_config.end_thinking_token_id]
            self.generation_config.max_new_tokens = self.generation_config.max_new_tokens_reasoning
            self.generation_config.stop_strings = self.generation_config.stop_strings + ["</think>"]
            return old_params
    
    @abstractmethod
    def _generate_batch(
        self,
        messages,
        generation_config,
        incomplete_last_bot_message,
        return_tokens,
        include_stop_str_in_output,
        enable_thinking,
        skip_special_tokens,
        **kwargs
    ):
        pass

    def _generate_reasoning(
        self,
        messages,
        generation_config=None,
        incomplete_last_bot_message=True,
        return_tokens=False,
        include_stop_str_in_output=False,
        add_assistant_prompt_to_output=True,
        skip_special_tokens=True,
        enable_thinking=True,
        add_reasoning_truncing_prompt=False,
        add_reasoning_info=True,
        **kwargs
    ):
        prompts, outputs, infos = self.generate_batch(
            [messages],
            generation_config=generation_config,
            incomplete_last_bot_message=incomplete_last_bot_message,
            return_tokens=return_tokens,
            enable_thinking=enable_thinking,
            add_reasoning_truncing_prompt=add_reasoning_truncing_prompt,
            add_reasoning_info=add_reasoning_info,
            add_assistant_prompt_to_output=add_assistant_prompt_to_output,
            include_stop_str_in_output=include_stop_str_in_output,
            **kwargs
        )
        return prompts[0], outputs[0], infos[0]
    
    def _generate_batch_reasoning(
        self,
        messages,
        generation_config=None,
        incomplete_last_bot_message=True,
        return_tokens=False,
        include_stop_str_in_output=False,
        add_assistant_prompt_to_output=True,
        skip_special_tokens=True,
        enable_thinking=True,
        add_reasoning_truncing_prompt=False,
        add_reasoning_info=True,
        **kwargs
    ):
        messages_batch = messages
        generation_config = self.generation_config if generation_config is None else generation_config
        if not enable_thinking:
            return self._generate_batch(
                messages_batch,
                generation_config=generation_config,
                incomplete_last_bot_message=incomplete_last_bot_message,
                return_tokens=return_tokens,
                include_stop_str_in_output=include_stop_str_in_output,
                enable_thinking=False,
                skip_special_tokens=skip_special_tokens,
                **kwargs
            )

        prompt_messages_batch = []
        assistant_messages_batch = []
        for messages in messages_batch:
            for i in range(len(messages) - 1, -1, -1):
                if messages[i]["role"] not in ["assistant", "bot"]:
                    prompt_messages_batch.append(messages[:i + 1])
                    assistant_messages_batch.append(messages[i + 1:])
                    break

        old_params = self.prepare_generation_config()
        reasoning_prompt_batch, reasoning_output_batch, reasoning_infos_batch = self._generate_batch(
            prompt_messages_batch,
            generation_config=generation_config,
            incomplete_last_bot_message=incomplete_last_bot_message,
            include_stop_str_in_output=True,
            return_tokens=False,
            enable_thinking=True,
            skip_special_tokens=False,
            **kwargs
        )
        self.prepare_generation_config(old_params)

        updated_messages_batch = []
        for i, messages in enumerate(prompt_messages_batch):
            reasoning_text = reasoning_output_batch[i]
            stopped_by_length = reasoning_infos_batch[i]["generated_len"][0] >= generation_config.max_new_tokens_reasoning and not reasoning_text.endswith("</think>")
            stopped_by_think = False
            if "</think>" in reasoning_text:
                stopped_by_think = True
                if not reasoning_text.endswith("</think>"):
                    raise Exception(f'Unexpected generation: </think> token is present but generation did not stop.\nReasoning text: "{reasoning_text}"')
            if not (stopped_by_think or stopped_by_length):
                    raise Exception(f'Unexpected generation: stopped by neither </think> token nor length limit.\nReasoning text: "{reasoning_text}"')

            if stopped_by_length:
                if add_reasoning_truncing_prompt:
                    reasoning_text += generation_config.reasoning_truncing_prompt
                if not reasoning_text.endswith("\n"):
                    reasoning_text += "\n"
                reasoning_text += "</think>\n"
            reasoning_output_batch[i] = reasoning_text

            new_messages = [message.copy() for message in messages]
            new_messages.append({
                'role': 'assistant',
                'content': reasoning_text
            })
            new_messages.extend(assistant_messages_batch[i])
            updated_messages_batch.append(new_messages)

        _, final_output_batch, final_infos_batch = self._generate_batch(
            updated_messages_batch,
            generation_config=generation_config,
            incomplete_last_bot_message=True,
            return_tokens=return_tokens,
            include_stop_str_in_output=include_stop_str_in_output,
            enable_thinking=False,
            skip_special_tokens=skip_special_tokens,
            **kwargs
        )

        if add_assistant_prompt_to_output:
            for i in range(len(assistant_messages_batch)):
                assistant_prompt = "".join(list(map(lambda x: x["content"], assistant_messages_batch[i])))
                final_output_batch[i] = assistant_prompt + final_output_batch[i]

        infos_batch = []
        for i in range(len(messages_batch)):
            info = {
                'reasoning': {
                    'prompt_len': reasoning_infos_batch[i]['prompt_len'],
                    'generated_len': reasoning_infos_batch[i]['generated_len'],
                    'generated_cumulative_logprob': reasoning_infos_batch[i]['generated_cumulative_logprob']
                },
                'response': {
                    'prompt_len': final_infos_batch[i]['prompt_len'],
                    'generated_len': final_infos_batch[i]['generated_len'],
                    'generated_cumulative_logprob': final_infos_batch[i]['generated_cumulative_logprob']
                }
            }
            if add_reasoning_info:
                info['reasoning']['text'] = reasoning_output_batch[i]
            infos_batch.append(info)

        return reasoning_prompt_batch, final_output_batch, infos_batch

    @abstractmethod
    def _calculate_tokens_proba_batch(
        self,
        messages,
        tokens_of_interest,
        incomplete_last_bot_message=True
    ):
        pass

    def _calculate_tokens_proba_reasoning(
        self,
        messages,
        tokens_of_interest,
        incomplete_last_bot_message=True,
        enable_thinking=True,
        add_reasoning_truncing_prompt=False,
        add_reasoning_info=True,
        generation_config=None
    ):
        prompts, outputs, infos = self._calculate_tokens_proba_batch_reasoning(
            [messages],
            [tokens_of_interest],
            incomplete_last_bot_message=incomplete_last_bot_message,
            enable_thinking=enable_thinking,
            add_reasoning_truncing_prompt=add_reasoning_truncing_prompt,
            add_reasoning_info=add_reasoning_info,
            generation_config=generation_config
        )
        return prompts[0], outputs[0], infos[0]
    
    def _calculate_tokens_proba_batch_reasoning(
        self,
        messages,
        tokens_of_interest,
        incomplete_last_bot_message=True,
        enable_thinking=True,
        add_reasoning_truncing_prompt=False,
        add_reasoning_info=True,
        generation_config=None
    ):
        messages_batch = messages
        generation_config = self.generation_config if generation_config is None else generation_config
        if not enable_thinking:
            return self._calculate_tokens_proba_batch(
                messages_batch,
                tokens_of_interest,
                incomplete_last_bot_message=incomplete_last_bot_message
            )

        prompt_messages_batch = []
        assistant_messages_batch = []
        for messages in messages_batch:
            for i in range(len(messages) - 1, -1, -1):
                if messages[i]["role"] not in ["assistant", "bot"]:
                    prompt_messages_batch.append(messages[:i + 1])
                    assistant_messages_batch.append(messages[i + 1:])
                    break

        old_params = self.prepare_generation_config()
        reasoning_prompt_batch, reasoning_output_batch, reasoning_infos_batch = self._generate_batch(
            prompt_messages_batch,
            generation_config=generation_config,
            incomplete_last_bot_message=incomplete_last_bot_message,
            include_stop_str_in_output=True,
            return_tokens=False,
            enable_thinking=True,
            skip_special_tokens=False
        )
        self.prepare_generation_config(old_params)

        updated_messages_batch = []
        for i, messages in enumerate(prompt_messages_batch):
            reasoning_text = reasoning_output_batch[i]
            stopped_by_length = reasoning_infos_batch[i]["generated_len"][0] >= generation_config.max_new_tokens_reasoning and not reasoning_text.endswith("</think>")
            stopped_by_think = False
            if "</think>" in reasoning_text:
                stopped_by_think = True
                if not reasoning_text.endswith("</think>"):
                    raise Exception(f'Unexpected generation: </think> token is present but generation did not stop.\nReasoning text: "{reasoning_text}"')
            if not (stopped_by_think or stopped_by_length):
                    raise Exception(f'Unexpected generation: stopped by neither </think> token nor length limit.\nReasoning text: "{reasoning_text}"')

            if stopped_by_length:
                if add_reasoning_truncing_prompt:
                    reasoning_text += generation_config.reasoning_truncing_prompt
                if not reasoning_text.endswith("\n"):
                    reasoning_text += "\n"
                reasoning_text += "</think>\n"
            reasoning_output_batch[i] = reasoning_text

            new_messages = [message.copy() for message in messages]
            new_messages.append({
                'role': 'assistant',
                'content': reasoning_text
            })
            new_messages.extend(assistant_messages_batch[i])
            updated_messages_batch.append(new_messages)

        _, final_output_batch, final_infos_batch = self._calculate_tokens_proba_batch(
            updated_messages_batch,
            tokens_of_interest,
            incomplete_last_bot_message=incomplete_last_bot_message
        )

        infos_batch = []
        for i in range(len(messages_batch)):
            info = {
                'reasoning': {
                    'prompt_len': reasoning_infos_batch[i]['prompt_len'],
                    'generated_len': reasoning_infos_batch[i]['generated_len'],
                    'generated_cumulative_logprob': reasoning_infos_batch[i]['generated_cumulative_logprob']
                },
                'response': {
                    'generated_len': final_infos_batch[i]['generated_len'],
                    'generated_token': final_infos_batch[i]['generated_token'],
                }
            }
            if add_reasoning_info:
                info['reasoning']['text'] = reasoning_output_batch[i]
            infos_batch.append(info)

        return reasoning_prompt_batch, final_output_batch, infos_batch


class ApiVLLMModel(LLM):
    def __init__(self, api_base, api_key='', **kwargs):
        super().__init__(**kwargs)
        # requests.packages.urllib3.util.connection.HAS_IPV6 = False
        self.logger.info('ATTENTION! Hosting vLLM server must have vllm 0.6.3+')
        self.api_base = api_base
        self.num_procs = os.getenv('OPENAI_MAX_CONCURRENCY', 20)
        self.api_key = api_key if api_base else os.getenv('OPENAI_API_KEY', '123')
        self.model_name = None
        self.max_model_len = None
        self.generation_config = None
        self._tokenize_warning_shown = False
        self.max_len_warning_shown = False

    def support_method(self, method):
        return method in ['generate', 'calculate_tokens_proba']
    
    def from_pretrained(self, model_dir):
        url = self.api_base + '/v1/models'
        r = requests.get(url, headers={'Authorization': 'Bearer ' + self.api_key})
        if r.status_code != 200:
            print(r.text)
        assert r.status_code == 200

        data = r.json()
        print(data['data'])
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
            'max_new_tokens': 64,
            'do_sample': True
        })
        self.eos_token_ids_base = copy.deepcopy(self.generation_config.eos_token_id)
        self.stop_strings_base = copy.deepcopy(self.generation_config.stop_strings)

    def generate(
        self,
        messages,
        generation_config=None,
        incomplete_last_bot_message=True,
        return_tokens=False,
        include_stop_str_in_output=False,
        skip_special_tokens=True,
        enable_thinking=False,
        **kwargs
    ):
        if return_tokens:
            raise NotImplementedError

        messages = self._preprocess_messages(messages)
        last_role = messages[-1]['role']

        generation_config = self.generation_config if generation_config is None else generation_config
        num_return_sequences = generation_config.num_return_sequences if generation_config.num_return_sequences is not None else 1
        completion_tokens = []
        outputs = []
        for _ in range(num_return_sequences):
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
                    'stop_token_ids': self.generation_config.stop_token_ids if hasattr(self.generation_config, "stop_token_ids") else None,
                    'n': 1,
                    'add_generation_prompt': last_role == 'user',
                    'skip_special_tokens': skip_special_tokens,
                    'continue_final_message': incomplete_last_bot_message and last_role == 'assistant',
                    'include_stop_str_in_output': include_stop_str_in_output,
                    'chat_template_kwargs': {'enable_thinking': enable_thinking}
                },
                headers={'Authorization': 'Bearer ' + self.api_key}
            )
            if r.status_code != 200:
                if "maximum context length" in r.text:
                    if not self._max_len_warning_shown:
                        print("You requested more tokens than maximum model context length. Generation result will be defaulted to empty string.")
                        self._max_len_warning_shown = True
                    info = {
                        'prompt_len': "error",
                        'generated_len': 0,
                        'generated_cumulative_logprob': 'TODO: implement'
                    }
                    return messages, "", info
                else:
                    print(r.text)
                    assert r.status_code == 200

            data = r.json()
            outputs.append(data['choices'][0]['message']['content'])
            completion_tokens.append(data['usage']['completion_tokens'])

        if len(outputs) == 1:
            outputs = outputs[0]

        info = {
            'prompt_len': data['usage']['prompt_tokens'],
            'generated_len': completion_tokens,
            'generated_cumulative_logprob': 'TODO: implement'
        }
        return messages, outputs, info

    def generate_batch(
        self,
        messages,
        generation_config=None,
        incomplete_last_bot_message=True,
        return_tokens=False,
        include_stop_str_in_output=False,
        skip_special_tokens=True,
        enable_thinking=False,
        **kwargs
    ):
        messages_batch = messages
        kwargs.update({
            'generation_config': generation_config,
            'incomplete_last_bot_message': incomplete_last_bot_message,
            'return_tokens': return_tokens,
            'enable_thinking': enable_thinking,
            'include_stop_str_in_output': include_stop_str_in_output,
            'skip_special_tokens': skip_special_tokens
        })

        # Список для хранения результатов в правильном порядке
        results_ordered = [None] * len(messages_batch)

        with ThreadPoolExecutor(max_workers=self.num_procs) as executor:
            # Создаем словарь для сопоставления future с исходным индексом
            # Это нужно, чтобы сохранить исходный порядок результатов
            future_to_idx = {
                executor.submit(ApiVLLMModel.generate, self, msg, **kwargs): i
                for i, msg in enumerate(messages_batch)
            }

            # Создаем tqdm, который будет итерироваться по завершающимся задачам
            pbar = tqdm.tqdm(
                as_completed(future_to_idx),
                total=len(messages_batch),
                desc="Generating Batches"
            )

            # Обрабатываем результаты по мере их поступления
            for future in pbar:
                idx = future_to_idx[future]  # Получаем исходный индекс задачи
                try:
                    result = future.result()  # Получаем результат выполнения
                    results_ordered[idx] = result
                except Exception as e:
                    # Важно обрабатывать возможные исключения в потоках
                    print(f"Задача {idx} завершилась с ошибкой: {e}")
                    # или другое значение-маркер ошибки
                    results_ordered[idx] = (None, None, e)

        # Разделяем упорядоченные результаты на отдельные списки
        # Фильтруем None на случай, если какая-то задача упала
        valid_results = [res for res in results_ordered if res is not None and not isinstance(res[2], Exception)]
        if not valid_results:
            return [], [], []

        prompts, outputs, infos = list(zip(*valid_results))
        return list(prompts), list(outputs), list(infos)

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

    def calculate_tokens_proba(self, messages, tokens_of_interest, incomplete_last_bot_message=True, **kwargs):
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
                'top_logprobs': 20,
                'chat_template_kwargs': {'enable_thinking': False}
            },
            headers={'Authorization': 'Bearer ' + self.api_key}
        )
        if r.status_code != 200:
            if "maximum context length" in r.text:
                if not self._max_len_warning_shown:
                    print("You requested more tokens than maximum model context length. Generation result will be defaulted to empty string.")
                    self._max_len_warning_shown = True
                info = {
                    'generated_len': 0,
                    'generated_token': ""
                }
                return messages, "", info
            else:
                print(r.text)
                assert r.status_code == 200

        data = r.json()
        logprobs = data['choices'][0]['logprobs']['content'][0]['top_logprobs']
        probs = {lp['token']: np.exp(lp['logprob']) for lp in logprobs}

        tokens_of_interest_augmented = [(token, [' ' + token, token]) for token in tokens_of_interest]
        probs = {token: max(*list(map(lambda x: probs.get(x, 0.0), tokens))) for token, tokens in tokens_of_interest_augmented}

        info = {
            'generated_len': 1,
            'generated_token': data['choices'][0]['logprobs']['content'][0]['token']
        }
        return messages, probs, info

    def calculate_tokens_proba_batch(self, messages, tokens_of_interest, incomplete_last_bot_message=True, **kwargs):
        kwargs = {'incomplete_last_bot_message': incomplete_last_bot_message}
        results_ordered = [None] * len(messages)

        with ThreadPoolExecutor(max_workers=self.num_procs) as executor:
            # Создаем словарь для сопоставления future с исходным индексом
            # Это нужно, чтобы сохранить исходный порядок результатов
            future_to_idx = {
                executor.submit(ApiVLLMModel.calculate_tokens_proba, self, msg, toi, **kwargs): i
                for i, (msg, toi) in enumerate(zip(messages, tokens_of_interest))
            }

            # Создаем tqdm, который будет итерироваться по завершающимся задачам
            pbar = tqdm.tqdm(
                as_completed(future_to_idx),
                total=len(messages),
                desc="Generating Batches"
            )

            # Обрабатываем результаты по мере их поступления
            for future in pbar:
                idx = future_to_idx[future]  # Получаем исходный индекс задачи
                try:
                    result = future.result()  # Получаем результат выполнения
                    results_ordered[idx] = result
                except Exception as e:
                    # Важно обрабатывать возможные исключения в потоках
                    print(f"Задача {idx} завершилась с ошибкой: {e}")
                    # или другое значение-маркер ошибки
                    results_ordered[idx] = (None, None, e)

        # Разделяем упорядоченные результаты на отдельные списки
        # Фильтруем None на случай, если какая-то задача упала
        valid_results = [res for res in results_ordered if res is not None and not isinstance(res[2], Exception)]
        if not valid_results:
            return [], [], []

        prompts, outputs, infos = list(zip(*valid_results))
        return list(prompts), list(outputs), list(infos)

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

    def apply_model_prompt(self, messages, incomplete_last_bot_message=True, add_think_token=False):
        if self._tokenize_warning_shown:
            return 0
        _messages = self._preprocess_messages(messages)
        last_role = _messages[-1]['role']
        r = requests.post(
            f'{self.api_base}/tokenize',
            json={
                'messages': _messages,
                'model': self.model_name,
                'add_generation_prompt': last_role == 'user',
                'continue_final_message': incomplete_last_bot_message and last_role == 'assistant',
                'chat_template_kwargs': {'enable_thinking': add_think_token}
            }
        )
        if r.status_code != 200:
            print('Can\'t tokenize, fallback to 0 len assumtion')
            self._tokenize_warning_shown = True
            return 0

        assert r.status_code == 200
        data = r.json()
        return data['count']

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


class ApiVLLMModelReasoning(ApiVLLMModel, ReasoningModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def from_pretrained(
        self,
        model_dir,
        max_new_tokens_reasoning=None,
        reasoning_truncing_prompt="\n…the rest of the reasoning chain is hidden due to limit on the length of the thoughts.\n",
        end_thinking_token_id=None,
    ):
        super().from_pretrained(model_dir)
        self.reasoning_from_pretrained(
            max_new_tokens_reasoning=max_new_tokens_reasoning,
            reasoning_truncing_prompt=reasoning_truncing_prompt,
            end_thinking_token_id=end_thinking_token_id
        )
    
    def get_end_thinking_token_id(self):
        request = requests.post(
            f'{self.api_base}/tokenize',
            json={
                'prompt': '</think>',
                'model': self.model_name
            }
        )

        if request.status_code == 200:
            self.generation_config.end_thinking_token_id = request.json()["tokens"][0]
        else:
            raise Exception('Unable to get </think> token id. Make sure your endpoint supports tokenizer requests.\n' + request.text)
    
    def _generate_batch(*args, **kwargs):
        return ApiVLLMModel.generate_batch(*args, **kwargs)
    
    def generate(
        self,
        messages,
        generation_config=None,
        incomplete_last_bot_message=True,
        return_tokens=False,
        include_stop_str_in_output=False,
        add_assistant_prompt_to_output=True,
        skip_special_tokens=True,
        enable_thinking=True,
        add_reasoning_truncing_prompt=False,
        add_reasoning_info=True
    ):
        return self._generate_reasoning(
            messages,
            generation_config=generation_config,
            incomplete_last_bot_message=incomplete_last_bot_message,
            return_tokens=return_tokens,
            include_stop_str_in_output=include_stop_str_in_output,
            add_assistant_prompt_to_output=add_assistant_prompt_to_output,
            skip_special_tokens=skip_special_tokens,
            enable_thinking=enable_thinking,
            add_reasoning_truncing_prompt=add_reasoning_truncing_prompt,
            add_reasoning_info=add_reasoning_info
        )
    
    def generate_batch(
        self,
        messages,
        generation_config=None,
        incomplete_last_bot_message=True,
        return_tokens=False,
        include_stop_str_in_output=False,
        add_assistant_prompt_to_output=True,
        skip_special_tokens=True,
        enable_thinking=True,
        add_reasoning_truncing_prompt=False,
        add_reasoning_info=True
    ):
        return self._generate_batch_reasoning(
            messages,
            generation_config=generation_config,
            incomplete_last_bot_message=incomplete_last_bot_message,
            return_tokens=return_tokens,
            include_stop_str_in_output=include_stop_str_in_output,
            add_assistant_prompt_to_output=add_assistant_prompt_to_output,
            skip_special_tokens=skip_special_tokens,
            enable_thinking=enable_thinking,
            add_reasoning_truncing_prompt=add_reasoning_truncing_prompt,
            add_reasoning_info=add_reasoning_info
        )
    
    def _calculate_tokens_proba_batch(*args, **kwargs):
        return ApiVLLMModel.calculate_tokens_proba_batch(*args, **kwargs)
    
    def calculate_tokens_proba(
        self,
        messages,
        tokens_of_interest,
        incomplete_last_bot_message=True,
        enable_thinking=True,
        add_reasoning_truncing_prompt=False,
        add_reasoning_info=True,
        generation_config=None
    ):
        return self._calculate_tokens_proba_reasoning(
            messages,
            tokens_of_interest,
            incomplete_last_bot_message=incomplete_last_bot_message,
            enable_thinking=enable_thinking,
            add_reasoning_truncing_prompt=add_reasoning_truncing_prompt,
            add_reasoning_info=add_reasoning_info,
            generation_config=generation_config
        )
    
    def calculate_tokens_proba_batch(
        self,
        messages,
        tokens_of_interest,
        incomplete_last_bot_message=True,
        enable_thinking=True,
        add_reasoning_truncing_prompt=False,
        add_reasoning_info=True,
        generation_config=None
    ):
        return self._calculate_tokens_proba_batch_reasoning(
            messages,
            tokens_of_interest,
            incomplete_last_bot_message=incomplete_last_bot_message,
            enable_thinking=enable_thinking,
            add_reasoning_truncing_prompt=add_reasoning_truncing_prompt,
            add_reasoning_info=add_reasoning_info,
            generation_config=generation_config
        )


class LocalHostedLLM(LLM):
    def __init__(
        self,
        **kwargs
    ):
        super().__init__(
            **kwargs
        )

    def support_method(self, method):
        return method in ['generate', 'calculate_tokens_proba', 'calculate_logsoftmax']

    def from_pretrained(
        self,
        model_dir,
        conversation_template_path="auto",
        is_foundational=False,
        **kwargs
    ):
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
        self.tokenizer.padding_side = 'left' #TODO: а нужно ли это вообще? нужно перепроверить имплементации.
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

        # if not hasattr(self.generation_config, "stop_strings"):
        #     self.generation_config.stop_strings = []
        
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

    def _augment_tokens_of_interest(self, tokens_of_interest):
        token_variants = []
        for token_str in tokens_of_interest:
            variants = []

            tokens = self.tokenizer(token_str, add_special_tokens=False)['input_ids']
            if len(tokens) == 1:
                variants.append(tokens[0])

            if token_str[0] != ' ':
                augmented_token = ' ' + token_str
            else:
                augmented_token = token_str[1:]
            augmented_token = self.tokenizer(augmented_token, add_special_tokens=False)['input_ids']

            variants.append(augmented_token[0])

            variants = list(set(variants))
            token_variants.append(variants)

        return token_variants

    def apply_model_prompt(self, messages, incomplete_last_bot_message=True, add_think_token=False):
        _messages = []
        for m in messages:
            if m['role'] == 'bot':
                _messages.append({'role': 'assistant', 'content': m['content']})
            else:
                _messages.append({'role': m['role'], 'content': m['content']})

        last_role = _messages[-1]['role']
        add_generation_prompt = last_role == 'user'

        prompt = self.tokenizer.apply_chat_template(
            _messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=incomplete_last_bot_message and last_role == 'assistant',
            enable_thinking=add_think_token
        )

        return prompt

    def count_tokens_for_prompt(self, prompt):
        return len(self.tokenizer(prompt, add_special_tokens=False)['input_ids'])

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
        self.generation_config.max_length = self.get_max_model_len()
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
        self,
        conversation_template_path="auto",
        is_foundational=False,
        load_in_8bit=False,
        torch_dtype='auto',
        device_map='auto',
        attn_implementation="flash_attention_2",
        use_fast_tokenizer=True,
        trust_remote_code=False,
        alpha_scale=1.0,
        not_scale_lm_head=False,
        **kwargs
    ):
        super().__init__(
            conversation_template_path=conversation_template_path,
            is_foundational=is_foundational,
            **kwargs
        )
        self.load_in_8bit = load_in_8bit
        self.torch_dtype = torch_dtype
        self.attn_implementation = attn_implementation
        self.device_map = device_map
        self.use_fast_tokenizer = use_fast_tokenizer
        self.trust_remote_code = trust_remote_code
        self.alpha_scale = alpha_scale
        self.not_scale_lm_head = not_scale_lm_head

    def get_params(self):
        return {
            'model_name_or_path': self.model_name_or_path,
            'generation_config': json.loads(self.generation_config.to_json_string(use_diff=True)),
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

    def generate(
        self,
        messages,
        generation_config=None,
        incomplete_last_bot_message=True,
        return_tokens=False,
        include_stop_str_in_output=False,
        skip_special_tokens=True,
        enable_thinking=False,
        **kwargs
    ):
        prompts, outputs, infos = self.generate_batch(
            [messages],
            generation_config=generation_config,
            incomplete_last_bot_message=incomplete_last_bot_message,
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
        generation_config=None,
        incomplete_last_bot_message=True,
        return_tokens=False,
        include_stop_str_in_output=False,
        skip_special_tokens=True,
        enable_thinking=False,
        **kwargs
    ):
        messages_batch = messages
        generation_config = self.generation_config if generation_config is None else generation_config
        
        prompts = []
        for messages in messages_batch:
            prompts.append(self.apply_model_prompt(messages, incomplete_last_bot_message=incomplete_last_bot_message, add_think_token=enable_thinking))
        tokens = self.tokenizer(
            prompts,
            return_tensors="pt",
            truncation=True,
            add_special_tokens=False,
            max_length=generation_config.max_length,
            padding=True
        )
        data = {k: v.to(self.model.device) for k, v in tokens.items()}

        # TODO: upgrade to 4.40+ version with propper testing
        stop_strings = generation_config.stop_strings if generation_config.stop_strings else []
        # generation_config.stop_strings = None
        with torch.no_grad():
            output_ids = self.model.generate(
                **data,
                generation_config=generation_config,
                tokenizer=self.tokenizer
            )
        # generation_config.stop_strings = stop_strings

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

                    # TODO: better stop strings tructation.
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
                # this is length of untruncted output!
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

    def calculate_tokens_proba(self, messages, tokens_of_interest, incomplete_last_bot_message=True, **kwargs):
        prompts, probs, infos = self.calculate_tokens_proba_batch([messages], [tokens_of_interest], incomplete_last_bot_message=incomplete_last_bot_message, **kwargs)
        return prompts[0], probs[0], infos[0]

    def calculate_tokens_proba_batch(self, messages, tokens_of_interest, incomplete_last_bot_message=True, **kwargs):
        prompts_batch = []
        tokens_of_interest_ids_batch = []
        for _messages, _tokens_of_interest in zip(messages, tokens_of_interest):
            prompt = self.apply_model_prompt(_messages, incomplete_last_bot_message=incomplete_last_bot_message)
            prompts_batch.append(prompt)
            token_variants = self._augment_tokens_of_interest(_tokens_of_interest)
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
                    'generated_cumulative_logprob': 'TODO: calculate for hf model',
                    'generated_token': self.tokenizer.decode([next_token_probs.argmax()])
                }
            )

        return prompts_batch, probs_batch, infos

    def calculate_logsoftmax_batch(self, messages, incomplete_last_bot_message=True, log_only_last=True):
        # TODO: transformers 4.38.2 will be ok for llama3
        prompts = []
        for _messages in messages:
            prompts.append(self.apply_model_prompt(_messages, incomplete_last_bot_message=incomplete_last_bot_message))

        data = self.tokenizer(
            prompts, return_tensors="pt", truncation=True, padding=True,
            add_special_tokens=False,
            max_length=self.generation_config.max_length, return_offsets_mapping=True
        )
        # print(data['input_ids'].shape)
        offset_mapping = data.pop('offset_mapping').tolist()
        # if 'llama3' in self.model.config._name_or_path.lower() or 'llama-3' in self.model.config._name_or_path.lower() or os.environ.get('FORCE_CALCULATE_OFFSET_MAPPING_CUSTOM', False):
        #    offset_mapping = calculate_offset_mapping_llama3_workaround(prompts, data['input_ids'], self.tokenizer)

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
                    'generated_cumulative_logprob': 'TODO: calculate for hf model',
                }
            )

        add_tokens_with_logsoftmax_messages(
            messages, prompts, tokens_with_logsoftmax, log_only_last)
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

    def _add_stop_string(self, stop_string):
        vocab = self.tokenizer.vocab
        for t in tqdm.tqdm(vocab):
            token = self.tokenizer.convert_tokens_to_string([t])
            if token.endswith(stop_string):
                self.add_stop_token([vocab[t]])
        self.generation_config.stop_strings.append(stop_string)

    def get_max_model_len(self):
        return self.model.config.max_position_embeddings


class HFModelReasoning(HFModel, ReasoningModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def from_pretrained(
        self,
        model_dir,
        max_new_tokens_reasoning=None,
        reasoning_truncing_prompt="\n…the rest of the reasoning chain is hidden due to limit on the length of the thoughts.\n",
        end_thinking_token_id=None,
        **kwargs
    ):
        super().from_pretrained(model_dir, **kwargs)
        self.reasoning_from_pretrained(
            max_new_tokens_reasoning=max_new_tokens_reasoning,
            reasoning_truncing_prompt=reasoning_truncing_prompt,
            end_thinking_token_id=end_thinking_token_id
        )
    
    def get_end_thinking_token_id(self):
        self.generation_config.end_thinking_token_id = self.tokenizer.convert_tokens_to_ids("</think>")
    
    def _generate_batch(*args, **kwargs):
        return HFModel.generate_batch(*args, **kwargs)
    
    def generate(
        self,
        messages,
        generation_config=None,
        incomplete_last_bot_message=True,
        return_tokens=False,
        include_stop_str_in_output=False,
        add_assistant_prompt_to_output=True,
        skip_special_tokens=True,
        enable_thinking=True,
        add_reasoning_truncing_prompt=False,
        add_reasoning_info=True
    ):
        return self._generate_reasoning(
            messages,
            generation_config=generation_config,
            incomplete_last_bot_message=incomplete_last_bot_message,
            return_tokens=return_tokens,
            include_stop_str_in_output=include_stop_str_in_output,
            add_assistant_prompt_to_output=add_assistant_prompt_to_output,
            skip_special_tokens=skip_special_tokens,
            enable_thinking=enable_thinking,
            add_reasoning_truncing_prompt=add_reasoning_truncing_prompt,
            add_reasoning_info=add_reasoning_info
        )
    
    def generate_batch(
        self,
        messages,
        generation_config=None,
        incomplete_last_bot_message=True,
        return_tokens=False,
        include_stop_str_in_output=False,
        add_assistant_prompt_to_output=True,
        skip_special_tokens=True,
        enable_thinking=True,
        add_reasoning_truncing_prompt=False,
        add_reasoning_info=True
    ):
        return self._generate_batch_reasoning(
            messages,
            generation_config=generation_config,
            incomplete_last_bot_message=incomplete_last_bot_message,
            return_tokens=return_tokens,
            include_stop_str_in_output=include_stop_str_in_output,
            add_assistant_prompt_to_output=add_assistant_prompt_to_output,
            skip_special_tokens=skip_special_tokens,
            enable_thinking=enable_thinking,
            add_reasoning_truncing_prompt=add_reasoning_truncing_prompt,
            add_reasoning_info=add_reasoning_info
        )

    def _calculate_tokens_proba_batch(*args, **kwargs):
        return HFModel.calculate_tokens_proba_batch(*args, **kwargs)
    
    def calculate_tokens_proba(
        self,
        messages,
        tokens_of_interest,
        incomplete_last_bot_message=True,
        enable_thinking=True,
        add_reasoning_truncing_prompt=False,
        add_reasoning_info=True,
        generation_config=None
    ):
        return self._calculate_tokens_proba_reasoning(
            messages,
            tokens_of_interest,
            incomplete_last_bot_message=incomplete_last_bot_message,
            enable_thinking=enable_thinking,
            add_reasoning_truncing_prompt=add_reasoning_truncing_prompt,
            add_reasoning_info=add_reasoning_info,
            generation_config=generation_config
        )
    
    def calculate_tokens_proba_batch(
        self,
        messages,
        tokens_of_interest,
        incomplete_last_bot_message=True,
        enable_thinking=True,
        add_reasoning_truncing_prompt=False,
        add_reasoning_info=True,
        generation_config=None
    ):
        return self._calculate_tokens_proba_batch_reasoning(
            messages,
            tokens_of_interest,
            incomplete_last_bot_message=incomplete_last_bot_message,
            enable_thinking=enable_thinking,
            add_reasoning_truncing_prompt=add_reasoning_truncing_prompt,
            add_reasoning_info=add_reasoning_info,
            generation_config=generation_config
        )


class VLLMModel(LocalHostedLLM):
    def __init__(
        self,
        conversation_template_path="auto",
        is_foundational=False,
        use_fast_tokenizer=True,
        device_map='auto',
        max_seq_len_to_capture=4096*2,
        gpu_memory_utilization=0.95,
        disable_sliding_window=True,
        enable_prefix_caching=True,
        trust_remote_code=False,
        calculate_tokens_proba_logprobs_count=50,
        tensor_parallel_size=1,
        **kwargs
    ):
        super().__init__(
            conversation_template_path=conversation_template_path,
            is_foundational=is_foundational,
            **kwargs
        )
        self.use_fast_tokenizer = use_fast_tokenizer
        self.device_map = device_map
        self.max_seq_len_to_capture = max_seq_len_to_capture
        self.gpu_memory_utilization = gpu_memory_utilization
        self.disable_sliding_window = disable_sliding_window
        self.enable_prefix_caching = enable_prefix_caching
        self.trust_remote_code = trust_remote_code
        self.calculate_tokens_proba_logprobs_count = calculate_tokens_proba_logprobs_count
        self.tensor_parallel_size = tensor_parallel_size

        assert 'CUDA_VISIBLE_DEVICES' in os.environ
        self.logger.info('CUDA_VISIBLE_DEVICES=' + os.environ['CUDA_VISIBLE_DEVICES'])
        self.logger.info('device_map=' + self.device_map)

    def support_method(self, method):
        return method in ['generate', 'calculate_tokens_proba']

    def from_pretrained(
        self,
        model_dir,
        conversation_template_path="auto",
        is_foundational=False,
        **kwargs
    ):
        self._load_model(
            model_dir,
            conversation_template_path=conversation_template_path,
            is_foundational=is_foundational
        )
        # self._check_if_leading_space()
        self.reset_stop_strings()
        self.logger.info(f'Leading space: {self.leading_space}')
        # self.logger.info(f'For calculate_tokens_proba batch always will be 1 because of possible errors in logprobs') # TODO: verify

        tokenizer = self.model.get_tokenizer()
        tokenizer.pad_token_id = self.tokenizer.pad_token_id
        # tokenizer.padding_side = self.tokenizer.padding_side ?????
        tokenizer.truncation_side = self.tokenizer.truncation_side

        self.attn_backend = self.model.llm_engine.model_executor.driver_worker.model_runner.attn_backend
        self.special_attn_warning_complete = False

    def generate(
        self,
        messages,
        generation_config=None,
        incomplete_last_bot_message=True,
        return_tokens=False,
        include_stop_str_in_output=False,
        skip_special_tokens=True,
        enable_thinking=False,
        **kwargs
    ):
        prompts, outputs, infos = self.generate_batch(
            [messages],
            generation_config=generation_config,
            incomplete_last_bot_message=incomplete_last_bot_message,
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
        generation_config=None,
        incomplete_last_bot_message=True,
        return_tokens=False,
        include_stop_str_in_output=False,
        skip_special_tokens=True,
        enable_thinking=False,
        **kwargs
    ):
        messages_batch = messages
        generation_config = self.generation_config if generation_config is None else generation_config
        
        prompts_tokens_batch = []
        # allowed_token_ids_batch = [] if allowed_token_ids is not None else None
        for i, messages in enumerate(messages_batch):
            prompt = self.apply_model_prompt(
                messages,
                incomplete_last_bot_message=incomplete_last_bot_message,
                add_think_token=enable_thinking
            )
            prompts_tokens_batch.append(
                self.tokenizer(
                    prompt,
                    add_special_tokens=False,
                    truncation=True,
                    max_length=self.generation_config.max_length
                )['input_ids']
            )
            # if allowed_token_ids_batch is not None:
            #     allowed_token_ids_batch += allowed_token_ids[i]

        # if allowed_token_ids_batch is not None:
        #     allowed_token_ids_batch = list(set(allowed_token_ids_batch))

        sampling_params = SamplingParams(
            temperature=generation_config.temperature,
            top_p=generation_config.top_p,
            top_k=generation_config.top_k,
            max_tokens=generation_config.max_new_tokens,
            repetition_penalty=generation_config.repetition_penalty,
            stop=generation_config.stop_strings,
            # stop_token_ids=self.generation_config.stop_token_ids if hasattr(self.generation_config, "stop_token_ids") else None,
            stop_token_ids=self.generation_config.eos_token_id,
            n=generation_config.num_return_sequences,
            include_stop_str_in_output=enable_thinking or include_stop_str_in_output# ,
            # allowed_token_ids=allowed_token_ids_batch
        )

        prompts_vllm = []
        outputs = []
        infos = []

        vllm_responses = self.model.generate(
            prompt_token_ids=prompts_tokens_batch,
            sampling_params=sampling_params,
            use_tqdm=False,
            lora_request=self._get_lora_request(),
        )

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

    def calculate_tokens_proba(self, messages, tokens_of_interest, incomplete_last_bot_message=True, **kwargs):
        prompts, probs, infos = self.calculate_tokens_proba_batch([messages], [tokens_of_interest], incomplete_last_bot_message, **kwargs)
        return prompts[0], probs[0], infos[0]

    def calculate_tokens_proba_batch(self, messages, tokens_of_interest, incomplete_last_bot_message=True, **kwargs):
        if len(messages) > 1 and self.attn_backend == FlashAttentionBackend:
            if not self.special_attn_warning_complete:
                self.logger.warning(
                    'Flash Attention 2 most probably can work incorrectly with logproba and batch size > 1 '
                    '(because of padding).\nHighly recommended to use Xformer backend in this case '
                    '(set VLLM_ATTENTION_BACKEND env var to XFORMERS) or batch size=1.'
                )
                self.special_attn_warning_complete = True

        prompts_tokens_batch = []
        tokens_of_interest_ids_batch = []
        for _messages, _tokens_of_interest in zip(messages, tokens_of_interest):
            prompt = self.apply_model_prompt(_messages, incomplete_last_bot_message=incomplete_last_bot_message)
            prompts_tokens_batch.append(self.tokenizer(prompt, add_special_tokens=False, truncation=True, max_length=self.generation_config.max_length)['input_ids'])
            tokens_of_interest_ids_batch.append(self._augment_tokens_of_interest(_tokens_of_interest))

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
            prompt_token_ids=prompts_tokens_batch,
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
                'generated_token': response.outputs[0].text
            })

        return prompts_vllm, probs_batch, infos

    def calculate_logsoftmax_batch(self, messages, incomplete_last_bot_message=True, log_only_last=True):
        # BUGGED https://github.com/vllm-project/vllm/pull/5355
        prompts_tokens_batch = []
        prompts = []
        offset_mapping = []
        for _messages in messages:
            prompt = self.apply_model_prompt(_messages, incomplete_last_bot_message=incomplete_last_bot_message)
            data = self.tokenizer(prompt, add_special_tokens=False, truncation=True,max_length=self.generation_config.max_length, return_offsets_mapping=True)

            prompts.append(prompt)
            prompts_tokens_batch.append(data['input_ids'])
            offset_mapping.append(data['offset_mapping'])

        # config = self.model.llm_engine.get_model_config().hf_config
        # if 'llama3' in config._name_or_path.lower() or 'llama-3' in config._name_or_path.lower() or os.environ.get('FORCE_CALCULATE_OFFSET_MAPPING_CUSTOM', False):
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

    def _load_plain_model(self, model_dir):
        self.model = vLLM(
            model=model_dir, device=self.device_map,  # max_model_len=self.max_model_len,
            max_model_len=self.max_seq_len_to_capture, max_seq_len_to_capture=self.max_seq_len_to_capture,
            gpu_memory_utilization=self.gpu_memory_utilization, max_logprobs=1000000,
            trust_remote_code=self.trust_remote_code, tensor_parallel_size=self.tensor_parallel_size,
            # rope_scaling='{"type": "extended", "factor": 8.0}'
        )

    def _load_lora(self, model_dir):
        # TODO: не работает с modules_to_save, и вообще пока не тестил
        config = PeftConfig.from_pretrained(model_dir)
        self.model = vLLM(
            model=config.base_model_name_or_path, device=self.device_map,
            max_model_len=self.max_seq_len_to_capture, max_seq_len_to_capture=self.max_seq_len_to_capture,
            gpu_memory_utilization=self.gpu_memory_utilization, max_logprobs=1000000,
            enable_lora=True, trust_remote_code=self.trust_remote_code, tensor_parallel_size=self.tensor_parallel_size,
            max_lora_rank=self._get_max_lora_rank(config)
        )

    def _get_lora_request(self):
        if not self.if_lora:
            return None
        return LoRARequest("lora", 1, self.model_name_or_path)

    def _get_max_lora_rank(self, lora_config):
        # TODO: случай с наличием не дефолтного ранга
        return lora_config.r

    def get_max_model_len(self):
        return min(self.model.llm_engine.model_config.max_model_len, self.max_seq_len_to_capture)


class VLLMModelReasoning(VLLMModel, ReasoningModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def from_pretrained(
        self,
        model_dir,
        max_new_tokens_reasoning=None,
        reasoning_truncing_prompt="\n…the rest of the reasoning chain is hidden due to limit on the length of the thoughts.\n",
        end_thinking_token_id=None,
        **kwargs
    ):
        super().from_pretrained(model_dir, **kwargs)
        self.reasoning_from_pretrained(
            max_new_tokens_reasoning=max_new_tokens_reasoning,
            reasoning_truncing_prompt=reasoning_truncing_prompt,
            end_thinking_token_id=end_thinking_token_id
        )
    
    def get_end_thinking_token_id(self):
        self.generation_config.end_thinking_token_id = self.tokenizer.convert_tokens_to_ids("</think>")
    
    def _generate_batch(*args, **kwargs):
        return VLLMModel.generate_batch(*args, **kwargs)
    
    def generate(
        self,
        messages,
        generation_config=None,
        incomplete_last_bot_message=True,
        return_tokens=False,
        include_stop_str_in_output=False,
        add_assistant_prompt_to_output=True,
        skip_special_tokens=True,
        enable_thinking=True,
        add_reasoning_truncing_prompt=False,
        add_reasoning_info=True
    ):
        return self._generate_reasoning(
            messages,
            generation_config=generation_config,
            incomplete_last_bot_message=incomplete_last_bot_message,
            return_tokens=return_tokens,
            include_stop_str_in_output=include_stop_str_in_output,
            add_assistant_prompt_to_output=add_assistant_prompt_to_output,
            skip_special_tokens=skip_special_tokens,
            enable_thinking=enable_thinking,
            add_reasoning_truncing_prompt=add_reasoning_truncing_prompt,
            add_reasoning_info=add_reasoning_info
        )
    
    def generate_batch(
        self,
        messages,
        generation_config=None,
        incomplete_last_bot_message=True,
        return_tokens=False,
        include_stop_str_in_output=False,
        add_assistant_prompt_to_output=True,
        skip_special_tokens=True,
        enable_thinking=True,
        add_reasoning_truncing_prompt=False,
        add_reasoning_info=True
    ):
        return self._generate_batch_reasoning(
            messages,
            generation_config=generation_config,
            incomplete_last_bot_message=incomplete_last_bot_message,
            return_tokens=return_tokens,
            include_stop_str_in_output=include_stop_str_in_output,
            add_assistant_prompt_to_output=add_assistant_prompt_to_output,
            skip_special_tokens=skip_special_tokens,
            enable_thinking=enable_thinking,
            add_reasoning_truncing_prompt=add_reasoning_truncing_prompt,
            add_reasoning_info=add_reasoning_info
        )

    def _calculate_tokens_proba_batch(*args, **kwargs):
        return VLLMModel.calculate_tokens_proba_batch(*args, **kwargs)
    
    def calculate_tokens_proba(
        self,
        messages,
        tokens_of_interest,
        incomplete_last_bot_message=True,
        enable_thinking=True,
        add_reasoning_truncing_prompt=False,
        add_reasoning_info=True,
        generation_config=None
    ):
        return self._calculate_tokens_proba_reasoning(
            messages,
            tokens_of_interest,
            incomplete_last_bot_message=incomplete_last_bot_message,
            enable_thinking=enable_thinking,
            add_reasoning_truncing_prompt=add_reasoning_truncing_prompt,
            add_reasoning_info=add_reasoning_info,
            generation_config=generation_config
        )
    
    def calculate_tokens_proba_batch(
        self,
        messages,
        tokens_of_interest,
        incomplete_last_bot_message=True,
        enable_thinking=True,
        add_reasoning_truncing_prompt=False,
        add_reasoning_info=True,
        generation_config=None
    ):
        return self._calculate_tokens_proba_batch_reasoning(
            messages,
            tokens_of_interest,
            incomplete_last_bot_message=incomplete_last_bot_message,
            enable_thinking=enable_thinking,
            add_reasoning_truncing_prompt=add_reasoning_truncing_prompt,
            add_reasoning_info=add_reasoning_info,
            generation_config=generation_config
        )
