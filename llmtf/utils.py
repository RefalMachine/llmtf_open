from typing import List, Dict, Union
import os
import codecs
import time
import logging
import json 
import copy
import re

def set_out_handler_to_main_logger(output_dir):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    default_log_name = 'evaluation_log.txt'
    logger = logging.getLogger('llmtf')

    for handler in logger.handlers:
        if handler.__class__ == logging.FileHandler and handler.baseFilename.endswith(default_log_name):
            logger.removeHandler(handler)

    fh = logging.FileHandler(os.path.join(output_dir, default_log_name))
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(levelname)s: %(asctime)s: %(name)s: %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

class SimpleTaskLogger():
    def __init__(self, output_dir, task_name, append=False):
        self.output_dir = output_dir
        self.task_name = task_name.replace('/', '_')
        self.append = append

    def __enter__(self):
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
        self.file = codecs.open(os.path.join(self.output_dir, self.task_name + '.jsonl'), 'a' if self.append else 'w', 'utf-8')
        return self
 
    def __exit__(self, *args):
        self.file.close()

    def log_sample(self, sample, pred, prompt, metric, info):
        self.log_json({'metric': metric, 'predict': pred, 'sample': sample, 'prompt': prompt, 'info': info})
    
    def log_json(self, json_data, indent=4):
        self.file.write(json.dumps(json_data, ensure_ascii=False, indent=indent) + '\n')

class CustomTimer():
    def __init__(self, logger, prefix):
        self.logger = logger
        self.prefix = prefix
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, *args):
        time_passed = time.time() - self.start_time
        self.logger.info(f'{self.prefix}: {time_passed:.2f}s')

class MaxLenContext():
    def __init__(self, task, model, max_prompt_len, custom_generation_config):
        self.task = task
        self.model = model
        self.max_prompt_len = max_prompt_len
        self.logger = logging.getLogger(__name__ + '.MaxLenContext')
        self.saved_max_new_tokens = self.model.generation_config.max_new_tokens
        self.reasoning = False
        if hasattr(self.model.generation_config, "max_new_tokens_reasoning"):
            self.saved_max_new_tokens_reasoning = self.model.generation_config.max_new_tokens_reasoning
            self.reasoning = True
        else:
            self.saved_max_new_tokens_reasoning = 0
        self.custom_generation_config = custom_generation_config

    def __enter__(self):
        model_max_len = self.model.get_max_model_len()
        max_prompt_len = self.max_prompt_len

        max_new_tokens = self.task.max_task_new_tokens if self.custom_generation_config is None else self.custom_generation_config.max_new_tokens
        max_new_tokens_reasoning = self.saved_max_new_tokens_reasoning

        if model_max_len < max_prompt_len + max_new_tokens + max_new_tokens_reasoning:
            self.logger.warning(f'model_max_len ({model_max_len}) < max_prompt_len ({self.max_prompt_len}) + max_new_tokens ({max_new_tokens})' + (f' + max_new_tokens_reasoning ({max_new_tokens_reasoning})' if self.reasoning else ''))
            if self.reasoning:
                max_new_tokens_reasoning = max(model_max_len - max_prompt_len - max_new_tokens, 0)
                self.logger.warning(f'Lowering max_new_tokens_reasoning to {max_new_tokens_reasoning}')
            if not self.reasoning or model_max_len < max_prompt_len + max_new_tokens + max_new_tokens_reasoning:
                max_prompt_len = model_max_len - max_new_tokens
                self.logger.warning(f'Lowering max_prompt_len to {max_prompt_len}')
                
        self.model.generation_config.max_new_tokens = max_new_tokens
        if self.reasoning:
            self.model.generation_config.max_new_tokens_reasoning = max_new_tokens_reasoning
        return max_prompt_len

    def __exit__(self, *args):
        self.model.generation_config.max_new_tokens = self.saved_max_new_tokens
        self.model.generation_config.max_new_tokens_reasoning = self.saved_max_new_tokens_reasoning


def calculate_offset_mapping_llama3_workaround(prompts, tokens, tokenizer):
    # https://github.com/huggingface/tokenizers/issues/1553
    offset_mapping = []
    for i in range(len(prompts)):
        offset_mapping.append([])
        substring_pos = -1
        for j in range(len(tokens[i])):
            substring = tokenizer.decode(tokens[i][j:])
            substring_pos = prompts[i].find(substring, max(0, substring_pos))
            if substring_pos == -1:
                token_pos = [0, 0]
            else:
                token_pos = [substring_pos, substring_pos + len(tokenizer.decode(tokens[i][j:j+1]))]
            offset_mapping[-1].append(token_pos)
    return offset_mapping

def add_tokens_with_logsoftmax_messages(messages, prompts, tokens_with_logsoftmax, log_only_last):
    for i in range(len(messages)):
        message_end = 0
        for j, m in enumerate(messages[i]):
            message_start = prompts[i].find(m['content'], message_end)
            message_end = message_start + len(m['content'])
            if log_only_last and j < len(messages[i]) - 1:
                continue
            message_tokens = []
            inside = False
            for token, score, positions in tokens_with_logsoftmax[i]:
                positions_set = set(range(*positions))
                if message_start in positions_set:
                    inside = True

                if inside:
                    message_tokens.append([token, score, positions])

                if message_end in positions_set:
                    break
            
            m['tokens'] = message_tokens

def check_if_system_standard(tokenizer):
    text = tokenizer.apply_chat_template([{'role': 'system', 'content': '{scontent}'}], tokenize=False)
    maybe_system_message_template = text
    
    text = tokenizer.apply_chat_template([{'role': 'system', 'content': '{scontent}'}, {'role': 'user', 'content': '{ucontent}'}], tokenize=False)
    return text.startswith(maybe_system_message_template) and '{scontent}' in maybe_system_message_template

class Multiset():
    def __init__(self, l: Union[List, Dict]=[]):
        if type(l) == list:
            data = {}
            for e in l:
                if e in data.keys():
                    data[e] += 1
                else:
                    data[e] = 1
            self.data = data
        elif type(l) == dict:
            self.data = l
        else:
            raise Exception("Multiset can be initialized only with list or dictionary")

    def count(self):
        count = 0
        for v in self.data.values():
            count += v
        return count
    
    def union(self, m):
        data = self.data.copy()
        for k, v in m.data.items():
            if k in self.data.keys():
                data[k] = max(data[k], v)
            else:
                data[k] = v
        return Multiset(data)

    def intersect(self, m):
        data = {}
        for k, v in self.data.items():
            if k in m.data.keys():
                data[k] = min(v, m.data[k])
        return Multiset(data)

    def subtract(self, m):
        data = {}
        for k, v in self.data.items():
            if k in m.data.keys():
                if self.data[k] > m.data[k]:
                    data[k] = v - m.data[k]
            else:
                data[k] = v
        return Multiset(data)

    def add(self, m):
        data = self.data.copy()
        for k, v in m.data.items():
            if k in self.data.keys():
                data[k] += v
            else:
                data[k] = v
        return Multiset(data)