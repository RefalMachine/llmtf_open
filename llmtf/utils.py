import os
import codecs
import time
import logging
import json 

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
    def __init__(self, task, model, max_len, custom_generation_config):
        self.task = task
        self.model = model
        self.max_len = max_len
        self.logger = logging.getLogger(__name__ + '.MaxLenContext')
        self.saved_max_new_tokens = self.model.generation_config.max_new_tokens
        self.custom_generation_config = custom_generation_config

    def __enter__(self):
        model_max_len = self.model.get_max_model_len()
        max_len = self.max_len

        max_new_tokens = self.task.max_new_tokens if self.custom_generation_config is None else self.custom_generation_config.max_new_tokens
        if max_new_tokens is None:
            max_new_tokens = self.task.max_new_tokens

        if model_max_len < self.max_len + max_new_tokens:
            self.logger.warning(f'model_max_len ({model_max_len}) < max_len ({self.max_len}) + max_new_tokens ({max_new_tokens})')
            max_len = model_max_len - max_new_tokens
            self.logger.warning(f'Lowering max_len to {max_len}')

        self.saved_max_new_tokens = self.model.generation_config.max_new_tokens
        self.model.generation_config.max_new_tokens = max_new_tokens
        return max_len

    def __exit__(self, *args):
        self.model.generation_config.max_new_tokens = self.saved_max_new_tokens