from llmtf.base import Task, SimpleFewShotHFTask, LLM
from sklearn.metrics import matthews_corrcoef
from tqdm import tqdm
from typing import Dict, List, Tuple
from datasets import load_dataset, Dataset
import copy
from llmtf.metrics import mean, rougel
import pandas as pd
import numpy as np
from sklearn.metrics import average_precision_score

class DaruFlores(SimpleFewShotHFTask):
    DARUMERU_HF_PATH = 'RefalMachine/darumeru'
    def __init__(self, input_lang, **kwargs):
        super().__init__(**kwargs)
        self.method = 'generate'
        self.dataset_name = 'flores'
        self.input_lang = input_lang
        self._max_new_tokens = 512

    def evaluate(self, sample, y_pred) -> Dict:
        output_lang = self.get_output_lang()
        y_true = sample['inputs'][output_lang]
        return {"rougel": rougel(y_true, y_pred).fmeasure}

    def task_name(self):
        return f'darumeru/flores_{self.input_lang}_{self.get_output_lang()}'

    def get_output_lang(self):
        if self.input_lang == 'ru':
            output_lang = 'en'
        else:
            output_lang = 'ru'
        return output_lang
    
    def aggregation(self) -> Dict:
        return {"rougel": mean}

    def dataset_args(self) -> Dict:
        return {'path': self.DARUMERU_HF_PATH, 'name': self.dataset_name}

    def test_split_name(self) -> str:
        return 'test'

    def prompt_split_name(self) -> str:
        return 'prompt'

    def create_messages(self, sample, with_answer):
        messages = sample['messages']
        if with_answer:
            messages[-1]['content'] += '{translation}'
            
        if self.input_lang == 'ru':
            inputs = {'input': 'русского', 'output': 'английский', 'input_text': sample['inputs']['ru'], 'translation': sample['inputs']['en']}
        elif self.input_lang == 'en':
            inputs = {'input': 'английского', 'output': 'русский', 'input_text': sample['inputs']['en'], 'translation': sample['inputs']['ru']}
        else:
            raise Exception('Incorrect input_lang')
            
        for m in messages:
            m['content'] = m['content'].format(**inputs)
            
        return messages

    def prompt_dataset_start_idx(self) -> int:
        return 0
    
    def get_answer(self, sample):
        output_lang = self.get_output_lang()
        return sample['inputs'][output_lang]