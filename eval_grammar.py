from llmtf.base import SimpleFewShotHFTask
import codecs
import json
import copy
from collections import OrderedDict, defaultdict
import numpy as np
from tqdm import tqdm
import os
from datasets import load_dataset, Dataset, DatasetDict
from typing import Dict, List, Tuple
from llmtf.metrics import mean, metric_max_over_ground_truths, f1_macro_score
import transformers.data.metrics.squad_metrics as squad_metrics
import re
from llmtf.base import Task, SimpleFewShotHFTask, LLM
from difflib import SequenceMatcher 
import pandas as pd
import fire
from llmtf.model import VLLMModel, HFModel
import json
from llmtf.evaluator import Evaluator

def load_dataset(dataset_path):
    df = pd.read_csv(dataset_path)
    df['correct_s'] = 1
    df['correct_r'] = 2
    #df_prompt = df.iloc[:10]
    df_test = df#df.iloc[10:]

    tds = Dataset.from_pandas(df_test)
    #vds = Dataset.from_pandas(df_prompt)

    ds = DatasetDict()

    ds['test'] = tds
    #ds['prompt'] = vds
    return ds

    
class GrammarTask1(SimpleFewShotHFTask):
    def __init__(self, instruction, **kwargs):
        super().__init__(**kwargs)
        self._max_new_tokens = 1
        self.instruction = instruction
        self.method = 'calculate_tokens_proba'

    def name(self):
        return 'grammar1'

    def dataset_args(self) -> Dict:
        return {'dataset_path': 'grammar_test_new.csv'}

    def test_split_name(self) -> str:
        return 'test'

    def prompt_split_name(self) -> str:
        return 'prompt'

    @property
    def choices(self):
        return ["1", "2"]

    def _confident_accuracy_mean(self, results: Dict) -> Dict:
        samples_ids = [r['id'] for r in results]
        samples_pred = [r['val'] for r in results]
        df = pd.DataFrame()
        df['id'] = samples_ids
        df['pred'] = samples_pred
        accuracy_list = []
        for idx, group in df.groupby('id'):
            assert group.shape[0] == 2
            accuracy_list.append(group['pred'].iloc[0] * group['pred'].iloc[1])
        return sum(accuracy_list) / len(accuracy_list)

    def aggregation(self) -> Dict:
        return {"acc": self._confident_accuracy_mean}

    def evaluate(self, sample, y_pred) -> Dict:
        y_true = '1' if sample['order'] == 's' else '2'
        y_pred = sorted([pair for pair in y_pred.items()], key=lambda x: -x[1])[0][0].strip()

        return {"acc": {'val': y_true == y_pred, 'id': sample['id']}}

    def _load_dataset(self, model: LLM, max_len: int, max_sample_per_dataset: int, few_shot_count: int) -> List:
        samples = []
        dataset = load_dataset(**self.dataset_args())
        test_dataset = dataset[self.test_split_name()]
        prompt_dataset = []#dataset[self.prompt_split_name()]

        test_dataset = test_dataset.select(range(min(max_sample_per_dataset, len(test_dataset))))
        #prompt_dataset = prompt_dataset.select(range(self.prompt_dataset_start_idx(), min(self.prompt_dataset_start_idx() + few_shot_count, len(prompt_dataset))))
        def change_order(example, idx):
            if idx % 2:
                example['order'] = 'r'
            return example
        #prompt_dataset = prompt_dataset.map(change_order, with_indices=True)
                
        for sample in tqdm(test_dataset):
            sample_lhs = copy.deepcopy(sample)
            sample_lhs['order'] = 's'
            samples.append({'messages': self._prepare_messages(sample_lhs, model, max_len, few_shot_count, prompt_dataset), 'sample': sample_lhs})

            sample_rhs = copy.deepcopy(sample)
            sample_rhs['order'] = 'r'
            samples.append({'messages': self._prepare_messages(sample_rhs, model, max_len, few_shot_count, prompt_dataset), 'sample': sample_rhs})

        return samples

    def create_messages(self, sample, with_answer=None) -> List[Dict]:
        sentences = [sample['gram'], sample['ungram']]
        if sample['order'] == 'r':
            sentences.reverse()

        inputs = {'sent_lhs': sentences[0], 'sent_rhs': sentences[1]}
        messages = [{'role': 'user', 'content': self.instruction.format(**inputs)}]
        if with_answer:
            messages.append({'role': 'bot', 'content': '1' if sample['order'] == 's' else '2'})
        return messages


def evaluate_model(model_name_or_path, conv_path, instructions_path, output_path, few_shot_count, device_map='cuda:0', vllm=False):
    MODEL_CLASS = VLLMModel if vllm else HFModel
    model = MODEL_CLASS(conversation_template_path=conv_path, device_map=device_map)
    model.from_pretrained(model_name_or_path)

    with codecs.open(instructions_path, 'r', 'utf-8') as file:
        instructions = json.loads(file.read())

    for i, instruction in enumerate(instructions):
        task = GrammarTask1(instruction)
        evaluator = Evaluator()
        evaluator.evaluate_dataset(task, model, output_path + f'_i{i}', 4000, few_shot_count, None, 16, 10000000000000000)

    
if __name__ == '__main__':
    fire.Fire(evaluate_model)