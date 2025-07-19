from llmtf.base import Task, SimpleFewShotHFTask, LLM
from sklearn.metrics import matthews_corrcoef
from tqdm import tqdm
from typing import Dict, List, Tuple
from datasets import load_dataset, Dataset, Features, Value
import copy
from llmtf.metrics import mean, rouge1, rouge2, r_precision
import pandas as pd
import numpy as np
from sklearn.metrics import average_precision_score
from llmtf.metrics import mean, metric_max_over_ground_truths, f1_macro_score

class HabrQASbS(SimpleFewShotHFTask):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.method = 'calculate_tokens_proba'
        self._max_new_tokens = 1

    def task_name(self):
        return 'vikhrmodels/habr_qa_sbs'

    @property
    def choices(self):
        return ["A", "B"]

    def aggregation(self) -> Dict:
        return {"acc": mean, "f1_macro": f1_macro_score}

    def evaluate(self, sample, y_pred) -> Dict:
        y_true = sample['outputs']
        y_pred = sorted([pair for pair in y_pred.items()], key=lambda x: -x[1])[0][0]
        return {"acc": y_true == y_pred, "f1_macro": (y_true, y_pred)}
    
    def dataset_args(self) -> Dict:
        return {'path': 'Vikhrmodels/habr_qa_sbs'}

    def _convert_dataset(self, dataset):
        dataset = dataset['train']
        answers = []
        outputs = []
        for sample in dataset:
            answer_vars = [sample['best'], sample['bad']]
            np.random.shuffle(answer_vars)
            output = self.choices[answer_vars.index(sample['best'])]
            outputs.append(output)

            answer_vars = '\n\n'.join('***' + self.choices[i] + '***\n' + v for i, v in enumerate(answer_vars))
            answers.append(answer_vars)

        dataset = dataset.add_column('answers', answers)
        dataset = dataset.add_column('outputs', outputs)
        return dataset

    def _load_dataset(self, model: LLM, max_len: int, max_sample_per_dataset: int, few_shot_count: int) -> List:
        samples = []
        dataset = self._convert_dataset(load_dataset(**self.dataset_args()))

        prompt_dataset = dataset.select(range(1000))
        test_dataset = dataset.select(range(1000, len(dataset)))

        test_dataset = test_dataset.select(range(min(max_sample_per_dataset, len(test_dataset))))
        prompt_dataset = prompt_dataset.select(range(self.prompt_dataset_start_idx(), min(self.prompt_dataset_start_idx() + few_shot_count, len(prompt_dataset))))
        for sample in tqdm(test_dataset):
            samples.append({'messages': self._prepare_messages(sample, model, max_len, few_shot_count, prompt_dataset), 'sample': sample})
        return samples
    
    def create_messages(self, sample, with_answer):
        messages = []
        instruction_user = 'Тебе будет показан вопрос и два варианта ответа на него с QA платформы. Твоя задача выбрать, какой ответ получил больше лайков (то есть больше понравился пользователям).\n\n***Вопрос***\n{question}\n\n***Варианты ответа***\n{answers}\n\nОтветь одной буквой.'
        instruction_bot = 'Буква правильного ответа: {outputs}'
        instruction_bot_incomplete = 'Буква правильного ответа:'

        user_content = instruction_user.replace('{question}', sample['question']).replace('{answers}', sample['answers'])
        bot_content = instruction_bot.replace('{outputs}', sample['outputs']) if with_answer else instruction_bot_incomplete

        messages.append({'role': 'user', 'content': user_content})
        messages.append({'role': 'bot', 'content': bot_content})

        return messages
    
    def test_split_name(self) -> str:
        return ''

    def prompt_split_name(self) -> str:
        return ''
    
    def get_answer(self, sample):
        return ' ' + str(sample['outputs'])