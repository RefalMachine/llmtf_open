import abc
import codecs
import json
import copy
from collections import OrderedDict, defaultdict
import numpy as np
from tqdm import tqdm
import os
from abc import abstractmethod
import abc
from datasets import DatasetDict, load_dataset
from llmtf.model import LLM
from typing import Dict, List
from llmtf.metrics import mean, metric_max_over_ground_truths, f1_macro_score
import transformers.data.metrics.squad_metrics as squad_metrics
import re

DARUMERU_HF_PATH = 'RefalMachine/darumeru'

class Task(abc.ABC):
    @abstractmethod
    def load_dataset(self, **kwargs) -> List:
        pass
    
    @abstractmethod
    def evaluate(self, **kwargs) -> Dict:
        pass

    @abstractmethod
    def aggregation(self, **kwargs) -> Dict:
        pass

    def _load_dataset(self, dataset: DatasetDict, model: LLM, max_len: int, few_shot_count: int) -> List:
        assert model.support_method(self.method)
        samples = [{'messages': self._prepare_messages(sample, model, max_len, few_shot_count, dataset['prompt']), 'sample': sample} for sample in tqdm(dataset['test'])]
        return samples

    def _apply_inputs(self, messages, inputs):
        prepared_messages = copy.deepcopy(messages)
        for m in prepared_messages:
            m['content'] = m['content'].format(**inputs)
        return prepared_messages

    def _prepare_messages(self, sample: Dict, model: LLM, max_len: int, few_shot_count: int, few_shot_samples: List) -> List:
        k = min(few_shot_count, len(few_shot_samples))

        zero_shot_messages = self._apply_inputs(sample['messages'], sample.get('inputs', {}))
        zero_shot_messages_len = model.count_tokens_for_prompt(model.apply_model_prompt(zero_shot_messages))
        if zero_shot_messages_len >= max_len:
            #TODO: logger ...
            print(f'WARNING: sample zero-shot len {zero_shot_messages_len} greater then {max_len}. Will be truncated.')
        
        messages = copy.deepcopy(zero_shot_messages)
        successful = 0
        for i in range(k):
            few_shot_messages = self._apply_inputs(few_shot_samples[i]['messages'], few_shot_samples[i].get('inputs', {}))
            few_shot_messages_len = model.count_tokens_for_prompt(model.apply_model_prompt(few_shot_messages + messages))
            if few_shot_messages_len >= max_len:
                break

            messages = few_shot_messages + messages
            successful += 1

        self._fix_double_slash_n(messages)
        return messages

    def _fix_double_slash_n(self, messages):
        for i in range(len(messages)):
            messages[i]['content'] = messages[i]['content'].replace('\\n', '\n')
    
class MultiQ(Task):
    def __init__(self):
        self.method = 'generate'

    @property
    def name(self):
        return 'darumeru/MultiQ'
    
    def load_dataset(self, model: LLM, max_len: int, few_shot_count: int) -> List:
        dataset = load_dataset(DARUMERU_HF_PATH, 'multiq')
        samples = self._load_dataset(dataset, model, max_len, few_shot_count)
        messages = [{'messages': s['messages']} for s in samples]
        samples = [{'sample': s['sample']} for s in samples]
        return messages, samples

    def aggregation(self) -> Dict:
        return {"f1": mean, "em": mean}

    def evaluate(self, sample, y_pred) -> Dict:
        y_true = [answer["segment"] for answer in sample['outputs']]
        f1 = metric_max_over_ground_truths(squad_metrics.compute_f1, y_pred, y_true)
        em = metric_max_over_ground_truths(squad_metrics.compute_exact, y_pred, y_true)

        return {
            "f1": f1,
            "em": em,
        }
    
class PARus(Task):
    def __init__(self):
        self.method = 'calculate_token_interest_probs'

    @property
    def name(self):
        return 'darumeru/PARus'

    def load_dataset(self, model: LLM, max_len: int, few_shot_count: int) -> List:
        dataset = load_dataset(DARUMERU_HF_PATH, 'parus')
        samples = self._load_dataset(dataset, model, max_len, few_shot_count)
        messages = [{'messages': s['messages']} for s in samples]
        samples = [{'sample': s['sample']} for s in samples]
        for m in messages:
            m['tokens_of_interest'] = self.choices
        return messages, samples
    
    @property
    def choices(self):
        return ["1", "2"]

    def aggregation(self) -> Dict:
        return {"acc": mean}

    def evaluate(self, sample, y_pred) -> Dict:
        y_true = sample['outputs']
        y_pred = sorted([pair for pair in y_pred.items()], key=lambda x: -x[1])[0][0]
        return {"acc": y_true == y_pred}


class RCB(Task):
    def __init__(self):
        self.method = 'calculate_token_interest_probs'

    @property
    def name(self):
        return 'darumeru/RCB'
    
    def load_dataset(self, model: LLM, max_len: int, few_shot_count: int) -> List:
        dataset = load_dataset(DARUMERU_HF_PATH, 'rcb')
        samples = self._load_dataset(dataset, model, max_len, few_shot_count)
        messages = [{'messages': s['messages']} for s in samples]
        samples = [{'sample': s['sample']} for s in samples]
        for m in messages:
            m['tokens_of_interest'] = self.choices
        return messages, samples
    
    @property
    def choices(self):
        return ["1", "2", "3"]

    def aggregation(self) -> Dict:
        return {"acc": mean, "f1_macro": f1_macro_score}

    def evaluate(self, sample, y_pred) -> Dict:
        y_true = sample['outputs']
        y_pred = sorted([pair for pair in y_pred.items()], key=lambda x: -x[1])[0][0]
        return {"acc": y_true == y_pred, "f1_macro": (y_true, y_pred)}

class ruMMLU(Task):
    def __init__(self):
        self.method = 'calculate_token_interest_probs'

    @property
    def name(self):
        return 'darumeru/ruMMLU'
    
    def load_dataset(self, model: LLM, max_len: int, few_shot_count: int) -> List:
        dataset = load_dataset(DARUMERU_HF_PATH, 'rummlu')
        samples = self._load_dataset(dataset, model, max_len, few_shot_count)
        messages = [{'messages': s['messages']} for s in samples]
        samples = [{'sample': s['sample']} for s in samples]
        for m in messages:
            m['tokens_of_interest'] = self.choices
        return messages, samples

    @property
    def choices(self):
        return ["A", "B", "C", "D"]

    #TODO: mmlu aggregation 
    '''
    def load_gold(self):
        errors = super().load_gold()
        if self._aggregation is None:
            criterias = []
            for doc_id in self.gold.doc_ids():
                doc = self.gold[doc_id]
                criterias.append(doc["meta"]["domain"])
            criterias = list(set(criterias))
            self._aggregation = {"acc": mean}
            for criteria in criterias:
                self._aggregation[f"acc.{criteria}"] = mean
        return errors
    '''
    def aggregation(self) -> Dict:
        return {"acc": mean}

    def evaluate(self, sample, y_pred) -> Dict:
        y_true = sample['outputs']
        y_pred = sorted([pair for pair in y_pred.items()], key=lambda x: -x[1])[0][0]
        res = y_true == y_pred
        #criteria = sample["meta"]["domain"]
        return {"acc": res}#, f"acc.{criteria}": res}

class ruOpenBookQA(Task):
    def __init__(self):
        self.method = 'calculate_token_interest_probs'

    @property
    def name(self):
        return 'darumeru/ruOpenBookQA'
    
    def load_dataset(self, model: LLM, max_len: int, few_shot_count: int) -> List:
        dataset = load_dataset(DARUMERU_HF_PATH, 'ruopenbookqa')
        samples = self._load_dataset(dataset, model, max_len, few_shot_count)
        messages = [{'messages': s['messages']} for s in samples]
        samples = [{'sample': s['sample']} for s in samples]
        for m in messages:
            m['tokens_of_interest'] = self.choices
        return messages, samples

    @property
    def choices(self):
        return ["A", "B", "C", "D"]

    def aggregation(self) -> Dict:
        return {"acc": mean, "f1_macro": f1_macro_score}

    def evaluate(self, sample, y_pred) -> Dict:
        y_true = sample['outputs']
        y_pred = sorted([pair for pair in y_pred.items()], key=lambda x: -x[1])[0][0]
        return {"acc": y_true == y_pred, "f1_macro": (y_true, y_pred)}

class ruTiE(Task):
    def __init__(self):
        self.method = 'calculate_token_interest_probs'
    
    @property
    def name(self):
        return 'darumeru/ruTiE'

    def load_dataset(self, model: LLM, max_len: int, *inputs, **kwargs) -> List:
        dataset = load_dataset(DARUMERU_HF_PATH, 'rutie')
        samples = self._load_dataset(dataset, model, max_len)
        messages = [{'messages': s['messages']} for s in samples]
        samples = [{'sample': s['sample']} for s in samples]
        for m in messages:
            m['tokens_of_interest'] = self.choices
        return messages, samples
    
    @property
    def choices(self):
        return ["1", "2"]

    def aggregation(self) -> Dict:
        return {"acc": mean}

    def evaluate(self, sample, y_pred) -> Dict:
        y_true = sample['outputs']
        y_pred = sorted([pair for pair in y_pred.items()], key=lambda x: -x[1])[0][0]
        return {"acc": y_true == y_pred}
    
    def _load_dataset(self, dataset: DatasetDict, model: LLM, max_len: int) -> List:
        assert model.support_method(self.method)
        samples = self._prepare_messages(dataset['test'], model, max_len)
        return samples
    
    def _prepare_messages(self, samples: List, model: LLM, max_len: int) -> List:
        samples = sorted(samples, key=lambda x: x['meta']['question_id'])
        all_dataset_messages = []
        for s in samples:
            sample = copy.deepcopy(s)
            query_id = int(sample["meta"]["question_id"])
            context = [
                "{question}\n1. {choice1}\n2. {choice2}".format(
                    **{
                        "question": elem["inputs"]["question"],
                        "choice1": elem["inputs"]["choice1"],
                        "choice2": elem["inputs"]["choice2"],
                    }
                )
                + f"\nОтвет: {elem['outputs']}"
                for i, elem in enumerate(samples[:query_id])
            ]
            sample['inputs']['context'] = "\n".join(context)
            messages = self._apply_inputs(sample['messages'], sample.get('inputs', {}))
            messages_len = model.count_tokens_for_prompt(model.apply_model_prompt(messages))
            if messages_len >= max_len:
                #log this
                pass
            self._fix_double_slash_n(messages)
            all_dataset_messages.append({'messages': messages, 'sample': s})

        return all_dataset_messages

class ruWorldTree(Task):
    def __init__(self):
        self.method = 'calculate_token_interest_probs'

    @property
    def name(self):
        return 'darumeru/ruWorldTree'
    
    def load_dataset(self, model: LLM, max_len: int, few_shot_count: int) -> List:
        dataset = load_dataset(DARUMERU_HF_PATH, 'ruworldtree')
        samples = self._load_dataset(dataset, model, max_len, few_shot_count)
        messages = [{'messages': s['messages']} for s in samples]
        samples = [{'sample': s['sample']} for s in samples]
        for m in messages:
            m['tokens_of_interest'] = self.choices
        return messages, samples

    @property
    def choices(self):
        return ["A", "B", "C", "D"]

    def aggregation(self) -> Dict:
        return {"acc": mean, "f1_macro": f1_macro_score}

    def evaluate(self, sample, y_pred) -> Dict:
        y_true = sample['outputs']
        y_pred = sorted([pair for pair in y_pred.items()], key=lambda x: -x[1])[0][0]
        return {"acc": y_true == y_pred, "f1_macro": (y_true, y_pred)}

class RWSD(Task):
    def __init__(self):
        self.method = 'calculate_token_interest_probs'

    @property
    def name(self):
        return 'darumeru/RWSD'

    def load_dataset(self, model: LLM, max_len: int, few_shot_count: int) -> List:
        dataset = load_dataset(DARUMERU_HF_PATH, 'rwsd')
        samples = self._load_dataset(dataset, model, max_len, few_shot_count)
        messages = [{'messages': s['messages']} for s in samples]
        samples = [{'sample': s['sample']} for s in samples]
        for m in messages:
            m['tokens_of_interest'] = self.choices
        return messages, samples

    @property
    def choices(self):
        return ["Да", "Нет"]

    def aggregation(self) -> Dict:
        return {"acc": mean}

    def evaluate(self, sample, y_pred) -> Dict:
        y_true = sample['outputs']
        y_pred = sorted([pair for pair in y_pred.items()], key=lambda x: -x[1])[0][0]
        return {"acc": y_true.startswith(y_pred)}


class USE(Task):
    def __init__(self):
        self.method = 'generate'
        self.max_grade_point = 34

    @property
    def name(self):
        return 'darumeru/USE'
    
    def load_dataset(self, model: LLM, max_len: int, few_shot_count: int) -> List:
        dataset = load_dataset(DARUMERU_HF_PATH, 'use')
        samples = self._load_dataset(dataset, model, max_len, few_shot_count)
        messages = [{'messages': s['messages']} for s in samples]
        samples = [{'sample': s['sample']} for s in samples]
        return messages, samples

    def evaluate(self, sample, y_pred) -> Dict:
        id_task = sample["meta"]["id_task"]
        task_type = sample["meta"]["type"]
        variant = sample["meta"]["variant"]
        answer = sample["outputs"]

        score = self.get_scores(task_type, id_task, answer, y_pred)

        return {
            "grade_norm": (score, variant)
        }

    @staticmethod
    def multiple_choice_score(answer: str, prediction: str, is_task16=False) -> int:
        pred = prediction.split(",")
        ans = answer.split(",")
        if is_task16:
            while len(pred) < len(ans):
                pred.append("-1")
            return max(
                0,
                len(set.intersection(set(ans), set(pred))) - len(pred) + len(ans),
            )
        else:
            ans = set(ans)
            pred = set(pred)
            return int(len(set.intersection(ans, pred)) == len(ans) == len(pred))

    @staticmethod
    def matching_score(answer: str, prediction: str) -> int:
        pred = prediction.split(",")
        ans = answer.split(",")
        score = 0
        if len(ans) != len(pred):
            # print('Format Error: The prediction must contain a string of 4 numbers separated by ","')
            return 0
        for idx, num in enumerate(ans):
            if num == pred[idx]:
                score += 1
        return score

    @staticmethod
    def text_score(answer: str, prediction: str) -> int:
        pred = re.sub(r"[\d+\W+]", "", prediction).lower()
        ans = answer.split(",")
        if pred in ans:
            return 1
        return 0

    def get_scores(self, task_type, id_task, answer, prediction):
        if task_type == "matching":
            score = self.matching_score(answer, prediction)
        elif task_type == "text":
            score = self.text_score(answer, prediction)
        else:
            is_task16 = False
            if id_task == "16":
                is_task16 = True
            score = self.multiple_choice_score(answer, prediction, is_task16)
        return score

    def overall_score(self, items):
        overall_scores = defaultdict(float)
        for item in items:
            score, variant = item[0], item[1]
            overall_scores[variant] += score

        average_overall_score = np.mean([score / self.max_grade_point for score in overall_scores.values()])
        return average_overall_score

    def aggregation(self):
        return {
            "grade_norm": self.overall_score,
        }