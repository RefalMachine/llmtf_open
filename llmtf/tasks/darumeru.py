import codecs
import json
import copy
from collections import OrderedDict, defaultdict
import numpy as np
from tqdm import tqdm
import os
from datasets import load_dataset, Dataset
from typing import Dict, List, Tuple
from llmtf.metrics import mean, metric_max_over_ground_truths, f1_macro_score
import transformers.data.metrics.squad_metrics as squad_metrics
import re
from llmtf.base import Task, SimpleFewShotHFTask, LLM

class DarumeruTask(SimpleFewShotHFTask):
    DARUMERU_HF_PATH = 'RefalMachine/darumeru'
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.additional_stop_strings.append('\n')
        self.additional_stop_strings.append('\n\n')
        self._max_new_tokens = 64

    def dataset_args(self) -> Dict:
        return {'path': self.DARUMERU_HF_PATH, 'name': self.dataset_name}

    def test_split_name(self) -> str:
        return 'test'

    def prompt_split_name(self) -> str:
        return 'prompt'

    def create_messages(self, sample, with_answer=None) -> List[Dict]:
        # ignoring with_answer because it's already taken into account in the darumeru dataset
        messages = sample['messages']
        inputs = sample['inputs']
        for m in messages:
            m['content'] = m['content'].format(**inputs)
        return messages

class MultiQ(DarumeruTask):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.method = 'generate'
        self.dataset_name = 'multiq'

    @property
    def name(self):
        return 'darumeru/MultiQ'

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

class PARus(DarumeruTask):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.method = 'calculate_tokens_proba'
        self.dataset_name = 'parus'

    @property
    def name(self):
        return 'darumeru/PARus'
    
    @property
    def choices(self):
        return ["1", "2"]

    def aggregation(self) -> Dict:
        return {"acc": mean}

    def evaluate(self, sample, y_pred) -> Dict:
        y_true = sample['outputs']
        y_pred = sorted([pair for pair in y_pred.items()], key=lambda x: -x[1])[0][0]
        return {"acc": y_true == y_pred}


class RCB(DarumeruTask):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.method = 'calculate_tokens_proba'
        self.dataset_name = 'rcb'

    @property
    def name(self):
        return 'darumeru/RCB'

    @property
    def choices(self):
        return ["1", "2", "3"]

    def aggregation(self) -> Dict:
        return {"acc": mean, "f1_macro": f1_macro_score}

    def evaluate(self, sample, y_pred) -> Dict:
        y_true = sample['outputs']
        y_pred = sorted([pair for pair in y_pred.items()], key=lambda x: -x[1])[0][0]
        return {"acc": y_true == y_pred, "f1_macro": (y_true, y_pred)}

class ruMMLU(DarumeruTask):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.method = 'calculate_tokens_proba'
        self.dataset_name = 'rummlu'

    @property
    def name(self):
        return 'darumeru/ruMMLU'

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

class ruOpenBookQA(DarumeruTask):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.method = 'calculate_tokens_proba'
        self.dataset_name = 'ruopenbookqa'

    @property
    def name(self):
        return 'darumeru/ruOpenBookQA'

    @property
    def choices(self):
        return ["A", "B", "C", "D"]

    def aggregation(self) -> Dict:
        return {"acc": mean, "f1_macro": f1_macro_score}

    def evaluate(self, sample, y_pred) -> Dict:
        y_true = sample['outputs']
        y_pred = sorted([pair for pair in y_pred.items()], key=lambda x: -x[1])[0][0]
        return {"acc": y_true == y_pred, "f1_macro": (y_true, y_pred)}

class ruTiE(DarumeruTask):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.method = 'calculate_tokens_proba'
        self.dataset_name = 'rutie'
    
    @property
    def name(self):
        return 'darumeru/ruTiE'

    @property
    def choices(self):
        return ["1", "2"]

    def aggregation(self) -> Dict:
        return {"acc": mean}

    def evaluate(self, sample, y_pred) -> Dict:
        y_true = sample['outputs']
        y_pred = sorted([pair for pair in y_pred.items()], key=lambda x: -x[1])[0][0]
        return {"acc": y_true == y_pred}
    
    def _load_dataset(self, model: LLM, max_len: int, max_sample_per_dataset: int, *args, **kwargs) -> List:
        dataset = load_dataset(**self.dataset_args())
        test_dataset = dataset[self.test_split_name()]
        samples = self._prepare_messages(test_dataset, model, max_len, max_sample_per_dataset)
        return samples
    
    def _prepare_messages(self, samples: Dataset, model: LLM, max_len: int, max_sample_per_dataset: int) -> List:
        samples = sorted(samples, key=lambda x: x['meta']['question_id'])
        all_dataset_messages = []
        dialog_shift = 0
        for s in samples[:max_sample_per_dataset]:
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
                for i, elem in enumerate(samples[dialog_shift:query_id])
            ]
            
            sample['inputs']['context'] = "\n".join(context)
            messages = self.create_messages(copy.deepcopy(sample)) #self.apply_inputs(sample['messages'], sample.get('inputs', {}))
            messages_len = model.count_tokens_for_prompt(model.apply_model_prompt(messages))
            while messages_len >= max_len:
                context = context[1:]
                sample['inputs']['context'] = "\n".join(context)
                messages = self.create_messages(copy.deepcopy(sample))#self.apply_inputs(sample['messages'], sample.get('inputs', {}))
                messages_len = model.count_tokens_for_prompt(model.apply_model_prompt(messages))
                dialog_shift += 1

            if messages_len >= max_len:
                #log this
                pass
            all_dataset_messages.append({'messages': messages, 'sample': s})

        return all_dataset_messages

class ruWorldTree(DarumeruTask):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.method = 'calculate_tokens_proba'
        self.dataset_name = 'ruworldtree'

    @property
    def name(self):
        return 'darumeru/ruWorldTree'

    @property
    def choices(self):
        return ["A", "B", "C", "D"]

    def aggregation(self) -> Dict:
        return {"acc": mean, "f1_macro": f1_macro_score}

    def evaluate(self, sample, y_pred) -> Dict:
        y_true = sample['outputs']
        y_pred = sorted([pair for pair in y_pred.items()], key=lambda x: -x[1])[0][0]
        return {"acc": y_true == y_pred, "f1_macro": (y_true, y_pred)}

class RWSD(DarumeruTask):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.method = 'calculate_tokens_proba'
        self.dataset_name = 'rwsd'

    @property
    def name(self):
        return 'darumeru/RWSD'
    @property
    def choices(self):
        return ["Да", "Нет"]

    def aggregation(self) -> Dict:
        return {"acc": mean}

    def evaluate(self, sample, y_pred) -> Dict:
        y_true = sample['outputs']
        y_pred = sorted([pair for pair in y_pred.items()], key=lambda x: -x[1])[0][0]
        return {"acc": y_true.startswith(y_pred)}


class USE(DarumeruTask):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.method = 'generate'
        self.dataset_name = 'use'
        self.max_grade_point = 34

    @property
    def name(self):
        return 'darumeru/USE'

    def evaluate(self, sample, y_pred) -> Dict:
        id_task = sample["meta"]["id_task"]
        task_type = sample["meta"]["type"]
        variant = sample["meta"]["variant"]
        answer = sample["outputs"]

        score = self.get_scores(task_type, id_task, answer, y_pred.strip())

        return {
            "grade_norm": (score, variant)
        }

    @staticmethod
    def multiple_choice_score(answer: str, prediction: str, is_task16=False) -> int:
        # Отличие от исходной меры. Добавлен .strip().
        # TODO: отразить отличие в readme

        pred = [p.strip() for p in prediction.split(",")]
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
        # Отличие от исходной меры. Добавлен .strip().
        # TODO: отразить отличие в readme

        pred = [p.strip() for p in prediction.split(",")]
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