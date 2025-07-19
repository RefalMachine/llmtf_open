from llmtf.base import Task

import string
import random
from llmtf.base import Task, SimpleFewShotHFTask, LLM
from llmtf.metrics import mean
from tqdm import tqdm
from typing import Dict, List, Tuple
from datasets import load_dataset, Dataset
import copy

def flatten(xss):
    return [x for xs in xss for x in xs]

#TODO: refactoring
class ShlepaSmallMMLU(Task):
    def __init__(self, dataset_name, **kwargs):
        super().__init__(**kwargs)
        self.dataset_name = dataset_name
        self.method = 'calculate_tokens_proba'
        self._max_new_tokens = 1

    def evaluate(self, sample, y_pred) -> Dict:
        y_true = str(sample['gold'])
        y_pred = sorted([pair for pair in y_pred.items()], key=lambda x: -x[1])[0][0]
        return {"acc": y_true == y_pred}

    def aggregation(self) -> Dict:
        return {'acc': mean}

    def task_name(self) -> str:
        dataset_name_short = self.dataset_name[self.dataset_name.find('/') + 1:]
        return f'shlepa/{dataset_name_short}'

    @property
    def choices(self) -> List:
        return ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"]

    def test_split_name(self) -> str:
        return 'train'

    def load_dataset(self, model: LLM, max_len: int, max_sample_per_dataset: int, few_shot_count: int) -> Tuple[List[Dict], List[Dict]]:
        assert model.support_method(self.method)

        samples = self._load_dataset(model, max_len, max_sample_per_dataset)
        messages = [{'messages': s['messages']} for s in samples]
        samples = [{'sample': s['sample']} for s in samples]
        for m in messages:
            m['tokens_of_interest'] = self.choices

        return messages, samples

    def _load_dataset(self, model: LLM, max_len: int, max_sample_per_dataset: int) -> List:
        samples = []
        dataset = load_dataset(self.dataset_name)
        test_dataset = dataset[self.test_split_name()]
        test_dataset = test_dataset.select(range(min(max_sample_per_dataset, len(test_dataset))))
        for i, sample in tqdm(enumerate(test_dataset)):
            additional_samples = self._get_additional_samples(i, test_dataset)
            messages, sample = self._prepare_messages(sample, model, max_len, additional_samples)
            if len(messages) > 0:
                samples.append({'messages': messages, 'sample': sample})
        return samples
        
    def _prepare_messages(self, sample: Dict, model: LLM, max_len: int, additional_samples: List[Dict]) -> List:
        zero_shot_messages, sample = self.create_messages(copy.deepcopy(sample), additional_samples)
        if len(zero_shot_messages) == 0:
            return [], sample
        zero_shot_messages_len = model.count_tokens_for_prompt(model.apply_model_prompt(zero_shot_messages))
        if zero_shot_messages_len >= max_len:
            self.logger.warning(f'WARNING: sample zero-shot len {zero_shot_messages_len} greater then {max_len}. Will be truncated.')

        return zero_shot_messages, sample

    def create_messages(self, sample: Dict, additional_samples: List[Dict]):
        sample = self._helper(sample, additional_samples)
        if len(sample['gold']) != 1:
            return [], sample

        instruction = '''{question}\nA. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}\nE. {choices[4]}\nF. {choices[5]}\nG. {choices[6]}\nH. {choices[7]}\nI. {choices[8]}\nJ. {choices[9]}\nK. {choices[10]}\nL. {choices[11]}\n\nОтветь одной буквой.'''
        messages = [{'role': 'user', 'content': instruction.format(**sample)}, {'role': 'bot', 'content': 'Ответ:'}]
        return messages, sample

    def _get_additional_samples(self, index: int, dataset: Dataset):
        next_doc_count = 5
        if index < len(dataset) - next_doc_count:
            next_docs = dataset[index+1:index+next_doc_count+1]
        else:
            next_docs = dataset[index-next_doc_count:index]
        assert len(next_docs[list(dataset[0].keys())[0]]) == next_doc_count
        return next_docs

    def _helper(self, doc, additional_samples):
        field = doc['correct_answer']
        fi = None
        if field == 'answerA' or field in ['A','А']:
            fi = "A"
        if field == 'answerB' or field in ['B','Б']:
            fi = "B"
        if field == 'answerC' or field in ['C','С']:
            fi = "C"
        if field == 'answerD' or field in ['D','Д']:
            fi = "D"
        if fi:
            doc["choices"] = [doc[f"answer{s}"] for s in list('ABCD')]

        else:
            for idx,(key,val) in enumerate(list(doc.items())[1:-1]):
                if field == val:
                    fi = {0:"A",1:"B",2:"C",3:"D"}.get(idx)

        doc["choices"] = [doc[f"answer{s}"] for s in list('ABCD')]

        extended_choices = flatten([additional_samples[f"answer{s}"] for s in list('ABCD')])
        extended_choices = [c for c in extended_choices if c not in doc["choices"]]
        assert len(extended_choices) >= 8
        doc["choices"].extend(extended_choices[:8])
        assert len(doc["choices"]) == 12

        label_map = {label: i for i, label in enumerate(string.ascii_uppercase[:12])}
        inv_label_map = {i: label for i, label in enumerate(string.ascii_uppercase[:12])}

        correct_label = label_map.get(fi)
        if not correct_label: return {"label":"failed row w/o answer","gold":""}

        random.shuffle(doc["choices"])
        gold = fi
        shuffled_label = doc["choices"].index(doc[f"answer{gold}"])
        doc["label"] = inv_label_map[shuffled_label]
        doc["gold"] = inv_label_map[shuffled_label]

        return doc
    
    def get_answer(self, sample):
        return ' ' + str(sample['gold'])
