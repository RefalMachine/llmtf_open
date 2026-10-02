import string
import random
from llmtf.base import Task, BaseLLM, PromptTooLongError, ensure_prompt_fits
from llmtf.metrics import mean
from tqdm import tqdm
from typing import Dict, List, Tuple
from datasets import load_dataset, Dataset
import copy

def flatten(xss):
    return [x for xs in xss for x in xs]

class ShlepaSmallMMLU(Task):
    def __init__(self, dataset_name, **kwargs):
        super().__init__(**kwargs)
        self.dataset_name = dataset_name
        self.method = 'calculate_tokens_proba'
        self._max_task_new_tokens = 1

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

    def dataset_args(self) -> Dict:
        return {'path': self.dataset_name}

    def get_task_provenance(self) -> Dict:
        return {
            'few_shot_protocol': 'shlepa_train_demonstrations_v1',
            'demonstration_selection': 'first_valid_distinct_train_questions',
            'demonstration_seed': 555,
            'evaluation_excludes': 'demonstration_questions',
            'context_overflow_policy': 'error_keep_all_demonstrations',
        }

    def load_dataset(self, model: BaseLLM, max_prompt_len: int, max_sample_per_dataset: int, few_shot_count: int) -> Tuple[List[Dict], List[Dict]]:
        self.require_model_method(model)

        samples = self._load_dataset(model, max_prompt_len, max_sample_per_dataset, few_shot_count)
        messages = [{'messages': s['messages']} for s in samples]
        samples = [{'sample': s['sample']} for s in samples]
        for m in messages:
            m['tokens_of_interest'] = self.choices

        return messages, samples

    @staticmethod
    def _question_key(sample):
        return ' '.join(sample['question'].split()).casefold()

    def _load_dataset(self, model: BaseLLM, max_prompt_len: int, max_sample_per_dataset: int, few_shot_count: int = 0) -> List:
        if isinstance(few_shot_count, bool) or not isinstance(few_shot_count, int) or few_shot_count < 0:
            raise ValueError('Shlepa requires a non-negative integer few_shot_count')
        samples = []
        dataset = load_dataset(**self.dataset_args())
        test_dataset = dataset[self.test_split_name()]
        demonstration_messages = []
        demonstration_indices = []
        demonstration_questions = set()
        if few_shot_count:
            demonstration_rng = random.Random(555)
            for i in range(len(test_dataset)):
                question = self._question_key(test_dataset[i])
                if question in demonstration_questions:
                    continue
                messages, _ = self.create_messages(
                    copy.deepcopy(test_dataset[i]),
                    self._get_additional_samples(i, test_dataset),
                    with_answer=True, rng=demonstration_rng,
                )
                if not messages:
                    continue
                demonstration_messages.extend(messages)
                demonstration_indices.append(i)
                demonstration_questions.add(question)
                if len(demonstration_indices) == few_shot_count:
                    break
            if len(demonstration_indices) != few_shot_count:
                raise ValueError(
                    f'{self.run_name()} cannot supply {few_shot_count} distinct valid demonstrations'
                )
        evaluation_indices = [
            i for i in range(len(test_dataset))
            if self._question_key(test_dataset[i]) not in demonstration_questions
        ][:max_sample_per_dataset]
        if few_shot_count and not evaluation_indices:
            raise ValueError(f'{self.run_name()} has no evaluation questions outside the demonstrations')
        evaluation_indices = set(evaluation_indices)
        # Consume the same query-shuffle sequence as zero-shot for shared rows.
        # Demonstrations use a separate RNG and never affect query choice order.
        stop_index = max(evaluation_indices) + 1 if evaluation_indices else 0
        for i in tqdm(range(stop_index)):
            sample = test_dataset[i]
            additional_samples = self._get_additional_samples(i, test_dataset)
            messages, sample = self.create_messages(copy.deepcopy(sample), additional_samples)
            if i not in evaluation_indices or not messages:
                continue
            messages, sample = self._prepare_messages(
                sample, model, max_prompt_len, additional_samples,
                demonstration_messages=demonstration_messages,
                demonstration_indices=demonstration_indices,
                query_messages=messages,
            )
            sample['_shlepa']['dataset_index'] = i
            if len(messages) > 0:
                samples.append({'messages': messages, 'sample': sample})
        return samples
        
    def _prepare_messages(self, sample: Dict, model: BaseLLM, max_prompt_len: int, additional_samples: List[Dict], *, demonstration_messages=(), demonstration_indices=(), query_messages=None) -> List:
        if query_messages is None:
            zero_shot_messages, sample = self.create_messages(copy.deepcopy(sample), additional_samples)
        else:
            zero_shot_messages = query_messages
        if len(zero_shot_messages) == 0:
            return [], sample
        messages = list(demonstration_messages) + zero_shot_messages
        token_count = model.count_tokens_for_messages(messages)
        if demonstration_indices:
            if token_count is not None and token_count > max_prompt_len:
                raise PromptTooLongError(
                    f'{self.run_name()} fixed {len(demonstration_indices)}-shot prompt has '
                    f'{token_count} tokens; budget {max_prompt_len}'
                )
        else:
            ensure_prompt_fits(token_count, max_prompt_len, self.run_name())
        sample['_shlepa'] = {
            'requested_shots': len(demonstration_indices),
            'effective_shots': len(demonstration_indices),
            'demonstration_indices': list(demonstration_indices),
            'prompt_token_count': token_count,
        }

        return messages, sample

    def create_messages(self, sample: Dict, additional_samples: List[Dict], with_answer=False, *, rng=None):
        sample = self._helper(sample, additional_samples, rng=rng)
        if len(sample['gold']) != 1:
            return [], sample

        instruction = '''{question}\nA. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}\nE. {choices[4]}\nF. {choices[5]}\nG. {choices[6]}\nH. {choices[7]}\nI. {choices[8]}\nJ. {choices[9]}\nK. {choices[10]}\nL. {choices[11]}\n\nОтветь одной буквой.'''
        answer = 'Ответ:' + (self.get_answer(sample) if with_answer else '')
        messages = [{'role': 'user', 'content': instruction.format(**sample)}, {'role': 'assistant', 'content': answer}]
        return messages, sample

    def _get_additional_samples(self, index: int, dataset: Dataset):
        next_doc_count = 5
        if len(dataset) <= next_doc_count:
            raise ValueError(
                f"{self.run_name()} requires at least {next_doc_count + 1} "
                "rows to construct distractor choices"
            )
        indexes = [
            (index + offset) % len(dataset)
            for offset in range(1, next_doc_count + 1)
        ]
        return dataset[indexes]

    def _helper(self, doc, additional_samples, *, rng=None):
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
        if fi is None:
            matching_labels = [label for label in 'ABCD' if field == doc[f'answer{label}']]
            fi = matching_labels[0] if len(matching_labels) == 1 else None
        if fi is None:
            return {"label": "failed row w/o answer", "gold": ""}

        doc["choices"] = [doc[f"answer{s}"] for s in list('ABCD')]

        extended_choices = flatten([additional_samples[f"answer{s}"] for s in list('ABCD')])
        extended_choices = [c for c in extended_choices if c not in doc["choices"]]
        assert len(extended_choices) >= 8
        doc["choices"].extend(extended_choices[:8])
        assert len(doc["choices"]) == 12

        inv_label_map = {i: label for i, label in enumerate(string.ascii_uppercase[:12])}

        (rng if rng is not None else random).shuffle(doc["choices"])
        gold = fi
        shuffled_label = doc["choices"].index(doc[f"answer{gold}"])
        doc["label"] = inv_label_map[shuffled_label]
        doc["gold"] = inv_label_map[shuffled_label]

        return doc
    
    def get_answer(self, sample):
        return ' ' + str(sample['gold'])
