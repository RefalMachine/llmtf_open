"""Closed-book binary tax reasoning with fixed demonstrations and probability scoring."""
import copy
import math
from pathlib import Path

from llmtf.base import SimpleFewShotHFTask, PromptTooLongError
from llmtf.metrics import mean
from .data import ROOT, load_rows, manifest, sha256, split_rows


class RuTaR(SimpleFewShotHFTask):
    method = 'calculate_tokens_proba'
    choices = ['0', '1']
    _max_task_new_tokens = 1

    def task_name(self):
        return 'rutar/closed'

    def dataset_args(self):
        spec = manifest()
        return {'path': spec['dataset_repo'], 'revision': spec['dataset_revision'],
                'data_files': spec['dataset_filename']}

    def __init__(self, data_path=None, **kwargs):
        super().__init__(**kwargs)
        self.data_path = data_path

    def test_split_name(self):
        return 'evaluation'

    def prompt_split_name(self):
        return 'demonstrations'

    def get_task_provenance(self):
        return {'manifest': manifest(),
                'resource_sha256': {name: sha256((ROOT / name).read_bytes())
                                    for name in ('manifest.json', 'data.py', 'task.py')},
                'local_data_sha256': sha256(Path(self.data_path).read_bytes())
                                     if self.data_path else None,
                'context_overflow_policy': 'error_keep_all_demonstrations',
                'answer_mapping': {'0': 'Нет', '1': 'Да'},
                'tie_policy': 'incorrect', 'answer_budget': self.max_task_new_tokens}

    def create_messages(self, sample, with_answer=False):
        content = ('Ответь на вопрос о налогообложении. Выбери 1, если ответ «Да», '
                   'или 0, если ответ «Нет». Ответь только одной цифрой.\n\n'
                   f"Вопрос: {sample['question']}\nОтвет:")
        messages = [{'role': 'user', 'content': content}]
        if with_answer:
            messages.append({'role': 'assistant', 'content': sample['answer']})
        return messages

    def _load_dataset(self, model, max_prompt_len, max_sample_per_dataset, few_shot_count):
        if isinstance(few_shot_count, bool) or not isinstance(few_shot_count, int) or not 0 <= few_shot_count <= 5:
            raise ValueError('RuTaR requires few_shot_count in 0..5')
        rows, pool = split_rows(load_rows(self.data_path))
        demos = pool[:few_shot_count]
        prefix = [message for demo in demos for message in self.create_messages(demo, with_answer=True)]
        result = []
        for raw in rows[:max_sample_per_dataset]:
            row = copy.deepcopy(raw)
            messages = copy.deepcopy(prefix) + self.create_messages(row)
            count = model.count_tokens_for_messages(messages)
            if count is not None and count > max_prompt_len:
                raise PromptTooLongError(f'RuTaR fixed {few_shot_count}-shot prompt has {count} tokens; budget {max_prompt_len}')
            row['_rutar'] = {'demonstration_ids': [demo['id'] for demo in demos],
                             'requested_shots': few_shot_count, 'effective_shots': len(demos),
                             'prompt_token_count': count, 'eligible_count': len(rows)}
            result.append({'messages': messages, 'sample': row})
        return result

    def evaluate(self, sample, y_pred):
        values = [y_pred[choice] for choice in self.choices]
        if any(isinstance(value, bool) or not isinstance(value, (int, float))
               or not math.isfinite(value) or not 0 <= value <= 1 for value in values):
            raise ValueError('Invalid RuTaR candidate probabilities')
        predicted = self.choices[values.index(max(values))] if values[0] != values[1] else None
        return {'acc': float(predicted == sample['answer'])}

    def aggregation(self):
        return {'acc': mean}
