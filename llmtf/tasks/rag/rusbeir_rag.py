from llmtf.base import SimpleFewShotHFTask
from typing import Any, Dict
from llmtf.metrics import metric_max_over_ground_truths, rougel, mean

def convert_context(context):
    segments = [f'**Сегмент №{i+1}**:\n' + c['chunk'] for i, c in enumerate(context)]
    if len(segments) == 0:
        return 'Поиск не вернул результатов (сегментов).'
    else:
        return '\n\n'.join(segments)
    
class RusbeirRag(SimpleFewShotHFTask):
    def __init__(self, instruction, dataset='bearberry/rubqqa', **kwargs):
        super().__init__(**kwargs)
        self.method = 'generate'
        self.dataset = dataset
        self.instruction = instruction
        self._max_new_tokens = 128

    def evaluate(self, sample, y_pred) -> Dict:
        rougel_metric = metric_max_over_ground_truths(lambda x, y: rougel(x, y).fmeasure, y_pred, sample['answers'])
        return {"rougel": rougel_metric}

    def task_name(self):
        return self.dataset

    def aggregation(self) -> Dict:
        return {"rougel": mean}

    def dataset_args(self) -> Dict:
        return {'path': self.dataset}

    def test_split_name(self) -> str:
        return 'train'

    def prompt_split_name(self) -> str:
        return 'train'

    def create_messages(self, sample, with_answer):
        messages = []
        
        answer = sample['answers'][0]
        question = sample['question'].strip()
        segments = convert_context(sample['context']).strip()
        
        instruction_user = self.instruction.replace('{question}', question).replace('{segments}', segments)

        messages.append({'role': 'user', 'content': instruction_user})
        if with_answer:
            messages.append({'role': 'bot', 'content': answer})

        return messages

    def prompt_dataset_start_idx(self) -> int:
        return 0
    
    def get_answer(self, sample):
        return sample['answer'].strip()