from llmtf.base import Task, SimpleFewShotHFTask, LLM
from sklearn.metrics import matthews_corrcoef
from tqdm import tqdm
from typing import Dict, List, Tuple
from datasets import load_dataset, Dataset
import copy
from llmtf.metrics import mean

class RuColaCustomTask(SimpleFewShotHFTask):
    RUCOLA_HF_PATH = 'RussianNLP/rucola'
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.method = 'calculate_tokens_proba'
        self._max_new_tokens = 1

    def evaluate(self, sample, y_pred) -> Dict:
        y_true = str(sample['label'])
        y_pred = sorted([pair for pair in y_pred.items()], key=lambda x: -x[1])[0][0]
        return {"acc": y_true == y_pred, "mcc": [y_true, y_pred]}

    @classmethod
    def name(cls):
        return 'russiannlp/rucola_custom'
    
    @property
    def choices(self):
        return ["0", "1"]

    def aggregation(self) -> Dict:
        return {"acc": mean, "mcc": lambda data: matthews_corrcoef([d[0] for d in data], [d[1] for d in data])}

    def dataset_args(self) -> Dict:
        return {'path': self.RUCOLA_HF_PATH}

    def test_split_name(self) -> str:
        return 'validation'

    def prompt_split_name(self) -> str:
        return 'train'

    def create_messages(self, sample, with_answer):
        messages = []
        instruction_user = 'Твоя задача определить приемлемость текста для русского языка с точки зрения синтаксиса, морфологии и семантики. Ответом должно служить одно число: 0 или 1, где 0 - предложение не приемлемо с точки зрения русского языка, 1 - приемлемо.\nТекст: {sentence}'
        instruction_bot = 'Ответ: {label}'
        instruction_bot_incomplete = 'Ответ:'

        bot_content = instruction_bot.format(**sample) if with_answer else instruction_bot_incomplete

        messages.append({'role': 'user', 'content': instruction_user.format(**sample)})
        messages.append({'role': 'bot', 'content': bot_content})

        return messages

    def prompt_dataset_start_idx(self) -> int:
        # в ближайших индексах после 29 сбалансировано по меткам классов, вот поэтому
        return 29

        

