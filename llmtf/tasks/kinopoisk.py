from llmtf.base import SimpleFewShotHFTask
from llmtf.metrics import mean, f1_macro_score

DEFAULT_KINOPOISK_INSTRUCTION = """\
Твоя задача определить тональность отзыва на фильм.

Ответом является цифра класса:
1. Негативный
2. Нейтральный
3. Позитивный

Формат ответа:
Ответ: 0

Отзыв для классификации:

{text}
"""

DEFAULT_KINOPOISK_INSTRUCTION_BOT = """\
Ответ: {label}
"""

class Kinopoisk(SimpleFewShotHFTask):
    DATASET_HF_PATH = "ai-forever/kinopoisk-sentiment-classification"
    
    def __init__(
        self,
        instruction=DEFAULT_KINOPOISK_INSTRUCTION,
        instruction_bot=DEFAULT_KINOPOISK_INSTRUCTION_BOT,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.instruction = instruction
        self.instruction_bot = instruction_bot
        self.method = "calculate_tokens_proba"
        self._max_task_new_tokens = 1

    def task_name(self):
        return 'ai-forever/kinopoisk'

    def dataset_args(self):
        return {'path': self.DATASET_HF_PATH}
    
    def test_split_name(self):
        return 'test'

    def prompt_split_name(self):
        return 'train'
    
    @property
    def choices(self):
        return ["1", "2", "3"]
    
    def create_messages(self, sample, with_answer):
        instruction = self.instruction.format(text=sample["text"])
        messages = [
            {"role": "user", "content": instruction},
        ]
        instruction_bot = self.instruction_bot
        if with_answer:
            instruction_bot = self.instruction_bot.format(label=(sample["label"] + 1))
        messages.append({"role": "assistant", "content": instruction_bot})
        return messages

    def evaluate(self, sample, y_pred):
        y_true = str(sample["label"] + 1)
        y_pred = sorted([pair for pair in y_pred.items()], key=lambda x: -x[1])[0][0]
        return {"acc": y_true == y_pred, "f1-macro": (y_true, y_pred)}
        
    def aggregation(self):
        return {"acc": mean,"f1-macro": f1_macro_score}

    def leaderboard_aggregation(self, metrics):
        return metrics["f1-macro"]
