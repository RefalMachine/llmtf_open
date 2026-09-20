"""Minimal local-dataset task using the current task contract."""

from pathlib import Path

from llmtf.base import SimpleFewShotHFTask
from llmtf.metrics import mean


DATA_DIR = Path(__file__).parent / "data"
TRAIN_DATA_PATH = DATA_DIR / "toy_sentiment_train.jsonl"
TEST_DATA_PATH = DATA_DIR / "toy_sentiment_test.jsonl"
TASK_NAME = "example/toy_sentiment"


class ToySentimentTask(SimpleFewShotHFTask):
    """Classify a Russian review as negative (0) or positive (1)."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.method = "calculate_tokens_proba"
        self._max_task_new_tokens = 1

    @property
    def choices(self):
        return ["0", "1"]

    def task_name(self):
        return TASK_NAME

    def dataset_args(self):
        return {
            "path": "json",
            "data_files": {
                "train": str(TRAIN_DATA_PATH),
                "test": str(TEST_DATA_PATH),
            },
        }

    def test_split_name(self):
        return "test"

    def prompt_split_name(self):
        return "train"

    def create_messages(self, sample, with_answer):
        messages = [{
            "role": "user",
            "content": (
                "Определи тональность отзыва. Ответь только числом: "
                "0 — отрицательная, 1 — положительная.\n"
                f"Отзыв: {sample['text']}"
            ),
        }]
        if with_answer:
            messages.append({
                "role": "assistant",
                "content": str(sample["label"]),
            })
        return messages

    def evaluate(self, sample, y_pred):
        predicted = max(y_pred, key=y_pred.get)
        return {"accuracy": predicted == str(sample["label"])}

    def aggregation(self):
        return {"accuracy": mean}
