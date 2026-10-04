"""Probability-scored MMLU-like view of the same legal microcases."""

import copy
import hashlib
import math
from numbers import Real
from pathlib import Path

from .scoring import score
from .task import RuLawProofBench


class RuLawProofBenchMCQ(RuLawProofBench):
    method = "calculate_tokens_proba"
    choices = list("ABCD")
    _max_task_new_tokens = 1
    dataset_config = "mcq"

    def __init__(self, *args, **kwargs):
        if kwargs.get("judge_model") is not None:
            raise ValueError("MCQ uses exact option scoring, without a judge")
        super().__init__(*args, **kwargs)

    def task_name(self):
        return "rulaw_proofbench/mcq_" + self.mode

    def get_task_provenance(self):
        result = super().get_task_provenance()
        result.update(
            answer_protocol="rulaw_mcq_v1",
            choice_count="3_binary_else_4",
            mcq_task_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        )
        return result

    def load_dataset(self, *args, **kwargs):
        messages, samples = super().load_dataset(*args, **kwargs)
        for message, sample in zip(messages, samples):
            message["tokens_of_interest"] = [
                option["label"] for option in sample["sample"]["mcq"]["options"]
            ]
        return messages, samples

    def create_messages(self, sample, with_answer=False):
        row = copy.deepcopy(sample)
        row["question"] = sample["mcq"]["question"]
        messages = super().create_messages(row)
        answer = "Ответ:"
        if with_answer:
            answer += " " + sample["mcq"]["gold_label"]
        messages.append({"role": "assistant", "content": answer})
        return messages

    def evaluate(self, sample, y_pred):
        options = sample["mcq"]["options"]
        labels = [option["label"] for option in options]
        if not isinstance(y_pred, dict) or set(y_pred) != set(labels):
            raise ValueError("MCQ probability candidates do not match displayed options")
        if any(
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(value)
            or not 0 <= value <= 1
            for value in y_pred.values()
        ):
            raise ValueError("Invalid MCQ probability")
        if not any(y_pred.values()):
            raise ValueError("No probability mass for any MCQ candidate")
        best = max(y_pred.values())
        ties = [label for label in labels if y_pred[label] == best]
        chosen = next(option for option in options if option["label"] == ties[0])
        result = score(sample, chosen["value"])
        result.update(
            choice_count=len(options),
            mcq_gold_label=sample["mcq"]["gold_label"],
            mcq_predicted_label=chosen["label"],
            probabilities=y_pred,
            tie_count=len(ties),
        )
        return {"score": result}

    def _aggregate(self, records):
        value, details = super()._aggregate(records)
        groups = {}
        for count in sorted({record["choice_count"] for record in records}):
            group = [record for record in records if record["choice_count"] == count]
            groups[str(count)] = {
                "count": len(group),
                "accuracy": sum(record["correct"] for record in group) / len(group),
            }
        details.update(
            answer_protocol="mcq",
            random_choice_micro_accuracy=sum(1 / record["choice_count"] for record in records) / len(records),
            by_choice_count=groups,
            tied_predictions=sum(record["tie_count"] > 1 for record in records),
        )
        return value, details
