import copy
import hashlib
import json
from collections import defaultdict
from typing import Dict, List, Optional

from datasets import load_dataset
from tqdm import tqdm

from llmtf.base import BaseLLM, SimpleFewShotHFTask


RUPARAM_DATASET_PATH = "RefalMachine/RuParam"
_TORFL_LEVELS = {"A1", "A2", "B1", "B2", "C1", "C2"}
_SOURCE_CONFUSABLES = str.maketrans({"А": "A", "С": "C"})


def _normalise_source(source: object) -> str:
    return str(source or "").strip().translate(_SOURCE_CONFUSABLES)


def _normalise_row(row: Dict, row_index: int) -> Dict:
    """Return a canonical grammatical/ungrammatical RuParam row.

    Historical RuParam exports use ``order=i`` when the values stored in the
    ``gram`` and ``ungram`` columns are inverted. Missing ``order`` is treated
    as standard because newer exports may rely on the column names alone.
    """
    order = str(row.get("order") or "s").strip().lower()
    if order not in {"s", "i"}:
        raise ValueError(
            f"RuParam row {row_index} has unsupported order {order!r}; "
            "expected 's' or 'i'"
        )

    gram = str(row.get("gram") or "")
    ungram = str(row.get("ungram") or "")
    if order == "i":
        gram, ungram = ungram, gram

    source_raw = str(row.get("source") or "").strip()
    source = _normalise_source(source_raw)
    level_candidate = source.removeprefix("torfl_")
    level = level_candidate if level_candidate in _TORFL_LEVELS else None
    category_raw = str(row.get("label") or "")
    category = category_raw.strip() or "__unlabeled__"

    identity = {
        "index": row_index,
        "id": str(row.get("id") or ""),
        "gram": gram,
        "ungram": ungram,
        "label": category_raw,
        "source": source_raw,
        "order": order,
    }
    digest = hashlib.sha256(
        json.dumps(
            identity,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()[:16]

    return {
        "row_id": f"{row_index}:{digest}",
        "original_id": identity["id"],
        "gram": gram,
        "ungram": ungram,
        "category": category,
        "category_raw": category_raw,
        "source": source,
        "source_raw": source_raw,
        "part": "torfl" if source.startswith("torfl_") else "parametric",
        "torfl_level": level,
        "source_order": order,
        "identical_sentences": gram.strip() == ungram.strip(),
    }


def _accuracy_details(records: List[Dict], field: str) -> Dict:
    buckets = defaultdict(list)
    for record in records:
        value = record.get(field)
        if value is not None:
            buckets[str(value)].append(record["pair_correct"])
    return {
        key: {
            "accuracy": sum(values) / len(values),
            "count": len(values),
        }
        for key, values in sorted(buckets.items())
    }


class RuParam(SimpleFewShotHFTask):
    """Zero-shot RuParam minimal-pair evaluation.

    A source row is scored as correct only if the model selects the grammatical
    sentence in both presentation orders. The primary score is micro accuracy
    over source rows; category and corpus slices are emitted as aggregation
    details and therefore do not accidentally change leaderboard weighting.
    """

    def __init__(
        self,
        instruction: str,
        dataset_path: str = RUPARAM_DATASET_PATH,
        dataset_data_files: Optional[object] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._max_task_new_tokens = 1
        self.instruction = instruction
        self.dataset_path = dataset_path
        self.dataset_data_files = dataset_data_files
        self.method = "calculate_tokens_proba"

    def task_name(self):
        return "ruparam"

    def dataset_args(self) -> Dict:
        if self.dataset_data_files is not None:
            return {"path": "csv", "data_files": self.dataset_data_files}
        return {"path": self.dataset_path}

    def test_split_name(self) -> str:
        return "test"

    def prompt_split_name(self) -> str:
        # RuParam is strictly zero-shot. This method only satisfies the common
        # task interface; no prompt split is read by this implementation.
        return "test"

    @property
    def choices(self):
        return ["1", "2"]

    @staticmethod
    def _aggregate_pair_accuracy(results: List[Dict]):
        if not results:
            raise ValueError("RuParam aggregation received no results")

        grouped = defaultdict(dict)
        for result in results:
            row_id = result["row_id"]
            presentation = result["presentation"]
            if presentation in grouped[row_id]:
                raise ValueError(
                    f"RuParam row {row_id!r} contains duplicate presentation "
                    f"{presentation!r}"
                )
            grouped[row_id][presentation] = result

        pair_records = []
        expected_presentations = {"grammatical_first", "grammatical_second"}
        for row_id, presentations in grouped.items():
            if set(presentations) != expected_presentations:
                raise ValueError(
                    f"RuParam row {row_id!r} must contain exactly both "
                    f"presentations; got {sorted(presentations)}"
                )
            first = presentations["grammatical_first"]
            second = presentations["grammatical_second"]
            for field in (
                "category",
                "source",
                "part",
                "torfl_level",
                "identical_sentences",
            ):
                if first.get(field) != second.get(field):
                    raise ValueError(
                        f"RuParam row {row_id!r} has inconsistent {field} "
                        "between presentations"
                    )
            pair_records.append({
                "row_id": row_id,
                "pair_correct": int(first["correct"] and second["correct"]),
                "category": first["category"],
                "source": first["source"],
                "part": first["part"],
                "torfl_level": first["torfl_level"],
                "identical_sentences": first["identical_sentences"],
            })

        micro_accuracy = (
            sum(record["pair_correct"] for record in pair_records)
            / len(pair_records)
        )
        by_category = _accuracy_details(pair_records, "category")
        category_macro_accuracy = (
            sum(item["accuracy"] for item in by_category.values())
            / len(by_category)
        )
        details = {
            "scoring_protocol": "correct_in_both_presentation_orders",
            "primary_aggregation": "micro_over_annotated_rows",
            "pair_count": len(pair_records),
            "presentation_count": len(results),
            "pair_micro_accuracy": micro_accuracy,
            "category_macro_accuracy": category_macro_accuracy,
            "identical_sentence_pair_count": sum(
                record["identical_sentences"] for record in pair_records
            ),
            "presentation_accuracy": {
                presentation: {
                    "accuracy": sum(
                        result["correct"]
                        for result in results
                        if result["presentation"] == presentation
                    )
                    / sum(
                        result["presentation"] == presentation
                        for result in results
                    ),
                    "count": sum(
                        result["presentation"] == presentation
                        for result in results
                    ),
                }
                for presentation in sorted(expected_presentations)
            },
            "by_category": by_category,
            "by_part": _accuracy_details(pair_records, "part"),
            "by_source": _accuracy_details(pair_records, "source"),
            "by_torfl_level": _accuracy_details(
                pair_records, "torfl_level"
            ),
        }
        return micro_accuracy, details

    def aggregation(self) -> Dict:
        return {"acc": self._aggregate_pair_accuracy}

    def leaderboard_aggregation(self, metrics: Dict) -> float:
        return metrics["acc"]

    def evaluate(self, sample, y_pred) -> Dict:
        if not isinstance(y_pred, dict) or not y_pred:
            predicted_choice = ""
        else:
            predicted_choice = max(y_pred.items(), key=lambda item: item[1])[0]
            predicted_choice = str(predicted_choice).strip()

        return {
            "acc": {
                "correct": predicted_choice == sample["correct_choice"],
                "row_id": sample["row_id"],
                "presentation": sample["presentation"],
                "category": sample["category"],
                "source": sample["source"],
                "part": sample["part"],
                "torfl_level": sample["torfl_level"],
                "identical_sentences": sample["identical_sentences"],
            }
        }

    def _select_test_dataset(self, dataset):
        if self.test_split_name() in dataset:
            return dataset[self.test_split_name()]
        if "train" in dataset:
            self.logger.warning(
                "RuParam dataset has no 'test' split; using 'train' as the "
                "evaluation split"
            )
            return dataset["train"]
        split_names = list(dataset.keys())
        if len(split_names) == 1:
            self.logger.warning(
                "RuParam dataset has no 'test' split; using its only split %r",
                split_names[0],
            )
            return dataset[split_names[0]]
        raise ValueError(
            "RuParam dataset must expose a 'test' split, a 'train' fallback, "
            f"or exactly one split; got {split_names}"
        )

    def _load_dataset(
        self,
        model: BaseLLM,
        max_prompt_len: int,
        max_sample_per_dataset: int,
        few_shot_count: int,
    ) -> List:
        if few_shot_count != 0:
            raise ValueError(
                "RuParam follows a zero-shot protocol and requires "
                "few_shot_count=0"
            )

        dataset = load_dataset(**self.dataset_args())
        test_dataset = self._select_test_dataset(dataset)
        selected_count = min(max_sample_per_dataset, len(test_dataset))
        test_dataset = test_dataset.select(range(selected_count))

        samples = []
        for row_index, raw_row in enumerate(tqdm(test_dataset)):
            row = _normalise_row(dict(raw_row), row_index)
            for presentation, correct_choice in (
                ("grammatical_first", "1"),
                ("grammatical_second", "2"),
            ):
                sample = copy.deepcopy(row)
                sample["presentation"] = presentation
                sample["correct_choice"] = correct_choice
                # Keep the historical field for result consumers while making
                # the less ambiguous presentation field authoritative.
                sample["order"] = "s" if correct_choice == "1" else "r"
                samples.append({
                    "messages": self._prepare_messages(
                        sample, model, max_prompt_len, 0, []
                    ),
                    "sample": sample,
                })
        return samples

    def create_messages(self, sample, with_answer=None) -> List[Dict]:
        sentences = [sample["gram"], sample["ungram"]]
        correct_choice = sample.get("correct_choice")
        if correct_choice is None:
            correct_choice = "1" if sample.get("order") == "s" else "2"
        if correct_choice == "2":
            sentences.reverse()

        inputs = {"sent_lhs": sentences[0], "sent_rhs": sentences[1]}
        messages = [{
            "role": "user",
            "content": self.instruction.format(**inputs),
        }]
        if with_answer:
            messages.append({"role": "assistant", "content": correct_choice})
        return messages

    def get_answer(self, sample):
        return sample.get(
            "correct_choice", "1" if sample.get("order") == "s" else "2"
        )
