"""Standalone RuLaw-ProofBench task for the shared legal suite."""

import copy
import hashlib
from pathlib import Path

from llmtf.base import PromptTooLongError, SimpleFewShotHFTask
from .data import load_rows, manifest, validate_split_separation
from .scoring import SCORER_NAME, aggregate, score


class RuLawProofBench(SimpleFewShotHFTask):
    method = "generate"
    ALLOW_BOOTSTRAPPING = False  # Confidence intervals must cluster by article.
    _max_task_new_tokens = 96
    dataset_config = "open"

    def __init__(self, mode="closed", judge_model=None, data_dir=None, **kwargs):
        super().__init__(**kwargs)
        if mode not in ("closed", "grounded"):
            raise ValueError("Unknown RuLaw-ProofBench mode")
        self.mode = mode
        self._judge_model = judge_model
        self.data_dir = data_dir

    def task_name(self):
        return "rulaw_proofbench/" + self.mode

    def dataset_args(self):
        spec = manifest()
        return {
            "path": spec["dataset_repo"],
            "name": self.dataset_config,
            "revision": spec["dataset_revision"],
        }

    def test_split_name(self):
        return "test"

    def prompt_split_name(self):
        return "train"

    def get_task_provenance(self):
        from .judge import provenance

        return {
            "mode": self.mode,
            "dataset": manifest(),
            "dataset_config": self.dataset_config,
            "implementation_sha256": {
                name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                for name in ("task.py", "data.py", "scoring.py", "judge.py")
            },
            "semantic_judge": provenance(self._judge_model) if self._judge_model is not None else None,
            "source_and_data_verification": "pinned_hub_revision_and_parquet_sha256",
            "primary": "article_macro_accuracy",
            "scorer": SCORER_NAME,
            "few_shot_policy": "fixed_train_order_article_disjoint_exact_k",
            "answer_budget": self.max_task_new_tokens,
        }

    def create_messages(self, sample, with_answer=False):
        prompt = sample["question"]
        if self.mode == "grounded":
            prompt = (
                "Ниже приведён текст нормы и связанных положений для контрольного режима чтения.\n\n"
                + sample["context"]
                + "\n\nВопрос:\n"
                + prompt
            )
        messages = [{"role": "user", "content": prompt}]
        if with_answer:
            messages.append({"role": "assistant", "content": sample["gold"]["value"]})
        return messages

    def _load_dataset(self, model, max_prompt_len, max_sample_per_dataset, few_shot_count):
        available = manifest()["demonstration_count"]
        if type(few_shot_count) is not int or not 0 <= few_shot_count <= available:
            raise ValueError(f"RuLaw-ProofBench requires few_shot_count between 0 and {available}")
        rows = load_rows(self.dataset_config, "test", self.data_dir)
        train = load_rows(self.dataset_config, "train", self.data_dir)
        if len(train) != available or len(rows) != manifest()["test_count"]:
            raise ValueError("Published split counts disagree with the pinned manifest")
        validate_split_separation(rows, train)
        demonstrations = train[:few_shot_count]
        prefix = []
        for demonstration in demonstrations:
            prefix.extend(self.create_messages(demonstration, with_answer=True))
        selected = rows[:max_sample_per_dataset]
        self.selected_ids = {row["id"] for row in selected}
        self.eligible_count = len(rows)
        result = []
        for raw in selected:
            row = copy.deepcopy(raw)
            messages = copy.deepcopy(prefix) + self.create_messages(row)
            count = model.count_tokens_for_messages(messages)
            if count is not None and count > max_prompt_len:
                raise PromptTooLongError(
                    f"RuLaw-ProofBench {self.mode} with {few_shot_count} demonstrations "
                    f"needs {count} prompt tokens; budget {max_prompt_len}"
                )
            row["_rulaw_proofbench"] = {
                "mode": self.mode,
                "requested_shots": few_shot_count,
                "effective_shots": len(demonstrations),
                "demonstration_ids": [demo["id"] for demo in demonstrations],
                "prompt_token_count": count,
            }
            result.append({"messages": messages, "sample": row})
        return result

    def evaluate(self, sample, y_pred):
        result = {"score": score(sample, y_pred)}
        if self._judge_model is not None:
            result["llm_judge_accuracy"] = {
                "id": sample["id"],
                "question": sample["question"],
                "reference": sample["gold"],
                "prediction": y_pred,
                "strict": result["score"],
            }
        return result

    def _aggregate(self, records):
        if {record["id"] for record in records} != self.selected_ids:
            raise ValueError("Incomplete RuLaw-ProofBench execution")
        value, details = aggregate(records)
        details.update(
            mode=self.mode,
            eligible_count=self.eligible_count,
            full_coverage=len(records) == self.eligible_count,
        )
        return value, details

    def _aggregate_judge(self, items):
        from .judge import aggregate_judgments, judge_records

        if {item["id"] for item in items} != self.selected_ids:
            raise ValueError("Incomplete judge coverage")
        return aggregate_judgments(items, judge_records(self._judge_model, items))

    def aggregation(self):
        result = {"score": self._aggregate}
        if self._judge_model is not None:
            result["llm_judge_accuracy"] = self._aggregate_judge
        return result

    def leaderboard_aggregation(self, metrics):
        return metrics["score"]
