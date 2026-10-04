"""Load public Hub tables at an immutable revision, with local hash checks."""

import hashlib
import json
from pathlib import Path
import unicodedata


def manifest():
    return json.loads(Path(__file__).with_name("manifest.json").read_text())


def question_key(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def load_rows(config="open", split="test", data_dir=None):
    import pyarrow.parquet as pq

    if config not in ("open", "mcq") or split not in ("train", "test"):
        raise ValueError("Unknown RuLaw-ProofBench configuration or split")
    spec = manifest()
    filename = f"data/{config}/{split}.parquet"
    expected = spec["files"].get(filename)
    if expected is None:
        if split == "train" and spec["demonstration_count"] == 0:
            return []
        raise ValueError(f"Missing pinned split: {filename}")
    if data_dir is None:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(
            repo_id=spec["dataset_repo"],
            repo_type="dataset",
            revision=spec["dataset_revision"],
            filename=filename,
            token=False,
        )
    else:
        path = Path(data_dir) / filename
    path = Path(path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected["sha256"]:
        raise ValueError(f"RuLaw-ProofBench SHA-256 mismatch: {filename}")
    public_rows = pq.read_table(path).to_pylist()
    if len(public_rows) != expected["rows"]:
        raise ValueError(f"Unexpected row count: {filename}")
    rows = []
    for public in public_rows:
        row = json.loads(public["record_json"])
        fields = {
            "id": row["id"],
            "article": row["article_key"],
            "domain": row["domain"],
            "as_of": row["as_of"],
            "answer": row["gold"]["value"],
            "answer_unit": row["gold"]["unit"],
        }
        if config == "mcq":
            mcq = json.loads(public["mcq_json"])
            fields.update(
                question=mcq["question"],
                choices=[option["text"] for option in mcq["options"]],
                answer_label=mcq["gold_label"],
            )
            options = mcq["options"]
            if (
                mcq["id"] != row["id"]
                or [option["label"] for option in options] != list("ABCD"[:len(options)])
                or sum(option["value"] == row["gold"]["value"] for option in options) != 1
                or next(option["value"] for option in options if option["label"] == mcq["gold_label"])
                != row["gold"]["value"]
            ):
                raise ValueError("MCQ parent/answer alignment failure")
            row["mcq"] = mcq
        else:
            fields["question"] = row["question"]
        if any(public[key] != value for key, value in fields.items()):
            raise ValueError(f"Public columns disagree with audit record: {row['id']}")
        if row["as_of"] != spec["as_of"] or not public["context"].strip():
            raise ValueError("Missing source context or wrong legal cutoff")
        row["context"] = public["context"]
        rows.append(row)
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate RuLaw-ProofBench IDs")
    if len({question_key(row["question"]) for row in rows}) != len(rows):
        raise ValueError("Duplicate RuLaw-ProofBench questions")
    return rows


def validate_split_separation(test, train):
    for key in ("id", "article_key"):
        if {row[key] for row in test} & {row[key] for row in train}:
            raise ValueError(f"Demonstration/test overlap: {key}")
    if {question_key(row["question"]) for row in test} & {
        question_key(row["question"]) for row in train
    }:
        raise ValueError("Demonstration/test question overlap")
