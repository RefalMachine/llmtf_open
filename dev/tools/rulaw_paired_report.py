"""Paired article-macro intervals from complete RuLaw evaluation artifacts.

python -m dev.tools.rulaw_paired_report LEFT_total.jsonl RIGHT_total.jsonl --output report.json
Uses only the standard library; performs no inference.
"""

import argparse
from collections import defaultdict
import importlib.util
import json
from pathlib import Path
import random

from dev.rulaw_proofbench.builder.packet import provision_key
from llmtf.provenance import fingerprint_run_config


def read_run(path):
    path = Path(path)
    if not path.name.endswith("_total.jsonl"):
        raise ValueError("Expected a task _total.jsonl file")
    stem = str(path).removesuffix("_total.jsonl")
    total = json.loads(path.read_text())
    params = json.loads(Path(stem + "_params.jsonl").read_text())
    config = total["run_config"]
    fingerprint = fingerprint_run_config(config)
    if total["run_fingerprint"] != fingerprint or params["run_fingerprint"] != fingerprint or params["run_config"] != config:
        raise ValueError("Params/total identity mismatch")
    provenance = config["task"]["provenance"]
    root = Path(__file__).resolve().parents[2] / "llmtf/tasks/rulaw_proofbench"
    spec = importlib.util.spec_from_file_location("rulaw_report_scoring", root / "scoring.py")
    scoring = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scoring)
    records, samples = {}, {}
    for artifact in json.loads(Path(stem + ".jsonl").read_text()):
        row, recorded = artifact["sample"], artifact["metric"]["score"]
        identity = row["id"]
        if identity in records:
            raise ValueError("Duplicate evaluated ID")
        prediction = artifact["predict"]
        if provenance["dataset_config"] == "mcq":
            options = row["mcq"]["options"]
            if not isinstance(prediction, dict) or set(prediction) != {o["label"] for o in options}:
                raise ValueError("MCQ candidates mismatch")
            if any(type(p) not in (int, float) or not 0 <= p <= 1 for p in prediction.values()) or not any(prediction.values()):
                raise ValueError("Invalid MCQ probabilities")
            chosen = max(options, key=lambda option: prediction[option["label"]])
            prediction = chosen["value"]
        replay = scoring.score(row, prediction)
        if any(recorded.get(key) != value for key, value in replay.items()):
            raise ValueError("Stored score differs from replay")
        records[identity], samples[identity] = replay, row
    value, details = scoring.aggregate(list(records.values()))
    if len(records) != provenance["dataset"]["test_count"]:
        raise ValueError("Paired reports require complete test runs")
    if abs(value - total["results"]["score"]) > 1e-12:
        raise ValueError("Replayed total differs")
    return dict(source=str(path), config=config, records=records, samples=samples, details=details)


def dependency_groups(samples):
    """Connected article components sharing a primary or dependent provision."""
    parents = {row["article_key"]: row["article_key"] for row in samples}

    def find(article):
        while parents[article] != article:
            parents[article] = parents[parents[article]]
            article = parents[article]
        return article

    owners = {}
    for row in samples:
        article = row["article_key"]
        refs = row["source_refs"] + [dict(url=row["source_url"], article=row["article"])]
        for ref in refs:
            key = provision_key(ref["url"], ref["article"])
            if key in owners:
                parents[find(article)] = find(owners[key])
            owners[key] = article
    groups = defaultdict(list)
    for article in sorted(parents):
        groups[find(article)].append(article)
    return sorted(groups.values())


def interval(deltas, groups, iterations, seed):
    """Resample clusters, retaining equal article weights within each draw."""
    if iterations < 100 or not groups:
        raise ValueError("Need nonempty clusters and at least 100 bootstrap iterations")
    if len(groups) == 1:
        return dict(cluster_count=1, iterations=0, seed=seed, percentile_95=None,
                    reason="Cannot estimate uncertainty from one independent cluster")
    rng = random.Random(seed)
    draws = []
    for _ in range(iterations):
        articles = [article for group in rng.choices(groups, k=len(groups)) for article in group]
        draws.append(sum(deltas[article] for article in articles) / len(articles))
    draws.sort()
    return dict(cluster_count=len(groups), iterations=iterations, seed=seed,
                percentile_95=[draws[int(0.025 * (iterations - 1))], draws[int(0.975 * (iterations - 1))]])


def paired(left, right, iterations=10000, seed=555):
    left_spec = left["config"]["task"]["provenance"]["dataset"]
    right_spec = right["config"]["task"]["provenance"]["dataset"]
    if (left_spec != right_spec or left["records"].keys() != right["records"].keys()
            or len(left["records"]) != left_spec["test_count"]):
        raise ValueError("Paired runs must use the same complete dataset version and IDs")
    by_article = defaultdict(list)
    for identity in sorted(left["records"]):
        a, b = left["samples"][identity], right["samples"][identity]
        for key in ("article_key", "question", "context", "facts", "gold", "source_refs", "source_url", "article", "mcq"):
            if a[key] != b[key]:
                raise ValueError("Paired samples differ: " + key)
        by_article[a["article_key"]].append(int(right["records"][identity]["correct"]) - int(left["records"][identity]["correct"]))
    deltas = {article: sum(values) / len(values) for article, values in by_article.items()}
    if not deltas:
        raise ValueError("Empty paired runs")
    groups = dependency_groups(list(left["samples"].values()))
    return dict(direction="right minus left", primary="article_macro_accuracy",
                delta=sum(deltas.values()) / len(deltas), article_count=len(deltas),
                article_bootstrap=interval(deltas, [[a] for a in sorted(deltas)], iterations, seed),
                dependency_bootstrap=interval(deltas, groups, iterations, seed),
                dependency_groups=groups)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=10000)
    args = parser.parse_args()
    left, right = read_run(args.left), read_run(args.right)
    report = dict(comparison=paired(left, right, args.iterations),
                  left={k: left[k] for k in ("source", "config", "details")},
                  right={k: right[k] for k in ("source", "config", "details")})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
