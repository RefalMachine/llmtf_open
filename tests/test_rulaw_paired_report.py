"""Clustered comparisons must preserve article weights and exact pairing."""

import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from dev.tools.rulaw_paired_report import dependency_groups, interval, paired, read_run
from llmtf.provenance import fingerprint_run_config


def sample(identity, article, dependencies=()):
    return dict(id=identity, article_key=article, article=article,
                source_url="http://pravo.gov.ru/proxy/ips/?nd=1&rdk=2",
                source_refs=[dict(url=url, article=number) for url, number in dependencies],
                question="case " + identity, context="norm", facts={}, gold={"value": "да"}, mcq={})


class PairedReportTests(unittest.TestCase):
    def test_reads_evaluator_files_and_rejects_score_or_identity_changes(self):
        spec = importlib.util.spec_from_file_location("rulaw_test_scoring",
                    Path(__file__).resolve().parents[1] / "llmtf/tasks/rulaw_proofbench/scoring.py")
        scoring = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(scoring)
        row = sample("1", "a")
        row.update(gold=dict(value="да", unit="none"), question_type="complete",
                   pair_ids=[], unknown_but_determinate=False, origin="retained",
                   mcq=dict(options=[dict(label="A", value="да"), dict(label="B", value="нет")]))
        for mode in ("open", "mcq"):
            config = {"task": {"provenance": {"dataset": {"test_count": 1}, "dataset_config": mode}}}
            identity = dict(run_config=config, run_fingerprint=fingerprint_run_config(config))
            prediction = "да" if mode == "open" else {"A": 0.5, "B": 0.5}
            record = dict(sample=row, predict=prediction, metric={"score": scoring.score(row, "да")})
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                total = root / "task_total.jsonl"
                total.write_text(json.dumps(dict(identity, results=dict(score=1.0))))
                params = root / "task_params.jsonl"
                params.write_text(json.dumps(identity))
                samples = root / "task.jsonl"
                samples.write_text(json.dumps([record]))
                self.assertTrue(read_run(total)["records"]["1"]["correct"])
                record["metric"]["score"]["correct"] = False
                samples.write_text(json.dumps([record]))
                with self.assertRaisesRegex(ValueError, "Stored score differs"):
                    read_run(total)
                params.write_text(json.dumps(dict(identity, run_fingerprint="changed")))
                with self.assertRaisesRegex(ValueError, "identity mismatch"):
                    read_run(total)

    def test_transitive_dependencies_ignore_edition_alias(self):
        rows = [sample("1", "a", [("http://pravo.gov.ru/proxy/ips/?nd=2&rdk=1", "7")]),
                sample("2", "b", [("https://pravo.gov.ru/proxy/ips/?rdk=9&nd=2", "7"),
                                  ("http://pravo.gov.ru/proxy/ips/?nd=3", "8")]),
                sample("3", "c", [("http://pravo.gov.ru/proxy/ips/?nd=3", "8")]),
                sample("4", "d")]
        self.assertEqual(dependency_groups(rows), [["a", "b", "c"], ["d"]])

    def test_point_estimate_weights_articles_not_questions_or_components(self):
        rows = [sample("1", "a"), sample("2", "a"), sample("3", "b"), sample("4", "c")]
        config = {"task": {"provenance": {"dataset": {"test_count": 4}}}}
        left = dict(config=config, samples={r["id"]: r for r in rows},
                    records={r["id"]: {"correct": r["article_key"] == "c"} for r in rows})
        right = copy.deepcopy(left)
        right["records"] = {r["id"]: {"correct": r["article_key"] != "c"} for r in rows}
        result = paired(left, right, iterations=100)
        self.assertAlmostEqual(result["delta"], 1 / 3)
        self.assertEqual(result, paired(left, right, iterations=100))
        # Different component sizes must not change the article weighting.
        with patch("random.Random.choices", return_value=[["a", "b"], ["c"]]):
            clustered = interval({"a": 1, "b": 1, "c": -1}, [["a", "b"], ["c"]], 100, 555)
        self.assertEqual(clustered["percentile_95"], [1 / 3, 1 / 3])
        self.assertIsNone(interval({"a": 1}, [["a"]], 100, 555)["percentile_95"])
        right["samples"]["1"]["question"] = "changed"
        with self.assertRaisesRegex(ValueError, "samples differ"):
            paired(left, right, iterations=100)

    def test_missing_or_different_dataset_is_rejected(self):
        config = {"task": {"provenance": {"dataset": {"test_count": 1}}}}
        left = dict(config=config, records={"1": {"correct": True}}, samples={"1": sample("1", "a")})
        right = copy.deepcopy(left)
        right["records"].clear()
        with self.assertRaisesRegex(ValueError, "same complete dataset"):
            paired(left, right)
        right = copy.deepcopy(left)
        right["config"]["task"]["provenance"]["dataset"]["version"] = "other"
        with self.assertRaisesRegex(ValueError, "same complete dataset"):
            paired(left, right)


if __name__ == "__main__":
    unittest.main()
