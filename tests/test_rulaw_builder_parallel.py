"""Scheduling, isolation and failure regression checks without an API."""

import copy
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dev.rulaw_proofbench.builder.common import Rejected, digest, read_json
from dev.rulaw_proofbench.builder.pipeline import process_group, process_article
from dev.rulaw_proofbench.builder.progress import Progress
from dev.rulaw_proofbench.builder.sources import Sources


def candidate(number):
    return dict(article=str(number), suitable=True, reason="fixture",
                source=dict(id="law", split="test"))


def row(number):
    return dict(id=str(number), question=f"Question {number}", article_key=f"law:{number}", pair_ids=[],
                source_refs=[dict(url="https://pravo.gov.ru/law", article=str(number))])


class ParallelTests(unittest.TestCase):
    def test_idle_worker_starts_reserved_article_before_slow_head_finishes(self):
        reserved_started = threading.Event()
        lock = threading.Lock()
        active = 0
        peak = 0

        def build(item, *args):
            nonlocal active, peak
            n = int(item["article"])
            with lock:
                active += 1
                peak = max(peak, active)
            try:
                if n == 0:
                    self.assertTrue(reserved_started.wait(timeout=3), "A free worker never started the reserve")
                elif n == 4:
                    reserved_started.set()
                return [row(n)], dict(article=f"law:{n}", status="accepted", count=1)
            finally:
                with lock:
                    active -= 1

        with tempfile.TemporaryDirectory() as tmp, patch(
                "dev.rulaw_proofbench.builder.pipeline.process_article", side_effect=build):
            splits = dict(test=[], train=[])
            decisions = []
            process_group([candidate(i) for i in range(12)], 10, "test", dict(questions_per_article=1),
                          Path(tmp), False, Progress(False), splits, decisions, 4)
            self.assertEqual([r["id"] for r in splits["test"]], [str(i) for i in range(10)])
            self.assertLessEqual(peak, 4)

    def test_prefetch_preserves_serial_result_with_rejections_and_uneven_yield(self):
        outputs = []
        for workers in (1, 4):
            def build(item, *args):
                n = int(item["article"])
                time.sleep(.02 if n == 0 else .001)
                if n % 3 == 1:
                    return [], dict(article=f"law:{n}", status="rejected")
                rows = [row(n * 10 + i) for i in range(1 if n % 2 else 2)]
                return rows, dict(article=f"law:{n}", status="accepted", count=len(rows))
            with tempfile.TemporaryDirectory() as tmp, patch(
                    "dev.rulaw_proofbench.builder.pipeline.process_article", side_effect=build):
                splits = dict(test=[], train=[])
                process_group([candidate(i) for i in range(25)], 11, "test", dict(questions_per_article=2),
                              Path(tmp), False, Progress(False), splits, [], workers)
                outputs.append(splits)
        self.assertEqual(*outputs)
        self.assertEqual(len(outputs[0]["test"]), 11)

    def test_completion_order_and_quota_do_not_change_dataset(self):
        results = []
        for workers in (1, 3):
            calls = []
            def build(item, *args):
                number = int(item["article"])
                calls.append(number)
                time.sleep(.025 if number == 0 else .001)
                return [row(number)], dict(article=f"law:{number}", status="accepted", count=1)
            with tempfile.TemporaryDirectory() as tmp, patch(
                    "dev.rulaw_proofbench.builder.pipeline.process_article", side_effect=build):
                splits = dict(test=[], train=[])
                decisions = []
                process_group([candidate(i) for i in range(20)], 2, "test", {}, Path(tmp),
                              False, Progress(False), splits, decisions, workers)
                results.append(splits)
                self.assertLessEqual(len(calls), 2 + workers - 1)
                self.assertEqual([r["id"] for r in splits["test"]], ["0", "1"])
                if workers > 1:
                    self.assertTrue(any(d["status"] == "saved_surplus" for d in decisions))
        self.assertEqual(*results)

    def test_dependency_leak_is_rejected_before_quota(self):
        with tempfile.TemporaryDirectory() as tmp:
            splits = dict(test=[], train=[row(0)])
            decisions = []
            def build(item, *args):
                n = int(item["article"])
                return [row(n)], dict(article=f"law:{n}", status="accepted", count=1)
            with patch("dev.rulaw_proofbench.builder.pipeline.process_article", side_effect=build):
                process_group([candidate(i) for i in range(3)], 1, "test", {}, Path(tmp),
                              False, Progress(False), splits, decisions, 2)
            self.assertEqual([r["id"] for r in splits["test"]], ["1"])
            self.assertEqual(decisions[0]["status"], "rejected")

    def test_article_error_keeps_other_clients_and_journals_isolated(self):
        clients = []
        barrier = threading.Barrier(2)
        class Client:
            def __init__(self, root, *args, **kwargs):
                self.root = root
                self.trace = []
                clients.append(self)
        def build(item, config, client, directory, count):
            number = item["article"]
            client.trace.append("request-" + number)
            barrier.wait(timeout=5)
            self.assertEqual(client.trace, ["request-" + number])
            if number == "0":
                raise RuntimeError("API transport failed")
            return [row(1)]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            splits = dict(test=[], train=[])
            decisions = []
            with patch("dev.rulaw_proofbench.builder.pipeline.Client", Client), patch(
                    "dev.rulaw_proofbench.builder.pipeline.build_candidate", side_effect=build):
                process_group([candidate(0), candidate(1)], 1, "test", dict(api={}, questions_per_article=1),
                              root, False, Progress(False), splits, decisions, 2)
            self.assertEqual([d["status"] for d in decisions], ["error", "accepted"])
            self.assertEqual(len(clients), 2)
            self.assertFalse(list(root.rglob("sources")))
            for number in (0, 1):
                directory = root / "stages" / digest(f"law:{number}")[:16]
                self.assertEqual(read_json(directory / "request-ids.json"), [f"request-{number}"])
                self.assertTrue((directory / "result.json").is_file())

    def test_http_failure_is_journalled_without_automatic_retry(self):
        import urllib.error
        from dev.rulaw_proofbench.builder.client import Client
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = Client(root, dict(model="fixture", request_options=dict(max_tokens=50)))
            client.key = "fixture-secret"
            error = urllib.error.HTTPError(client.url, 502, "failure", {}, None)
            with patch.object(client.opener, "open", side_effect=error) as call:
                with self.assertRaisesRegex(RuntimeError, "HTTP 502"):
                    client.complete([dict(role="user", content="test")])
                with self.assertRaisesRegex(Rejected, "Previous request failed"):
                    client.complete([dict(role="user", content="test")])
                self.assertEqual(call.call_count, 1)
            self.assertIsNone(client.spent())


if __name__ == "__main__":
    unittest.main()
