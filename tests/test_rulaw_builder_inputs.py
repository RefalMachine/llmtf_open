"""The preparation/build boundary: manual inputs, immutability and missing norms."""

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_rulaw_builder import prepared_fixture, input_scope, fixture
from dev.rulaw_proofbench.builder.common import Rejected, digest, read_json, write_json
from dev.rulaw_proofbench.builder.packet import load_packet, seal_packet
from dev.rulaw_proofbench.builder.pipeline import run
from dev.rulaw_proofbench.builder.preparation import run as prepare


def build_config():
    return dict(as_of="2025-01-01", seed=555, test_count=2, train_count=0,
                questions_per_article=2, workers=2, api={})


class PacketTests(unittest.TestCase):
    def test_manual_seal_needs_no_documents_api_or_old_pilots(self):
        with tempfile.TemporaryDirectory() as tmp, patch(
                "urllib.request.OpenerDirector.open", side_effect=AssertionError("No network")):
            path = Path(tmp) / "input.json"
            sealed = seal_packet(prepared_fixture(), path)
            self.assertEqual(load_packet(path), sealed)
            changed = prepared_fixture()
            changed["articles"][0]["target"] = "Another target"
            with self.assertRaisesRegex(Rejected, "new version"):
                seal_packet(changed, path)

    def test_package_and_fragment_hashes_are_separate_gates(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "input.json"
            sealed = seal_packet(prepared_fixture(), path)
            sealed["packet"]["articles"][0]["target"] = "tampered"
            write_json(path, sealed)
            with self.assertRaisesRegex(Rejected, "package checksum"):
                load_packet(path)
            sealed["sha256"] = digest(sealed["packet"])
            sealed["packet"]["articles"][0]["fragments"][0]["text"] += "tampered"
            sealed["sha256"] = digest(sealed["packet"])
            write_json(path, sealed)
            with self.assertRaisesRegex(Rejected, "text checksum"):
                load_packet(path)

    def test_scope_source_version_and_split_are_mandatory(self):
        variants = []
        p = prepared_fixture()
        p["articles"][0]["input_scope"]["temporal_constraints"] = ""
        variants.append(p)
        p = prepared_fixture()
        p["articles"][0]["source"]["sha256"] = "0" * 64
        variants.append(p)
        p = prepared_fixture()
        duplicate = copy.deepcopy(p["articles"][0])
        duplicate["source"].update(id="other", split="train")
        p["articles"].append(duplicate)
        variants.append(p)
        p = prepared_fixture()
        p["status"] = "draft"
        variants.append(p)
        p = prepared_fixture()
        p["articles"][0]["source_basis"] = {"documents": []}
        variants.append(p)
        with tempfile.TemporaryDirectory() as tmp:
            for i, value in enumerate(variants):
                with self.subTest(i=i), self.assertRaises((Rejected, KeyError)):
                    seal_packet(value, Path(tmp) / f"{i}.json")

    def test_input_gap_returns_to_preparation_without_download_or_revision(self):
        observed = []
        class Client:
            def __init__(self, *args, **kwargs):
                self.trace = []

            def ask(self, stage, data, validate):
                observed.append((stage, data))
                return dict(reject="Missing a required exception provision", kind="input_gap")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inputs = root / "input.json"
            seal_packet(prepared_fixture(), inputs)
            before = inputs.read_bytes()
            write_json(root / "config.json", build_config())
            with patch("dev.rulaw_proofbench.builder.pipeline.Client", Client), patch(
                    "dev.rulaw_proofbench.builder.sources.Sources.fetch", side_effect=AssertionError("Build fetched a document")):
                report = run(root / "config.json", root / "run", input_path=inputs, progress=False)
                self.assertEqual(report["status"], "shortfall")
                self.assertEqual(report["decisions"][0]["status"], "input_gap")
                self.assertTrue(report["decisions"][0]["return_to_preparation"])
                self.assertEqual([stage for stage, data in observed], ["formalize"])
                self.assertNotIn("source_basis", observed[0][1])
                self.assertFalse(list((root / "run").rglob("revisions")))
                self.assertEqual(before, inputs.read_bytes())
                # Offline resume has only the frozen package, without the original inputs.
                inputs.unlink()
                replayed = run(root / "config.json", root / "run", offline=True, progress=False)
                self.assertEqual(report, replayed)

    def test_resealed_new_input_requires_new_output_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            p = prepared_fixture()
            seal_packet(p, root / "first.json")
            config = build_config()
            config["test_quotas"] = {"sk": 2}
            write_json(root / "config.json", config)
            with patch("dev.rulaw_proofbench.builder.pipeline.process_group"):
                run(root / "config.json", root / "out", input_path=root / "first.json", progress=False)
                p["articles"][0]["input_scope"]["description"] += " Changed boundary."
                seal_packet(p, root / "second.json")
                with self.assertRaisesRegex(Rejected, "inputs/code changed"):
                    run(root / "config.json", root / "out", input_path=root / "second.json", progress=False)

    def test_final_audit_input_gap_takes_priority_over_card_revision(self):
        stages = []
        class Client:
            def __init__(self, output, *args, **kwargs):
                self.output = output
                self.trace = []

            def ask(self, stage, data, validate):
                stages.append(stage)
                if stage in ("formalize", "reconstruct"):
                    value = fixture(data["fragments"][0]["source_id"])
                elif stage == "align":
                    value = dict(mapping=[dict(index=w["index"], worlds=[w["facts"]], reason="Test")
                                          for w in data["primary_worlds"]])
                elif stage == "blind_solve":
                    saved = read_json(self.output / "candidates.json")
                    value = dict(answers=[dict(id=r["id"], answer=r["gold"]["value"], reason="Test") for r in saved])
                else:
                    assert stage == "language_review", stage
                    assert "source_basis" not in data
                    value = dict(verdicts=[dict(id=q["id"], accept=True, reason="Test") for q in data["questions"]],
                                 input_issues=["Missing transition rule"], card_issues=["Revise scope"], suggestions=[])
                validate(value)
                return value

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            seal_packet(prepared_fixture(), root / "inputs.json")
            write_json(root / "config.json", build_config())
            with patch("dev.rulaw_proofbench.builder.pipeline.Client", Client), patch(
                    "dev.rulaw_proofbench.builder.sources.Sources.fetch", side_effect=AssertionError("No downloads")):
                report = run(root / "config.json", root / "out", input_path=root / "inputs.json", progress=False)
            self.assertEqual(report["decisions"][0]["status"], "input_gap")
            self.assertEqual(report["actual"]["test"], 0)
            self.assertEqual(stages.count("formalize"), 1)
            self.assertFalse(list((root / "out/stages").glob("*/revisions")))

    def test_preparation_emits_norms_without_edition_evidence(self):
        class Client:
            def __init__(self, *args, **kwargs):
                self.trace = []

            def ask(self, stage, data, validate):
                if stage == "select":
                    value = dict(candidates=[dict(article="57", suitable=True, locator="test", target="Target?", reason="test")])
                else:
                    self.assert_source_stage(stage)
                    evidence = [dict(source_id=data["documents"][0]["source_id"], quote="учёт мнения")]
                    value = dict(status="accept", reason="HISTORY_ONLY", evidence=evidence, dependencies=[],
                                 requests=[], unresolved=[], task_scope=input_scope(),
                                 temporal=dict(status="verified", reason="HISTORY_ONLY", evidence=evidence))
                validate(value)
                return value

            def assert_source_stage(self, stage):
                assert stage == "source_review", stage

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = prepared_fixture()["articles"][0]["source"]
            raw = prepared_fixture()["articles"][0]["fragments"][0]["text"].encode()
            (root / "law.txt").write_bytes(raw)
            source = dict(source, articles=["57"], path="law.txt")
            config = dict(as_of="2025-01-01", sources=[source], workers=2, api=dict(model="fixture"))
            write_json(root / "config.json", config)
            with patch("dev.rulaw_proofbench.builder.preparation.Client", Client):
                report = prepare(root / "config.json", root / "prep", progress=False)
                self.assertEqual(report["status"], "prepared")
                sealed = load_packet(root / "prep/input-package.json")
                self.assertEqual(len(sealed["packet"]["articles"]), 1)
                self.assertNotIn("HISTORY_ONLY", json.dumps(sealed))
                self.assertTrue(list((root / "prep/stages").glob("*/source-basis.json")))
                before = (root / "prep/input-package.json").read_bytes()
                (root / "law.txt").unlink()
                replayed = prepare(root / "config.json", root / "prep", offline=True, progress=False)
                self.assertEqual(report, replayed)
                self.assertEqual(before, (root / "prep/input-package.json").read_bytes())


if __name__ == "__main__":
    unittest.main()
