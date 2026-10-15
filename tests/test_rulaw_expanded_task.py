"""Expanded task contracts; synthetic rows, no Hub or model requests."""

import copy
import os
import unittest
from unittest.mock import patch

from test_rulaw_proofbench import Model
from llmtf.tasks import TASK_REGISTRY
from llmtf.tasks.rulaw_proofbench import RuLawProofBench, RuLawProofBenchMCQ
from llmtf.tasks.rulaw_proofbench import task as task_module
from llmtf.tasks.rulaw_proofbench.data import load_rows, validate_split_separation
from llmtf.tasks.rulaw_proofbench.prompts import PROMPT_VERSION
from llmtf.base import PromptTooLongError
from llmtf.provenance import fingerprint_run_config


def row(identity):
    return dict(id=identity, article_key=identity, question="Public case " + identity,
                title="Test act", article="1", as_of="2025-01-01",
                prompt_version="expanded_v1", answer_kind="binary",
                case_text="Test act, статья 1. Применяется выбранное положение в состоянии на 2025-01-01.\n"
                          "Public scope\nPublic fact\nPublic target " + identity,
                context="Public norm " + identity, gold=dict(value="да", unit="none"),
                facts={"HIDDEN_FACT": True}, proof="HIDDEN_PROOF", question_type="complete",
                pair_ids=[], unknown_but_determinate=False,
                source_refs=[dict(url="https://pravo.gov.ru/?nd=" + identity, article="1")],
                mcq=dict(question="Public options " + identity, gold_label="A",
                         options=[dict(label="A", text="да", value="да"),
                                  dict(label="B", text="нет", value="нет")]))


class ExpandedTaskTests(unittest.TestCase):
    def test_fixed_ids_zero_five_and_no_hidden_fields(self):
        train = [row(str(i)) for i in range(8)]
        test = [row("test")]
        selected_ids = ["5", "2", "7", "0", "3"]
        spec = dict(test_count=1, demonstration_count=8, demonstration_ids=selected_ids)

        def load(config, split, data_dir, version):
            self.assertEqual(version, "expanded")
            return copy.deepcopy(train if split == "train" else test)

        with patch.object(task_module, "manifest", return_value=spec), \
                patch.object(task_module, "load_rows", side_effect=load):
            for cls in (RuLawProofBench, RuLawProofBenchMCQ):
                task = cls(dataset_version="expanded")
                for shots in (0, 5):
                    messages, samples = task.load_dataset(Model(), 16000, 1, shots)
                    trace = samples[0]["sample"]["_rulaw_proofbench"]
                    self.assertEqual(trace["demonstration_ids"], selected_ids if shots else [])
                    self.assertNotIn("HIDDEN", str(messages))
                    sequence = messages[0]["messages"]
                    self.assertEqual(len(sequence), 2 * shots + 1)
                    self.assertEqual(sequence[-1]["role"], "user")
                    self.assertEqual(trace["prompt_version"], PROMPT_VERSION)
                    self.assertIsNone(trace["assistant_prefill"])
                    for index, identity in enumerate(selected_ids if shots else []):
                        self.assertIn("Public target " + identity, sequence[2 * index]["content"])
                        self.assertEqual(sequence[2 * index + 1], {
                            "role": "assistant", "content": "A" if cls is RuLawProofBenchMCQ else "да",
                        })
                    if cls is RuLawProofBenchMCQ:
                        self.assertEqual(messages[0]["tokens_of_interest"], ["A", "B"])
                for shots in (-1, 1, 4, 6, True):
                    with self.assertRaisesRegex(ValueError, "only few_shot_count 0 or 5"):
                        task.load_dataset(Model(), 16000, 1, shots)
            spec["demonstration_ids"] = ["0"] * 5
            with self.assertRaisesRegex(ValueError, "Invalid fixed"):
                RuLawProofBench(dataset_version="expanded").load_dataset(Model(), 16000, 1, 5)

    def test_dependency_overlap_across_distinct_articles_and_editions(self):
        test, train = [row("test")], [row("train")]
        test[0]["source_refs"].append(dict(url="http://pravo.gov.ru/proxy/ips/?nd=102&rdk=5", article="12"))
        train[0]["source_refs"].append(dict(url="https://pravo.gov.ru/proxy/ips/?page=all&rdk=9&nd=102", article="12"))
        with self.assertRaisesRegex(ValueError, "normative dependency overlap"):
            validate_split_separation(test, train, dependencies=True)

    def test_registry_has_separate_explicit_tasks(self):
        for view in ("closed", "grounded", "mcq_closed", "mcq_grounded"):
            name = "rulaw_proofbench_expanded/" + view
            spec = TASK_REGISTRY[name]
            self.assertFalse(spec["include_in_all"])
            task = spec["class"](**spec["params"])
            self.assertEqual(task.task_name(), name)
            self.assertIn("rulaw_proofbench/" + view, TASK_REGISTRY)

    def test_prompt_provenance_changes_cache_identity(self):
        for cls in (RuLawProofBench, RuLawProofBenchMCQ):
            task = cls(dataset_version="expanded")
            provenance = task.get_task_provenance()
            self.assertEqual(provenance["prompt_version"], PROMPT_VERSION)
            self.assertIsNone(provenance["assistant_prefill"])
            self.assertEqual(len(provenance["implementation_sha256"]["prompts.py"]), 64)
            self.assertEqual(provenance["dataset"]["prompt_version"], "expanded_v1")
            old = copy.deepcopy(provenance)
            old.pop("prompt_version")
            old.pop("assistant_prefill")
            old["implementation_sha256"].pop("prompts.py")
            self.assertNotEqual(fingerprint_run_config(old), fingerprint_run_config(provenance))
            changed_renderer = copy.deepcopy(provenance)
            changed_renderer["implementation_sha256"]["prompts.py"] = "0" * 64
            self.assertNotEqual(fingerprint_run_config(changed_renderer), fingerprint_run_config(provenance))

    def test_legacy_prefill_is_preserved(self):
        sample = row("legacy")
        task = RuLawProofBenchMCQ()
        self.assertEqual(task.create_messages(sample)[-1], {"role": "assistant", "content": "Ответ:"})
        self.assertEqual(task.create_messages(sample, with_answer=True)[-1]["content"], "Ответ: A")
        self.assertEqual(RuLawProofBench().create_messages(sample)[0]["content"], sample["question"])

    def test_judge_receives_the_rendered_question(self):
        sample = row("judge")
        task = RuLawProofBench(mode="grounded", dataset_version="expanded", judge_model=object())
        result = task.evaluate(sample, "да")
        self.assertEqual(result["llm_judge_accuracy"]["question"], task.create_messages(sample)[0]["content"])
        self.assertIn(sample["context"], result["llm_judge_accuracy"]["question"])

    def test_full_prompt_budget_does_not_trim_demonstrations(self):
        train = [row(str(i)) for i in range(5)]
        test = [row("test")]
        spec = dict(test_count=1, demonstration_count=5, demonstration_ids=[r["id"] for r in train])
        seen = []
        model = Model()

        def count(messages):
            seen.append(copy.deepcopy(messages))
            return sum(len(message["content"]) for message in messages)

        model.count_tokens_for_messages = count
        with patch.object(task_module, "manifest", return_value=spec), patch.object(
            task_module, "load_rows", side_effect=lambda config, split, *a, **k: copy.deepcopy(
                train if split == "train" else test
            )
        ):
            for cls in (RuLawProofBench, RuLawProofBenchMCQ):
                task = cls(mode="grounded", dataset_version="expanded")
                with self.assertRaises(PromptTooLongError):
                    task.load_dataset(model, 10, 1, 5)
                self.assertEqual(len(seen[-1]), 11)
                self.assertEqual(seen[-1][-1]["role"], "user")
                for message in seen[-1][::2]:
                    self.assertIn("Public norm", message["content"])

    @unittest.skipUnless(os.environ.get("RULAW_EXPANDED_DATA_DIR"), "Requires pinned local expanded Parquet")
    def test_all_published_records_and_few_shot_messages(self):
        data_dir = os.environ["RULAW_EXPANDED_DATA_DIR"]
        for cls, config in ((RuLawProofBench, "open"), (RuLawProofBenchMCQ, "mcq")):
            test = load_rows(config, "test", data_dir, version="expanded")
            train = load_rows(config, "train", data_dir, version="expanded")
            self.assertEqual((len(test), len(train)), (505, 20))
            for mode in ("closed", "grounded"):
                task = cls(mode=mode, dataset_version="expanded", data_dir=data_dir)
                originals = copy.deepcopy(test + train)
                for sample in test + train:
                    prompt = task.create_messages(sample)[0]["content"]
                    parts = sample["case_text"].split("\n")
                    self.assertIn("\n".join(parts[:2]), prompt)
                    self.assertIn("— " + "\n— ".join(parts[2:-1]), prompt)
                    self.assertIn("ЗАДАНИЕ\n" + parts[-1], prompt)
                    self.assertEqual(sample["context"] in prompt, mode == "grounded")
                    if config == "mcq":
                        self.assertIn("\n".join(o["label"] + ". " + o["text"] for o in sample["mcq"]["options"]), prompt)
                    self.assertEqual(task.create_messages(sample)[-1]["role"], "user")
                    answer = sample["mcq"]["gold_label"] if config == "mcq" else sample["gold"]["value"]
                    self.assertEqual(task.create_messages(sample, with_answer=True)[-1]["content"], answer)
                self.assertEqual(test + train, originals)
                for shots in (0, 5):
                    messages, samples = task.load_dataset(Model(), 16000, 505, shots)
                    self.assertEqual([s["sample"]["id"] for s in samples], [s["id"] for s in test])
                    for message, sample, original in zip(messages, samples, test):
                        self.assertEqual(len(message["messages"]), 2 * shots + 1)
                        self.assertEqual(message["messages"][-1]["role"], "user")
                        for key in ("gold", "proof", "facts", "mcq", "context", "case_text"):
                            self.assertEqual(sample["sample"][key], original[key])
                        trace = sample["sample"]["_rulaw_proofbench"]
                        self.assertEqual(trace["demonstration_ids"], task._manifest()["demonstration_ids"] if shots else [])


if __name__ == "__main__":
    unittest.main()
