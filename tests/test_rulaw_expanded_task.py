"""Expanded task contracts; synthetic rows, no Hub or model requests."""

import copy
import unittest
from unittest.mock import patch

from test_rulaw_proofbench import Model
from llmtf.tasks import TASK_REGISTRY
from llmtf.tasks.rulaw_proofbench import RuLawProofBench, RuLawProofBenchMCQ
from llmtf.tasks.rulaw_proofbench import task as task_module
from llmtf.tasks.rulaw_proofbench.data import validate_split_separation


def row(identity):
    return dict(id=identity, article_key=identity, question="Public case " + identity,
                context="Public norm " + identity, gold=dict(value="да", unit="none"),
                facts={"HIDDEN_FACT": True}, proof="HIDDEN_PROOF", question_type="complete",
                source_refs=[dict(url="https://pravo.gov.ru/?nd=" + identity, article="1")],
                mcq=dict(question="Public options " + identity, gold_label="A",
                         options=[dict(label="A", value="да"), dict(label="B", value="нет")]))


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
                    self.assertEqual(len(messages[0]["messages"]), 2 * shots + (2 if cls is RuLawProofBenchMCQ else 1))
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


if __name__ == "__main__":
    unittest.main()
