import ast
import json
import re
import unittest
from pathlib import Path

from benchmark.config import load_benchmark_config
from examples.run_custom_task import build_parser


ROOT = Path(__file__).resolve().parents[1]


class ExampleTaskTests(unittest.TestCase):
    def test_custom_task_contract(self):
        source = (ROOT / "examples" / "custom_task.py").read_text(
            encoding="utf-8"
        )
        tree = ast.parse(source)
        task_class = next(
            node for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "ToySentimentTask"
        )
        self.assertEqual(
            [base.id for base in task_class.bases if isinstance(base, ast.Name)],
            ["SimpleFewShotHFTask"],
        )
        methods = {
            node.name for node in task_class.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        self.assertTrue({
            "dataset_args", "test_split_name", "prompt_split_name",
            "create_messages", "evaluate", "aggregation",
        }.issubset(methods))
        self.assertIn("self._max_task_new_tokens = 1", source)

    def test_toy_data_has_balanced_train_and_test_splits(self):
        for split in ("train", "test"):
            path = ROOT / "examples" / "data" / f"toy_sentiment_{split}.jsonl"
            rows = [json.loads(line) for line in path.read_text(
                encoding="utf-8"
            ).splitlines()]
            self.assertEqual(len(rows), 4)
            self.assertEqual({row["label"] for row in rows}, {0, 1})
            self.assertTrue(all(row["text"] for row in rows))

    def test_custom_task_loads_when_runtime_dependencies_are_available(self):
        try:
            import datasets
        except ModuleNotFoundError:
            self.skipTest("datasets is installed in runtime profiles only")
        if not getattr(datasets, "__file__", None):
            self.skipTest("another dependency-free test installed a stub")

        from examples.custom_task import ToySentimentTask

        class FakeModel:
            @staticmethod
            def support_method(method):
                return method == "calculate_tokens_proba"

            @staticmethod
            def count_tokens_for_messages(messages):
                return sum(len(message["content"]) for message in messages)

        messages, samples = ToySentimentTask().load_dataset(
            model=FakeModel(), max_prompt_len=4096,
            max_sample_per_dataset=4, few_shot_count=2,
        )
        self.assertEqual(len(messages), 4)
        self.assertEqual(len(samples), 4)
        self.assertTrue(all(
            item["tokens_of_interest"] == ["0", "1"] for item in messages
        ))

    def test_smoke_yaml_uses_current_schema(self):
        config = load_benchmark_config(ROOT / "examples" / "benchmark_smoke.yaml")
        self.assertEqual(config.model.model_kind, "plain")
        self.assertEqual(len(config.tasks), 2)
        self.assertTrue(all(task.evaluation["max_sample_per_dataset"] == 8 for task in config.tasks))

    def test_runner_parser(self):
        args = build_parser().parse_args([
            "--base-url", "http://localhost:8000",
            "--model-name", "example-model",
        ])
        self.assertEqual(args.api_profile, "auto")
        self.assertEqual(args.backend, "api")
        self.assertEqual(args.gpu_memory_utilization, 0.92)
        self.assertEqual(args.max_samples, 4)

    def test_readme_relative_links_exist(self):
        readme = ROOT / "examples" / "README.md"
        links = re.findall(r"\[[^]]+\]\(([^)]+)\)", readme.read_text(
            encoding="utf-8"
        ))
        relative_links = [link for link in links if "://" not in link]
        self.assertGreater(len(relative_links), 5)
        missing = [
            link for link in relative_links
            if not (readme.parent / link.split("#", 1)[0]).exists()
        ]
        self.assertEqual(missing, [])


if __name__ == "__main__":
    unittest.main()
