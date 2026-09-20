import json
from pathlib import Path
import tempfile
import unittest

from show_results import read_jsonl, save_table


class ShowResultsFormatTests(unittest.TestCase):
    def _write(self, content):
        directory = tempfile.TemporaryDirectory()
        path = Path(directory.name) / "samples.jsonl"
        path.write_text(content, encoding="utf-8")
        self.addCleanup(directory.cleanup)
        return path

    def test_reads_current_json_array(self):
        records = [
            {"metric": {"acc": 1}},
            {"metric": {"acc": 0}},
        ]
        path = self._write(json.dumps(records, indent=4))
        self.assertEqual(read_jsonl(path), records)

    def test_reads_historical_concatenated_objects(self):
        path = self._write(
            '{"metric": {"acc": 1}}\n{"metric": {"acc": 0}}\n'
        )
        self.assertEqual(
            read_jsonl(path),
            [{"metric": {"acc": 1}}, {"metric": {"acc": 0}}],
        )

    def test_rejects_non_object_samples(self):
        path = self._write('[1, 2]')
        with self.assertRaisesRegex(ValueError, "sample record"):
            read_jsonl(path)

    def test_table_allows_missing_time(self):
        with tempfile.TemporaryDirectory() as directory:
            save_table(
                {"model": {"task": {"score": "0.500"}}},
                directory,
                show_time=True,
            )
            report = (Path(directory) / "results.md").read_text("utf-8")
        self.assertIn("| model | 0.500 | — | 0.500 | — |", report)


if __name__ == "__main__":
    unittest.main()
