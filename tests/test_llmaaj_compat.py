import json
import os
import tempfile
import unittest

from benchmark.llmaaj.judge_llmaaj import load_llmtf_results


class LLMAsJudgeCompatibilityTests(unittest.TestCase):
    def test_new_json_array_sample_format_is_consumed(self):
        handle, path = tempfile.mkstemp(suffix='.jsonl')
        os.close(handle)
        try:
            with open(path, 'w', encoding='utf-8') as output:
                json.dump([
                    {'metric': {'score': {'winner': 'model'}}},
                    {'metric': {'score': {'winner': 'reference'}}},
                ], output)
            self.assertEqual(
                load_llmtf_results(path),
                [{'winner': 'model'}, {'winner': 'reference'}],
            )
        finally:
            os.unlink(path)

    def test_legacy_concatenated_object_format_is_rejected(self):
        handle, path = tempfile.mkstemp(suffix='.jsonl')
        os.close(handle)
        try:
            with open(path, 'w', encoding='utf-8') as output:
                output.write('{}\n{}\n')
            with self.assertRaises(json.JSONDecodeError):
                load_llmtf_results(path)
        finally:
            os.unlink(path)


if __name__ == '__main__':
    unittest.main()
