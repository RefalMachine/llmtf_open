import unittest
from pathlib import Path
import json
import tempfile

from llmtf.provenance import (
    CacheMismatchError, canonical_json, fingerprint_run_config, sanitize,
    validate_cache,
)


class RunProvenanceTests(unittest.TestCase):
    def test_fingerprint_is_stable_under_key_order(self):
        left = {'b': 2, 'a': {'y': 1, 'x': 0}}
        right = {'a': {'x': 0, 'y': 1}, 'b': 2}
        self.assertEqual(
            fingerprint_run_config(left), fingerprint_run_config(right)
        )

    def test_fingerprint_changes_with_execution_mode(self):
        off = {'execution': {'enable_thinking': False}}
        on = {'execution': {'enable_thinking': True}}
        self.assertNotEqual(
            fingerprint_run_config(off), fingerprint_run_config(on)
        )

    def test_secrets_are_redacted_recursively(self):
        data = sanitize({
            'api_key': 'top-secret',
            'nested': {'Authorization': 'Bearer top-secret'},
            'max_new_tokens': 10,
        })
        serialized = canonical_json(data)
        self.assertNotIn('top-secret', serialized)
        self.assertEqual(data['max_new_tokens'], 10)

    def test_cache_hit_mismatch_and_force(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'task_total.jsonl'
            path.write_text(json.dumps({'run_fingerprint': 'abc'}), encoding='utf-8')
            self.assertTrue(validate_cache(path, 'abc'))
            with self.assertRaises(CacheMismatchError):
                validate_cache(path, 'other')
            self.assertFalse(validate_cache(path, 'other', force_recalc=True))


if __name__ == '__main__':
    unittest.main()
