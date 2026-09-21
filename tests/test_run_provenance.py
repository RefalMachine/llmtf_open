import unittest
from pathlib import Path
import json
import tempfile

from llmtf.provenance import (
    CacheMismatchError, build_run_config, canonical_json,
    fingerprint_run_config, sanitize, validate_cache,
)
from llmtf.base import Task
from llmtf.reasoning import ReasoningConfig


class ProvenanceTask(Task):
    method = 'generate'
    _max_task_new_tokens = 4
    def task_name(self): return 'test/provenance'
    def load_dataset(self, **kwargs): return [], []
    def evaluate(self, **kwargs): return {'score': 0.0}
    def aggregation(self, **kwargs): return {'score': lambda values: 0.0}
    def dataset_args(self): return {'path': 'json', 'revision': 'fixed'}


class ProvenanceModel:
    reasoning_config = ReasoningConfig()
    generation_config = type('Config', (), {
        'to_dict': lambda self: {'max_new_tokens': 4},
    })()
    def get_params(self): return {'backend_class': 'FakeBackend'}


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

    def test_task_params_and_implementation_participate_in_fingerprint(self):
        common = dict(
            model=ProvenanceModel(), task=ProvenanceTask(),
            enable_thinking=False, generation_config=None,
            few_shot_count=0, batch_size=1, max_sample_per_dataset=1,
            max_prompt_len=100, effective_reasoning_tokens=0,
            scoring_method='generate', registry_name='test/provenance',
        )
        left = build_run_config(
            **common, task_init_params={'instruction': 'left'}
        )
        right = build_run_config(
            **common, task_init_params={'instruction': 'right'}
        )
        self.assertNotEqual(
            fingerprint_run_config(left), fingerprint_run_config(right)
        )
        self.assertEqual(left['task']['dataset_args']['revision'], 'fixed')
        self.assertIn('module_sha256', left['task']['implementation'])


if __name__ == '__main__':
    unittest.main()
