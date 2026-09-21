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

    def test_fingerprint_ignores_lazily_discovered_api_capabilities(self):
        unknown = {
            'model': {
                'server_capabilities': {
                    'tokenize': True,
                    'detokenize': None,
                    'logprobs': None,
                },
            },
        }
        discovered = {
            'model': {
                'server_capabilities': {
                    'tokenize': True,
                    'detokenize': True,
                    'logprobs': True,
                },
            },
        }
        self.assertEqual(
            fingerprint_run_config(unknown),
            fingerprint_run_config(discovered),
        )

        unavailable_tokenizer = json.loads(json.dumps(discovered))
        unavailable_tokenizer['model']['server_capabilities']['tokenize'] = False
        self.assertNotEqual(
            fingerprint_run_config(unknown),
            fingerprint_run_config(unavailable_tokenizer),
        )

    def test_secrets_are_redacted_recursively(self):
        data = sanitize({
            'api_key': 'top-secret',
            'nested': {'Authorization': 'Bearer top-secret'},
            'access_token': 'credential-token',
            'refreshToken': 'camel-case-token',
            'max_new_tokens': 10,
            'configured_max_new_tokens_reasoning': 4096,
            'effective_max_new_tokens_reasoning': 0,
            'max_task_new_tokens': 128,
            'tokenize': True,
            'detokenize': False,
        })
        serialized = canonical_json(data)
        self.assertNotIn('top-secret', serialized)
        self.assertNotIn('credential-token', serialized)
        self.assertNotIn('camel-case-token', serialized)
        self.assertEqual(data['max_new_tokens'], 10)
        self.assertEqual(data['configured_max_new_tokens_reasoning'], 4096)
        self.assertEqual(data['effective_max_new_tokens_reasoning'], 0)
        self.assertEqual(data['max_task_new_tokens'], 128)
        self.assertIs(data['tokenize'], True)
        self.assertIs(data['detokenize'], False)

    def test_cache_hit_mismatch_and_force(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'task_total.jsonl'
            path.write_text(json.dumps({'run_fingerprint': 'abc'}), encoding='utf-8')
            self.assertTrue(validate_cache(path, 'abc'))
            with self.assertRaises(CacheMismatchError):
                validate_cache(path, 'other')
            self.assertFalse(validate_cache(path, 'other', force_recalc=True))

    def test_legacy_cache_accepts_only_volatile_capability_differences(self):
        expected = {
            'model': {
                'server_capabilities': {
                    'tokenize': True,
                    'detokenize': None,
                    'logprobs': None,
                },
            },
            'task': {'batch_size': 20},
        }
        cached = json.loads(json.dumps(expected))
        cached['model']['server_capabilities']['detokenize'] = True
        cached['model']['server_capabilities']['logprobs'] = True
        expected_fingerprint = fingerprint_run_config(expected)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'task_total.jsonl'
            path.write_text(json.dumps({
                'run_fingerprint': 'legacy-order-dependent-fingerprint',
                'run_config': cached,
            }), encoding='utf-8')
            self.assertTrue(validate_cache(
                path,
                expected_fingerprint,
                expected_run_config=expected,
            ))

            incompatible = json.loads(json.dumps(expected))
            incompatible['task']['batch_size'] = 10
            with self.assertRaises(CacheMismatchError):
                validate_cache(
                    path,
                    fingerprint_run_config(incompatible),
                    expected_run_config=incompatible,
                )

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
