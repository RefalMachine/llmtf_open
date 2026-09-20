from pathlib import Path
import tempfile
import unittest
from unittest import mock

from test_refactor_logic import _stub_third_party
_stub_third_party()
import sys
import types
datasets_utils = types.ModuleType('datasets.utils')
datasets_logging = types.ModuleType('datasets.utils.logging')
datasets_logging.disable_progress_bar = lambda: None
sys.modules['datasets.utils'] = datasets_utils
sys.modules['datasets.utils.logging'] = datasets_logging
tasks_stub = types.ModuleType('llmtf.tasks')
tasks_stub.TASK_REGISTRY = {}
sys.modules['llmtf.tasks'] = tasks_stub

from llmtf.backends.base import BackendBatchError
from llmtf.base import Task
from llmtf.evaluator import Evaluator
from llmtf.reasoning import ReasoningConfig


class FailingTask(Task):
    method = 'generate'
    _max_task_new_tokens = 8
    def task_name(self): return 'test/failing_batch'
    def load_dataset(self, **kwargs):
        return ([{'messages': [{'role': 'user', 'content': 'q'}]}],
                [{'sample': {'id': 1}}])
    def evaluate(self, sample, prediction): return {'metric': 1}
    def aggregation(self): return {'metric': lambda values: sum(values) / len(values)}


class FailingModel:
    def __init__(self):
        self.generation_config = type('Config', (), {
            'max_new_tokens': 64, 'to_dict': lambda inner_self: {'max_new_tokens': inner_self.max_new_tokens}
        })()
        self.reasoning_config = ReasoningConfig()
        self.logger = mock.Mock()
    def get_model_context_len(self): return 1024
    def get_params(self): return {'backend_class': 'FailingBackend'}
    def add_stop_strings(self, values): pass
    def reset_stop_strings(self): pass
    def generate_batch(self, **kwargs):
        raise BackendBatchError({0: RuntimeError('failed request')})


class EvaluatorIntegrityTests(unittest.TestCase):
    def test_partial_batch_failure_has_nonzero_summary_and_no_total(self):
        with mock.patch('llmtf.evaluator.set_random_seed', lambda seed: None):
            evaluator = Evaluator()
        evaluator.add_new_task('test/failing_batch', FailingTask, {})
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch('llmtf.evaluator.tqdm', lambda iterable, **kwargs: iterable):
                summary = evaluator.evaluate(
                    FailingModel(), directory,
                    datasets_names=['test/failing_batch'], few_shot_count=0,
                    batch_size=1, max_sample_per_dataset=1,
                    enable_thinking=False, force_recalc=True,
                )
            self.assertEqual(summary.exit_code, 1)
            self.assertIn('test/failing_batch', summary.failed)
            self.assertIn('BackendBatchError', summary.failed['test/failing_batch'])
            self.assertFalse(
                (Path(directory) / 'test_failing_batch_total.jsonl').exists()
            )


if __name__ == '__main__':
    unittest.main()
