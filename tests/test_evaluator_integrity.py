from pathlib import Path
import copy
import json
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
tasks_stub.__path__ = [str(Path(__file__).resolve().parents[1] / 'llmtf' / 'tasks')]
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
    def support_method(self, method): return method == 'generate'
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

    def test_duplicate_task_registration_requires_explicit_override(self):
        with mock.patch('llmtf.evaluator.set_random_seed', lambda seed: None):
            evaluator = Evaluator()
        name = 'test/duplicate_registration'
        evaluator.add_new_task(name, FailingTask, {})
        with self.assertRaisesRegex(ValueError, 'already registered'):
            evaluator.add_new_task(name, FailingTask, {})
        evaluator.add_new_task(
            name, FailingTask, {}, allow_override=True
        )

    def test_force_recalc_removes_prior_total_without_stale_archive(self):
        with mock.patch('llmtf.evaluator.set_random_seed', lambda seed: None):
            evaluator = Evaluator()
        with tempfile.TemporaryDirectory() as directory:
            total_path = Path(directory) / 'test_failing_batch_total.jsonl'
            total_path.write_text('{}', encoding='utf-8')
            self.assertFalse(
                evaluator._cache_hit(
                    directory, FailingTask(), 'new-fingerprint', True
                )
            )
            self.assertFalse(total_path.exists())
            self.assertEqual(list(Path(directory).glob('*.stale-*')), [])

    def test_report_includes_all_totals_in_output_directory(self):
        with mock.patch('llmtf.evaluator.set_random_seed', lambda seed: None):
            evaluator = Evaluator()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'current_total.jsonl').write_text(json.dumps({
                'task_name': 'current', 'leaderboard_result': 1.0,
            }), encoding='utf-8')
            (root / 'stale_total.jsonl').write_text(json.dumps({
                'task_name': 'stale', 'leaderboard_result': 0.0,
            }), encoding='utf-8')
            evaluator.create_report(directory)
            report = (root / 'evaluation_results.txt').read_text(
                encoding='utf-8'
            )
            self.assertEqual(
                report,
                'mean\tcurrent\tstale\n0.500\t1.000\t0.000',
            )


class LogsoftTask(Task):
    method = 'calculate_logsoftmax'
    _max_task_new_tokens = 1
    def task_name(self): return 'test/logsoft_dispatch'
    def load_dataset(self, **kwargs):
        return (
            [{'messages': [{'role': 'assistant', 'content': 'answer'}]}],
            [{'sample': {'id': 1}}],
        )
    def evaluate(self, sample, prediction): return {'metric': 1.0}
    def aggregation(self): return {'metric': lambda values: sum(values) / len(values)}


class PPLTask(Task):
    method = 'generate'
    _max_task_new_tokens = 4
    def task_name(self): return 'test/ppl_boundary'
    def load_dataset(self, **kwargs):
        return (
            [{'messages': [
                {'role': 'user', 'content': 'question'},
                {'role': 'assistant', 'content': 'Answer: '},
            ]}],
            [{'sample': {'answer': 'yes'}}],
        )
    def get_answer(self, sample): return sample['answer']
    def evaluate(self, sample, prediction): return {'unused': 0.0}
    def aggregation(self): return {'unused': lambda values: 0.0}


class LogsoftModel(FailingModel):
    def support_method(self, method): return method in {'generate', 'calculate_logsoftmax'}
    def apply_model_prompt(self, messages):
        return ''.join(message['content'] for message in messages)
    def calculate_logsoftmax_batch(self, messages, **kwargs):
        prompts = [self.apply_model_prompt(item) for item in messages]
        outputs = []
        infos = []
        for prompt, item in zip(prompts, messages):
            rendered = copy.deepcopy(item)
            answer_start = len(prompt) - len(item[-1]['content']) + len('Answer: ')
            rendered[-1]['tokens'] = [
                [10, -9.0, [answer_start - 2, answer_start]],
                [11, -0.2, [answer_start, len(prompt)]],
            ]
            outputs.append(rendered)
            infos.append({'generated_len': 1})
        return prompts, outputs, infos


class EvaluatorLogsoftTests(unittest.TestCase):
    def setUp(self):
        with mock.patch('llmtf.evaluator.set_random_seed', lambda seed: None):
            self.evaluator = Evaluator()

    def test_normal_evaluator_dispatches_calculate_logsoftmax(self):
        name = 'test/logsoft_dispatch_registry'
        self.evaluator.add_new_task(name, LogsoftTask, {})
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch(
                'llmtf.evaluator.tqdm', lambda iterable, **kwargs: iterable
            ):
                summary = self.evaluator.evaluate(
                    LogsoftModel(), directory, datasets_names=[name],
                    few_shot_count=0, batch_size=1,
                    max_sample_per_dataset=1,
                )
            self.assertTrue(summary.ok, summary.failed)
            total = json.loads(
                (Path(directory) / 'test_logsoft_dispatch_total.jsonl')
                .read_text(encoding='utf-8')
            )
            self.assertEqual(total['results']['metric'], 1.0)

    def test_ppl_excludes_assistant_prefill_tokens(self):
        name = 'test/ppl_boundary_registry'
        self.evaluator.add_new_task(name, PPLTask, {})
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch(
                'llmtf.evaluator.tqdm', lambda iterable, **kwargs: iterable
            ), mock.patch(
                'llmtf.evaluator.np.mean',
                lambda values: sum(values) / len(values),
            ):
                summary = self.evaluator.evaluate_ppl(
                    LogsoftModel(), directory, datasets_names=[name],
                    few_shot_count=0, batch_size=1,
                    max_sample_per_dataset=1,
                )
            self.assertTrue(summary.ok, summary.failed)
            total = json.loads(
                (Path(directory) / 'test_ppl_boundary_total.jsonl')
                .read_text(encoding='utf-8')
            )
            self.assertAlmostEqual(total['results']['ppl'], -0.2)


if __name__ == '__main__':
    unittest.main()
