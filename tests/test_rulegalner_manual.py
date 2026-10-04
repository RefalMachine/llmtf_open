"""Manual RuLegalNER data, prompts, cache and evaluator contracts."""
import copy
import hashlib
import json
import logging
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from llmtf.base import PromptTooLongError
from llmtf.config import SamplingConfig
from llmtf.evaluator import Evaluator
from llmtf.provenance import build_run_config, fingerprint_run_config
from llmtf.reasoning import ReasoningConfig
from llmtf.task_selection import resolve_task_names
from llmtf.tasks import TASK_REGISTRY
from llmtf.tasks.rulegalner_manual import RuLegalNERManual
from llmtf.tasks.rulegalner_manual.data import load_splits, manifest, validate_rows


class Model:
    logger = logging.getLogger('rulegalner_manual_test')
    reasoning_config = ReasoningConfig()
    generation_config = SamplingConfig(max_new_tokens=1024)

    def get_params(self): return {'model': 'synthetic'}
    def count_tokens_for_messages(self, messages): return None
    def support_method(self, method): return True
    def get_model_context_len(self): return 16000
    def add_stop_strings(self, stops): pass
    def reset_stop_strings(self): pass


def row(id=1):
    return {'id': id, 'text': f'УК РФ, ст. 1; пример {id}',
            'label': [[0, 5, 'LAW'], [7, 12, 'PROVISION']],
            'meta': {'source_split': 'source', 'doc_id': id, 'case_type': 'уголовное'}}


def splits():
    return {'train': [row(id) for id in manifest()['demonstration_ids']],
            'val': [row(9000)], 'test': [row(9001), dict(row(9002), text='Пустой пример', label=[])]}


class RuLegalNERManualTests(unittest.TestCase):
    def test_uses_shared_ner_f1_and_separate_diagnostics(self):
        task = RuLegalNERManual()
        sample = row()
        gold = task.get_answer_str(sample)
        result = task.evaluate(sample, gold)
        self.assertEqual(result, {'f1-macro': ({'LAW': 1, 'PROVISION': 1}, {}, {}),
                                  'format_valid': 1.0, 'exact_match': 1.0})
        for invalid in ('', '{}', '[null]', '[["PERSON", "Иван"]]'):
            r = task.evaluate(dict(sample, label=[]), invalid)
            self.assertEqual(r['format_valid'], 0)
            self.assertEqual(r['exact_match'], 0)
        self.assertEqual(task.evaluate(dict(sample, label=[]), '[]')['exact_match'], 1)
        self.assertEqual(task.leaderboard_aggregation({'f1-macro': .25, 'exact_match': 1, 'format_valid': 1}), .25)

    def test_per_class_counts_and_repeats(self):
        task = RuLegalNERManual()
        record = task.evaluate(row(), '[["LAW", "УК РФ"], ["LAW", "УК РФ"]]')['f1-macro']
        value, details = task.aggregation()['f1-macro']([record])
        self.assertAlmostEqual(value, (2 / 3) / 3)
        self.assertEqual(details['per_class']['LAW'], {'tp': 1, 'fn': 0, 'fp': 1, 'f1': 2 / 3})
        self.assertEqual(details['per_class']['PROVISION']['fn'], 1)

    def test_fixed_shots_and_no_test_label_leakage(self):
        task = RuLegalNERManual()
        data = splits()
        with patch('llmtf.tasks.rulegalner_manual.task.load_splits', return_value=data):
            for shots in (0, 1, 5):
                messages, samples = task.load_dataset(Model(), 15000, 2, shots)
                self.assertEqual(len(messages), 2)
                self.assertEqual(len(messages[0]['messages']), 2 * shots + 1)
                self.assertEqual(messages[0]['messages'][-1]['role'], 'user')
                self.assertEqual(samples[0]['sample']['_rulegalner_manual']['effective_shots'], shots)
                before = copy.deepcopy(messages)
                data['test'][0]['label'] = []
                after, _ = task.load_dataset(Model(), 15000, 2, shots)
                self.assertEqual(before, after)
                model = Model(); model.count_tokens_for_messages = lambda _: 15001
                with self.assertRaises(PromptTooLongError):
                    task.load_dataset(model, 15000, 1, shots)
        for invalid in (-1, 6, True, 1.5):
            with self.assertRaises(ValueError):
                task.load_dataset(Model(), 15000, 1, invalid)

    def test_hashes_schema_and_split_overlap(self):
        data = splits()
        spec = copy.deepcopy(manifest())
        with tempfile.TemporaryDirectory() as directory:
            for split, rows in data.items():
                raw = ('\n'.join(json.dumps(r, ensure_ascii=False) for r in rows) + '\n').encode()
                Path(directory, spec['files'][split]['filename']).write_bytes(raw)
                spec['files'][split].update(sha256=hashlib.sha256(raw).hexdigest(), rows=len(rows))
            with patch('llmtf.tasks.rulegalner_manual.data.manifest', return_value=spec):
                self.assertEqual(load_splits(directory), data)
                # A new hash is insufficient to make overlapping source documents safe.
                overlapping = dict(row(9001), meta=data['train'][0]['meta'])
                raw = (json.dumps(overlapping) + '\n').encode()
                test = spec['files']['test']; test.update(sha256=hashlib.sha256(raw).hexdigest(), rows=1)
                path = Path(directory, test['filename']); path.write_bytes(raw)
                with self.assertRaisesRegex(ValueError, 'split overlap'):
                    load_splits(directory)
                path.write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError, 'SHA-256'):
                    load_splits(directory)
        for label in ([[0, 999, 'LAW']], [[True, 4, 'LAW']], [[0, 4, 'BAD']], [[0, 5, 'LAW'], [1, 4, 'LAW']]):
            with self.assertRaises(ValueError):
                validate_rows([dict(row(), label=label)], 'test', 1)

    def test_provenance_stable_before_and_after_loading(self):
        task = RuLegalNERManual()
        def config():
            return build_run_config(model=Model(), task=task, enable_thinking=False,
                generation_config=None, few_shot_count=0, batch_size=1,
                max_sample_per_dataset=8, max_prompt_len=14976,
                effective_reasoning_tokens=0, scoring_method=task.method)
        before = config()
        with patch('llmtf.tasks.rulegalner_manual.task.load_splits', return_value=splits()):
            task.load_dataset(Model(), 14976, 2, 0)
        self.assertEqual(fingerprint_run_config(before), fingerprint_run_config(config()))
        self.assertIn('shared_scorer_sha256', before['task']['provenance'])

    def test_evaluator_logs_invalid_empty_answer_and_transport_failure(self):
        from llmtf.backends.base import BackendBatchError
        data = splits(); data['test'] = [data['test'][1]]
        for failure in (False, True):
            model = Model()
            def generate_batch(**kwargs):
                if failure:
                    raise BackendBatchError({0: RuntimeError('synthetic transport failure')})
                return ['prompt'], [''], [{}]
            model.generate_batch = generate_batch
            with patch('llmtf.tasks.rulegalner_manual.task.load_splits', return_value=data), tempfile.TemporaryDirectory() as directory:
                summary = Evaluator().evaluate(model, directory, datasets_names=['rulegalner_manual/legal'],
                    few_shot_count=0, max_sample_per_dataset=1)
                self.assertEqual(summary.exit_code, int(failure))
                totals = list(Path(directory).glob('*_total.jsonl'))
                self.assertEqual(len(totals), 0 if failure else 1)
                if not failure:
                    total = json.loads(totals[0].read_text())
                    self.assertEqual(total['leaderboard_result'], 0)
                    self.assertEqual(total['results']['exact_match'], 0)
                    self.assertEqual(total['results']['format_valid'], 0)

    def test_registry_is_explicit_only(self):
        self.assertNotIn('rulegalner_manual/legal', resolve_task_names('all', TASK_REGISTRY))
        self.assertEqual(resolve_task_names('rulegalner_manual/legal', TASK_REGISTRY), ['rulegalner_manual/legal'])

    @unittest.skipUnless(os.environ.get('LLMTF_RULEGALNER_MANUAL_DATA_DIR'), 'optional pinned upstream data audit')
    def test_real_pinned_snapshot(self):
        data = load_splits()
        self.assertEqual([len(data[s]) for s in ('train', 'val', 'test')], [1000, 134, 201])
        task = RuLegalNERManual()
        messages, rows = task.load_dataset(Model(), 14976, 201, 5)
        self.assertEqual(len(rows), 201)
        self.assertEqual(sum(not task.get_gold_entities(r['sample']) for r in rows), 95)
        metrics = [task.evaluate(r['sample'], task.get_answer_str(r['sample'])) for r in rows]
        self.assertEqual(task.aggregation()['f1-macro']([m['f1-macro'] for m in metrics])[0], 1)
        self.assertTrue(all(m['exact_match'] == m['format_valid'] == 1 for m in metrics))
        self.assertTrue(all(len(m['messages']) == 11 for m in messages))


if __name__ == '__main__':
    unittest.main()
