"""RuTaR integration contracts; run inside the existing API Docker profile."""
import json
import logging
from io import BytesIO
import hashlib
from zipfile import ZipFile
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
from llmtf.tasks.rutar import RuTaR
from llmtf.tasks.rutar.data import manifest, validate_rows, split_rows, read_workbook, load_rows


class Model:
    logger = logging.getLogger('rutar_test')
    reasoning_config = ReasoningConfig()
    generation_config = SamplingConfig(max_new_tokens=1)

    def get_params(self): return {'model': 'synthetic'}
    def count_tokens_for_messages(self, messages): return None
    def support_method(self, method): return True
    def get_model_context_len(self): return 32768
    def add_stop_strings(self, stops): pass
    def reset_stop_strings(self): pass


def rows():
    return [dict(id=str(i), question=f'Question {i}?', answer=str(i % 2),
                 source_url=f'http://example.test/{i}', title=f'Letter {i}')
            for i in range(207)]


class RuTaRTests(unittest.TestCase):
    def test_workbook_sparse_cells_and_empty_rows(self):
        namespace = 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'
        data = BytesIO()
        with ZipFile(data, 'w') as z:
            z.writestr('xl/sharedStrings.xml', f'<sst xmlns="{namespace}"><si><t>question_for_llm</t></si><si><t>Question?</t></si></sst>')
            z.writestr('xl/worksheets/sheet1.xml', f'''<worksheet xmlns="{namespace}"><sheetData>
                <row r="1"><c r="I1" t="s"><v>0</v></c>
                  <c r="K1" t="inlineStr"><is><t>true_answer</t></is></c>
                  <c r="H1" t="inlineStr"><is><t>source_url</t></is></c></row>
                <row r="2"><c r="A2"><v>42.0</v></c><c r="I2" t="s"><v>1</v></c>
                  <c r="K2"><v>1.0</v></c></row><row r="3"/>
                </sheetData></worksheet>''')
        parsed = read_workbook(data.getvalue())
        self.assertEqual(parsed, [{'id': '42', 'question_for_llm': 'Question?', 'true_answer': '1.0'}])

    def test_parquet_loader_matches_workbook_and_pins_hf_download(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        raw = [dict(id='1', question_for_llm='Question?', true_answer='1.0',
                    title='Letter', date_publication='date', letter_type='minfin', source_url=None)]
        spec = dict(raw_row_count=1, excluded_missing_question_ids=[],
                    excluded_duplicate_question_ids=[], dataset_repo='RefalMachine/RuTaR',
                    dataset_revision='pinned', dataset_filename='data/test.parquet')
        with tempfile.TemporaryDirectory() as directory:
            workbook = Path(directory) / 'original.xlsx'; workbook.write_bytes(b'workbook')
            parquet = Path(directory) / 'test.parquet'
            pq.write_table(pa.Table.from_pylist([dict(raw[0], true_answer=1)]), parquet)
            spec.update(sha256=hashlib.sha256(workbook.read_bytes()).hexdigest(),
                        dataset_sha256=hashlib.sha256(parquet.read_bytes()).hexdigest())
            with patch('llmtf.tasks.rutar.data.manifest', return_value=spec), \
                    patch('llmtf.tasks.rutar.data.read_workbook', return_value=raw), \
                    patch('huggingface_hub.hf_hub_download', return_value=str(parquet)) as download:
                self.assertEqual(load_rows(workbook), load_rows(parquet))
                self.assertEqual(load_rows(), load_rows(parquet))
                download.assert_called_once_with(repo_id='RefalMachine/RuTaR', repo_type='dataset',
                    revision='pinned', filename='data/test.parquet')

    def test_fixed_split_even_zero_shot_and_source_exclusion(self):
        evaluation, demos = split_rows(rows())
        self.assertEqual([r['id'] for r in demos], ['0', '1', '2', '3', '4'])
        self.assertEqual(len(evaluation), 202)
        duplicate = rows()
        duplicate[-1]['title'] = duplicate[0]['title']
        with self.assertRaisesRegex(ValueError, 'split changed'):
            split_rows(duplicate)

    def test_exclusions_and_conflicting_duplicate_gold(self):
        spec = dict(raw_row_count=3, excluded_missing_question_ids=['119'],
                    excluded_duplicate_question_ids=['117'])
        raw = dict(id='114', question_for_llm=' Вопрос? ', true_answer='1.0',
                   title='Letter', date_publication='date', letter_type='minfin')
        records = [raw, dict(raw, id='117', question_for_llm='вопрос?'),
                   dict(raw, id='119', question_for_llm=None)]
        self.assertEqual([r['id'] for r in validate_rows(records, spec)], ['114'])
        records[1]['true_answer'] = '0.0'
        with self.assertRaisesRegex(ValueError, 'Conflicting'):
            validate_rows(records, spec)
        records[1]['true_answer'] = '2'
        with self.assertRaisesRegex(ValueError, 'gold'):
            validate_rows(records, spec)

    def test_hash_failure_before_parsing(self):
        with tempfile.NamedTemporaryFile() as f:
            f.write(b'not the pinned workbook'); f.flush()
            with self.assertRaisesRegex(ValueError, 'SHA-256'):
                load_rows(f.name)

    def test_exact_shots_no_gold_leakage_and_overflow(self):
        raw = rows()
        raw[5]['full_text'] = raw[5]['answer_letter'] = 'DO NOT LEAK'
        with patch('llmtf.tasks.rutar.task.load_rows', return_value=raw):
            for shots in (0, 1, 5):
                messages, samples = RuTaR().load_dataset(Model(), 1000, 2, shots)
                self.assertEqual(messages[0]['tokens_of_interest'], ['0', '1'])
                self.assertEqual(len(messages[0]['messages']), shots * 2 + 1)
                self.assertEqual(messages[0]['messages'][-1]['role'], 'user')
                self.assertNotIn('DO NOT LEAK', str(messages))
                self.assertEqual(samples[0]['sample']['id'], '5')
                self.assertEqual(samples[0]['sample']['_rutar']['effective_shots'], shots)
            model = Model(); model.count_tokens_for_messages = lambda _: 1001
            with self.assertRaises(PromptTooLongError):
                RuTaR().load_dataset(model, 1000, 2, 5)
            for k in (-1, 6, True):
                with self.assertRaises(ValueError):
                    RuTaR().load_dataset(Model(), 1000, 2, k)

    def test_probability_scoring_ties_and_invalid_values(self):
        task = RuTaR()
        self.assertEqual(task.evaluate({'answer': '1'}, {'0': .2, '1': .8})['acc'], 1)
        self.assertEqual(task.evaluate({'answer': '0'}, {'0': .8, '1': .2})['acc'], 1)
        self.assertEqual(task.evaluate({'answer': '1'}, {'0': .8, '1': .2})['acc'], 0)
        for label in ('0', '1'):
            self.assertEqual(task.evaluate({'answer': label}, {'0': 0, '1': 0})['acc'], 0)
        for invalid in (float('nan'), float('inf'), -1, 2, True):
            with self.assertRaises(ValueError):
                task.evaluate({'answer': '1'}, {'0': invalid, '1': .1})
        with self.assertRaises(KeyError):
            task.evaluate({'answer': '1'}, {'0': .5})

    def test_registry_and_provenance_before_loading(self):
        self.assertNotIn('rutar/closed', resolve_task_names('all', TASK_REGISTRY))
        self.assertEqual(resolve_task_names('rutar/closed', TASK_REGISTRY), ['rutar/closed'])
        def config(task):
            return build_run_config(model=Model(), task=task, enable_thinking=False,
                generation_config=None, few_shot_count=0, batch_size=1,
                max_sample_per_dataset=8, max_prompt_len=1000,
                effective_reasoning_tokens=0, scoring_method=task.method)
        before = config(RuTaR())
        self.assertEqual(before, config(RuTaR()))
        self.assertIn('data.py', before['task']['provenance']['resource_sha256'])
        with patch('llmtf.tasks.rutar.task.manifest', return_value=dict(manifest(), scorer_version='changed')):
            self.assertNotEqual(fingerprint_run_config(before), fingerprint_run_config(config(RuTaR())))
        with tempfile.NamedTemporaryFile() as f:
            first = config(RuTaR(data_path=f.name))
            f.write(b'changed'); f.flush()
            self.assertNotEqual(fingerprint_run_config(first), fingerprint_run_config(config(RuTaR(data_path=f.name))))

    def test_evaluator_probability_dispatch_artifacts_and_cache(self):
        model = Model()
        def calculate_tokens_proba_batch(messages, tokens_of_interest, **kwargs):
            return ['prompt'] * len(messages), [{'0': .2, '1': .8}] * len(messages), [{}] * len(messages)
        model.calculate_tokens_proba_batch = calculate_tokens_proba_batch
        with patch('llmtf.tasks.rutar.task.load_rows', return_value=rows()), tempfile.TemporaryDirectory() as directory:
            for iteration in range(2):
                summary = Evaluator().evaluate(model, directory, datasets_names=['rutar/closed'],
                    few_shot_count=5, batch_size=2, max_sample_per_dataset=2)
                self.assertEqual(summary.exit_code, 0)
            total = json.loads(next(Path(directory).glob('*_total.jsonl')).read_text())
            self.assertEqual(total['results']['acc'], .5)
            self.assertEqual(len(json.loads(next(p for p in Path(directory).glob('*.jsonl')
                if p.name == 'rutar_closed.jsonl').read_text())), 2)
            summary = Evaluator().evaluate_ppl(model, directory, datasets_names=['rutar/closed'])
            self.assertIn('rutar/closed', summary.skipped)


if __name__ == '__main__':
    unittest.main()
