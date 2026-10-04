"""Framework integration contracts (run in the API profile, no inference)."""
import copy
import json
import logging
import runpy
import sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from llmtf.base import PromptTooLongError
from llmtf.config import SamplingConfig
from llmtf.reasoning import ReasoningConfig
from llmtf.provenance import build_run_config, fingerprint_run_config
from llmtf.tasks import TASK_REGISTRY
from llmtf.tasks.legalbench_ru import LegalBenchRU
from llmtf.task_selection import resolve_task_names
from llmtf.evaluator import Evaluator
from benchmark.config import load_benchmark_config, build_evaluate_command


class Model:
    logger = logging.getLogger("legalbench_test")
    reasoning_config=ReasoningConfig()
    generation_config=SamplingConfig(max_new_tokens=512)
    def get_params(self):return {'model':'synthetic'}
    def count_tokens_for_messages(self,messages):return None
    def support_method(self,method):return True
    def get_model_context_len(self):return 32768
    def add_stop_strings(self,stops):pass
    def reset_stop_strings(self):pass


class IntegrationTests(unittest.TestCase):
    def sample(self):
        return dict(task='test',id='eval',question='Synthetic?',answer='Да',answer_type='binary',track='reasoning',domain='test',norm_text='Synthetic norm')

    def config(self,task,shots=0):
        return build_run_config(model=Model(),task=task,enable_thinking=False,generation_config=None,few_shot_count=shots,batch_size=1,max_sample_per_dataset=8,max_prompt_len=32256,effective_reasoning_tokens=0,scoring_method='generate')

    def test_provenance_precedes_loading_and_covers_helpers(self):
        task=LegalBenchRU()
        a=self.config(task)
        self.assertEqual(a,self.config(task))
        self.assertIn('scoring.py',a['task']['provenance']['resource_sha256'])
        self.assertNotEqual(fingerprint_run_config(a),fingerprint_run_config(self.config(task,5)))
        self.assertNotEqual(fingerprint_run_config(a),fingerprint_run_config(self.config(LegalBenchRU(mode='grounded'))))
        with tempfile.NamedTemporaryFile() as f:
            task=LegalBenchRU(data_path=f.name)
            before=self.config(task)
            f.write(b'changed');f.flush()
            self.assertNotEqual(fingerprint_run_config(before),fingerprint_run_config(self.config(task)))

    def test_fixed_shots_context_and_overflow(self):
        row=self.sample();demos=[dict(row,id=str(i),question='Demo '+str(i)) for i in range(5)]
        with patch('llmtf.tasks.legalbench_ru.task.load_rows',return_value=[]),patch('llmtf.tasks.legalbench_ru.task.select_rows',return_value=([row],{'binary':demos})):
            for k in range(6):
                task=LegalBenchRU(mode='grounded')
                messages,samples=task.load_dataset(Model(),1000,1,k)
                self.assertEqual(len(messages[0]['messages']),k*2+1)
                self.assertEqual(samples[0]['sample']['_legalbench']['effective_shots'],k)
                self.assertNotIn('Synthetic norm',messages[0]['messages'][0]['content'] if k else '')
                self.assertIn('Synthetic norm',messages[0]['messages'][-1]['content'])
            model=Model();model.count_tokens_for_messages=lambda _:1001
            with self.assertRaises(PromptTooLongError):task.load_dataset(model,1000,1,5)
            with self.assertRaises(ValueError):task.load_dataset(Model(),1000,1,6)
            with self.assertRaises(ValueError):LegalBenchRU(mode='upstream_all_zero_shot').load_dataset(Model(),1000,1,1)

    def test_backend_failure_vs_empty_output(self):
        from llmtf.backends.base import BackendBatchError
        row = self.sample()
        prepared = [{'messages': [{'role': 'user', 'content': 'Synthetic?'}],
                     'sample': row}]
        with patch.object(LegalBenchRU, '_load_dataset', return_value=prepared):
            for fail in (True, False):
                model = Model()
                def generate_batch(**kwargs):
                    if fail:
                        raise BackendBatchError({0: RuntimeError('synthetic transport error')})
                    return ['Synthetic?'], [''], [{}]
                model.generate_batch = generate_batch
                with tempfile.TemporaryDirectory() as directory:
                    result = Evaluator().evaluate(
                        model, directory, datasets_names=['legalbench_ru/closed'],
                        few_shot_count=0, max_sample_per_dataset=1)
                    totals = list(Path(directory).glob('*_total.jsonl'))
                    self.assertEqual(result.exit_code, int(fail))
                    self.assertEqual(len(totals), 0 if fail else 1)
                    if not fail:
                        self.assertEqual(json.loads(totals[0].read_text())['results']['score'], 0)

    def test_registry_all_and_explicit(self):
        selected=resolve_task_names('all',TASK_REGISTRY)
        self.assertTrue(all(not n.startswith('legalbench_ru/') for n in selected))
        self.assertEqual(len(selected), sum(spec.get('include_in_all', True) for spec in TASK_REGISTRY.values()))
        self.assertEqual(resolve_task_names('legalbench_ru/closed',TASK_REGISTRY),['legalbench_ru/closed'])

    def test_ppl_skipped(self):
        with tempfile.TemporaryDirectory() as directory:
            result=Evaluator().evaluate_ppl(Model(),directory,datasets_names=['legalbench_ru/closed'])
            self.assertIn('legalbench_ru/closed',result.skipped)
            self.assertEqual(result.exit_code,0)

    def test_yaml_and_all_command_builders(self):
        paths = list(Path('benchmark').glob('llmtf_legal_*.yaml'))
        self.assertEqual(len(paths), 2)
        for path in paths:
            cfg=load_benchmark_config(path)
            selected = [name for task in cfg.tasks for name in task.datasets]
            self.assertEqual(set(selected), {
                'shlepa/lawmc', 'legalbench_ru/closed', 'legalbench_ru/grounded',
                'legalbench_ru/distractor', 'legalbench_ru/temporal', 'rutar/closed',
                'rulegalner_manual/legal',
            })
            self.assertEqual(len(selected), len(set(selected)))
            reference_path = ('benchmark/llmtf_benchmark_foundational.yaml'
                              if cfg.model.is_foundational else
                              'benchmark/llmtf_benchmark_instruct_fast.yaml')
            reference = load_benchmark_config(reference_path)
            self.assertEqual(cfg.model, reference.model)
            reference_law = next(t for t in reference.tasks
                                 if 'shlepa/lawmc' in t.datasets)
            artifact_names = []
            for task in cfg.tasks:
                self.assertNotIn('batch_size', task.evaluation)
                if 'shlepa/lawmc' in task.datasets:
                    self.assertEqual(task.generation, reference_law.generation)
                    self.assertEqual(task.evaluation['few_shot_count'],
                                     5 if cfg.model.is_foundational else 0)
                else:
                    self.assertEqual(task.generation, reference.tasks[0].generation)
                    self.assertEqual(task.evaluation['few_shot_count'],
                                     5 if cfg.model.is_foundational else 0)
                for name in task.datasets:
                    spec = TASK_REGISTRY[name]
                    instance = spec['class'](**spec['params'],
                                             name_suffix=task.evaluation.get('name_suffix'))
                    artifact_names.append(instance.run_name())
                for api,backend in ((False,'hf'),(False,'vllm'),(True,'vllm')):
                    command=build_evaluate_command(cfg,task,model_name='synthetic',output_dir='/tmp/unused',api=api,base_url='http://localhost:1',backend=backend)
                    for name in task.datasets:
                        self.assertIn(name,command)
                    self.assertIn('--few_shot_count',command)
                    self.assertNotIn('--batch_size',command)
            self.assertEqual(len(artifact_names), len(set(artifact_names)))

    def test_existing_api_runner_executes_complete_legal_config(self):
        argv = ['benchmark.calculate_benchmark_existing_api',
                '--model_name', 'synthetic', '--base_url', 'http://localhost:1/v1',
                '--benchmark_config', 'benchmark/llmtf_legal_instruct.yaml']
        with tempfile.TemporaryDirectory() as directory:
            argv.extend(['--output_dir', directory])
            with patch.object(sys, 'argv', argv), patch('requests.get'), \
                    patch('subprocess.run') as execute, \
                    patch('llmtf.evaluator.Evaluator'):
                runpy.run_module('benchmark.calculate_benchmark_existing_api',
                                 run_name='__main__')
            self.assertEqual(execute.call_count, 4)
            commands = [call.args[0] for call in execute.call_args_list]
            self.assertIn('shlepa/lawmc', commands[0])
            for mode in ('closed', 'grounded', 'distractor', 'temporal'):
                self.assertIn('legalbench_ru/' + mode, commands[1])
            self.assertIn('rutar/closed', commands[2])
            self.assertIn('rulegalner_manual/legal', commands[3])
            for command in commands:
                self.assertEqual(command[command.index('--batch_size') + 1], '10000000')


if __name__=='__main__':unittest.main()
