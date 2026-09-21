from pathlib import Path
import json
import tempfile
import textwrap
import unittest

from benchmark.config import build_evaluate_command, load_benchmark_config


class BenchmarkConfigTests(unittest.TestCase):
    def _write(self, directory, body):
        path = Path(directory) / 'benchmark.yaml'
        path.write_text(textwrap.dedent(body), encoding='utf-8')
        return path

    def test_reasoning_fields_propagate_to_local_and_api_commands(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._write(directory, '''
                model:
                  model_kind: hybrid
                  enable_thinking: true
                  model_context_len: 8192
                  max_new_tokens_reasoning: 2048
                  min_new_tokens_reasoning: 512
                  end_thinking_token_id: 42
                  assistant_prefill_policy: exact
                  probe_api_prefill: true
                  api_profile: openai
                defaults:
                  evaluation: {few_shot_count: 0}
                  generation: {temperature: 0.0}
                tasks:
                  - name: smoke
                    datasets: [russiannlp/rucola_custom]
            ''')
            config = load_benchmark_config(path)
        task = config.tasks[0]
        commands = (
            build_evaluate_command(
                config, task, model_name='model', output_dir='/tmp/out', backend='hf'
            ),
            build_evaluate_command(
                config, task, model_name='model', output_dir='/tmp/out', api=True,
                base_url='http://localhost:8000/v1'
            ),
        )
        for command in commands:
            self.assertIn('--model_kind', command)
            self.assertIn('hybrid', command)
            self.assertIn('--enable_thinking', command)
            self.assertIn('--model_context_len', command)
            self.assertIn('8192', command)
            self.assertIn('--end_thinking_token_id', command)
            self.assertIn('42', command)
            self.assertIn('2048', command)
            self.assertIn('512', command)
            self.assertIn('--assistant_prefill_policy', command)
            self.assertIn('exact', command)
        self.assertNotIn('--probe_api_prefill', commands[0])
        self.assertIn('--probe_api_prefill', commands[1])
        self.assertNotIn('--api_profile', commands[0])
        self.assertEqual(
            commands[1][commands[1].index('--api_profile') + 1], 'openai'
        )

    def test_task_override_and_legacy_think_conflict(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._write(directory, '''
                model: {model_kind: hybrid, enable_thinking: false, end_thinking_token_id: 42}
                tasks:
                  - name: conflict
                    datasets: [x]
                    enable_thinking: true
                    extra_args: {think: false}
            ''')
            with self.assertRaisesRegex(ValueError, 'conflicting'):
                load_benchmark_config(path)

    def test_unknown_yaml_key_is_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._write(directory, '''
                model: {model_kind: plain}
                tasks:
                  - name: bad
                    datasets: [x]
                    max_prompt_len: 10
            ''')
            with self.assertRaisesRegex(ValueError, 'max_prompt_len'):
                load_benchmark_config(path)

    def test_shipped_instruct_ruparam_group_is_zero_shot(self):
        config_path = (
            Path(__file__).resolve().parents[1]
            / 'benchmark'
            / 'llmtf_benchmark_instruct_fast.yaml'
        )
        config = load_benchmark_config(config_path)
        ruparam = next(task for task in config.tasks if task.name == 'ruparam')
        self.assertEqual(ruparam.evaluation['few_shot_count'], 0)
        command = build_evaluate_command(
            config, ruparam, model_name='model', output_dir='/tmp/out',
            api=True, base_url='http://localhost:8000/v1',
        )
        option_index = command.index('--few_shot_count')
        self.assertEqual(command[option_index + 1], '0')

    def test_removed_libra_slices_are_not_shipped(self):
        root = Path(__file__).resolve().parents[1]
        removed = {'ru_gsm100', 'ru_trec', 'ru_qasper'}
        libra_config = json.loads(
            (root / 'llmtf/tasks/libra/libra_config.json').read_text(
                encoding='utf-8'
            )
        )
        self.assertTrue(removed.isdisjoint(libra_config))

        removed_dataset_names = {f'libra/{name}' for name in removed}
        for config_path in (root / 'benchmark').glob('*.yaml'):
            config = load_benchmark_config(config_path)
            shipped = {
                dataset
                for task in config.tasks
                for dataset in task.datasets
            }
            self.assertTrue(
                removed_dataset_names.isdisjoint(shipped),
                msg=str(config_path),
            )

    def test_main_benchmarks_use_libra_mini_and_full(self):
        root = Path(__file__).resolve().parents[1]
        benchmark_dir = root / 'benchmark'
        expected_mini = {
            'libra/ru_babilong_qa3',
            'libra/ru_sci_passage_count',
            'libra/librusec_mhqa',
            'libra/long_context_multiq',
            'libra/ru_2wikimultihopqa',
            'libra/matreshka_names',
        }
        expected_full = {
            f'libra/{name}'
            for name in json.loads(
                (root / 'llmtf/tasks/libra/libra_config.json').read_text(
                    encoding='utf-8'
                )
            )
        }

        def libra_datasets(filename):
            config = load_benchmark_config(benchmark_dir / filename)
            return {
                dataset
                for task in config.tasks
                for dataset in task.datasets
                if dataset.startswith('libra/')
            }

        self.assertEqual(
            libra_datasets('llmtf_benchmark_instruct_fast.yaml'),
            expected_mini,
        )
        self.assertEqual(
            libra_datasets('llmtf_benchmark_instruct.yaml'),
            expected_full,
        )

    def test_foundational_copy_group_is_last(self):
        root = Path(__file__).resolve().parents[1]
        config = load_benchmark_config(
            root / 'benchmark' / 'llmtf_benchmark_foundational.yaml'
        )
        self.assertEqual(config.tasks[-1].name, 'internal_copy')
        self.assertEqual(config.tasks[-1].datasets, (
            'darumeru/cp_para_ru',
            'darumeru/cp_para_en',
            'darumeru/cp_doc_ru',
        ))


if __name__ == '__main__':
    unittest.main()
