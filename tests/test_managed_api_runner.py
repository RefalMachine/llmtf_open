import io
import signal
import subprocess
import tempfile
from pathlib import Path
from queue import Queue
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import requests

from benchmark import calculate_benchmark_api as runner
from benchmark.config import BenchmarkConfig, BenchmarkModelConfig


class ManagedAPIRunnerTests(unittest.TestCase):
    def test_internal_starting_ports_are_distinct_and_override_inherited_port(self):
        args = runner.build_parser().parse_args(['--model_dir', 'model', '--benchmark_config', 'unused'])
        with patch.dict(runner.os.environ, {'VLLM_PORT': '8000', 'CUDA_VISIBLE_DEVICES': 'all'}):
            envs = [runner.server_environment(args, index, str(index)) for index in range(8)]
            self.assertEqual([env['VLLM_PORT'] for env in envs], [str(20000 + index * 100) for index in range(8)])
            self.assertEqual([env['CUDA_VISIBLE_DEVICES'] for env in envs], [str(index) for index in range(8)])
            self.assertEqual(runner.os.environ['VLLM_PORT'], '8000')
        args.internal_base_port = 24000
        self.assertEqual(runner.server_environment(args, 3, 'GPU-a,GPU-b')['VLLM_PORT'], '24300')

    def test_invalid_or_overlapping_port_ranges_fail_before_loading_config(self):
        modules = {'llmtf.evaluator': SimpleNamespace(Evaluator=Mock()),
                   'llmtf.utils': SimpleNamespace(json_to_jinja=Mock())}
        for options in (['--internal_base_port', '0'], ['--internal_base_port', '65500'],
                        ['--base_port', '20000'], ['--base_port', '20799']):
            with self.subTest(options=options), patch.dict('sys.modules', modules), \
                 patch.object(runner, 'load_benchmark_config') as load, \
                 patch('sys.stderr', new=io.StringIO()), \
                 patch.dict(runner.os.environ, {'CUDA_VISIBLE_DEVICES': '0,1,2,3,4,5,6,7'}):
                with self.assertRaises(SystemExit):
                    runner.main(['--num_gpus', '8', '--model_dir', 'model', '--benchmark_config', 'unused', *options])
                load.assert_not_called()

    def test_slurm_noncontiguous_and_uuid_devices_are_preserved(self):
        self.assertEqual(runner.gpu_assignments(4, 2, '2,5,6,7'), ['2,5', '6,7'])
        self.assertEqual(runner.gpu_assignments(2, 1, 'GPU-a,GPU-b'), ['GPU-a', 'GPU-b'])
        self.assertEqual(runner.gpu_assignments(2, 2), ['0,1'])

    def test_invalid_allocations_fail_before_starting_servers(self):
        for args in ((0, 1), (1, 0), (3, 2), (2, 1, '3'), (1, 1, ''), (1, 1, '-1')):
            with self.subTest(args=args), self.assertRaises(ValueError):
                runner.gpu_assignments(*args)

    def test_server_revision_context_and_sampling_defaults(self):
        args = runner.build_parser().parse_args([
            '--model_dir', 'model', '--benchmark_config', 'x.yaml',
            '--revision', 'a' * 40, '--seed', '555', '--generation_config', 'vllm',
        ])
        config = BenchmarkConfig(BenchmarkModelConfig(model_context_len=16000), ())
        command = runner.server_command(args, config, 8123)
        self.assertEqual(command[command.index('--served-model-name') + 1], 'model@' + 'a' * 40)
        self.assertEqual(runner.served_model_name(args), 'model@' + 'a' * 40)
        for key, expected in (('--revision', 'a' * 40), ('--tokenizer-revision', 'a' * 40),
                              ('--max-model-len', '16000'), ('--seed', '555'),
                              ('--generation-config', 'vllm'), ('--host', '127.0.0.1')):
            self.assertEqual(command[command.index(key) + 1], expected)

    def test_readiness_retries_http_failure_and_rejects_dead_process(self):
        server = Mock()
        server.poll.return_value = None
        bad = Mock()
        bad.raise_for_status.side_effect = requests.HTTPError('503')
        good = Mock()
        with patch.object(runner.requests, 'get', side_effect=[bad, good]) as get, patch.object(runner.time, 'sleep'):
            runner.wait_for_server(server, 'http://localhost:8000/v1', 10)
        self.assertEqual(get.call_count, 2)
        server.poll.return_value = 1
        with self.assertRaises(RuntimeError):
            runner.wait_for_server(server, 'http://localhost:8000/v1', 10)

    def test_workers_consume_until_sentinel_and_propagate_failure(self):
        queue = Queue()
        task = SimpleNamespace(name='a')
        for value in (task, task, None):
            queue.put(value)
        with patch.object(runner.os, 'setsid'), patch.object(runner, 'run_eval', return_value=True) as run:
            runner.worker(0, queue, None, None, 'url')
        self.assertEqual(run.call_count, 2)
        queue.put(task)
        with patch.object(runner.os, 'setsid'), patch.object(runner, 'run_eval', return_value=False):
            with self.assertRaises(RuntimeError):
                runner.worker(0, queue, None, None, 'url')

    def test_cleanup_kills_engine_group_and_removes_template(self):
        server = Mock(pid=12345)
        server.wait.side_effect = [subprocess.TimeoutExpired('vllm', 15), 0]
        with tempfile.TemporaryDirectory() as directory:
            template = Path(directory) / 'chat.j2'
            template.write_text('template')
            with patch.object(runner, 'stop_group') as stop:
                runner.shutdown_servers([server], [8000], [template])
            self.assertFalse(template.exists())
        self.assertEqual(stop.call_args_list[0].args, (12345, signal.SIGTERM))
        self.assertEqual(stop.call_args_list[1].args, (12345, signal.SIGKILL))

    def test_startup_failure_still_cleans_every_started_server(self):
        config = BenchmarkConfig(BenchmarkModelConfig(), (Mock(),))
        server = Mock(pid=12345)
        modules = {
            'llmtf.evaluator': SimpleNamespace(Evaluator=Mock()),
            'llmtf.utils': SimpleNamespace(json_to_jinja=Mock()),
        }
        with patch.dict('sys.modules', modules), \
             patch.object(runner, 'load_benchmark_config', return_value=config), \
             patch.object(runner.mp, 'get_context'), \
             patch.object(runner, 'is_port_in_use', return_value=False), \
             patch.object(runner.subprocess, 'Popen', return_value=server), \
             patch.object(runner, 'wait_for_server', side_effect=TimeoutError), \
             patch.object(runner, 'shutdown_servers') as shutdown, \
             patch.dict(runner.os.environ, {'CUDA_VISIBLE_DEVICES': '3,5'}):
            with self.assertRaises(TimeoutError):
                runner.main(['--num_gpus', '2', '--model_dir', 'model', '--benchmark_config', 'unused'])
        self.assertEqual(shutdown.call_args.args[0], [server])

    def test_client_receives_same_revision_identity_and_phrase(self):
        from benchmark.config import BenchmarkTaskConfig
        args = runner.build_parser().parse_args(['--model_dir', 'model', '--benchmark_config', 'unused', '--revision', 'a' * 40])
        task = BenchmarkTaskConfig('test', ('x',), True, {}, {})
        config = BenchmarkConfig(BenchmarkModelConfig(reasoning_truncing_prompt=''), (task,))
        with patch.object(runner.subprocess, 'run', return_value=SimpleNamespace(returncode=0)) as run:
            self.assertTrue(runner.run_eval(args, config, task, 'http://localhost:8000'))
        command = run.call_args.args[0]
        self.assertEqual(command[command.index('--model_name_or_path') + 1], 'model@' + 'a' * 40)
        self.assertEqual(command[command.index('--reasoning_truncing_prompt') + 1], '')


if __name__ == '__main__':
    unittest.main()
