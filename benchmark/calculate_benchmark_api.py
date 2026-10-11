"""Start local vLLM servers and evaluate the shared benchmark through API."""

import argparse
from contextlib import closing
import json
import multiprocessing as mp
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import time

import requests

from benchmark.config import build_evaluate_command, load_benchmark_config
from llmtf.config import DEFAULT_VLLM_GPU_MEMORY_UTILIZATION


INTERNAL_PORT_STRIDE = 100


def gpu_assignments(num_gpus, tensor_parallel_size, visible_devices=None):
    """Partition allocated CUDA identifiers, including Slurm GPU UUIDs."""
    if num_gpus <= 0 or tensor_parallel_size <= 0:
        raise ValueError('num_gpus and tensor_parallel_size must be positive')
    if num_gpus % tensor_parallel_size:
        raise ValueError('num_gpus must be divisible by tensor_parallel_size')
    devices = (visible_devices.split(',') if visible_devices is not None
               else [str(i) for i in range(num_gpus)])
    devices = [device.strip() for device in devices if device.strip()]
    if len(devices) < num_gpus or any(d == '-1' for d in devices):
        raise ValueError('Requested GPUs exceed CUDA_VISIBLE_DEVICES allocation')
    return [','.join(devices[i:i + tensor_parallel_size])
            for i in range(0, num_gpus, tensor_parallel_size)]


def served_model_name(args):
    # Revision must enter API provenance, otherwise a reused output directory
    # could accept cached scores from a different checkpoint with the same name.
    return f'{args.model_dir}@{args.revision}' if args.revision else args.model_dir


def run_eval(args, config, task, base_url):
    output_dir = args.output_dir or os.path.join(args.model_dir, 'llmtf_eval')
    if 'batch_size' not in task.evaluation:
        task = type(task)(task.name, task.datasets, task.enable_thinking,
                          {**task.evaluation, 'batch_size': 10000000}, task.generation)
    command = build_evaluate_command(
        config, task, model_name=served_model_name(args), output_dir=output_dir,
        api=True, base_url=base_url, force_recalc=args.force_recalc,
        conv_path=args.conv_path, is_foundational=args.is_foundational,
        api_profile='vllm', calculate_tokens_proba_logprobs_count=args.max_logprobs,
    )
    command[0] = sys.executable
    env = os.environ.copy()
    if args.api_key:
        env['OPENAI_API_KEY'] = args.api_key
    for name in ('TORCHELASTIC_USE_AGENT_STORE', 'OMP_NUM_THREADS', 'GROUP_RANK',
                 'ROLE_RANK', 'ROLE_NAME', 'LOCAL_WORLD_SIZE', 'GROUP_WORLD_SIZE',
                 'ROLE_WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT', 'WORLD_SIZE',
                 'LOCAL_RANK', 'RANK'):
        env.pop(name, None)
    print(f'Running evaluation group: {task.name}', flush=True)
    return subprocess.run(command, env=env).returncode == 0


def worker(worker_id, task_queue, args, config, base_url):
    # A worker owns its evaluator subprocesses; cleanup terminates the group.
    os.setsid()
    while True:
        task = task_queue.get()
        if task is None:
            return
        if not run_eval(args, config, task, base_url):
            raise RuntimeError(f'Worker {worker_id}: task {task.name} failed')


def is_port_in_use(port):
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        return sock.connect_ex(('127.0.0.1', port)) == 0


def stop_group(pid, sig):
    try:
        os.killpg(pid, sig)
    except ProcessLookupError:
        pass


def shutdown_servers(servers, server_ports, template_paths):
    # Engine children can remain after the front-end exits.
    for process in servers:
        stop_group(process.pid, signal.SIGTERM)
    for process in servers:
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            pass
        stop_group(process.pid, signal.SIGKILL)
        process.wait()
    for path in template_paths:
        Path(path).unlink(missing_ok=True)


def server_command(args, config, port):
    command = [
        sys.executable, '-m', 'vllm.entrypoints.openai.api_server',
        '--model', args.model_dir, '--host', '127.0.0.1', '--port', str(port),
        '--tensor-parallel-size', str(args.tensor_parallel_size),
        '--no-enable-log-requests', '--uvicorn-log-level', 'error',
        '--disable-uvicorn-access-log', '--disable-log-stats', '--language-model-only',
        '--gpu-memory-utilization', str(args.gpu_memory_utilization),
        '--max-model-len', str(config.model.model_context_len or 32000),
        '--max-logprobs', str(args.max_logprobs), '--dtype', args.dtype,
        '--seed', str(args.seed), '--generation-config', args.generation_config,
    ]
    if args.revision:
        command += ['--revision', args.revision, '--tokenizer-revision', args.revision,
                    '--served-model-name', served_model_name(args)]
    return command


def server_environment(args, index, devices):
    """Give each engine a separate starting port for TCP rendezvous.

    vLLM may increment this port when it is busy or needs another port.
    The spacing keeps our engines apart; it is not an OS-level reservation.
    """
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = devices
    env['VLLM_PORT'] = str(args.internal_base_port + index * INTERNAL_PORT_STRIDE)
    return env


def wait_for_server(server, url, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if server.poll() is not None:
            raise RuntimeError(f'vLLM exited with code {server.returncode}')
        try:
            response = requests.get(url.removesuffix('/v1') + '/health', timeout=5)
            response.raise_for_status()
            return
        except requests.RequestException:
            time.sleep(2)
    raise TimeoutError(f'vLLM failed readiness within {timeout} seconds')


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--num_gpus', type=int, default=None)
    parser.add_argument('--tensor_parallel_size', type=int, default=1)
    parser.add_argument('--base_port', type=int, default=8000)
    parser.add_argument('--internal_base_port', type=int, default=20000,
                        help='Starting internal TCP port; each engine gets +100. Separate from API base_port.')
    parser.add_argument('--gpu_memory_utilization', type=float, default=DEFAULT_VLLM_GPU_MEMORY_UTILIZATION)
    parser.add_argument('--max_logprobs', type=int, default=100)
    parser.add_argument('--model_dir', required=True)
    parser.add_argument('--benchmark_config', required=True)
    parser.add_argument('--api_key', default=None, help='Prefer OPENAI_API_KEY')
    parser.add_argument('--output_dir', default=None)
    parser.add_argument('--force_recalc', action='store_true')
    parser.add_argument('--conv_path', default='auto')
    parser.add_argument('--is_foundational', action='store_true')
    parser.add_argument('--revision', default=None, help='Pin model and tokenizer revision')
    parser.add_argument('--dtype', default='auto', choices=('auto', 'half', 'float16', 'bfloat16', 'float', 'float32'))
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--generation_config', default='auto', help='vllm disables model-specific generation defaults')
    parser.add_argument('--startup_timeout', type=float, default=1000)
    return parser


def main(argv=None):
    from llmtf.evaluator import Evaluator
    from llmtf.utils import json_to_jinja

    parser = build_parser()
    args = parser.parse_args(argv)
    if args.num_gpus is None:
        import torch
        args.num_gpus = torch.cuda.device_count()
    assignments = gpu_assignments(args.num_gpus, args.tensor_parallel_size,
                                  os.getenv('CUDA_VISIBLE_DEVICES'))
    if not 0 < args.gpu_memory_utilization <= 1 or not 1 <= args.max_logprobs <= 100:
        parser.error('Invalid memory utilization or max_logprobs')
    if args.startup_timeout <= 0 or not 1 <= args.base_port <= 65535 - len(assignments) + 1:
        parser.error('Invalid timeout or server port range')
    internal_end = args.internal_base_port + len(assignments) * INTERNAL_PORT_STRIDE - 1
    if not 1 <= args.internal_base_port <= internal_end <= 65535:
        parser.error('Invalid internal port range')
    api_end = args.base_port + len(assignments) - 1
    if args.base_port <= internal_end and args.internal_base_port <= api_end:
        parser.error('API port range overlaps internal port range; choose another base_port/internal_base_port')
    config = load_benchmark_config(args.benchmark_config)
    if not config.tasks:
        parser.error('Benchmark contains no tasks')
    servers, ports, templates, workers = [], [], [], []
    context = mp.get_context('spawn')
    task_queue = context.Queue()

    def interrupted(signum, frame):
        raise SystemExit(128 + signum)

    old_handlers = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        for index, devices in enumerate(assignments):
            port = args.base_port + index
            if is_port_in_use(port):
                raise RuntimeError(f'Port {port} is already in use; select another base_port')
            command = server_command(args, config, port)
            if args.is_foundational or config.model.is_foundational:
                conv_path = args.conv_path
                if conv_path == 'auto':
                    conv_path = Path(__file__).resolve().parents[1] / 'conversation_configs/default_foundational.json'
                with open(conv_path, encoding='utf-8') as source:
                    template, _ = json_to_jinja(json.load(source))
                with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', suffix='.j2', delete=False) as target:
                    templates.append(target.name)
                    target.write(template)
                command += ['--chat-template', target.name]
            env = server_environment(args, index, devices)
            print(f'Starting server {index}: GPUs={devices}, API port={port}, '
                  f'internal starting port={env["VLLM_PORT"]}', flush=True)
            servers.append(subprocess.Popen(command, env=env, start_new_session=True))
            ports.append(port)
            # Startup discovers free rendezvous ports before binding them.
            # Serial readiness prevents our own eight engines racing at once.
            wait_for_server(servers[-1], f'http://127.0.0.1:{port}/v1', args.startup_timeout)
        for task in config.tasks:
            task_queue.put(task)
        for _ in servers:
            task_queue.put(None)
        for index, port in enumerate(ports):
            process = context.Process(target=worker, args=(index, task_queue, args, config, f'http://127.0.0.1:{port}/v1'))
            process.start()
            workers.append(process)
        while any(process.is_alive() for process in workers):
            if any(server.poll() is not None for server in servers):
                raise RuntimeError('A managed vLLM server exited during evaluation')
            if any(process.exitcode not in (None, 0) for process in workers):
                raise RuntimeError('A benchmark worker failed')
            time.sleep(0.2)
        for process in workers:
            process.join()
        if any(process.exitcode != 0 for process in workers):
            raise RuntimeError('One or more benchmark workers failed')
        output = args.output_dir or os.path.join(args.model_dir, 'llmtf_eval')
        if os.path.isdir(output):
            Evaluator().create_report(output)
        return 0
    finally:
        for sig in old_handlers:
            signal.signal(sig, signal.SIG_IGN)
        for process in workers:
            stop_group(process.pid, signal.SIGTERM)
            if process.is_alive():
                process.terminate()
        for process in workers:
            process.join(timeout=5)
            stop_group(process.pid, signal.SIGKILL)
            if process.is_alive():
                process.kill()
            process.join()
        shutdown_servers(servers, ports, templates)
        task_queue.cancel_join_thread()
        task_queue.close()
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)


if __name__ == '__main__':
    sys.exit(main())
