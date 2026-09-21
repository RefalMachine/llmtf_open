import os
import sys
import time
import argparse
import subprocess
import torch.multiprocessing as mp
import torch
from multiprocessing import Queue
from queue import Empty
import json
import requests
from contextlib import closing
import socket
import tempfile
from pathlib import Path
from benchmark.config import build_evaluate_command, load_benchmark_config
from llmtf.config import DEFAULT_VLLM_GPU_MEMORY_UTILIZATION
from llmtf.evaluator import Evaluator
from llmtf.utils import json_to_jinja

# Функция run_eval теперь принимает base_url как явный аргумент
def run_eval(args, config, task, base_url):
    """Запускает один таск оценки модели через API."""
    if args.output_dir is not None:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(args.model_dir, 'llmtf_eval')
    if 'batch_size' not in task.evaluation:
        task = type(task)(task.name, task.datasets, task.enable_thinking,
                          {**task.evaluation, 'batch_size': 10000000}, task.generation)
    command = build_evaluate_command(
        config, task, model_name=args.model_dir, output_dir=output_dir,
        api=True, base_url=base_url, force_recalc=args.force_recalc,
        conv_path=args.conv_path, is_foundational=args.is_foundational,
        api_profile='vllm',
        calculate_tokens_proba_logprobs_count=args.max_logprobs,
    )

    env = os.environ.copy()
    if args.api_key:
        env['OPENAI_API_KEY'] = args.api_key
    torchrun_env_names = {'TORCHELASTIC_USE_AGENT_STORE', 'OMP_NUM_THREADS', 'GROUP_RANK', 'ROLE_RANK', 'ROLE_NAME', 'LOCAL_WORLD_SIZE', 'GROUP_WORLD_SIZE', 'ROLE_WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT', 'WORLD_SIZE', 'LOCAL_RANK', 'RANK'}
    for var_name in torchrun_env_names:
        if var_name in env:
            del env[var_name]
    command = [str(c) for c in command]
    print(f"[{base_url}] Running command: {' '.join(command)}", flush=True)

    try:
        subprocess.run(command, env=env, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[{base_url}] Error executing task '{task.name}': {e}", flush=True)
        return False

    return True

def worker(worker_id, task_queue, args, config, base_url):
    """
    Функция-воркер. Получает задачи из очереди и выполняет их,
    используя закрепленный за ним base_url.
    """
    print(f"[Worker-{worker_id}] Started, using API at {base_url}", flush=True)
    while True:
        try:
            # Неблокирующее получение задачи из очереди
            task_group = task_queue.get_nowait()
            print(f"[Worker-{worker_id}] Took task: {task_group.name}", flush=True)
            if not run_eval(args, config, task_group, base_url):
                raise RuntimeError(f"Task {task_group.name} failed")
        except Empty:
            # Если очередь пуста, воркер завершает работу
            print(f"[Worker-{worker_id}] No more tasks. Exiting.", flush=True)
            break
        except Exception as e:
            print(f"[Worker-{worker_id}] An unexpected error occurred: {e}", flush=True)
            raise

def is_port_in_use(port):
    """Проверяет, занят ли порт."""
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        return s.connect_ex(('localhost', port)) == 0


def shutdown_servers(servers, server_ports, template_paths):
    """Stop every managed server and remove generated chat templates."""
    if servers:
        print("\nShutting down vLLM servers...")
    for i, server_process in enumerate(servers):
        if server_process.poll() is None:
            print(f"Terminating server on port {server_ports[i]}...")
            server_process.terminate()
    if any(server_process.poll() is None for server_process in servers):
        time.sleep(5)
    for i, server_process in enumerate(servers):
        if server_process.poll() is None:
            print(f"Forcefully killing server on port {server_ports[i]}...")
            server_process.kill()
    for template_path in template_paths:
        try:
            os.unlink(template_path)
        except FileNotFoundError:
            pass
    if servers:
        print("All servers have been shut down.")

if __name__ == '__main__':
    # Используем 'spawn' для безопасности при работе с CUDA
    mp.set_start_method('spawn', force=True)

    parser = argparse.ArgumentParser(description="Run vLLM servers and distribute evaluation tasks.")
    # Аргументы для запуска серверов
    parser.add_argument('--num_gpus', type=int, default=torch.cuda.device_count())
    parser.add_argument('--tensor_parallel_size', type=int, default=1, help="Tensor parallel size for each vLLM instance.")
    parser.add_argument('--base_port', type=int, default=8000, help="Base port for the first vLLM server.")
    parser.add_argument(
        '--gpu_memory_utilization', type=float,
        default=DEFAULT_VLLM_GPU_MEMORY_UTILIZATION,
        help='Per-instance vLLM GPU memory fraction (default: 0.92).',
    )
    parser.add_argument(
        '--max_logprobs', type=int, default=100,
        help=(
            'Shared vLLM server limit and API-client top-logprobs count '
            '(default: 100).'
        ),
    )
    
    # Существующие аргументы
    parser.add_argument('--model_dir', required=True)
    parser.add_argument('--benchmark_config', required=True)
    parser.add_argument('--api_key', default=None,
                        help='Compatibility option; prefer OPENAI_API_KEY')
    parser.add_argument('--output_dir', default=None)
    parser.add_argument('--force_recalc', action='store_true')
    parser.add_argument('--conv_path', default='auto')
    parser.add_argument('--is_foundational', action='store_true')

    # Старый аргумент base_url больше не нужен
    # parser.add_argument('--base_url')

    args = parser.parse_args()
    print("Arguments parsed (API credentials redacted)")
    
    # Проверка корректности аргументов
    if args.num_gpus % args.tensor_parallel_size != 0:
        raise ValueError("`num_gpus` must be divisible by `tensor_parallel_size`")
    if args.num_gpus <= 0:
        raise ValueError("No CUDA devices detected; pass --num_gpus explicitly on a GPU host")
    if not 0.0 < args.gpu_memory_utilization <= 1.0:
        raise ValueError('gpu_memory_utilization must be in the interval (0, 1]')
    if not 1 <= args.max_logprobs <= 100:
        raise ValueError('max_logprobs must be between 1 and 100')

    num_instances = args.num_gpus // args.tensor_parallel_size
    print(f"Planning to start {num_instances} vLLM instances.")

    # --- 1. Запуск серверов vLLM ---
    servers = []
    server_urls = []
    server_ports = []
    template_paths = []
    config = load_benchmark_config(args.benchmark_config)
    foundational = args.is_foundational or config.model.is_foundational
    
    for i in range(num_instances):
        port = args.base_port + i
        if is_port_in_use(port):
            shutdown_servers(servers, server_ports, template_paths)
            raise RuntimeError(
                f"Port {port} is already in use; choose a different "
                "--base_port or free the port"
            )
            
        gpus_for_instance = ",".join(map(str, range(i * args.tensor_parallel_size, (i + 1) * args.tensor_parallel_size)))
        
        server_env = os.environ.copy()
        server_env["CUDA_VISIBLE_DEVICES"] = gpus_for_instance
        #server_env['VLLM_LOGGING_LEVEL'] = 'ERROR'
        
        command = [
            sys.executable, '-m', 'vllm.entrypoints.openai.api_server',
            '--model', args.model_dir,
            '--port', str(port),
            '--tensor-parallel-size', str(args.tensor_parallel_size),
            '--no-enable-log-requests',
            '--uvicorn-log-level', 'error',
            '--disable-uvicorn-access-log',
            '--disable-log-stats',
            '--language-model-only',
        ]
        command += [
            '--gpu-memory-utilization', str(args.gpu_memory_utilization),
            '--max-model-len', '32000',
            '--max-logprobs', str(args.max_logprobs),
        ]
        if config.model.model_context_len is not None:
            command[command.index('--max-model-len') + 1] = str(config.model.model_context_len)
        try:
            if foundational:
                conv_path = args.conv_path
                if conv_path == "auto":
                    conv_path = str(Path(__file__).parent.parent / 'conversation_configs' / 'default_foundational.json')
                with open(conv_path, "r", encoding="utf-8") as file:
                    template = json.load(file)
                chat_template, _ = json_to_jinja(template)
                with tempfile.NamedTemporaryFile(mode='w', suffix='.j2', delete=False) as f:
                    f.write(chat_template)
                    template_path = f.name
                template_paths.append(template_path)
                command += ['--chat-template', template_path]
            print(f"Starting vLLM server instance {i+1}/{num_instances} on port {port} with GPUs: {gpus_for_instance}...")

            # Запускаем сервер в фоновом режиме
            p = subprocess.Popen(command, env=server_env)
        except Exception:
            shutdown_servers(servers, server_ports, template_paths)
            raise
        servers.append(p)
        server_ports.append(port)
        server_urls.append(f"http://localhost:{port}/v1")

    # --- 2. Ожидание готовности серверов ---
    print("\nWaiting for all vLLM servers to be ready...")
    for i, url in enumerate(server_urls):
        retries = 100
        while retries > 0:
            return_code = servers[i].poll()
            if return_code is not None:
                shutdown_servers(servers, server_ports, template_paths)
                raise RuntimeError(
                    f"vLLM server on port {server_ports[i]} exited early "
                    f"with code {return_code}"
                )
            try:
                # Проверяем health-эндпоинт или просто доступность
                response = requests.get(url.replace("/v1", "/health"), timeout=5)
                response.raise_for_status()
                print(f"Server on port {server_ports[i]} is ready.")
                break
            except requests.RequestException:
                time.sleep(10)
                retries -= 1
                if retries == 0:
                    shutdown_servers(servers, server_ports, template_paths)
                    raise RuntimeError(
                        f"vLLM server on port {server_ports[i]} failed "
                        "its health check"
                    )

    # --- 3. Основная логика выполнения задач ---
    try:
        # Создаем и заполняем очередь задач
        task_queue = Queue()
        for task in config.tasks:
            task_queue.put(task)

        # Создаем и запускаем процессы-воркеры
        processes = []
        for i in range(num_instances):
            p = mp.Process(target=worker, args=(i, task_queue, args, config, server_urls[i]))
            processes.append(p)
            p.start()

        # Ожидаем завершения всех воркеров
        for p in processes:
            p.join()
        output_dir = args.output_dir or os.path.join(
            args.model_dir, 'llmtf_eval'
        )
        if os.path.isdir(output_dir):
            Evaluator().create_report(output_dir)
        if any(p.exitcode != 0 for p in processes):
            raise RuntimeError('One or more benchmark workers failed')

        print("\nAll evaluation tasks completed.")

    finally:
        # --- 4. Завершение работы серверов ---
        shutdown_servers(servers, server_ports, template_paths)
