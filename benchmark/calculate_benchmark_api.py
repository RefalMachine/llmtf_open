import os
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
        is_foundational=args.is_foundational, api_profile='vllm',
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

# def read_json(file_name):
#     with open(file_name, encoding="utf-8") as r:
#         return json.load(r)

# НОВАЯ ФУНКЦИЯ: Воркер, который будет выполняться в отдельном процессе
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

# a copy from llmtf/utils.py
def json_to_jinja(template_config):
    roles_mapping = {
        template_config.get("system_role", "system"): template_config.get("system_message_template", ""),
        template_config.get("user_role", "user"): template_config.get("user_message_template", ""),
        template_config.get("bot_role", "assistant"): template_config.get("bot_message_template", "")
    }

    jinja_template = []
    if template_config.get("global_prefix"):
        jinja_template.append(template_config["global_prefix"])
    jinja_template.append("{% for message in messages %}")

    for role, template in roles_mapping.items():
        if template:
            formatted_template = template.replace("{role}", role).replace("{content}", "{{ message['content'] }}")
            jinja_template.append(f"{{% if message['role'] == '{role}' %}}{formatted_template}{{% endif %}}")

    jinja_template.append("{% endfor %}")

    if template_config.get("suffix"):
        jinja_template.append("{% if add_generation_prompt %}")
        jinja_template.append(template_config["suffix"])
        jinja_template.append("{% endif %}")

    eos_token = template_config.get("eos_token")
    if eos_token and type(eos_token) == list:
        eos_token = eos_token[0]
    return ("\n".join(jinja_template), eos_token)
        
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
            print(f"Port {port} is already in use. Please choose a different base_port or free the port.")
            exit(1)
            
        gpus_for_instance = ",".join(map(str, range(i * args.tensor_parallel_size, (i + 1) * args.tensor_parallel_size)))
        
        server_env = os.environ.copy()
        server_env["CUDA_VISIBLE_DEVICES"] = gpus_for_instance
        #server_env['VLLM_LOGGING_LEVEL'] = 'ERROR'
        
        command = [
            'python', '-m', 'vllm.entrypoints.openai.api_server',
            '--model', args.model_dir,
            '--port', str(port),
            '--tensor-parallel-size', str(args.tensor_parallel_size),
            '--disable-log-requests',
            '--uvicorn-log-level', 'error',
            '--disable-uvicorn-access-log',
            '--disable-log-stats'
        ]
        command += [
            '--gpu-memory-utilization', str(args.gpu_memory_utilization),
            '--max-model-len', '32000',
            '--max-logprobs', '50',
        ]
        if config.model.model_context_len is not None:
            command[command.index('--max-model-len') + 1] = str(config.model.model_context_len)
        if foundational:
            if args.conv_path == "auto":
                args.conv_path = str(Path(__file__).parent.parent / 'conversation_configs' / 'default_foundational.json')
            with open(args.conv_path, "r", encoding="utf-8") as file:
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
        servers.append(p)
        server_ports.append(port)
        server_urls.append(f"http://localhost:{port}/v1")

    # --- 2. Ожидание готовности серверов ---
    print("\nWaiting for all vLLM servers to be ready...")
    for i, url in enumerate(server_urls):
        retries = 100
        while retries > 0:
            try:
                # Проверяем health-эндпоинт или просто доступность
                requests.get(url.replace("/v1", "/health"), timeout=5)
                print(f"Server on port {server_ports[i]} is ready.")
                break
            except requests.ConnectionError:
                time.sleep(10)
                retries -= 1
                if retries == 0:
                    print(f"Server on port {server_ports[i]} failed to start!")
                    # Завершаем все уже запущенные серверы и выходим
                    for s in servers:
                        s.terminate()
                    exit(1)

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
        if any(p.exitcode != 0 for p in processes):
            raise RuntimeError('One or more benchmark workers failed')

        print("\nAll evaluation tasks completed.")

    finally:
        # --- 4. Завершение работы серверов ---
        print("\nShutting down vLLM servers...")
        for i, server_process in enumerate(servers):
            print(f"Terminating server on port {server_ports[i]}...")
            server_process.terminate()
        # Даем время на завершение
        time.sleep(5)
        for i, server_process in enumerate(servers):
            if server_process.poll() is None: # Если процесс все еще жив
                print(f"Forcefully killing server on port {server_ports[i]}...")
                server_process.kill()
        for template_path in template_paths:
            try:
                os.unlink(template_path)
            except FileNotFoundError:
                pass
        print("All servers have been shut down.")
