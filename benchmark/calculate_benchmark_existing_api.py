import os
import argparse
from multiprocessing import Queue
import subprocess
from queue import Empty
import requests
import time
from benchmark.config import build_evaluate_command, load_benchmark_config

def run_eval(args, config, task):
    """Запускает один таск оценки модели через API."""
    if args.output_dir is not None:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(os.getcwd(), 'llmtf_eval')
    if 'batch_size' not in task.evaluation:
        task = type(task)(task.name, task.datasets, task.enable_thinking,
                          {**task.evaluation, 'batch_size': 10000000}, task.generation)
    command = build_evaluate_command(
        config, task, model_name=args.model_name, output_dir=output_dir,
        api=True, base_url=args.base_url, force_recalc=args.force_recalc,
        api_profile=args.api_profile,
    )

    command = [str(c) for c in command]
    print(f"[worker] Running command: {' '.join(command)}", flush=True)

    try:
        env = os.environ.copy()
        if args.api_key:
            env['OPENAI_API_KEY'] = args.api_key
        subprocess.run(command, check=True, env=env)
    except subprocess.CalledProcessError as e:
        print(f"[worker] Error executing task '{task.name}': {e}", flush=True)
        return False

    return True

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run tasks evaluation with an OpenAI-compatible API")

    parser.add_argument('--model_name', required=True)
    parser.add_argument('--base_url', required=True)
    parser.add_argument('--api_profile', choices=['auto', 'openai', 'vllm'],
                        default=None,
                        help='Overrides model.api_profile from benchmark YAML')
    parser.add_argument('--api_key', default=None,
                        help='Compatibility option; prefer OPENAI_API_KEY')
    parser.add_argument('--benchmark_config', required=True)
    parser.add_argument('--output_dir', default=None)
    parser.add_argument('--force_recalc', action='store_true')
    args = parser.parse_args()

    retries = 100
    while retries > 0:
        try:
            # Проверяем health-эндпоинт или просто доступность
            requests.get(args.base_url.replace("/v1", "/health"), timeout=5)
            print(f"Server is available")
            break
        except requests.ConnectionError:
            time.sleep(10)
            retries -= 1
            if retries == 0:
                print(f"Failed to connect to the server!")
                exit(1)
    
    config = load_benchmark_config(args.benchmark_config)

    # Создаем и заполняем очередь задач
    task_queue = Queue()
    for task in config.tasks:
        task_queue.put(task)

    print(f"[Worker] Started", flush=True)
    while True:
        try:
            # Неблокирующее получение задачи из очереди
            task_group = task_queue.get_nowait()
            print(f"[Worker] Took task: {task_group.name}", flush=True)
            if not run_eval(args, config, task_group):
                raise SystemExit(1)
        except Empty:
            # Если очередь пуста, воркер завершает работу
            print(f"[Worker] No more tasks. Exiting.", flush=True)
            break
        except Exception as e:
            print(f"[Worker] An unexpected error occurred: {e}", flush=True)
            raise

    print("\nAll evaluation tasks completed.")
