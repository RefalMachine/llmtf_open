import os
import time
import argparse
import subprocess
import torch.multiprocessing as mp
import torch
from multiprocessing import Queue, Lock
from queue import Empty
import nltk
from benchmark.config import build_evaluate_command, load_benchmark_config
from llmtf.evaluator import Evaluator

class GPUManager:
    def __init__(self, num_gpus):
        self.available_gpus = Queue()
        self.lock = Lock()
        for i in range(num_gpus):
            self.available_gpus.put(i)
    
    def acquire_gpu(self, count=1, timeout=5):
        with self.lock:
            try:
                gpuids = []
                for i in range(count):
                    gpuids.append(self.available_gpus.get(timeout=timeout))
                print(f'Acquired GPUs {gpuids}')
                return gpuids
            except Empty:
                for gpu_id in gpuids:
                    self.available_gpus.put(gpu_id)
                    
                print('Not enough GPU available')
                return None
    
    def release_gpu(self, gpu_id):
        with self.lock:
            self.available_gpus.put(gpu_id)
            print(f'Released GPU {gpu_id}')

class TaskQueue:
    def __init__(self, task_groups):
        self.tasks = Queue()
        self.lock = Lock()
        for task in task_groups:
            self.tasks.put(task)
    
    def get_task(self):
        with self.lock:
            try:
                return self.tasks.get_nowait()
            except Empty:
                return None


def run_eval(args, config, task, gpu_manager):
    """Запускает один таск оценки модели через прямой вызов evaluate_model.py."""
    gpu_ids = gpu_manager.acquire_gpu(count=args.tensor_parallel_size)
    if gpu_ids is None:
        return False

    if args.backend == 'vllm':
        default_bs = 10000000
    else:
        default_bs = 8

    if args.output_dir is not None:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(args.model_dir, 'llmtf_eval')
    if 'batch_size' not in task.evaluation:
        task = type(task)(task.name, task.datasets, task.enable_thinking,
                          {**task.evaluation, 'batch_size': default_bs}, task.generation)
    command = build_evaluate_command(
        config, task, model_name=args.model_dir, output_dir=output_dir,
        conv_path=args.conv_path, backend=args.backend,
        tensor_parallel_size=args.tensor_parallel_size,
        force_recalc=args.force_recalc, ppl=args.ppl,
        is_foundational=args.is_foundational,
    )

    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = ','.join([str(g) for g in gpu_ids])

    torchrun_env_names = {'TORCHELASTIC_USE_AGENT_STORE', 'OMP_NUM_THREADS', 'GROUP_RANK', 'ROLE_RANK', 'ROLE_NAME', 'LOCAL_WORLD_SIZE', 'GROUP_WORLD_SIZE', 'ROLE_WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT', 'WORLD_SIZE', 'LOCAL_RANK', 'RANK'}
    for var_name in torchrun_env_names:
        if var_name in env:
            del env[var_name]
    command = [str(c) for c in command]
    print(f"Running on GPUs {gpu_ids}: {' '.join(command)}", flush=True)

    try:
        subprocess.run(command, env=env, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error executing task '{task.name}': {e}", flush=True)
        return False
    finally:
        for gpu_id in gpu_ids:
            gpu_manager.release_gpu(gpu_id)
    return True

def worker(worker_id, task_queue, args, config, gpu_manager):
    """
    Функция-воркер. Получает задачи из очереди и выполняет их,
    используя закрепленные за ним GPU.
    """
    print(f"[Worker-{worker_id}] Started")
    while True:
        try:
            # Неблокирующее получение задачи из очереди
            task_group = task_queue.get_task()
            if task_group is None:
                print(f"[Worker-{worker_id}] No more tasks. Exiting.", flush=True)
                break
            print(f"[Worker-{worker_id}] Took task: {task_group.name}", flush=True)
            if not run_eval(args, config, task_group, gpu_manager):
                raise RuntimeError(f"Task {task_group.name} failed")
                
        except Exception as e:
            print(f"[Worker-{worker_id}] An unexpected error occurred: {e}", flush=True)
            raise

if __name__ == '__main__':
    nltk.download('punkt_tab', quiet=True)
    # Используем 'spawn' для безопасности при работе с CUDA
    mp.set_start_method('spawn', force=True)
    
    parser = argparse.ArgumentParser(description="Run local model evaluation and distribute tasks across GPUs.")
    parser.add_argument('--model_dir', required=True)
    parser.add_argument('--benchmark_config', required=True)
    parser.add_argument('--conv_path', default='auto')
    parser.add_argument('--output_dir', default=None)
    parser.add_argument('--force_recalc', action='store_true')
    parser.add_argument('--tensor_parallel_size', default=1, type=int)
    parser.add_argument('--num_gpus', type=int, default=torch.cuda.device_count())
    parser.add_argument('--is_foundational', action='store_true')
    parser.add_argument('--backend', choices=['hf', 'vllm'], default='vllm')
    parser.add_argument('--ppl', action='store_true')

    args = parser.parse_args()
    print("Parsed arguments:", args)
    
    # Проверка корректности аргументов
    if args.num_gpus % args.tensor_parallel_size != 0:
        raise ValueError("`num_gpus` must be divisible by `tensor_parallel_size`")
    if args.num_gpus <= 0:
        raise ValueError("No CUDA devices detected; pass --num_gpus explicitly on a GPU host")

    num_workers = args.num_gpus // args.tensor_parallel_size
    print(f"Planning to use {num_workers} workers with {args.tensor_parallel_size} GPU(s) each.")

    gpu_manager = GPUManager(args.num_gpus)
        
    # Создаем и заполняем очередь задач
    config = load_benchmark_config(args.benchmark_config)
    task_queue = TaskQueue(config.tasks)
    
    print(f'TOTAL WORKERS: {num_workers}')
    print(f'TOTAL TASKS: {len(config.tasks)}')
    print(f'BACKEND: {args.backend}')
    
    # Создаем и запускаем процессы-воркеры
    processes = []
    for i in range(num_workers):
        p = mp.Process(target=worker, args=(i, task_queue, args, config, gpu_manager))
        processes.append(p)
        p.start()
    
    # Ожидаем завершения всех воркеров
    for p in processes:
        p.join()
    output_dir = args.output_dir or os.path.join(args.model_dir, 'llmtf_eval')
    if os.path.isdir(output_dir):
        Evaluator().create_report(output_dir)
    if any(p.exitcode != 0 for p in processes):
        raise SystemExit(1)
        
    print("\nAll evaluation tasks completed.")
