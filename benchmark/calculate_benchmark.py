import os
import time
import argparse
import subprocess
import torch.multiprocessing as mp
import torch
from multiprocessing import Queue, Lock
from queue import Empty
import json
import nltk
nltk.download('punkt_tab')
import yaml

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
                    self.release_gpu(gpu_id)
                    
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


def run_eval(args, group, gpu_manager, gen_config_settings):
    """Запускает один таск оценки модели через прямой вызов evaluate_model.py."""
    gpu_ids = gpu_manager.acquire_gpu(count=args.tensor_parallel_size)
    if gpu_ids is None:
        return False

    if args.backend == 'vllm':
        default_bs = 10000000
    else:
        default_bs = 8

    batch_size = group['params'].get('batch_size', default_bs)
    few_shot_count = group['params'].get('few_shot_count', 0)
    max_len = group['params'].get('max_len', args.max_len)
    name_suffix = group['params'].get('name_suffix', None)

    conv_path = args.conv_path
    command = ['python', 'evaluate_model.py', '--model_name_or_path', args.model_dir, '--conv_path', conv_path, '--max_len', str(max_len), '--few_shot_count', str(few_shot_count), '--batch_size', str(batch_size)]
    command += ['--dataset_names'] + group['params']['dataset_names'].split()
    
    if args.backend == 'vllm':
        command += ['--vllm', '--tensor_parallel_size', str(args.tensor_parallel_size)]
    
    # Поддержка thinking режима (для задач с think=True используем специальные конфиги)
    if not group.get('think', False):
        command += ['--disable_thinking']
    
    if 'max_sample_per_dataset' in group['params']:
        command += ['--max_sample_per_dataset', str(group['params']['max_sample_per_dataset'])]
    
    if 'max_new_tokens_reasoning' in group['params']:
        command += ['--max_new_tokens_reasoning', str(group['params']['max_new_tokens_reasoning'])]

    if args.output_dir is not None:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(args.model_dir, 'llmtf_eval')
    command += ['--device_map', f'auto', '--output_dir', output_dir]
    if args.force_recalc:
        command += ['--force_recalc']
    
    if args.is_foundational:
        command += ['--is_foundational']

    if name_suffix is not None:
        command += ['--name_suffix', name_suffix]

    group_custom_gen_config = gen_config_settings.get(group['name'], {})
    for param in group_custom_gen_config:
        command += [f'--{param}', str(group_custom_gen_config[param])]

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
        print(f"Error executing task '{group['name']}': {e}", flush=True)
        return False
    finally:
        for gpu_id in gpu_ids:
            gpu_manager.release_gpu(gpu_id)
    return True

def worker(worker_id, task_queue, args, gpu_manager, gen_config_settings):
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
            print(f"[Worker-{worker_id}] Took task: {task_group['name']}", flush=True)
            
            # Повторяем попытки выполнения задачи до успеха
            while True:
                if run_eval(args, task_group, gpu_manager, gen_config_settings):
                    break
                time.sleep(5)  # Wait before retrying
                
        except Exception as e:
            print(f"[Worker-{worker_id}] An unexpected error occurred: {e}", flush=True)
            # Можно добавить логику повтора или просто пропустить задачу
            continue

def read_json(file_name):
    with open(file_name, encoding="utf-8") as r:
        return json.load(r)

def load_benchmark_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    defaults = config.get('defaults', {})
    gen_defaults = defaults.get('generation', {})
    
    task_groups = []
    gen_config_settings = {}
    
    for task in config.get('tasks', []):
        # Reconstruct task group dict
        group = {
            'name': task['name'],
            'params': {
                'dataset_names': ' '.join(task['datasets'])
            }
        }
        
        # Copy optional params
        param_keys = ['few_shot_count', 'max_len', 'name_suffix', 'max_sample_per_dataset', 'max_new_tokens_reasoning', 'batch_size']
        for key in param_keys:
            if key in defaults:
                group['params'][key] = defaults[key]
        
        for key in param_keys:
            if key in task:
                group['params'][key] = task[key]
        
        # Handle extra args (like think)
        if 'extra_args' in task:
            for k, v in task['extra_args'].items():
                group[k] = v
                
        task_groups.append(group)
        
        # Reconstruct generation config
        # Start with defaults
        gen_conf = gen_defaults.copy()
        # Update with task specific
        if 'generation' in task:
            gen_conf.update(task['generation'])
            
        gen_config_settings[task['name']] = gen_conf
        
    return task_groups, gen_config_settings
    
if __name__ == '__main__':
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
    parser.add_argument('--max_len', type=int, default=4000)
    parser.add_argument('--is_foundational', action='store_true')
    parser.add_argument('--backend', choices=['hf', 'vllm'], default='vllm')

    args = parser.parse_args()
    print("Parsed arguments:", args)
    
    # Проверка корректности аргументов
    if args.num_gpus % args.tensor_parallel_size != 0:
        raise ValueError("`num_gpus` must be divisible by `tensor_parallel_size`")

    num_workers = args.num_gpus // args.tensor_parallel_size
    print(f"Planning to use {num_workers} workers with {args.tensor_parallel_size} GPU(s) each.")

    gpu_manager = GPUManager(args.num_gpus)
        
    # Создаем и заполняем очередь задач
    task_groups, gen_config_settings = load_benchmark_config(args.benchmark_config)
    task_queue = TaskQueue(task_groups)
    
    print(f'TOTAL WORKERS: {num_workers}')
    print(f'TOTAL TASKS: {len(task_groups)}')
    print(f'BACKEND: {args.backend}')
    
    # Создаем и запускаем процессы-воркеры
    processes = []
    for i in range(num_workers):
        p = mp.Process(target=worker, args=(i, task_queue, args, gpu_manager, gen_config_settings))
        processes.append(p)
        p.start()
    
    # Ожидаем завершения всех воркеров
    for p in processes:
        p.join()
        
    print("\nAll evaluation tasks completed.")
