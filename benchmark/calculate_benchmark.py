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

task_groups_knowledge = [
    {'name': 'nlpcoreteam_mmlu_ru_zero_shot', 'params': {'dataset_names': 'nlpcoreteam/rummlu', 'few_shot_count': 0, 'name_suffix': 'zero_shot'}},
    {'name': 'nlpcoreteam_mmlu_en_zero_shot', 'params': {'dataset_names': 'nlpcoreteam/enmmlu', 'few_shot_count': 0, 'name_suffix': 'zero_shot'}},
    {'name': 'nlpcoreteam_mmlu_ru_few_shot', 'params': {'dataset_names': 'nlpcoreteam/rummlu', 'few_shot_count': 5, 'name_suffix': 'few_shot'}},
    {'name': 'nlpcoreteam_mmlu_en_few_shot', 'params': {'dataset_names': 'nlpcoreteam/enmmlu', 'few_shot_count': 5, 'name_suffix': 'few_shot'}},
    {'name': 'shlepa', 'params': {'dataset_names': 'shlepa/moviesmc shlepa/musicmc shlepa/lawmc shlepa/booksmc'}},
]

task_groups_skills = [
    {'name': 'translation', 'params': {'dataset_names': 'darumeru/flores_ru_en darumeru/flores_en_ru'}},
    {'name': 'summarization', 'params': {'dataset_names': 'daru/treewayabstractive ilyagusev/gazeta', 'max_sample_per_dataset': 1000}},
    {'name': 'sentiment', 'params': {'dataset_names': 'ruopinionne ruopinionne_simple', 'max_sample_per_dataset': 1000}},
    {'name': 'ner', 'params': {'dataset_names': 'nerel-bio nerel', 'max_sample_per_dataset': 500, 'few_shot_count': 5, 'max_len': 12000}},
    {'name': 'rag', 'params': {'dataset_names': 'rusbeirrag/rubqqa rusbeirrag/rus_tydiqa rusbeirrag/sberquadqa rusbeirrag/rus_xquadqa', 'max_sample_per_dataset': 500, 'max_len': 12000}},
    {'name': 'rag_data_first', 'params': {'dataset_names': 'rusbeirrag/rubqqa_data_first rusbeirrag/rus_tydiqa_data_first rusbeirrag/sberquadqa_data_first rusbeirrag/rus_xquadqa_data_first', 'max_sample_per_dataset': 500, 'max_len': 12000}},
]

task_groups_ifeval = [
    {'name': 'ruifeval', 'params': {'dataset_names': 'ruifeval', 'few_shot_count': 0}},
    {'name': 'enifeval', 'params': {'dataset_names': 'enifeval', 'few_shot_count': 0}},
]

task_groups_long = [
    {'name': 'libra_rubabilong1', 'params': {'dataset_names': 'libra/rubabilong1', 'max_sample_per_dataset': 1000, 'few_shot_count': 0, 'max_len': 32000}},
    {'name': 'libra_rubabilong2', 'params': {'dataset_names': 'libra/rubabilong2', 'max_sample_per_dataset': 1000, 'few_shot_count': 0, 'max_len': 32000}},
    {'name': 'libra_rubabilong3', 'params': {'dataset_names': 'libra/rubabilong3', 'max_sample_per_dataset': 1000, 'few_shot_count': 0, 'max_len': 32000}},
    {'name': 'libra_rubabilong4', 'params': {'dataset_names': 'libra/rubabilong4', 'max_sample_per_dataset': 1000, 'few_shot_count': 0, 'max_len': 32000}},
    {'name': 'libra_rubabilong5', 'params': {'dataset_names': 'libra/rubabilong5', 'max_sample_per_dataset': 1000, 'few_shot_count': 0, 'max_len': 32000}}
]

def run_eval(args, group, gpu_manager, gen_config_settings):
    gpu_ids = gpu_manager.acquire_gpu(count=args.tensor_parallel_size)
    if gpu_ids is None:
        return False

    batch_size = group['params'].get('batch_size', 10000000) # vllm
    few_shot_count = group['params'].get('few_shot_count', 0)
    max_len = group['params'].get('max_len', args.max_len)
    name_suffix = group['params'].get('name_suffix', None)

    conv_path = args.conv_path
    command = ['python', 'evaluate_model.py', '--model_name_or_path', args.model_dir, '--conv_path', conv_path, '--max_len', str(max_len), '--few_shot_count', str(few_shot_count), '--batch_size', str(batch_size)]
    command += ['--dataset_names'] + group['params']['dataset_names'].split()
    command += ['--vllm', '--tensor_parallel_size', args.tensor_parallel_size]
    if 'max_sample_per_dataset' in group['params']:
        command += ['--max_sample_per_dataset', group['params']['max_sample_per_dataset']]

    if args.output_dir is not None:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(args.model_dir, 'llmtf_eval')
    command += ['--device_map', f'auto', '--output_dir', output_dir]
    if args.force_recalc:
        command += ['--force_recalc']

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
    print(f"Running on GPUs {gpu_ids}: {command}")

    try:
        subprocess.run(command, env=env, check=True)
    finally:
        for gpu_id in gpu_ids:
            gpu_manager.release_gpu(gpu_id)
    return True

def worker(args, task_queue, gpu_manager, gen_config_settings):
    while True:
        group = task_queue.get_task()
        if group is None:
            break
        while True:
            if run_eval(args, group, gpu_manager, gen_config_settings):
                break
            time.sleep(5)  # Wait before retrying

def read_json(file_name):
    with open(file_name, encoding="utf-8") as r:
        return json.load(r)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir')
    parser.add_argument('--gen_config_settings')
    parser.add_argument('--conv_path')
    parser.add_argument('--output_dir', default=None)
    parser.add_argument('--force_recalc', action='store_true')
    parser.add_argument('--tensor_parallel_size', default=1, type=int)
    parser.add_argument('--num_gpus', type=int, default=torch.cuda.device_count())
    parser.add_argument('--max_len', type=int, default=4000)

    args = parser.parse_args()
    print(args)

    gpu_manager = GPUManager(args.num_gpus)
    task_groups = task_groups_knowledge + task_groups_skills + task_groups_ifeval + task_groups_long
        
    task_queue = TaskQueue(task_groups)
    gen_config_settings = read_json(args.gen_config_settings)
    # Create worker processes
    num_workers = min(args.num_gpus // args.tensor_parallel_size, len(task_groups))
    processes = []
    for _ in range(num_workers):
        p = mp.Process(target=worker, args=(args, task_queue, gpu_manager, gen_config_settings))
        p.start()
        processes.append(p)
    
    # Wait for all processes to complete
    for p in processes:
        p.join()
