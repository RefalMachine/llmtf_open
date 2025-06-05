import os
import time
import argparse
import subprocess
import torch.multiprocessing as mp
import torch
from multiprocessing import Queue, Lock
from queue import Empty

import nltk
nltk.download('punkt_tab')

class GPUManager:
    def __init__(self, num_gpus):
        self.available_gpus = Queue()
        self.lock = Lock()
        for i in range(num_gpus):
            self.available_gpus.put(i)
    
    def acquire_gpu(self, timeout=5):
        with self.lock:
            try:
                gpuid = self.available_gpus.get(timeout=timeout)
                print(f'Acquired GPU {gpuid}')
                return gpuid
            except Empty:
                print('No GPU available')
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


task_groups_default = [
    {'name': 'flores_use_multiq', 'params': {'dataset_names': 'darumeru/flores_ru_en darumeru/flores_en_ru darumeru/use darumeru/multiq', 'allow_vllm': True}},
    {'name': 'habr_ruparam_shlepa', 'params': {'dataset_names': 'vikhrmodels/habr_qa_sbs ruparam shlepa/moviesmc shlepa/musicmc shlepa/lawmc shlepa/booksmc', 'allow_vllm': True, 'max_sample_per_dataset': 1000}},
    {'name': 'nlpcoreteam_mmlu_ru', 'params': {'dataset_names': 'nlpcoreteam/rummlu', 'allow_vllm': True}},
    {'name': 'nlpcoreteam_mmlu_en', 'params': {'dataset_names': 'nlpcoreteam/enmmlu', 'allow_vllm': True}},
    {'name': 'cp_doc_ru', 'params': {'dataset_names': 'darumeru/cp_doc_ru', 'allow_vllm': True}},
    {'name': 'cp_para_ru', 'params': {'dataset_names': 'darumeru/cp_para_ru', 'allow_vllm': True}},
    {'name': 'nerel_ruopinionne_treewayabstractive', 'params': {'dataset_names': 'nerel ruopinionne daru/treewayabstractive', 'allow_vllm': True, 'max_sample_per_dataset': 500}}
]

task_groups_zero_shot_additional = [
    {'name': 'ruifeval', 'params': {'dataset_names': 'ruifeval', 'allow_vllm': True, 'few_shot_count': 0}},
    {'name': 'enifeval', 'params': {'dataset_names': 'enifeval', 'allow_vllm': True, 'few_shot_count': 0}},
    {'name': 'libra_rubabilong1', 'params': {'dataset_names': 'libra/rubabilong1', 'allow_vllm': True, 'max_sample_per_dataset': 200, 'few_shot_count': 0, 'max_len': 32000}},
    {'name': 'libra_rubabilong2', 'params': {'dataset_names': 'libra/rubabilong2', 'allow_vllm': True, 'max_sample_per_dataset': 200, 'few_shot_count': 0, 'max_len': 32000}},
    {'name': 'libra_rubabilong3', 'params': {'dataset_names': 'libra/rubabilong3', 'allow_vllm': True, 'max_sample_per_dataset': 200, 'few_shot_count': 0, 'max_len': 32000}},
    {'name': 'libra_rubabilong4', 'params': {'dataset_names': 'libra/rubabilong4', 'allow_vllm': True, 'max_sample_per_dataset': 200, 'few_shot_count': 0, 'max_len': 32000}},
    {'name': 'libra_rubabilong5', 'params': {'dataset_names': 'libra/rubabilong5', 'allow_vllm': True, 'max_sample_per_dataset': 200, 'few_shot_count': 0, 'max_len': 32000}}
]

def run_eval(args, group, gpu_manager):
    gpu_id = gpu_manager.acquire_gpu()
    if gpu_id is None:
        return False

    batch_size = group['params'].get('batch_size', args.batch_size)
    few_shot_count = group['params'].get('few_shot_count', args.few_shot_count)
    max_len = group['params'].get('max_len', args.max_len)

    command = ['python', 'evaluate_model.py', '--model_name_or_path', args.model_dir, '--conv_path', args.conv_path, '--max_len', str(max_len), '--few_shot_count', str(few_shot_count), '--batch_size', str(batch_size)]
    command += ['--dataset_names'] + group['params']['dataset_names'].split()
    if args.vllm and group['params']['allow_vllm']:
        command += ['--vllm', '--disable_sliding_window']
    if 'max_sample_per_dataset' in group['params']:
        command += ['--max_sample_per_dataset', group['params']['max_sample_per_dataset']]

    if args.output_dir is not None:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(args.model_dir, 'llmtf_eval')
    command += ['--device_map', f'cuda:{gpu_id}', '--output_dir', output_dir]
    if args.force_recalc:
        command += ['--force_recalc']

    command += ['--alpha_scale', str(args.alpha_scale)]
    if args.not_scale_lm_head:
        command += ['--not_scale_lm_head']
    
    if args.ppl_scoring:
        command += ['--ppl_scoring']

    env = os.environ.copy()
    #env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    torchrun_env_names = {'TORCHELASTIC_USE_AGENT_STORE', 'OMP_NUM_THREADS', 'GROUP_RANK', 'ROLE_RANK', 'ROLE_NAME', 'LOCAL_WORLD_SIZE', 'GROUP_WORLD_SIZE', 'ROLE_WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT', 'WORLD_SIZE', 'LOCAL_RANK', 'RANK'}
    for var_name in torchrun_env_names:
        if var_name in env:
            del env[var_name]
    command = [str(c) for c in command]
    print(f"Running on GPU {gpu_id}: {command}")

    try:
        subprocess.run(command, env=env, check=True)
    finally:
        gpu_manager.release_gpu(gpu_id)
    return True

def worker(args, task_queue, gpu_manager):
    while True:
        group = task_queue.get_task()
        if group is None:
            break
        while True:
            if run_eval(args, group, gpu_manager):
                break
            time.sleep(5)  # Wait before retrying

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir')
    parser.add_argument('--conv_path')
    parser.add_argument('--output_dir', default=None)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--max_len', default=4000, type=int)
    parser.add_argument('--few_shot_count', default=0, type=int)
    parser.add_argument('--vllm', action='store_true')
    parser.add_argument('--force_recalc', action='store_true')
    parser.add_argument('--alpha_scale', type=float, default=1.0)
    parser.add_argument('--not_scale_lm_head', action='store_true')
    parser.add_argument('--ppl_scoring', action='store_true')
    parser.add_argument('--num_gpus', type=int, default=torch.cuda.device_count())

    args = parser.parse_args()
    print(args)

    gpu_manager = GPUManager(args.num_gpus)
    task_groups = task_groups_default
    if int(args.few_shot_count) == 0 and not args.ppl_scoring:
        task_groups += task_groups_zero_shot_additional
        
    task_queue = TaskQueue(task_groups)

    # Create worker processes
    num_workers = min(args.num_gpus, len(task_groups))
    processes = []
    for _ in range(num_workers):
        p = mp.Process(target=worker, args=(args, task_queue, gpu_manager))
        p.start()
        processes.append(p)
    
    # Wait for all processes to complete
    for p in processes:
        p.join()
