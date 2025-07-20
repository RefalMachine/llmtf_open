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


task_groups_math_no_think = [
    {'name': 'doom_math_no_think', 'params': {'dataset_names': 'doom/math', 'few_shot_count': 0, 'max_len': 32000, 'name_suffix': 'no_think', 'batch_size': 100000000000}},
    {'name': 'doom_phys_no_think', 'params': {'dataset_names': 'doom/phys', 'few_shot_count': 0, 'max_len': 32000, 'name_suffix': 'no_think', 'batch_size': 100000000000}},
    {'name': 't-bank_t-math_no_think', 'params': {'dataset_names': 't-bank/t-math', 'few_shot_count': 0, 'max_len': 32000, 'name_suffix': 'no_think', 'batch_size': 100000000000}}
]

task_groups_math_think = [
    {'name': 'doom_math', 'params': {'dataset_names': 'doom/math', 'few_shot_count': 0, 'max_len': 32000, 'name_suffix': 'think', 'batch_size': 100000000000}, 'think': True},
    {'name': 'doom_phys', 'params': {'dataset_names': 'doom/phys', 'few_shot_count': 0, 'max_len': 32000, 'name_suffix': 'think', 'batch_size': 100000000000}, 'think': True},
    {'name': 't-bank_t-math', 'params': {'dataset_names': 't-bank/t-math', 'few_shot_count': 0, 'max_len': 32000, 'name_suffix': 'think', 'batch_size': 100000000000}, 'think': True}
]

def run_eval(args, group, gen_config_settings):
    batch_size = group['params'].get('batch_size', 10000000)
    few_shot_count = group['params'].get('few_shot_count', 0)
    max_len = group['params'].get('max_len', args.max_len)
    name_suffix = group['params'].get('name_suffix', None)

    command = ['python', 'evaluate_model_api.py', '--base_url', args.base_url, '--model_name_or_path', args.model_dir, '--api_key', args.api_key, '--max_len', str(max_len), '--few_shot_count', str(few_shot_count), '--batch_size', str(batch_size)]
    command += ['--dataset_names'] + group['params']['dataset_names'].split()
    if not group.get('think', False):
        command += ['--disable_thinking']
    if 'max_sample_per_dataset' in group['params']:
        command += ['--max_sample_per_dataset', group['params']['max_sample_per_dataset']]

    if args.output_dir is not None:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(args.model_dir, 'llmtf_eval')
    command += ['--output_dir', output_dir]
    if args.force_recalc:
        command += ['--force_recalc']

    if name_suffix is not None:
        command += ['--name_suffix', name_suffix]

    group_custom_gen_config = gen_config_settings.get(group['name'], {})
    for param in group_custom_gen_config:
        command += [f'--{param}', str(group_custom_gen_config[param])]

    env = os.environ.copy()
    torchrun_env_names = {'TORCHELASTIC_USE_AGENT_STORE', 'OMP_NUM_THREADS', 'GROUP_RANK', 'ROLE_RANK', 'ROLE_NAME', 'LOCAL_WORLD_SIZE', 'GROUP_WORLD_SIZE', 'ROLE_WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT', 'WORLD_SIZE', 'LOCAL_RANK', 'RANK'}
    for var_name in torchrun_env_names:
        if var_name in env:
            del env[var_name]
    command = [str(c) for c in command]
    print(f"Running on API: {command}")

    try:
        subprocess.run(command, env=env, check=True)
    finally:
        pass

    return True

def read_json(file_name):
    with open(file_name, encoding="utf-8") as r:
        return json.load(r)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir')
    parser.add_argument('--base_url')
    parser.add_argument('--gen_config_settings')
    parser.add_argument('--api_key', default='1')
    parser.add_argument('--output_dir', default=None)
    parser.add_argument('--force_recalc', action='store_true')
    parser.add_argument('--add_reasoning_tasks', action='store_true')
    parser.add_argument('--max_len', type=int, default=4000)

    args = parser.parse_args()
    print(args)

    task_groups = task_groups_math_no_think
    if args.add_reasoning_tasks:
        task_groups += task_groups_math_think
        
    gen_config_settings = read_json(args.gen_config_settings)

    for task_group in task_groups:
        run_eval(args, task_group, gen_config_settings)

