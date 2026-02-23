import os
import argparse
import yaml
from multiprocessing import Queue
import subprocess
from queue import Empty
import requests
import time

def run_eval(args, group, gen_config_settings):
    """Запускает один таск оценки модели через API."""
    batch_size = group['params'].get('batch_size', 10000000)
    few_shot_count = group['params'].get('few_shot_count', 0)
    max_prompt_len = group['params'].get('max_prompt_len', args.max_prompt_len)
    name_suffix = group['params'].get('name_suffix', None)

    command = ['python', 'evaluate_model_api.py', '--base_url', args.base_url.replace('/v1', ''), '--model_name_or_path', args.model_name, '--api_key', args.api_key, '--max_prompt_len', str(max_prompt_len), '--few_shot_count', str(few_shot_count), '--batch_size', str(batch_size)]
    command += ['--dataset_names'] + group['params']['dataset_names'].split()
    if not group.get('think', False):
        command += ['--disable_thinking']
    if 'max_sample_per_dataset' in group['params']:
        command += ['--max_sample_per_dataset', group['params']['max_sample_per_dataset']]
    if 'max_new_tokens_reasoning' in group['params']:
        command += ['--max_new_tokens_reasoning', group['params']['max_new_tokens_reasoning']]

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

    command = [str(c) for c in command]
    print(f"[worker] Running command: {' '.join(command)}", flush=True)

    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[worker] Error executing task '{group['name']}': {e}", flush=True)
        return False

    return True

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
        param_keys = ['few_shot_count', 'max_prompt_len', 'name_suffix', 'max_sample_per_dataset', 'max_new_tokens_reasoning', 'batch_size']
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
    parser = argparse.ArgumentParser(description="Run tasks evaluation with vllm api")

    parser.add_argument('--model_name', required=True)
    parser.add_argument('--base_url', required=True)
    parser.add_argument('--api_key', required=True)
    parser.add_argument('--benchmark_config', required=True)
    parser.add_argument('--output_dir', default=None)
    parser.add_argument('--force_recalc', action='store_true')
    parser.add_argument('--add_reasoning_tasks', action='store_true')
    parser.add_argument('--max_prompt_len', type=int, default=4000)
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
    
    task_groups, gen_config_settings = load_benchmark_config(args.benchmark_config)

    # Создаем и заполняем очередь задач
    task_queue = Queue()
    for task in task_groups:
        task_queue.put(task)

    print(f"[Worker] Started", flush=True)
    while True:
        try:
            # Неблокирующее получение задачи из очереди
            task_group = task_queue.get_nowait()
            print(f"[Worker] Took task: {task_group['name']}", flush=True)
            run_eval(args, task_group, gen_config_settings)
        except Empty:
            # Если очередь пуста, воркер завершает работу
            print(f"[Worker] No more tasks. Exiting.", flush=True)
            break
        except Exception as e:
            print(f"[Worker] An unexpected error occurred: {e}", flush=True)
            # Можно добавить логику повтора или просто пропустить задачу
            continue

    print("\nAll evaluation tasks completed.")