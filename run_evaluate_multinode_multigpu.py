import os
import time
import argparse
import subprocess
import torch.multiprocessing as mp

#TODO: refactoring and vllm
task_groups_few_shot = [
    {'name': 'darumeru', 'params': {'dataset_names': 'darumeru/multiq darumeru/parus darumeru/rcb darumeru/rwsd darumeru/use', 'allow_vllm': True}},
    {'name': 'nlpcoreteam_mmlu_ru', 'params': {'dataset_names': 'nlpcoreteam/rummlu', 'allow_vllm': True}},
    {'name': 'nlpcoreteam_mmlu_en', 'params': {'dataset_names': 'nlpcoreteam/enmmlu', 'allow_vllm': True}},
    {'name': 'treewayabstractive', 'params': {'dataset_names': 'daru/treewayabstractive', 'allow_vllm': True, 'max_sample_per_dataset': 500}},
    {'name': 'copy_tasks', 'params': {'dataset_names': 'darumeru/cp_para_ru darumeru/cp_para_en', 'allow_vllm': True}},
    {'name': 'ruopinionne_habr_ruparam', 'params': {'dataset_names': 'vikhrmodels/habr_qa_sbs ruopinionne ruparam', 'allow_vllm': True, 'max_sample_per_dataset': 5000}},
    {'name': 'nerel', 'params': {'dataset_names': 'nerel', 'allow_vllm': True, 'max_sample_per_dataset': 500}}
]

task_groups_zero_shot = [
    {'name': 'darumeru', 'params': {'dataset_names': 'darumeru/multiq darumeru/parus darumeru/rcb darumeru/rwsd darumeru/use', 'allow_vllm': True}},
    {'name': 'nlpcoreteam_mmlu_ru', 'params': {'dataset_names': 'nlpcoreteam/rummlu', 'allow_vllm': True}},
    {'name': 'nlpcoreteam_mmlu_en', 'params': {'dataset_names': 'nlpcoreteam/enmmlu', 'allow_vllm': True}},
    {'name': 'treewayabstractive', 'params': {'dataset_names': 'daru/treewayabstractive', 'allow_vllm': True, 'max_sample_per_dataset': 500}},
    {'name': 'copy_tasks', 'params': {'dataset_names': 'darumeru/cp_para_ru darumeru/cp_para_en', 'allow_vllm': True}},
    {'name': 'habr_ruparam_shlepa', 'params': {'dataset_names': 'vikhrmodels/habr_qa_sbs ruparam shlepa/moviesmc shlepa/musicmc shlepa/lawmc shlepa/booksmc', 'allow_vllm': True, 'max_sample_per_dataset': 5000}},
    {'name': 'ruopinionne', 'params': {'dataset_names': 'ruopinionne', 'allow_vllm': True, 'max_sample_per_dataset': 5000}},
    {'name': 'nerel', 'params': {'dataset_names': 'nerel', 'allow_vllm': True, 'max_sample_per_dataset': 500}},
]

task_groups_zero_shot_short_ver = [
    {'name': 'darumeru', 'params': {'dataset_names': 'darumeru/multiq darumeru/parus darumeru/rcb darumeru/rwsd darumeru/use', 'allow_vllm': True}},
    {'name': 'nlpcoreteam_rummlu', 'params': {'dataset_names': 'nlpcoreteam/rummlu', 'allow_vllm': True}},
    {'name': 'nlpcoreteam_enmmlu', 'params': {'dataset_names': 'nlpcoreteam/enmmlu', 'allow_vllm': True}},
    {'name': 'treewayabstractive', 'params': {'dataset_names': 'daru/treewayabstractive', 'allow_vllm': True, 'max_sample_per_dataset': 200}},
    {'name': 'cp_para_ru', 'params': {'dataset_names': 'darumeru/cp_para_ru', 'allow_vllm': True}},
    {'name': 'habr_ruparam_shlepa', 'params': {'dataset_names': 'vikhrmodels/habr_qa_sbs ruparam shlepa/moviesmc shlepa/musicmc shlepa/lawmc shlepa/booksmc', 'allow_vllm': True, 'max_sample_per_dataset': 5000}},
    {'name': 'ruopinionne', 'params': {'dataset_names': 'ruopinionne', 'allow_vllm': True, 'max_sample_per_dataset': 200}},
    {'name': 'nerel', 'params': {'dataset_names': 'nerel', 'allow_vllm': True, 'max_sample_per_dataset': 200}}
]

task_groups = None
def get_current_groups(rank, total_workers):
    current_idx = [i for i in range(rank, len(task_groups), total_workers)]
    return [task_groups[i] for i in current_idx]

def run_eval(args, group, local_rank):
    command = ['python', 'evaluate_model.py', '--model_name_or_path', args.model_dir, '--conv_path', args.conv_path, '--max_len', str(args.max_len), '--few_shot_count', str(args.few_shot_count), '--batch_size', str(args.batch_size)]
    command += ['--dataset_names'] + group['params']['dataset_names'].split()
    if args.vllm and group['params']['allow_vllm']:
        command += ['--vllm', '--disable_sliding_window']
    if 'max_sample_per_dataset' in group['params']:
        command += ['--max_sample_per_dataset', group['params']['max_sample_per_dataset']]

    if args.output_dir is not None:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(args.model_dir, 'llmtf_eval')
    command += ['--device_map', f'cuda:{0}', '--output_dir', output_dir]
    if args.force_recalc:
        command += ['--force_recalc']

    command += ['--alpha_scale', str(args.alpha_scale)]
    if args.not_scale_lm_head:
        command += ['--not_scale_lm_head']
        
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(local_rank)
    torchrun_env_names = {'TORCHELASTIC_USE_AGENT_STORE', 'OMP_NUM_THREADS', 'GROUP_RANK', 'ROLE_RANK', 'ROLE_NAME', 'LOCAL_WORLD_SIZE', 'GROUP_WORLD_SIZE', 'ROLE_WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT', 'WORLD_SIZE', 'LOCAL_RANK', 'RANK'}
    for var_name in torchrun_env_names:
        if var_name in env:
            del env[var_name]
    command = [str(c) for c in command]
    print(command)

    def func(command, env):
        subprocess.run(command, env=env, start_new_session=True)

    p = mp.Process(target=func, args=(command, env,))
    p.start()
    p.join()


import torch
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
    parser.add_argument('--short', action='store_true')

    args = parser.parse_args()

    print(torch.distributed.is_initialized())

    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    rank = int(os.environ.get('RANK', 0))
    workers = int(os.environ.get('WORLD_SIZE', 1))
    print(f'info: lr={local_rank}, r={rank}, total={workers}')
    time.sleep(rank*2)

    if int(args.few_shot_count) > 0:
        task_groups = task_groups_few_shot
    else:
        if args.short:
            task_groups = task_groups_zero_shot_short_ver
        else:
            task_groups = task_groups_zero_shot

    for group in get_current_groups(rank, workers):
        print(f'RANK {rank} starting {group}')
        run_eval(args, group, local_rank)
