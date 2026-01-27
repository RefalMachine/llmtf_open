#from llmtf.task import *
from llmtf.tasks import TASK_REGISTRY
from llmtf.base import Task, Base
from llmtf.utils import CustomTimer, SimpleTaskLogger, MaxLenContext, set_out_handler_to_main_logger
import os
import json
from tqdm import tqdm
import codecs
import inspect
import time
import numpy as np
import traceback
from pathlib import Path
import json
import os
import random
import torch
import numpy as np
import re
from typing import List, Dict
from sklearn.utils import resample


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:2"
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

class Evaluator(Base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        set_random_seed(555)

    def add_new_task(self, task_name, task_cls, task_params):
        assert issubclass(task_cls, Task)
        TASK_REGISTRY[task_name] = {'class': task_cls, 'params': task_params}

    def evaluate(
        self,
        model,
        output_dir,
        datasets_names='all',
        max_prompt_len=4096,
        few_shot_count=5,
        generation_config=None,
        batch_size=1,
        max_sample_per_dataset=100000000,
        include_stop_str_in_output=False,
        enable_thinking=False,
        add_reasoning_truncing_prompt=True,
        add_reasoning_info=True,
        add_assistant_prompt_to_output=True,
        force_recalc=False,
        name_suffix=None
    ):
        set_out_handler_to_main_logger(output_dir)
        try:
            if generation_config is not None:
                model.logger.warning('Custom generation_config receives full priority over internal generation config. Compose it carefully. Not tested yet!')

            if datasets_names == 'all':
                datasets_names = list(TASK_REGISTRY.keys())

            self.logger.info(f'Starting eval on {datasets_names}')
            for dataset_name in datasets_names:
                task_class = TASK_REGISTRY[dataset_name]['class']
                task_init_params = TASK_REGISTRY[dataset_name].get('params', {})
                task_init_params['name_suffix'] = task_init_params.get('name_suffix', name_suffix)
                task = task_class(**task_init_params)
                if (Path(output_dir) / f"{task.run_name().replace('/', '_')}_total.jsonl").exists() and not force_recalc:
                    self.logger.info(f"Found precomputed {task.run_name()}_total")
                    continue

                try:
                    # MaxLenContext changes model.generation_config.max_new_tokens param based on task.max_new_tokens
                    with MaxLenContext(task, model, max_prompt_len, generation_config) as max_prompt_len:
                        self.evaluate_dataset(
                            task,
                            model,
                            output_dir,
                            max_prompt_len,
                            few_shot_count,
                            generation_config,
                            batch_size,
                            max_sample_per_dataset,
                            enable_thinking,
                            include_stop_str_in_output,
                            add_reasoning_truncing_prompt,
                            add_reasoning_info,
                            add_assistant_prompt_to_output,
                        )
                except Exception as e:
                    self.logger.error(f"Failed to evaluate on {dataset_name}: {e}")
                    self.logger.error(traceback.format_exc())

            self.logger.info(f'Ended eval')
            self.create_report(output_dir)
        except Exception as e:
            self.logger.error(e)
            self.logger.error(traceback.format_exc()) 
        
    def evaluate_dataset(
        self,
        task,
        model,
        output_dir,
        max_prompt_len,
        few_shot_count,
        generation_config,
        batch_size,
        max_sample_per_dataset,
        enable_thinking,
        include_stop_str_in_output,
        add_reasoning_truncing_prompt,
        add_reasoning_info,
        add_assistant_prompt_to_output
    ):
        model.add_stop_strings(task.additional_stop_strings)
        with CustomTimer(task.logger, 'Loading Dataset'):
            messages, samples = task.load_dataset(model=model, max_prompt_len=max_prompt_len, max_sample_per_dataset=max_sample_per_dataset, few_shot_count=few_shot_count)

        metrics = []
        with SimpleTaskLogger(output_dir, task.run_name()) as logger, CustomTimer(task.logger, 'Processing Dataset') as timer:
            for i in tqdm(range(0, len(messages), batch_size)):
                messages_batch = messages[i:i+batch_size]
                messages_batch = {k: [subdict[k] for subdict in messages_batch] for k in messages_batch[0]}
                if generation_config is not None:
                    messages_batch['generation_config'] = generation_config
                for k, v in task.method_additional_args.items():
                    messages_batch[k] = v
                if task.method in ['generate', 'calculate_tokens_proba']:
                    messages_batch['enable_thinking'] = enable_thinking
                    messages_batch['add_reasoning_truncing_prompt'] = add_reasoning_truncing_prompt
                    messages_batch['add_reasoning_info'] = add_reasoning_info
                if task.method == 'generate':
                    messages_batch['include_stop_str_in_output'] = include_stop_str_in_output
                    messages_batch['add_assistant_prompt_to_output'] = add_assistant_prompt_to_output

                if task.method == 'generate':
                    prompts, y_preds, infos = model.generate_batch(**messages_batch)
                elif task.method == 'calculate_tokens_proba':
                    prompts, y_preds, infos = model.calculate_tokens_proba_batch(**messages_batch)

                for j in range(len(y_preds)):
                    metrics.append(task.evaluate(samples[i+j]['sample'], y_preds[j]))
                    logger.log_sample(samples[i+j]['sample'], y_preds[j], prompts[j], metrics[-1], infos[j])
            processing_time = timer.time()
        
        task.logger.info(f'Results for {task.run_name()}:')
        
        # Агрегация с возможностью сохранения деталей
        metrics_res = {}
        aggregation_details = {}
        for metric in metrics[0].keys():
            agg_result = task.aggregation()[metric]([m[metric] for m in metrics])
            
            # Проверяем, вернула ли функция агрегации детали (tuple из 2 элементов)
            if isinstance(agg_result, tuple) and len(agg_result) == 2:
                metrics_res[metric] = agg_result[0]  # Агрегированное значение
                aggregation_details[metric] = agg_result[1]  # Детали агрегации
            else:
                metrics_res[metric] = agg_result  # Обратная совместимость

        # bootstrapping
        if hasattr(task, "n_bags"):
            assert hasattr(task, "n_samples")
            metrics_res.update(self._bootstrap(task, metrics))

        with SimpleTaskLogger(output_dir, task.run_name() + '_total') as logger:
            logger.log_json({'task_name': task.run_name(), 'results': metrics_res, 'leaderboard_result': task.leaderboard_aggregation(metrics_res), 'time': processing_time})
        
        # Сохранение детальных результатов агрегации, если они есть
        if aggregation_details:
            with SimpleTaskLogger(output_dir, task.run_name() + '_aggregation_details') as logger:
                logger.log_json({'task_name': task.run_name(), 'aggregation_details': aggregation_details})

        with SimpleTaskLogger(output_dir, task.run_name() + '_params') as logger:
            params = {}
            params['custom_generation_config'] = generation_config
            params['model_params'] = model.get_params()
            params['task_params'] = {'max_prompt_len': max_prompt_len, 'few_shot_count': few_shot_count, 'batch_size': batch_size, 'max_sample_per_dataset': max_sample_per_dataset, 'method': task.method}
            logger.log_json(params)

        model.reset_stop_strings()
        task.logger.info(str(metrics_res))

    def evaluate_ppl(self, model, output_dir, datasets_names='all', max_prompt_len=4096, few_shot_count=5, batch_size=1, max_sample_per_dataset=100000000, force_recalc=False, name_suffix=None):
        set_out_handler_to_main_logger(output_dir)
        try:
            if datasets_names == 'all':
                datasets_names = list(TASK_REGISTRY.keys())

            self.logger.info(f'Starting eval on {datasets_names}')
            for dataset_name in datasets_names:
                task_class = TASK_REGISTRY[dataset_name]['class']
                task_init_params = TASK_REGISTRY[dataset_name].get('params', {})
                task_init_params['name_suffix'] = task_init_params.get('name_suffix', name_suffix)
                task = task_class(**task_init_params)
                if (Path(output_dir) / f"{task.run_name().replace('/', '_')}_total.jsonl").exists() and not force_recalc:
                    self.logger.info(f"Found precomputed {task.run_name()}_total")
                    continue
                
                if 'get_answer' not in dir(task):
                    self.logger.info(f"Skip task {task.run_name()} because method get_answer not implemented")
                    continue

                with MaxLenContext(task, model, max_prompt_len, None) as max_prompt_len:
                    self.evaluate_dataset_ppl(task, model, output_dir, max_prompt_len, few_shot_count, batch_size, max_sample_per_dataset)

            self.logger.info(f'Ended eval')
            self.create_report(output_dir)
        except Exception as e:
            self.logger.error(e)
            self.logger.error(traceback.format_exc()) 

    def evaluate_dataset_ppl(self, task, model, output_dir, max_prompt_len, few_shot_count, batch_size, max_sample_per_dataset):
        assert 'get_answer' in dir(task)
        assert model.support_method('calculate_logsoftmax')

        messages, samples = task.load_dataset(model=model, max_prompt_len=max_prompt_len, max_sample_per_dataset=max_sample_per_dataset, few_shot_count=few_shot_count)
        shifts = []
        for m, s in zip(*[messages, samples]):
            shift = len(model.apply_model_prompt(m['messages']))
            shifts.append(shift)
            if m['messages'][-1]['role'] == 'assistant':
                m['messages'][-1]['content'] += task.get_answer(s['sample'])
            else:
                m['messages'].append({'role': 'assistant', 'content': task.get_answer(s['sample'])})

        metrics = []
        with SimpleTaskLogger(output_dir, task.run_name()) as logger, CustomTimer(task.logger, 'Processing Dataset'):
            for i in tqdm(range(0, len(messages), batch_size)):
                messages_batch = messages[i:i+batch_size]
                messages_batch = {k: [subdict[k] for subdict in messages_batch] for k in messages_batch[0]}
                for k, v in task.method_additional_args.items():
                    messages_batch[k] = v
                    
                if 'tokens_of_interest' in messages_batch:
                    del messages_batch['tokens_of_interest']
                if 'return_tokens' in messages_batch:
                    del messages_batch['return_tokens']

                prompts, y_preds, infos = model.calculate_logsoftmax_batch(**messages_batch)

                for j in range(len(y_preds)):
                    tokens = [t for t in y_preds[j][-1]['tokens'] if t[2][0] >= shifts[i+j]]
                    metrics.append({'ppl': np.mean([t[1] for t in tokens])})
                    logger.log_sample(samples[i+j]['sample'], y_preds[j], prompts[j], metrics[-1], infos[j])

        task.logger.info(f'Results for {task.run_name()}:')
        metrics_res = {'ppl': np.mean([m['ppl'] for m in metrics])}

        # bootstrapping
        if hasattr(task, "n_bags"):
            assert hasattr(task, "n_samples")
            metrics_res.update(self._bootstrap(task, metrics))

        with SimpleTaskLogger(output_dir, task.run_name() + '_total') as logger:
            logger.log_json({'task_name': task.run_name(), 'results': metrics_res, 'leaderboard_result': metrics_res['ppl']})

        with SimpleTaskLogger(output_dir, task.run_name() + '_params') as logger:
            params = {}
            params['model_params'] = model.get_params()
            params['task_params'] = {'max_prompt_len': max_prompt_len, 'few_shot_count': few_shot_count, 'batch_size': batch_size, 'max_sample_per_dataset': max_sample_per_dataset, 'method': 'calculate_logsoftmax'}
            logger.log_json(params)

        task.logger.info(str(metrics_res))

    def create_report(self, output_dir):
        reports = {}
        for file_name in os.listdir(output_dir):
            if file_name.endswith('_total.jsonl'):
                with codecs.open(os.path.join(output_dir, file_name), 'r', 'utf-8') as file:
                    task_report = json.load(file)
                reports[task_report['task_name']] = task_report['leaderboard_result']
        task_names = sorted(list(reports.keys()))
        task_metrics = [reports[t] for t in task_names]
        task_names = ['mean'] + task_names
        task_metrics = [np.mean(task_metrics)] + task_metrics

        with codecs.open(os.path.join(output_dir, 'evaluation_results.txt'), 'w', 'utf-8') as file:
            file.write('\t'.join(task_names) + '\n')
            file.write('\t'.join([f'{m:.3f}' for m in task_metrics]))
            
        self.logger.info('\n' + '\t'.join(task_names) + '\n' + '\t'.join([f'{m:.3f}' for m in task_metrics]))
                

    def _bootstrap(self, task, metrics: List[Dict]) -> Dict:
        aggregation = task.aggregation()
        metrics_bags_res = {}
        for metric in metrics[0].keys():
            metrics_bags_res[metric] = []

        for _ in range(task.n_bags):
            metrics_bag = resample(metrics, replace=True, n_samples=task.n_samples) # random_state?
            assert metrics_bag is not None
            for metric in metrics[0].keys():
                metrics_bags_res[metric].append(aggregation[metric]([m[metric] for m in metrics_bag]))

        bootstrapped_metrics = {}
        for metric in metrics[0].keys():
            metric_bags_res_mean = np.mean(metrics_bags_res[metric])
            metric_bags_res_std = np.std(metrics_bags_res[metric])
            bootstrapped_metrics[metric + "_std"] = metric_bags_res_std
            bootstrapped_metrics[metric + "_lower"] = metric_bags_res_mean - metric_bags_res_std
            bootstrapped_metrics[metric + "_upper"] = metric_bags_res_mean + metric_bags_res_std

        return bootstrapped_metrics
