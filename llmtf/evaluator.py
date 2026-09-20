from llmtf.tasks import TASK_REGISTRY
from llmtf.base import Task, Base
from llmtf.utils import CustomTimer, MaxLenContext, set_out_handler_to_main_logger
from llmtf.sample_logger import JsonArrayLogger, PrettyJsonLogger
import os
import json
import codecs
import copy
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
import random
import numpy as np
from typing import List, Dict
from sklearn.utils import resample
from tqdm import tqdm
from llmtf.provenance import build_run_config, fingerprint_run_config, validate_cache
from llmtf.reasoning import resolve_reasoning_execution


@dataclass
class EvaluationSummary:
    succeeded: List[str] = field(default_factory=list)
    skipped: List[str] = field(default_factory=list)
    failed: Dict[str, str] = field(default_factory=dict)

    @property
    def exit_code(self):
        return 1 if self.failed else 0

    @property
    def ok(self):
        return not self.failed


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:2"
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
    except ImportError:
        return
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
        few_shot_count=5,
        generation_config=None,
        batch_size=1,
        max_sample_per_dataset=100000000,
        include_stop_str_in_output=False,
        enable_thinking=False,
        add_reasoning_truncing_prompt=True,
        add_reasoning_info=True,
        force_recalc=False,
        name_suffix=None
    ):
        set_out_handler_to_main_logger(output_dir)
        summary = EvaluationSummary()
        if generation_config is not None:
            model.logger.warning('Custom generation_config receives full priority over internal generation config. Compose it carefully.')
        if datasets_names == 'all':
            datasets_names = list(TASK_REGISTRY.keys())
        mode = resolve_reasoning_execution(
            model.reasoning_config.model_kind, enable_thinking, "generate"
        )
        self.logger.info(f'Starting eval on {datasets_names}')
        for dataset_name in datasets_names:
            try:
                task_class = TASK_REGISTRY[dataset_name]['class']
                task_init_params = dict(TASK_REGISTRY[dataset_name].get('params', {}))
                task_init_params['name_suffix'] = task_init_params.get('name_suffix', name_suffix)
                task = task_class(**task_init_params)
                budget = MaxLenContext(
                    task, model, generation_config,
                    reasoning_enabled=mode.reasoning_enabled,
                    scoring_method=task.method,
                )
                with budget as max_prompt_len:
                    effective_thinking = budget.effective_enable_thinking
                    run_config = build_run_config(
                        model=model, task=task,
                        enable_thinking=effective_thinking,
                        generation_config=generation_config,
                        few_shot_count=few_shot_count,
                        batch_size=batch_size,
                        max_sample_per_dataset=max_sample_per_dataset,
                        max_prompt_len=max_prompt_len,
                        effective_reasoning_tokens=budget.effective_reasoning_tokens,
                        scoring_method=task.method,
                        requested_enable_thinking=enable_thinking,
                    )
                    run_fingerprint = fingerprint_run_config(run_config)
                    if self._cache_hit(
                        output_dir, task, run_fingerprint, force_recalc
                    ):
                        summary.skipped.append(dataset_name)
                        continue
                    self.evaluate_dataset(
                        task, model, output_dir, max_prompt_len, few_shot_count,
                        generation_config, batch_size, max_sample_per_dataset,
                        effective_thinking, include_stop_str_in_output,
                        add_reasoning_truncing_prompt, add_reasoning_info,
                        run_config, run_fingerprint,
                    )
                summary.succeeded.append(dataset_name)
            except Exception as exc:
                summary.failed[dataset_name] = f"{type(exc).__name__}: {exc}"
                self.logger.error(f"Failed to evaluate on {dataset_name}: {exc}")
                self.logger.error(traceback.format_exc())
        self.logger.info('Ended eval')
        self.create_report(output_dir)
        return summary

    def _cache_hit(self, output_dir, task, expected_fingerprint, force_recalc):
        total_path = Path(output_dir) / f"{task.run_name().replace('/', '_')}_total.jsonl"
        if force_recalc and total_path.exists():
            stale_path = total_path.with_name(
                total_path.name + f".stale-{time.time_ns()}"
            )
            total_path.rename(stale_path)
            self.logger.info(
                "Archived prior total before forced recalculation: %s", stale_path
            )
            return False
        if validate_cache(total_path, expected_fingerprint, force_recalc):
            self.logger.info(f"Found compatible precomputed {task.run_name()}_total")
            return True
        return False
        
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
        run_config,
        run_fingerprint,
    ):
        model.add_stop_strings(task.additional_stop_strings)
        custom_stop_strings = None
        if generation_config is not None:
            custom_stop_strings = copy.deepcopy(
                getattr(generation_config, 'stop_strings', None)
            )
            merged_stops = list(custom_stop_strings or [])
            for stop_string in task.additional_stop_strings:
                if stop_string not in merged_stops:
                    merged_stops.append(stop_string)
            generation_config.stop_strings = merged_stops
        try:
            with PrettyJsonLogger(output_dir, task.run_name() + '_params') as logger:
                logger.log_json({
                    'run_config': run_config,
                    'run_fingerprint': run_fingerprint,
                    'model_params': model.get_params(),
                    'task_params': run_config['task'],
                })
            with CustomTimer(task.logger, 'Loading Dataset'):
                messages, samples = task.load_dataset(model=model, max_prompt_len=max_prompt_len, max_sample_per_dataset=max_sample_per_dataset, few_shot_count=few_shot_count)

            metrics = []
            with JsonArrayLogger(output_dir, task.run_name()) as logger, CustomTimer(task.logger, 'Processing Dataset') as timer:
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

            with PrettyJsonLogger(output_dir, task.run_name() + '_total') as logger:
                logger.log_json({'task_name': task.run_name(), 'results': metrics_res, 'leaderboard_result': task.leaderboard_aggregation(metrics_res), 'time': processing_time, 'run_config': run_config, 'run_fingerprint': run_fingerprint})

            # Сохранение детальных результатов агрегации, если они есть
            if aggregation_details:
                with PrettyJsonLogger(output_dir, task.run_name() + '_aggregation_details') as logger:
                    logger.log_json({'task_name': task.run_name(), 'aggregation_details': aggregation_details})

            task.logger.info(str(metrics_res))
        finally:
            if generation_config is not None:
                generation_config.stop_strings = custom_stop_strings
            model.reset_stop_strings()

    def evaluate_ppl(self, model, output_dir, datasets_names='all', few_shot_count=5, batch_size=1, max_sample_per_dataset=100000000, force_recalc=False, name_suffix=None):
        set_out_handler_to_main_logger(output_dir)
        summary = EvaluationSummary()
        if datasets_names == 'all':
            datasets_names = list(TASK_REGISTRY.keys())
        resolve_reasoning_execution(
            model.reasoning_config.model_kind, False, "calculate_logsoftmax"
        )
        self.logger.info(f'Starting eval on {datasets_names}')
        for dataset_name in datasets_names:
            try:
                task_class = TASK_REGISTRY[dataset_name]['class']
                task_init_params = dict(TASK_REGISTRY[dataset_name].get('params', {}))
                task_init_params['name_suffix'] = task_init_params.get('name_suffix', name_suffix)
                task = task_class(**task_init_params)
                if 'get_answer' not in dir(task):
                    self.logger.info(f"Skip task {task.run_name()} because method get_answer not implemented")
                    summary.skipped.append(dataset_name)
                    continue

                if not model.support_method('calculate_logsoftmax'):
                    raise NotImplementedError(
                        f"{type(model.backend).__name__} does not support "
                        f"calculate_logsoftmax; PPL is HF-only"
                    )
                budget = MaxLenContext(
                    task, model, None, reasoning_enabled=False,
                    scoring_method="calculate_logsoftmax",
                )
                with budget as max_prompt_len:
                    run_config = build_run_config(
                        model=model, task=task, enable_thinking=False,
                        generation_config=None, few_shot_count=few_shot_count,
                        batch_size=batch_size,
                        max_sample_per_dataset=max_sample_per_dataset,
                        max_prompt_len=max_prompt_len,
                        effective_reasoning_tokens=0,
                        scoring_method="calculate_logsoftmax",
                        requested_enable_thinking=False,
                    )
                    run_fingerprint = fingerprint_run_config(run_config)
                    if self._cache_hit(output_dir, task, run_fingerprint, force_recalc):
                        summary.skipped.append(dataset_name)
                        continue
                    self.evaluate_dataset_ppl(
                        task, model, output_dir, max_prompt_len, few_shot_count,
                        batch_size, max_sample_per_dataset, run_config,
                        run_fingerprint,
                    )
                summary.succeeded.append(dataset_name)
            except Exception as exc:
                summary.failed[dataset_name] = f"{type(exc).__name__}: {exc}"
                self.logger.error(f"Failed to evaluate PPL on {dataset_name}: {exc}")
                self.logger.error(traceback.format_exc())
        self.logger.info('Ended eval')
        self.create_report(output_dir)
        return summary

    def evaluate_dataset_ppl(self, task, model, output_dir, max_prompt_len,
                             few_shot_count, batch_size, max_sample_per_dataset,
                             run_config, run_fingerprint):
        assert 'get_answer' in dir(task)
        with PrettyJsonLogger(output_dir, task.run_name() + '_params') as logger:
            logger.log_json({
                'run_config': run_config,
                'run_fingerprint': run_fingerprint,
                'model_params': model.get_params(),
                'task_params': run_config['task'],
            })

        messages, samples = task.load_dataset(model=model, max_prompt_len=max_prompt_len, max_sample_per_dataset=max_sample_per_dataset, few_shot_count=few_shot_count)
        shifts = []
        for m, s in zip(*[messages, samples]):
            shift = model.count_tokens_for_messages(m['messages'])
            if shift is None:
                raise NotImplementedError(
                    "PPL preparation requires exact prompt token counts"
                )
            shifts.append(shift)
            if m['messages'][-1]['role'] == 'assistant':
                m['messages'][-1]['content'] += task.get_answer(s['sample'])
            else:
                m['messages'].append({'role': 'assistant', 'content': task.get_answer(s['sample'])})

        metrics = []
        with JsonArrayLogger(output_dir, task.run_name()) as logger, CustomTimer(task.logger, 'Processing Dataset'):
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

        with PrettyJsonLogger(output_dir, task.run_name() + '_total') as logger:
            logger.log_json({'task_name': task.run_name(), 'results': metrics_res, 'leaderboard_result': metrics_res['ppl'], 'run_config': run_config, 'run_fingerprint': run_fingerprint})

        task.logger.info(str(metrics_res))

    def create_report(self, output_dir):
        reports = {}
        for file_name in os.listdir(output_dir):
            if file_name.endswith('_total.jsonl'):
                with codecs.open(os.path.join(output_dir, file_name), 'r', 'utf-8') as file:
                    task_report = json.load(file)
                reports[task_report['task_name']] = task_report['leaderboard_result']
        if not reports:
            self.logger.warning("No successful task totals found; report not created")
            return
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
