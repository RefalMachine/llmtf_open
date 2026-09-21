from llmtf.tasks import TASK_REGISTRY
from llmtf.base import Task, Base
from llmtf.utils import (
    CustomTimer, MaxLenContext, normalize_message_roles,
    set_out_handler_to_main_logger,
)
from llmtf.sample_logger import JsonArrayLogger, PrettyJsonLogger
import os
import json
import codecs
import copy
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


SUPPORTED_TASK_METHODS = frozenset({
    'generate', 'calculate_tokens_proba', 'calculate_logsoftmax'
})


def _validate_evaluation_limits(few_shot_count, batch_size,
                                max_sample_per_dataset):
    if not isinstance(few_shot_count, int) or few_shot_count < 0:
        raise ValueError("few_shot_count must be a non-negative integer")
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if not isinstance(max_sample_per_dataset, int) \
            or max_sample_per_dataset <= 0:
        raise ValueError(
            "max_sample_per_dataset must be a positive integer"
        )


def _validate_task_contract(task):
    run_name = task.run_name()
    if not isinstance(run_name, str) or not run_name.strip():
        raise ValueError("task.run_name() must return a non-empty string")
    if getattr(task, 'method', None) not in SUPPORTED_TASK_METHODS:
        raise ValueError(
            f"{run_name} has unsupported task method "
            f"{getattr(task, 'method', None)!r}"
        )
    budget = task.max_task_new_tokens
    if not isinstance(budget, int) or budget <= 0:
        raise ValueError(
            f"{run_name} max_task_new_tokens must be a positive integer"
        )


def _validate_loaded_dataset(task, messages, samples):
    if not isinstance(messages, list) or not isinstance(samples, list):
        raise TypeError(
            f"{task.run_name()} load_dataset must return two lists"
        )
    if len(messages) != len(samples):
        raise ValueError(
            f"{task.run_name()} returned misaligned messages/samples: "
            f"{len(messages)} != {len(samples)}"
        )
    if not messages:
        raise ValueError(f"{task.run_name()} produced no evaluation samples")
    expected_keys = set(messages[0])
    if 'messages' not in expected_keys:
        raise ValueError(
            f"{task.run_name()} message payload lacks 'messages'"
        )
    for index, payload in enumerate(messages):
        if not isinstance(payload, dict):
            raise TypeError(
                f"{task.run_name()} message payload {index} is not a dict"
            )
        if set(payload) != expected_keys:
            raise ValueError(
                f"{task.run_name()} message payload {index} has inconsistent "
                f"keys {sorted(payload)}; expected {sorted(expected_keys)}"
            )
        if not isinstance(payload.get('messages'), list) \
                or not payload['messages']:
            raise ValueError(
                f"{task.run_name()} sample {index} has no chat messages"
            )
        payload['messages'] = normalize_message_roles(payload['messages'])
        for message_index, message in enumerate(payload['messages']):
            if 'content' not in message:
                raise ValueError(
                    f"{task.run_name()} sample {index} message "
                    f"{message_index} lacks 'content'"
                )
    for index, payload in enumerate(samples):
        if not isinstance(payload, dict) or 'sample' not in payload:
            raise ValueError(
                f"{task.run_name()} sample payload {index} lacks 'sample'"
            )


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
    numpy_random = getattr(np, 'random', None)
    if numpy_random is not None:
        numpy_random.seed(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:2"
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
    except ImportError:
        return
    manual_seed = getattr(torch, 'manual_seed', None)
    if manual_seed is not None:
        manual_seed(seed)
    torch_cuda = getattr(torch, 'cuda', None)
    if torch_cuda is not None \
            and hasattr(torch_cuda, 'manual_seed_all'):
        torch_cuda.manual_seed_all(seed)
    cudnn = getattr(getattr(torch, 'backends', None), 'cudnn', None)
    if cudnn is not None:
        cudnn.benchmark = False
        cudnn.deterministic = True

class Evaluator(Base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        set_random_seed(555)

    def add_new_task(self, task_name, task_cls, task_params, *, allow_override=False):
        if not isinstance(task_name, str) or not task_name.strip():
            raise ValueError("task_name must be a non-empty string")
        if not isinstance(task_cls, type) or not issubclass(task_cls, Task):
            raise TypeError("task_cls must be a Task subclass")
        if not isinstance(task_params, dict):
            raise TypeError("task_params must be a dict")
        if task_name in TASK_REGISTRY and not allow_override:
            raise ValueError(
                f"Task {task_name!r} is already registered; pass "
                "allow_override=True to replace it explicitly"
            )
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
        _validate_evaluation_limits(
            few_shot_count, batch_size, max_sample_per_dataset
        )
        set_out_handler_to_main_logger(output_dir)
        summary = EvaluationSummary()
        if generation_config is not None:
            model.logger.warning('Custom generation_config receives full priority over internal generation config. Compose it carefully.')
        if datasets_names == 'all':
            datasets_names = list(TASK_REGISTRY.keys())
        elif isinstance(datasets_names, str):
            datasets_names = [datasets_names]
        self.logger.info(f'Starting eval on {datasets_names}')
        for dataset_name in datasets_names:
            try:
                # Dataset sampling must not depend on which tasks ran before it.
                set_random_seed(555)
                task_class = TASK_REGISTRY[dataset_name]['class']
                task_init_params = dict(TASK_REGISTRY[dataset_name].get('params', {}))
                task_init_params['name_suffix'] = task_init_params.get('name_suffix', name_suffix)
                task = task_class(**task_init_params)
                _validate_task_contract(task)
                task.require_model_method(model)
                mode = resolve_reasoning_execution(
                    model.reasoning_config.model_kind,
                    enable_thinking,
                    task.method,
                )
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
                        registry_name=dataset_name,
                        task_init_params=task_init_params,
                    )
                    run_fingerprint = fingerprint_run_config(run_config)
                    if self._cache_hit(
                        output_dir, task, run_fingerprint, force_recalc,
                        expected_run_config=run_config,
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

    def _cache_hit(self, output_dir, task, expected_fingerprint, force_recalc,
                   expected_run_config=None):
        total_path = Path(output_dir) / f"{task.run_name().replace('/', '_')}_total.jsonl"
        if force_recalc and total_path.exists():
            total_path.unlink()
            self.logger.info(
                "Removed prior total before forced recalculation: %s",
                total_path,
            )
            return False
        if validate_cache(
            total_path, expected_fingerprint, force_recalc,
            expected_run_config=expected_run_config,
        ):
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
            _validate_loaded_dataset(task, messages, samples)

            metrics = []
            with JsonArrayLogger(output_dir, task.run_name()) as logger, CustomTimer(task.logger, 'Processing Dataset') as timer:
                for i in tqdm(range(0, len(messages), batch_size)):
                    messages_batch = messages[i:i+batch_size]
                    messages_batch = {k: [subdict[k] for subdict in messages_batch] for k in messages_batch[0]}
                    if generation_config is not None \
                            and task.method != 'calculate_logsoftmax':
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
                    elif task.method == 'calculate_logsoftmax':
                        prompts, y_preds, infos = model.calculate_logsoftmax_batch(
                            **messages_batch
                        )
                    else:
                        raise ValueError(
                            f"Unsupported task method {task.method!r}"
                        )

                    for j in range(len(y_preds)):
                        metric_result = task.evaluate(
                            samples[i+j]['sample'], y_preds[j]
                        )
                        if not isinstance(metric_result, dict) or not metric_result:
                            raise TypeError(
                                f"{task.run_name()} evaluate() must return a "
                                "non-empty metric dict"
                            )
                        if metrics and set(metric_result) != set(metrics[0]):
                            raise ValueError(
                                f"{task.run_name()} returned inconsistent metric "
                                f"keys {sorted(metric_result)}; expected "
                                f"{sorted(metrics[0])}"
                            )
                        metrics.append(metric_result)
                        logger.log_sample(samples[i+j]['sample'], y_preds[j], prompts[j], metrics[-1], infos[j])
                processing_time = timer.time()

            task.logger.info(f'Results for {task.run_name()}:')

            # Агрегация с возможностью сохранения деталей
            metrics_res = {}
            aggregation_details = {}
            aggregation = task.aggregation()
            if not isinstance(aggregation, dict):
                raise TypeError(
                    f"{task.run_name()} aggregation() must return a dict"
                )
            if set(aggregation) != set(metrics[0]):
                raise ValueError(
                    f"{task.run_name()} aggregation keys {sorted(aggregation)} "
                    f"do not match metric keys {sorted(metrics[0])}"
                )
            for metric in metrics[0].keys():
                agg_result = aggregation[metric]([m[metric] for m in metrics])

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
        _validate_evaluation_limits(
            few_shot_count, batch_size, max_sample_per_dataset
        )
        set_out_handler_to_main_logger(output_dir)
        summary = EvaluationSummary()
        if datasets_names == 'all':
            datasets_names = list(TASK_REGISTRY.keys())
        elif isinstance(datasets_names, str):
            datasets_names = [datasets_names]
        resolve_reasoning_execution(
            model.reasoning_config.model_kind, False, "calculate_logsoftmax"
        )
        self.logger.info(f'Starting eval on {datasets_names}')
        for dataset_name in datasets_names:
            try:
                set_random_seed(555)
                task_class = TASK_REGISTRY[dataset_name]['class']
                task_init_params = dict(TASK_REGISTRY[dataset_name].get('params', {}))
                task_init_params['name_suffix'] = task_init_params.get('name_suffix', name_suffix)
                task = task_class(**task_init_params)
                _validate_task_contract(task)
                if not callable(getattr(task, 'get_answer', None)):
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
                        registry_name=dataset_name,
                        task_init_params=task_init_params,
                    )
                    run_fingerprint = fingerprint_run_config(run_config)
                    if self._cache_hit(
                        output_dir, task, run_fingerprint, force_recalc,
                        expected_run_config=run_config,
                    ):
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
        _validate_loaded_dataset(task, messages, samples)
        prompt_prefixes = []
        for m, s in zip(messages, samples):
            prompt_prefixes.append(model.apply_model_prompt(m['messages']))
            answer = task.get_answer(s['sample'])
            if not isinstance(answer, str):
                raise TypeError(
                    f"{task.run_name()} get_answer() must return a string"
                )
            if m['messages'][-1]['role'] == 'assistant':
                m['messages'][-1]['content'] += answer
            else:
                m['messages'].append({'role': 'assistant', 'content': answer})

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
                    prompt_prefix = prompt_prefixes[i+j]
                    if not prompts[j].startswith(prompt_prefix):
                        raise ValueError(
                            "PPL answer boundary is not stable: the rendered "
                            "prompt before the reference answer is not an exact "
                            "prefix of the scored prompt"
                        )
                    answer_start = len(prompt_prefix)
                    tokens = [
                        token for token in y_preds[j][-1]['tokens']
                        if token[2][1] > answer_start
                    ]
                    if not tokens:
                        raise ValueError(
                            f"{task.run_name()} reference answer produced no "
                            "scored tokens"
                        )
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
        """Rebuild the summary from every completed task in output_dir."""
        reports = {}
        for file_name in os.listdir(output_dir):
            if file_name.endswith('_total.jsonl'):
                with codecs.open(os.path.join(output_dir, file_name), 'r', 'utf-8') as file:
                    task_report = json.load(file)
                reports[task_report['task_name']] = task_report['leaderboard_result']
        if not reports:
            self.logger.warning("No successful task totals found; report not created")
            report_path = Path(output_dir) / 'evaluation_results.txt'
            if report_path.exists():
                report_path.unlink()
            return
        task_names = sorted(list(reports.keys()))
        task_metrics = [reports[t] for t in task_names]
        task_names = ['mean'] + task_names
        task_metrics = [sum(task_metrics) / len(task_metrics)] + task_metrics

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
