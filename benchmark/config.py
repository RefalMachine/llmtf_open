"""Strict shared benchmark YAML normalization and command construction."""

from dataclasses import dataclass, field
import json
import warnings

import yaml


MODEL_KEYS = {
    'model_kind', 'enable_thinking', 'model_context_len',
    'max_new_tokens_reasoning', 'min_new_tokens_reasoning',
    'end_thinking_token_id', 'backend_kwargs', 'is_foundational',
    'assistant_prefill_policy', 'probe_api_prefill', 'api_profile',
}
EVALUATION_KEYS = {
    'few_shot_count', 'batch_size', 'max_sample_per_dataset', 'name_suffix',
}
GENERATION_KEYS = {
    'temperature', 'repetition_penalty', 'presence_penalty',
    'num_return_sequences',
}
TOP_KEYS = {'model', 'defaults', 'tasks'}
DEFAULT_KEYS = EVALUATION_KEYS | {'evaluation', 'generation'}
TASK_KEYS = EVALUATION_KEYS | {
    'name', 'datasets', 'enable_thinking', 'evaluation', 'generation',
    'extra_args',
}


def _reject_unknown(mapping, allowed, where):
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise ValueError(f"Unknown key(s) in {where}: {', '.join(unknown)}")


def _require_mapping(value, where):
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{where} must be a mapping")
    return value


@dataclass(frozen=True)
class BenchmarkModelConfig:
    model_kind: str = 'plain'
    enable_thinking: bool = False
    model_context_len: int = None
    max_new_tokens_reasoning: int = 4096
    min_new_tokens_reasoning: int = 1024
    end_thinking_token_id: int = None
    backend_kwargs: dict = field(default_factory=dict)
    is_foundational: bool = False
    assistant_prefill_policy: str = 'auto'
    probe_api_prefill: bool = False
    api_profile: str = 'auto'


@dataclass(frozen=True)
class BenchmarkTaskConfig:
    name: str
    datasets: tuple
    enable_thinking: bool
    evaluation: dict
    generation: dict


@dataclass(frozen=True)
class BenchmarkConfig:
    model: BenchmarkModelConfig
    tasks: tuple


def load_benchmark_config(path):
    with open(path, 'r', encoding='utf-8') as source:
        raw = yaml.safe_load(source) or {}
    raw = _require_mapping(raw, 'benchmark config')
    _reject_unknown(raw, TOP_KEYS, 'benchmark config')

    model_raw = _require_mapping(raw.get('model'), 'model')
    _reject_unknown(model_raw, MODEL_KEYS, 'model')
    model = BenchmarkModelConfig(**model_raw)
    if model.model_kind not in {'plain', 'hybrid', 'reasoning'}:
        raise ValueError(f"Invalid model.model_kind: {model.model_kind!r}")
    if model.assistant_prefill_policy not in {
        'auto', 'exact', 'portable', 'best_effort'
    }:
        raise ValueError(
            "model.assistant_prefill_policy must be one of auto, exact, "
            "portable, best_effort"
        )
    if not isinstance(model.probe_api_prefill, bool):
        raise ValueError('model.probe_api_prefill must be boolean')
    if model.api_profile not in {'auto', 'openai', 'vllm'}:
        raise ValueError('model.api_profile must be one of auto, openai, vllm')
    if model.model_kind == 'reasoning' and not model.enable_thinking:
        raise ValueError("model_kind='reasoning' requires model.enable_thinking=true")
    if model.enable_thinking and model.model_kind != 'plain' \
            and model.end_thinking_token_id is None:
        raise ValueError(
            "thinking-enabled benchmark requires model.end_thinking_token_id"
        )
    if not isinstance(model.backend_kwargs, dict):
        raise ValueError('model.backend_kwargs must be a mapping')
    secret_keys = {
        key for key in model.backend_kwargs
        if any(fragment in key.lower() for fragment in (
            'api_key', 'authorization', 'password', 'secret', 'credential'
        ))
    }
    if secret_keys:
        raise ValueError(
            'Secrets are forbidden in model.backend_kwargs; pass them through '
            'runtime environment variables: ' + ', '.join(sorted(secret_keys))
        )

    defaults = _require_mapping(raw.get('defaults'), 'defaults')
    _reject_unknown(defaults, DEFAULT_KEYS, 'defaults')
    default_eval = _require_mapping(defaults.get('evaluation'), 'defaults.evaluation')
    _reject_unknown(default_eval, EVALUATION_KEYS, 'defaults.evaluation')
    for key in EVALUATION_KEYS:
        if key in defaults:
            default_eval[key] = defaults[key]
    default_gen = _require_mapping(defaults.get('generation'), 'defaults.generation')
    _reject_unknown(default_gen, GENERATION_KEYS, 'defaults.generation')

    tasks = []
    raw_tasks = raw.get('tasks', [])
    if not isinstance(raw_tasks, list):
        raise ValueError('tasks must be a list')
    for index, task_raw in enumerate(raw_tasks):
        task_raw = _require_mapping(task_raw, f'tasks[{index}]')
        _reject_unknown(task_raw, TASK_KEYS, f'tasks[{index}]')
        if not task_raw.get('name'):
            raise ValueError(f'tasks[{index}].name is required')
        datasets = task_raw.get('datasets')
        if not isinstance(datasets, list) or not datasets or not all(
            isinstance(item, str) and item for item in datasets
        ):
            raise ValueError(f'tasks[{index}].datasets must be a non-empty string list')

        evaluation = dict(default_eval)
        task_eval = _require_mapping(task_raw.get('evaluation'), f'tasks[{index}].evaluation')
        _reject_unknown(task_eval, EVALUATION_KEYS, f'tasks[{index}].evaluation')
        evaluation.update(task_eval)
        for key in EVALUATION_KEYS:
            if key in task_raw:
                evaluation[key] = task_raw[key]

        generation = dict(default_gen)
        task_gen = _require_mapping(task_raw.get('generation'), f'tasks[{index}].generation')
        _reject_unknown(task_gen, GENERATION_KEYS, f'tasks[{index}].generation')
        generation.update(task_gen)

        enable_thinking = task_raw.get('enable_thinking', model.enable_thinking)
        extra_args = _require_mapping(task_raw.get('extra_args'), f'tasks[{index}].extra_args')
        _reject_unknown(extra_args, {'think'}, f'tasks[{index}].extra_args')
        if 'think' in extra_args:
            warnings.warn(
                "extra_args.think is deprecated; use task.enable_thinking",
                DeprecationWarning,
                stacklevel=2,
            )
            if 'enable_thinking' in task_raw and task_raw['enable_thinking'] != extra_args['think']:
                raise ValueError(
                    f"tasks[{index}] has conflicting enable_thinking and extra_args.think"
                )
            enable_thinking = extra_args['think']
        if not isinstance(enable_thinking, bool):
            raise ValueError(f'tasks[{index}].enable_thinking must be boolean')
        if model.model_kind == 'reasoning' and not enable_thinking:
            raise ValueError(
                f"tasks[{index}] disables thinking for strict reasoning model"
            )
        if enable_thinking and model.model_kind != 'plain' \
                and model.end_thinking_token_id is None:
            raise ValueError(
                f"tasks[{index}] enables thinking without model.end_thinking_token_id"
            )
        if enable_thinking and generation.get('num_return_sequences', 1) != 1:
            raise ValueError(
                f"tasks[{index}] two-pass reasoning requires num_return_sequences=1"
            )
        tasks.append(BenchmarkTaskConfig(
            name=task_raw['name'], datasets=tuple(datasets),
            enable_thinking=enable_thinking, evaluation=evaluation,
            generation=generation,
        ))
    return BenchmarkConfig(model=model, tasks=tuple(tasks))


def build_evaluate_command(config, task, *, model_name, output_dir,
                           api=False, base_url=None, conv_path='auto',
                           backend='vllm', tensor_parallel_size=1,
                           force_recalc=False, ppl=False,
                           is_foundational=False, api_profile=None,
                           calculate_tokens_proba_logprobs_count=None):
    entrypoint = 'evaluate_model_api.py' if api else 'evaluate_model.py'
    command = ['python', entrypoint]
    if api:
        if not base_url:
            raise ValueError('base_url is required for API command')
        command += ['--base_url', base_url.rstrip('/').removesuffix('/v1')]
        command += ['--api_profile', api_profile or config.model.api_profile]
    command += [
        '--model_name_or_path', model_name,
        '--output_dir', output_dir,
        '--dataset_names', *task.datasets,
        '--model_kind', config.model.model_kind,
        '--max_new_tokens_reasoning', str(config.model.max_new_tokens_reasoning),
        '--min_new_tokens_reasoning', str(config.model.min_new_tokens_reasoning),
        '--enable_thinking' if task.enable_thinking else '--disable_thinking',
        '--assistant_prefill_policy', config.model.assistant_prefill_policy,
    ]
    if api and config.model.probe_api_prefill:
        command.append('--probe_api_prefill')
    if api and calculate_tokens_proba_logprobs_count is not None:
        command += [
            '--calculate_tokens_proba_logprobs_count',
            str(calculate_tokens_proba_logprobs_count),
        ]
    if api and (is_foundational or config.model.is_foundational):
        command += ['--conv_path', conv_path]
    if not api:
        command += ['--conv_path', conv_path]
        if backend == 'vllm':
            command += ['--vllm', '--tensor_parallel_size', str(tensor_parallel_size)]
    if config.model.model_context_len is not None:
        command += ['--model_context_len', str(config.model.model_context_len)]
    if config.model.end_thinking_token_id is not None:
        command += ['--end_thinking_token_id', str(config.model.end_thinking_token_id)]
    if config.model.backend_kwargs:
        command += [
            '--backend_kwargs',
            json.dumps(config.model.backend_kwargs, separators=(',', ':')),
        ]
    for key, value in task.evaluation.items():
        if value is not None:
            command += [f'--{key}', str(value)]
    for key, value in task.generation.items():
        command += [f'--{key}', str(value)]
    if force_recalc:
        command.append('--force_recalc')
    if ppl:
        command.append('--ppl_scoring')
    if is_foundational or config.model.is_foundational:
        command.append('--is_foundational')
    return command
