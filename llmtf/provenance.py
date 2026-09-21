"""Canonical run configuration, redaction and cache fingerprint helpers."""

import hashlib
import inspect
import json
import re
from pathlib import Path
from typing import Any, Dict

from llmtf.config import config_to_dict


RUN_CONFIG_SCHEMA_VERSION = 2
_VOLATILE_SERVER_CAPABILITIES = frozenset({
    "detokenize",
    "logprobs",
})
_SECRET_FRAGMENTS = (
    "api_key", "apikey", "authorization", "password", "secret",
    "credential",
)


def _is_secret_key(key: str) -> bool:
    normalized = re.sub(r"(?<!^)(?=[A-Z])", "_", key)
    normalized = normalized.lower().replace("-", "_").replace(" ", "_")
    # A credential token is conventionally named ``token`` or ``*_token``.
    # Token counts, token ids, tokenizer settings and API capabilities are
    # ordinary run provenance and must remain visible.
    if normalized == "token" or normalized.endswith("_token"):
        return True
    return any(fragment in normalized for fragment in _SECRET_FRAGMENTS)


def sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): ("<redacted>" if _is_secret_key(str(key)) else sanitize(item))
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [sanitize(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "value") and isinstance(getattr(value, "value"), (str, int, float, bool)):
        return value.value
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def canonical_json(config: Dict[str, Any]) -> str:
    return json.dumps(sanitize(config), ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def _fingerprint_payload(config: Dict[str, Any]) -> Dict[str, Any]:
    """Return the semantic, order-independent part of a run configuration.

    API capability discovery is intentionally lazy. In particular, logprobs
    becomes known only after a probability request and detokenize only after a
    continuation probe. Their observed values therefore depend on which
    earlier tasks were evaluated instead of loaded from cache. Keep these
    observations in result provenance, but do not let them invalidate otherwise
    identical cached results.
    """
    payload = sanitize(config)
    model = payload.get("model") if isinstance(payload, dict) else None
    capabilities = (
        model.get("server_capabilities")
        if isinstance(model, dict) else None
    )
    if isinstance(capabilities, dict):
        for capability in _VOLATILE_SERVER_CAPABILITIES:
            capabilities.pop(capability, None)
    return payload


def fingerprint_run_config(config: Dict[str, Any]) -> str:
    return hashlib.sha256(
        canonical_json(_fingerprint_payload(config)).encode("utf-8")
    ).hexdigest()


def _stable_task_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _stable_task_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_stable_task_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_stable_task_value(item) for item in value), key=str)
    if isinstance(value, Path) or value is None \
            or isinstance(value, (str, int, float, bool)):
        return value
    get_params = getattr(value, 'get_params', None)
    if callable(get_params):
        return {
            'class': f"{type(value).__module__}.{type(value).__qualname__}",
            'params': _stable_task_value(get_params()),
        }
    return {'class': f"{type(value).__module__}.{type(value).__qualname__}"}


def _task_implementation_identity(task) -> Dict[str, Any]:
    task_cls = type(task)
    identity = {
        'class': f"{task_cls.__module__}.{task_cls.__qualname__}",
    }
    source_path = inspect.getsourcefile(task_cls)
    if source_path:
        try:
            source = Path(source_path).read_bytes()
        except OSError:
            source = None
        if source is not None:
            identity['module_sha256'] = hashlib.sha256(source).hexdigest()
    return identity


def _task_dataset_args(task):
    dataset_args = getattr(task, 'dataset_args', None)
    if not callable(dataset_args):
        return None
    try:
        value = dataset_args()
        if not isinstance(value, dict):
            value = list(value)
    except TypeError:
        return None
    return _stable_task_value(value)


def build_run_config(*, model, task, enable_thinking: bool,
                     generation_config, few_shot_count: int, batch_size: int,
                     max_sample_per_dataset: int, max_prompt_len: int,
                     effective_reasoning_tokens: int, scoring_method: str,
                     requested_enable_thinking=None, registry_name=None,
                     task_init_params=None) -> Dict[str, Any]:
    rc = model.reasoning_config
    configured_reasoning = getattr(rc, "configured_max_new_tokens_reasoning",
                                   rc.max_new_tokens_reasoning)
    effective_generation = config_to_dict(
        generation_config or model.generation_config
    )
    effective_stops = list(effective_generation.get('stop_strings') or [])
    for stop_string in task.additional_stop_strings:
        if stop_string not in effective_stops:
            effective_stops.append(stop_string)
    effective_generation['stop_strings'] = effective_stops
    result = {
        "schema_version": RUN_CONFIG_SCHEMA_VERSION,
        "model": model.get_params(),
        "execution": {
            "model_kind": rc.model_kind.value,
            "enable_thinking": bool(enable_thinking),
            "requested_enable_thinking": bool(
                enable_thinking if requested_enable_thinking is None
                else requested_enable_thinking
            ),
            "scoring_method": scoring_method,
            "reasoning": {
                "configured_max_new_tokens_reasoning": configured_reasoning,
                "effective_max_new_tokens_reasoning": effective_reasoning_tokens,
                "min_new_tokens_reasoning": rc.min_new_tokens_reasoning,
                "end_thinking_token_id": rc.fmt.end_thinking_token_id,
                "think_close": rc.fmt.think_close,
                "truncation_prompt": rc.fmt.truncation_prompt,
                "answer_separator": rc.fmt.answer_separator,
            },
        },
        "task": {
            "registry_name": registry_name,
            "name": task.run_name(),
            "implementation": _task_implementation_identity(task),
            "init_params": _stable_task_value(task_init_params or {}),
            "dataset_args": _task_dataset_args(task),
            "method": scoring_method,
            "few_shot_count": few_shot_count,
            "batch_size": batch_size,
            "max_sample_per_dataset": max_sample_per_dataset,
            "max_prompt_len": max_prompt_len,
            "max_task_new_tokens": task.max_task_new_tokens,
            "additional_stop_strings": list(task.additional_stop_strings),
            "method_additional_args": sanitize(task.method_additional_args),
        },
        "generation_config": effective_generation,
    }
    return sanitize(result)


class CacheMismatchError(RuntimeError):
    pass


def read_cached_fingerprint(total_path: Path):
    with total_path.open("r", encoding="utf-8") as source:
        payload = json.load(source)
    return payload.get("run_fingerprint")


def validate_cache(total_path: Path, expected_fingerprint: str,
                   force_recalc=False, expected_run_config=None) -> bool:
    if not total_path.exists() or force_recalc:
        return False
    with total_path.open("r", encoding="utf-8") as source:
        payload = json.load(source)
    actual = payload.get("run_fingerprint")
    if actual == expected_fingerprint:
        return True
    # Compatibility with schema-v2 artifacts written before volatile API
    # capability observations were excluded from the fingerprint. Compare the
    # embedded configuration using the current semantic fingerprint instead of
    # requiring users to recalculate valid benchmark results.
    cached_run_config = payload.get("run_config")
    if expected_run_config is not None and isinstance(cached_run_config, dict):
        if fingerprint_run_config(cached_run_config) == expected_fingerprint:
            return True
    raise CacheMismatchError(
        f"Cached result at {total_path} has fingerprint {actual!r}, expected "
        f"{expected_fingerprint!r}. Use --force_recalc, a different "
        f"--output_dir, or --name_suffix."
    )
