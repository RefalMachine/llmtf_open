"""Canonical run configuration, redaction and cache fingerprint helpers."""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict

from llmtf.config import config_to_dict


RUN_CONFIG_SCHEMA_VERSION = 1
_SECRET_FRAGMENTS = (
    "api_key", "apikey", "authorization", "password", "secret", "token",
    "credential",
)


def _is_secret_key(key: str) -> bool:
    normalized = key.lower().replace("-", "_")
    # Token budgets/ids and model names are provenance, not credentials.
    if normalized in {
        "max_new_tokens", "max_new_tokens_reasoning", "min_new_tokens_reasoning",
        "end_thinking_token_id", "eos_token_id", "pad_token_id", "bos_token_id",
        "stop_token_ids", "tokenizer", "use_fast_tokenizer",
        "calculate_tokens_proba_logprobs_count",
        "token_probability_surface_forms", "token_probability_aggregation",
    }:
        return False
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


def fingerprint_run_config(config: Dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(config).encode("utf-8")).hexdigest()


def build_run_config(*, model, task, enable_thinking: bool,
                     generation_config, few_shot_count: int, batch_size: int,
                     max_sample_per_dataset: int, max_prompt_len: int,
                     effective_reasoning_tokens: int, scoring_method: str,
                     requested_enable_thinking=None) -> Dict[str, Any]:
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
            "name": task.run_name(),
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
                   force_recalc=False) -> bool:
    if not total_path.exists() or force_recalc:
        return False
    actual = read_cached_fingerprint(total_path)
    if actual == expected_fingerprint:
        return True
    raise CacheMismatchError(
        f"Cached result at {total_path} has fingerprint {actual!r}, expected "
        f"{expected_fingerprint!r}. Use --force_recalc, a different "
        f"--output_dir, or --name_suffix."
    )
