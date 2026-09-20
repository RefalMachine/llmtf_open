"""Shared CLI normalization for backend kwargs and thinking flags."""

import argparse
import inspect
import json
from typing import Dict, Type


def parse_json_object(raw, option_name="--backend_kwargs") -> Dict:
    if raw is None:
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{option_name} must be valid JSON: {exc.msg}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{option_name} must contain a JSON object")
    return value


def constructor_keys(backend_cls: Type) -> set:
    signature = inspect.signature(backend_cls.__init__)
    return {
        name for name, parameter in signature.parameters.items()
        if name != "self" and parameter.kind not in {
            inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD
        }
    }


def merge_backend_kwargs(backend_cls: Type, raw_json, explicit: Dict) -> Dict:
    merged = parse_json_object(raw_json)
    allowed = constructor_keys(backend_cls)
    unknown = sorted(set(merged) - allowed)
    if unknown:
        raise ValueError(
            f"Unknown {backend_cls.__name__} constructor option(s): "
            + ", ".join(unknown)
        )
    explicit = {key: value for key, value in explicit.items() if value is not None}
    unknown_explicit = sorted(set(explicit) - allowed)
    if unknown_explicit:
        raise ValueError(
            f"Invalid explicit {backend_cls.__name__} option(s): "
            + ", ".join(unknown_explicit)
        )
    merged.update(explicit)
    return merged


def add_thinking_flags(parser: argparse.ArgumentParser):
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--enable_thinking", action="store_true", dest="enable_thinking")
    group.add_argument(
        "--disable_thinking", action="store_false", dest="enable_thinking",
        help="Deprecated compatibility alias; thinking is disabled by default.",
    )
    parser.set_defaults(enable_thinking=False)


def add_continuation_flags(parser: argparse.ArgumentParser, *, api=False):
    parser.add_argument(
        "--assistant_prefill_policy",
        choices=["auto", "exact", "portable", "best_effort"],
        default="auto",
        help=(
            "Handling of a continued assistant message. auto requires declared "
            "API continuation support and checks hazardous whitespace; exact "
            "requires strong API evidence for hazardous whitespace; portable "
            "rejects all assistant prefills; best_effort explicitly accepts "
            "unverified server handling."
        ),
    )
    if api:
        parser.add_argument(
            "--probe_api_prefill",
            action="store_true",
            help=(
                "Use synthetic /tokenize and optional /detokenize requests to "
                "check trailing-whitespace handling when a task needs it."
            ),
        )


def validate_execution_args(model_kind: str, enable_thinking: bool, *, ppl=False):
    if model_kind == "reasoning" and not enable_thinking:
        raise ValueError(
            "model_kind='reasoning' requires --enable_thinking; use "
            "model_kind='hybrid' for a switchable model"
        )
    if ppl and model_kind == "reasoning":
        raise ValueError(
            "PPL is a one-pass HF-only method and cannot satisfy the strict "
            "model_kind='reasoning' contract"
        )
