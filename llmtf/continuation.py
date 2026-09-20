"""Assistant continuation and next-token surface-form policy.

This module is the only place that owns trailing-whitespace handling for an
assistant prefill.  Backends provide transport/tokenizer operations; they do
not independently trim messages or invent space-prefixed label variants.
"""

import copy
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Optional


_CONTINUATION_BOUNDARY = "__LLMTF_CONTINUATION_BOUNDARY_7f5d7e30__"


class PrefillPolicy(str, Enum):
    AUTO = "auto"
    EXACT = "exact"
    PORTABLE = "portable"
    BEST_EFFORT = "best_effort"


class PrefillVerification(str, Enum):
    NOT_APPLICABLE = "not_applicable"
    NOT_REQUIRED = "not_required"
    EXACT_LOCAL = "exact_local"
    VERIFIED_EXACT = "verified_exact"
    VERIFIED_TOKEN_EFFECT = "verified_token_effect"
    VERIFIED_STRIPPED = "verified_stripped"
    INCONCLUSIVE = "inconclusive"
    UNVERIFIED = "unverified"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True)
class ContinuationConfig:
    prefill_policy: PrefillPolicy = PrefillPolicy.AUTO
    probe_api_prefill: bool = False

    def __post_init__(self):
        object.__setattr__(self, "prefill_policy", PrefillPolicy(self.prefill_policy))

    def to_dict(self):
        return {
            "prefill_policy": self.prefill_policy.value,
            "probe_api_prefill": self.probe_api_prefill,
            "token_probability_surface_forms": "candidate_and_single_leading_space",
            "token_probability_aggregation": "max",
        }


@dataclass(frozen=True)
class PrefillAnalysis:
    present: bool
    trailing_whitespace: str = ""

    @property
    def hazardous(self):
        return bool(self.trailing_whitespace)


@dataclass(frozen=True)
class PrefillProbeResult:
    verification: PrefillVerification
    evidence: str

    def __post_init__(self):
        object.__setattr__(self, "verification", PrefillVerification(self.verification))


@dataclass(frozen=True)
class PrefillDecision:
    present: bool
    trailing_whitespace: str
    policy: PrefillPolicy
    handling: str
    verification: PrefillVerification
    evidence: Optional[str] = None

    def to_dict(self):
        result = asdict(self)
        result["policy"] = self.policy.value
        result["verification"] = self.verification.value
        result["trailing_whitespace"] = escape_whitespace(
            self.trailing_whitespace
        )
        if self.evidence is None:
            result.pop("evidence")
        return result


class PrefillCompatibilityError(ValueError):
    pass


def escape_whitespace(value):
    return value.encode("unicode_escape").decode("ascii")


def analyze_assistant_prefill(messages, *, continue_last_assistant_message=True):
    if not messages:
        raise ValueError("messages must not be empty")
    for message in messages:
        if message.get("role") not in {"user", "assistant", "system"}:
            raise ValueError(f"Unknown role {message.get('role')}")
    if not continue_last_assistant_message or messages[-1]["role"] != "assistant":
        return PrefillAnalysis(False)
    content = messages[-1].get("content")
    if not isinstance(content, str):
        raise TypeError("the final assistant message content must be a string")
    return PrefillAnalysis(
        present=True,
        trailing_whitespace=content[len(content.rstrip()):],
    )


def resolve_prefill(
    analysis,
    config,
    *,
    local_exact=False,
    continuation_supported=True,
    probe_result=None,
):
    """Resolve one prefill without mutating its content.

    ``continuation_supported`` describes the more fundamental capability:
    whether the backend can continue a final assistant message at all.
    Trailing-whitespace fidelity is checked only after that capability is
    established.
    """
    policy = config.prefill_policy
    if not analysis.present:
        return PrefillDecision(
            False, "", policy, "none", PrefillVerification.NOT_APPLICABLE
        )
    if policy == PrefillPolicy.PORTABLE:
        raise PrefillCompatibilityError(
            "portable mode does not allow an assistant prefill. Rewrite the "
            "task so the request ends with a user message and asks the model "
            "to emit only the answer continuation."
        )
    if not continuation_supported:
        if policy == PrefillPolicy.BEST_EFFORT:
            return PrefillDecision(
                True, analysis.trailing_whitespace, policy,
                "server_managed_unverified", PrefillVerification.UNVERIFIED,
                "the API has no declared assistant-continuation capability",
            )
        raise PrefillCompatibilityError(
            "the selected API profile does not declare support for continuing "
            "a final assistant message. Rewrite the task without an assistant "
            "prefill, select an API profile that supports continuation, or "
            "explicitly choose best_effort."
        )
    if local_exact:
        return PrefillDecision(
            True, analysis.trailing_whitespace, policy, "preserved_local",
            PrefillVerification.EXACT_LOCAL,
        )
    if not analysis.hazardous:
        return PrefillDecision(
            True, "", policy, "direct", PrefillVerification.NOT_REQUIRED
        )
    if policy == PrefillPolicy.BEST_EFFORT:
        return PrefillDecision(
            True, analysis.trailing_whitespace, policy, "server_managed",
            PrefillVerification.UNVERIFIED,
            "API behavior was explicitly accepted without verification",
        )
    if probe_result is None:
        raise PrefillCompatibilityError(
            "assistant-prefill ends in whitespace, but exact API handling is "
            "unknown. Enable --probe_api_prefill, rewrite the prompt without "
            "trailing whitespace, or explicitly choose best_effort."
        )
    verification = probe_result.verification
    if verification == PrefillVerification.VERIFIED_EXACT:
        return PrefillDecision(
            True, analysis.trailing_whitespace, policy, "server_managed",
            verification, probe_result.evidence,
        )
    if verification == PrefillVerification.VERIFIED_TOKEN_EFFECT \
            and policy == PrefillPolicy.AUTO:
        return PrefillDecision(
            True, analysis.trailing_whitespace, policy, "server_managed",
            verification, probe_result.evidence,
        )
    if verification == PrefillVerification.VERIFIED_STRIPPED:
        detail = "the API probe indicates that the suffix is stripped"
    elif verification == PrefillVerification.VERIFIED_TOKEN_EFFECT:
        detail = (
            "the API probe only verified a token-level effect, while policy "
            "exact requires prompt reconstruction"
        )
    else:
        detail = "the API probe was inconclusive or unsupported"
    raise PrefillCompatibilityError(
        f"Cannot satisfy assistant-prefill policy {policy.value!r}: {detail}. "
        "Rewrite the prompt without trailing whitespace or explicitly choose "
        "best_effort."
    )


def render_local_chat_prompt(
    tokenizer,
    messages,
    *,
    config=None,
    continue_last_assistant_message=True,
    enable_thinking=False,
):
    """Render a local prompt and preserve assistant-prefill bytes exactly."""
    config = config or ContinuationConfig()
    analysis = analyze_assistant_prefill(
        messages,
        continue_last_assistant_message=continue_last_assistant_message,
    )
    decision = resolve_prefill(analysis, config, local_exact=True)
    last_role = messages[-1]["role"]
    if not analysis.present:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=last_role == "user",
            continue_final_message=False,
            enable_thinking=enable_thinking,
        )
        return prompt, decision

    content = messages[-1]["content"]
    if any(
        isinstance(message.get("content"), str)
        and _CONTINUATION_BOUNDARY in message["content"]
        for message in messages
    ):
        raise ValueError("messages contain the internal continuation boundary")
    marked_messages = copy.deepcopy(messages)
    marked_messages[-1]["content"] = content + _CONTINUATION_BOUNDARY
    rendered = tokenizer.apply_chat_template(
        marked_messages,
        tokenize=False,
        add_generation_prompt=False,
        continue_final_message=False,
        enable_thinking=enable_thinking,
    )
    boundary_index = rendered.rfind(_CONTINUATION_BOUNDARY)
    if boundary_index < 0:
        raise ValueError(
            "chat template altered the internal continuation boundary; "
            "cannot preserve assistant prefill exactly"
        )
    return rendered[:boundary_index], decision


def candidate_surface_forms(candidate):
    """Return label spellings with and without one leading ASCII space."""
    if not isinstance(candidate, str) or not candidate:
        raise ValueError("tokens_of_interest must contain non-empty strings")
    alternate = candidate[1:] if candidate.startswith(" ") else " " + candidate
    return tuple(dict.fromkeys((candidate, alternate)))


def single_token_candidate_ids(tokenizer, candidates):
    """Map semantic candidates to all surface forms that are one token."""
    result = []
    for candidate in candidates:
        ids = []
        for surface in candidate_surface_forms(candidate):
            token_ids = tokenizer(surface, add_special_tokens=False)["input_ids"]
            if len(token_ids) == 1 and token_ids[0] not in ids:
                ids.append(token_ids[0])
        if not ids:
            raise ValueError(
                f"No single-token surface form exists for candidate {candidate!r}"
            )
        result.append(ids)
    return result
