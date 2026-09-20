from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List
import copy
import logging
from llmtf.backends.base import validate_batch_result


THINK_CLOSE_MARKER = "</think>"


class ModelKind(str, Enum):
    plain = "plain"
    reasoning = "reasoning"
    hybrid = "hybrid"


@dataclass(frozen=True)
class ExecutionMode:
    model_kind: ModelKind
    enable_thinking: bool
    scoring_method: str
    reasoning_enabled: bool


def resolve_reasoning_execution(model_kind, enable_thinking, scoring_method="generate"):
    """Resolve the one/two-pass decision used by budgeting and dispatch."""
    kind = ModelKind(model_kind)
    if scoring_method == "calculate_logsoftmax":
        if kind == ModelKind.reasoning:
            raise ValueError(
                "calculate_logsoftmax is one-pass and cannot satisfy "
                "model_kind='reasoning'"
            )
        return ExecutionMode(kind, False, scoring_method, False)
    if not enable_thinking:
        if kind == ModelKind.reasoning:
            raise ValueError(
                "model_kind='reasoning' but enable_thinking=False — use "
                "model_kind='hybrid' or enable thinking"
            )
        return ExecutionMode(kind, False, scoring_method, False)
    return ExecutionMode(kind, True, scoring_method, kind != ModelKind.plain)


@dataclass
class ReasoningFormat:
    """Format of the emulated reasoning block (Qwen3-think style by default).

    think_close: stop-string used to end the reasoning phase (text marker).
    truncation_prompt: text appended when the reasoning phase hit the length
        limit instead of the close marker.
    answer_separator: canonical separator between the reasoning close marker
        and the visible answer inside one assistant turn.
    end_thinking_token_id: optional id of the think-close token. If None, the
        reasoning phase falls back to stopping on the text thinkClose marker
        only (less robust, especially on the HF backend where the model may
        emit the close token as part of the text rather than stopping on it).
        For model_kind='reasoning' it must be provided explicitly.
    """
    think_close: str = THINK_CLOSE_MARKER
    truncation_prompt: str = "\n\u2026the rest of the reasoning chain is hidden due to limit on the length of the thoughts.\n"
    answer_separator: str = "\n\n"
    end_thinking_token_id: Optional[int] = None

    def __post_init__(self):
        if not isinstance(self.think_close, str) or not self.think_close:
            raise ValueError("ReasoningFormat.think_close must be a non-empty string")
        if not isinstance(self.answer_separator, str):
            raise ValueError("ReasoningFormat.answer_separator must be a string")


@dataclass
class ReasoningConfig:
    """LLM-level reasoning configuration.

    Holds the reasoning mode (model_kind), the reasoning-token budget
    (max_new_tokens_reasoning, upper bound) and its floor (min_new_tokens_reasoning),
    and the reasoning format (fmt). These fields are NOT stored on the backend's
    generation_config (which is sampling-only). MaxLenContext reads/trims
    max_new_tokens_reasoning here (clamped to [min, max]); the strategy reads it
    live at call time, so context-pressure trims take effect.

    If MaxLenContext cannot reserve at least min_new_tokens_reasoning for the
    reasoning phase, it forces one-pass (enable_thinking=False) for that turn
    with a loud warning — no silent silent trimming to 0.
    """
    model_kind: ModelKind = ModelKind.plain
    max_new_tokens_reasoning: int = 4096
    min_new_tokens_reasoning: int = 1024
    fmt: ReasoningFormat = field(default_factory=ReasoningFormat)
    configured_max_new_tokens_reasoning: Optional[int] = None

    def __post_init__(self):
        self.model_kind = ModelKind(self.model_kind)
        if self.configured_max_new_tokens_reasoning is None:
            self.configured_max_new_tokens_reasoning = self.max_new_tokens_reasoning
        if self.max_new_tokens_reasoning < 0 or self.min_new_tokens_reasoning < 0:
            raise ValueError("Reasoning token budgets must be non-negative")

    @property
    def is_reasoning(self) -> bool:
        return self.model_kind != ModelKind.plain

    @property
    def is_hybrid(self) -> bool:
        return self.model_kind == ModelKind.hybrid


@dataclass
class ReasoningResult:
    """Raw result of EmulatedReasoningStrategy.run — one batch-level field per
    phase output. The caller (LLM layer) shapes the per-sample info payload from
    the raw reasoning_infos / final_infos batches.
    """
    reasoning_prompts: List
    final_outputs: List
    reasoning_outputs: List
    reasoning_infos: List
    final_infos: List


class EmulatedReasoningStrategy:
    """Composition object implementing the emulated reasoning flow: reasoning
    phase then continuation phase, as a single indivisible operation "obtain an
    answer with a reasoning block".

    Responsibilities:
      - build a deepcopy of the sampling-only generation_config for the
        reasoning phase (stop_strings += think_close, max_new_tokens =
        max_new_tokens_reasoning, eos_token_id = [end_thinking_token_id]);
        the base config is never mutated;
      - split incoming messages into the prompt prefix and the trailing
        assistant block;
      - invoke the reasoning_fn (a backend primitive) on the prefix;
      - validate the stop condition (think-close marker or length limit) and,
        on length-limit, optionally append the truncation_prompt and synthesize
        a closing marker so the continuation phase sees a well-formed block;
      - assemble one continuation turn as `updated_messages` = prefix +
        {assistant, reasoning_text + trailing_assistant_prefill};
      - invoke continuation_fn(updated_messages, **continuation_phase_kwargs)
        — the caller decides what continuation_fn is (generate or ctp) and
        threads any extra arguments (generation_config, tokens_of_interest, ...)
        via continuation_phase_kwargs;
      - return raw results to the caller: reasoning prompts, final continuation
        outputs, raw reasoning infos and raw final infos.

    What this class deliberately does NOT own:
      - the generate-vs-ctp distinction (no `mode` flag): the caller wraps its
        continuation primitive as a continuation_fn accepting (messages_batch,
        **kwargs) and supplies any extra arguments via continuation_phase_kwargs;
      - the shape of the per-sample info payload (assembled by the LLM layer
        from the raw reasoning_infos / final_infos returned here).

    Reasoning-specific fields (max_new_tokens_reasoning, truncation_prompt,
    end_thinking_token_id) are read from the ReasoningConfig live, so that
    MaxLenContext trims to max_new_tokens_reasoning take effect at call time.

    One implementation works for all three backends (HF/vLLM/API) and for both
    scoring methods (generate / calculate_tokens_proba).
    """

    def __init__(self, reasoning_config: ReasoningConfig, logger: Optional[logging.Logger] = None):
        self._rc = reasoning_config
        self.fmt = reasoning_config.fmt
        self.logger = logger or logging.getLogger(__name__)

    @property
    def max_new_tokens_reasoning(self):
        return self._rc.max_new_tokens_reasoning

    def _build_reasoning_config(self, base_config):
        cfg = copy.deepcopy(base_config)
        stop = getattr(cfg, "stop_strings", None)
        if not stop:
            stop = []
        cfg.stop_strings = list(stop) + [self.fmt.think_close]
        if self._rc.max_new_tokens_reasoning is not None:
            cfg.max_new_tokens = self._rc.max_new_tokens_reasoning
        if self.fmt.end_thinking_token_id is not None:
            cfg.eos_token_id = [self.fmt.end_thinking_token_id]
        return cfg

    def _split_messages(self, messages_batch):
        prompt_messages_batch = []
        assistant_messages_batch = []
        for messages in messages_batch:
            split_at = len(messages)
            while split_at > 0 and messages[split_at - 1]["role"] == "assistant":
                split_at -= 1
            if split_at == 0:
                raise ValueError(
                    "Reasoning input must contain a non-assistant prompt before "
                    "the optional assistant prefill"
                )
            prompt_messages_batch.append(messages[:split_at])
            assistant_messages_batch.append(messages[split_at:])
        return prompt_messages_batch, assistant_messages_batch

    @staticmethod
    def _join_assistant_prefill(messages):
        parts = []
        for message in messages:
            content = message.get("content", "")
            if not isinstance(content, str):
                raise TypeError(
                    "Reasoning assistant prefill currently requires string content"
                )
            parts.append(content)
        return "".join(parts)

    def _validate_and_normalize_reasoning_text(self, reasoning_text, generated_len):
        """Apply stop-condition validation and length-limit normalization."""
        max_new_tokens_reasoning = self._rc.max_new_tokens_reasoning
        stopped_by_length = (
            max_new_tokens_reasoning is not None
            and generated_len >= max_new_tokens_reasoning
            and not reasoning_text.endswith(self.fmt.think_close)
        )
        stopped_by_think = False
        if self.fmt.think_close in reasoning_text:
            stopped_by_think = True
            if not reasoning_text.endswith(self.fmt.think_close):
                raise Exception(
                    f"Unexpected generation: {self.fmt.think_close} token is present but generation did not stop.\n"
                    f'Reasoning text: "{reasoning_text}"'
                )
        if not (stopped_by_think or stopped_by_length):
            raise Exception(
                f"Unexpected generation: stopped by neither {self.fmt.think_close} token nor length limit.\n"
                f'Reasoning text: "{reasoning_text}"'
            )
        return stopped_by_length

    def run(self, messages_batch, *, reasoning_fn, continuation_fn, generation_config,
            reasoning_phase_kwargs, continuation_phase_kwargs,
            add_reasoning_truncing_prompt=True) -> "ReasoningResult":
        """Run the emulated reasoning flow: reasoning phase then continuation
        phase. Returns a ReasoningResult with batch-level raw fields:

            - reasoning_prompts:   prompts of the reasoning phase (== the prompt
              prefix of each sample); the LLM layer surfaces them as the outer prompts.
            - final_outputs:       continuation-phase outputs per sample.
            - reasoning_outputs:   normalized reasoning text per sample.
            - reasoning_infos:     raw infos returned by reasoning_fn, for the
              LLM layer to compose into the final info payload.
            - final_infos:         raw infos returned by continuation_fn, for the
              LLM layer to compose into the final info payload.
        The caller is responsible for shaping the per-sample info payload from
        the raw info batches. ``final_outputs`` are always generated
        continuations; the assistant prefill remains input-only.

        The caller supplies a flat kwargs dict per phase (already merged with
        any outer **kwargs, with phase-specific overrides taking priority).
        No `mode` flag: the caller picks the continuation primitive
        (generate_batch / calculate_tokens_proba_batch / ...) and threads any
        extra arguments (generation_config, tokens_of_interest, ...) via
        continuation_phase_kwargs.

        - reasoning_fn(prompt_messages, generation_config=reasoning_cfg,
            **reasoning_phase_kwargs) -> (prompts, outputs, infos);
        - continuation_fn(updated_messages, **continuation_phase_kwargs) ->
          (prompts, outputs, infos).
        """
        num_return_sequences = getattr(generation_config, "num_return_sequences", 1) or 1
        if num_return_sequences != 1:
            raise NotImplementedError(
                "Two-pass reasoning requires num_return_sequences=1; branching "
                "reasoning continuations is not defined"
            )
        reasoning_cfg = self._build_reasoning_config(generation_config)
        prompt_messages_batch, assistant_messages_batch = self._split_messages(messages_batch)

        reasoning_result = validate_batch_result(
            reasoning_fn(
                prompt_messages_batch, generation_config=reasoning_cfg,
                **reasoning_phase_kwargs
            ),
            len(messages_batch),
            method="reasoning.generate",
        )
        reasoning_prompt_batch, reasoning_output_batch, reasoning_infos_batch = reasoning_result

        truncation_prompt = self.fmt.truncation_prompt
        updated_messages_batch = []
        for i, messages in enumerate(prompt_messages_batch):
            reasoning_text = reasoning_output_batch[i]
            generated_len = reasoning_infos_batch[i]["generated_len"][0]
            stopped_by_length = self._validate_and_normalize_reasoning_text(reasoning_text, generated_len)
            if stopped_by_length:
                if add_reasoning_truncing_prompt:
                    reasoning_text += truncation_prompt
                if not reasoning_text.endswith("\n"):
                    reasoning_text += "\n"
                reasoning_text += self.fmt.think_close
            # Qwen chat templates render reasoning and the visible answer in
            # one assistant turn with this canonical boundary.  Normalizing it
            # also lets continue_final_message match the rendered final turn.
            reasoning_text += self.fmt.answer_separator
            reasoning_output_batch[i] = reasoning_text

            # Reasoning and the answer prefill are parts of one assistant turn.
            # Keeping them as adjacent assistant messages inserts an end-of-turn
            # boundary between reasoning and continuation in chat templates.
            assistant_prefill = self._join_assistant_prefill(
                assistant_messages_batch[i]
            )
            new_messages = [message.copy() for message in messages]
            new_messages.append({
                "role": "assistant",
                "content": reasoning_text + assistant_prefill,
            })
            updated_messages_batch.append(new_messages)

        continuation_result = validate_batch_result(
            continuation_fn(updated_messages_batch, **continuation_phase_kwargs),
            len(messages_batch),
            method="reasoning.continuation",
        )
        _, final_output_batch, final_infos_batch = continuation_result

        return ReasoningResult(
            reasoning_prompts=reasoning_prompt_batch,
            final_outputs=final_output_batch,
            reasoning_outputs=reasoning_output_batch,
            reasoning_infos=reasoning_infos_batch,
            final_infos=final_infos_batch,
        )
