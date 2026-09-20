from llmtf.base import BaseLLM
from llmtf.backends.base import validate_batch_result
from llmtf.continuation import ContinuationConfig
from llmtf.reasoning import (
    ModelKind, ReasoningFormat, ReasoningConfig, EmulatedReasoningStrategy,
    THINK_CLOSE_MARKER, resolve_reasoning_execution,
)


class LLM(BaseLLM):
    """Concrete LLM: composes a Backend and runs reasoning orchestration on top.

    Primitives (generate_batch / calculate_tokens_proba_batch / logsoftmax /
    tokenization / loading) live on self.backend (a llmtf.backends.Backend).
    This class is responsible ONLY for the reasoning dispatcher (two-pass
    emulated flow) and for proxying the backend contract to consumers.

    This is the single entry point regardless of backend; reasoning logic is
    NOT duplicated across backends (one dispatcher, bound to self.backend).
    """

    def __init__(self, backend, *, assistant_prefill_policy="auto",
                 probe_api_prefill=False, **kwargs):
        super().__init__(**kwargs)
        self.backend = backend
        self.continuation_config = ContinuationConfig(
            prefill_policy=assistant_prefill_policy,
            probe_api_prefill=probe_api_prefill,
        )
        configure_continuation = getattr(
            self.backend, "configure_continuation", None
        )
        if configure_continuation is None:
            # Lightweight test/provider adapters may implement the structural
            # backend contract without inheriting Backend.
            self.backend.continuation_config = self.continuation_config
        else:
            configure_continuation(self.continuation_config)
        self._reasoning_config: "ReasoningConfig" = ReasoningConfig()
        self._reasoning: "EmulatedReasoningStrategy | None" = None

    # --- lifecycle ---

    def from_pretrained(self, model_dir, *, conversation_template_path="auto",
                        is_foundational=False, model_kind="plain",
                        max_new_tokens_reasoning=4096, min_new_tokens_reasoning=1024,
                        reasoning_truncing_prompt=None, end_thinking_token_id=None,
                        reasoning_format=None,
                        **kwargs):
        """Load the backend and configure reasoning mode.

        Delegates model loading to self.backend.from_pretrained (passing only
        backend-relevant kwargs: conversation_template_path, is_foundational,
        and any **kwargs the backend accepts), then builds the reasoning
        config via _setup_reasoning. Reasoning kwargs (model_kind,
        max_new_tokens_reasoning, min_new_tokens_reasoning,
        reasoning_truncing_prompt, end_thinking_token_id) are consumed here,
        NOT forwarded to the backend.

        'auto' is NOT a valid model_kind — auto-detection was removed in v3.
        Pick one of 'plain' (default, one-pass) / 'reasoning' / 'hybrid' explicitly.
        """
        self.backend.from_pretrained(
            model_dir,
            conversation_template_path=conversation_template_path,
            is_foundational=is_foundational,
            **kwargs
        )
        self._setup_reasoning(
            model_kind=model_kind,
            max_new_tokens_reasoning=max_new_tokens_reasoning,
            min_new_tokens_reasoning=min_new_tokens_reasoning,
            reasoning_truncing_prompt=reasoning_truncing_prompt,
            end_thinking_token_id=end_thinking_token_id,
            reasoning_format=reasoning_format,
        )

    # --- reasoning setup (user-driven; no auto-detection) ---

    def _setup_reasoning(self, model_kind="plain", max_new_tokens_reasoning=4096,
                         min_new_tokens_reasoning=1024, reasoning_truncing_prompt=None,
                         end_thinking_token_id=None, reasoning_format=None):
        """Configure reasoning mode after the backend is loaded.

        Builds self._reasoning_config (a ReasoningConfig holding model_kind,
        max_new_tokens_reasoning [upper bound], min_new_tokens_reasoning [floor],
        and a ReasoningFormat) and self._reasoning (an EmulatedReasoningStrategy
        bound to that config). Reasoning fields are NOT written onto the
        backend's generation_config (which stays sampling-only); MaxLenContext
        and the strategy read them from self._reasoning_config.

        Responsibilities on the user: choose model_kind explicitly
        ('plain' | 'reasoning' | 'hybrid'). 'auto' is NOT supported — if a
        reasoning/two-pass flow is desired, pass model_kind='reasoning' or
        'hybrid' (plus --end_thinking_token_id). The default 'plain' gives a
        one-pass flow, which for hybrid models is correct when enable_thinking
        is False.

        Defaults:
          - max_new_tokens_reasoning (upper bound): 4096 if not provided.
          - min_new_tokens_reasoning (floor, below which reasoning is skipped):
            1024 if not provided.

        MaxLenContext may trim max_new_tokens_reasoning down at call time
        (per-task / per-turn), but never below min_new_tokens_reasoning: if
        even the floor cannot be reserved, reasoning is skipped for that turn
        (one-pass with a loud warning).
        """
        kind = ModelKind(model_kind)

        if reasoning_format is not None and not isinstance(reasoning_format, ReasoningFormat):
            raise TypeError("reasoning_format must be a ReasoningFormat instance")
        if reasoning_format is not None and any(value is not None for value in (
            reasoning_truncing_prompt, end_thinking_token_id
        )):
            raise ValueError(
                "reasoning_format cannot be combined with legacy format arguments"
            )
        if reasoning_truncing_prompt is None:
            reasoning_truncing_prompt = ReasoningFormat.truncation_prompt

        if kind == ModelKind.plain:
            self._reasoning_config = ReasoningConfig(model_kind=ModelKind.plain)
            self._reasoning = None
            return

        if max_new_tokens_reasoning <= 0:
            raise ValueError(
                "Reasoning-capable model kinds require max_new_tokens_reasoning > 0"
            )

        if min_new_tokens_reasoning > max_new_tokens_reasoning:
            self.logger.warning(
                "min_new_tokens_reasoning (%d) > max_new_tokens_reasoning (%d); "
                "raising max to min.", min_new_tokens_reasoning, max_new_tokens_reasoning
            )
            max_new_tokens_reasoning = min_new_tokens_reasoning

        effective_end_token_id = (
            reasoning_format.end_thinking_token_id
            if reasoning_format is not None else end_thinking_token_id
        )
        if effective_end_token_id is None:
            if kind == ModelKind.reasoning:
                raise ValueError(
                    "model_kind='reasoning' requires an explicit end_thinking_token_id "
                    "(the id of the think-close token). Pass it via --end_thinking_token_id "
                    "or from_pretrained(end_thinking_token_id=...). Without it the reasoning "
                    "phase cannot reliably stop."
                )
            # hybrid: warn; the reasoning phase will fall back to the text
            # stop-string THINK_CLOSE_MARKER only (less robust on HF, where
            # without an eos_token_id the model might generate the close token
            # as part of the text rather than stopping on it).
            self.logger.warning(
                "model_kind=%s without an explicit end_thinking_token_id; the reasoning "
                "phase will stop on the text marker %s only, which is less robust "
                "(especially on the HF backend). Pass --end_thinking_token_id for reliable "
                "two-pass reasoning.",
                kind.value, THINK_CLOSE_MARKER,
            )

        fmt = reasoning_format or ReasoningFormat(
            think_close=THINK_CLOSE_MARKER,
            truncation_prompt=reasoning_truncing_prompt,
            end_thinking_token_id=end_thinking_token_id,
        )
        self._reasoning_config = ReasoningConfig(
            model_kind=kind,
            max_new_tokens_reasoning=max_new_tokens_reasoning,
            min_new_tokens_reasoning=min_new_tokens_reasoning,
            fmt=fmt,
        )
        self._reasoning = EmulatedReasoningStrategy(self._reasoning_config, logger=self.logger)

    # --- reasoning state accessors ---

    @property
    def reasoning_config(self) -> "ReasoningConfig":
        return self._reasoning_config

    # --- reasoning dispatcher (single copy for all backends) ---

    def generate(
        self,
        messages,
        generation_config=None,
        continue_last_assistant_message=True,
        return_tokens=False,
        include_stop_str_in_output=False,
        skip_special_tokens=True,
        enable_thinking=False,
        **kwargs
    ):
        prompts, outputs, infos = self.generate_batch(
            [messages],
            generation_config=generation_config,
            continue_last_assistant_message=continue_last_assistant_message,
            return_tokens=return_tokens,
            include_stop_str_in_output=include_stop_str_in_output,
            skip_special_tokens=skip_special_tokens,
            enable_thinking=enable_thinking,
            **kwargs
        )
        return prompts[0], outputs[0], infos[0]

    def _resolve_dispatch(self, enable_thinking):
        """Decide one-pass vs two-pass for a reasoning dispatcher.

        Returns True for a one-pass call to the backend (no emulated
        reasoning), False for the two-pass emulated flow. Raises ValueError
        for the invalid combination (model_kind='reasoning', thinking off).
        Warns when enable_thinking is requested but the model is plain.
        Forces one-pass with a loud warning when enable_thinking is requested
        but max_new_tokens_reasoning==0 (typically because MaxLenContext
        trimmed the reasoning budget to 0 — running two-pass would synthesize
        a fake truncation marker with no actual reasoning).
        """
        rc = self.reasoning_config
        mode = resolve_reasoning_execution(rc.model_kind, enable_thinking, "generate")
        if enable_thinking and not rc.is_reasoning:
            self.logger.warning('enable_thinking ignored on plain model (no reasoning support)')
        if not mode.reasoning_enabled:
            return True
        if rc.max_new_tokens_reasoning == 0:
            self.logger.warning(
                'enable_thinking=True but max_new_tokens_reasoning=0 (reasoning budget '
                'trimmed to 0 for this task, typically due to model_max_len constraints); '
                'running one-pass WITHOUT reasoning — no two-pass emulated flow, no '
                'synthesized reasoning block. Lower the answer budget on this task, disable '
                'reasoning, or use --model_kind plain.'
            )
            return True
        return False

    def _require_two_pass_continuation(self):
        if getattr(self.backend, 'supports_assistant_continuation', True):
            return
        if self.continuation_config.prefill_policy.value == 'best_effort':
            return
        raise NotImplementedError(
            "two-pass emulated reasoning requires assistant continuation, but "
            "the selected API profile does not declare that capability. Use "
            "api_profile='vllm', disable emulated reasoning, or explicitly "
            "select best_effort for an unverified run."
        )

    @staticmethod
    def _reasoning_info_entry(reasoning_info, reasoning_text, add_reasoning_info):
        """Compose the `reasoning` sub-payload of the per-sample info dict."""
        entry = {
            "prompt_len": reasoning_info["prompt_len"],
            "generated_len": reasoning_info["generated_len"],
            "generated_cumulative_logprob": reasoning_info["generated_cumulative_logprob"],
        }
        if add_reasoning_info:
            entry["text"] = reasoning_text
        return entry

    def generate_batch(
        self,
        messages,
        generation_config=None,
        continue_last_assistant_message=True,
        return_tokens=False,
        include_stop_str_in_output=False,
        skip_special_tokens=True,
        enable_thinking=False,
        add_reasoning_truncing_prompt=True,
        add_reasoning_info=True,
        **kwargs
    ):
        if "add_assistant_prompt_to_output" in kwargs:
            raise TypeError(
                "add_assistant_prompt_to_output was removed: predict contains "
                "only tokens generated by the model"
            )
        if self._resolve_dispatch(enable_thinking):
            result = self.backend.generate_batch(
                messages,
                generation_config=generation_config,
                continue_last_assistant_message=continue_last_assistant_message,
                return_tokens=return_tokens,
                include_stop_str_in_output=include_stop_str_in_output,
                skip_special_tokens=skip_special_tokens,
                enable_thinking=False,
                **kwargs
            )
            return validate_batch_result(result, len(messages), method="generate")
        self._require_two_pass_continuation()
        # hybrid + True, or reasoning + True: two-pass emulated reasoning.
        # The strategy owns the reasoning+continuation flow; this layer merges
        # outer **kwargs with phase-specific overrides (overrides win), supplies
        # the continuation primitive + extra args, then post-processes the output
        # and shapes the per-sample info payload. final_outputs remain the raw
        # continuation generated after the assistant prefill.
        generation_config = self.backend.generation_config if generation_config is None else generation_config
        reasoning_phase_kwargs = {
            **kwargs,
            'continue_last_assistant_message': continue_last_assistant_message,
            'include_stop_str_in_output': True,
            'return_tokens': False,
            'enable_thinking': True,
            'skip_special_tokens': False,
        }
        continuation_phase_kwargs = {
            **kwargs,
            'generation_config': generation_config,
            'continue_last_assistant_message': True,
            'return_tokens': return_tokens,
            'include_stop_str_in_output': include_stop_str_in_output,
            'enable_thinking': False,
            'skip_special_tokens': skip_special_tokens,
        }
        result = self._reasoning.run(
            messages,
            reasoning_fn=self.backend.generate_batch,
            continuation_fn=self.backend.generate_batch,
            generation_config=generation_config,
            reasoning_phase_kwargs=reasoning_phase_kwargs,
            continuation_phase_kwargs=continuation_phase_kwargs,
            add_reasoning_truncing_prompt=add_reasoning_truncing_prompt,
        )
        infos_batch = []
        for i in range(len(messages)):
            response_info = {
                "prompt_len": result.final_infos[i]["prompt_len"],
                "generated_len": result.final_infos[i]["generated_len"],
                "generated_cumulative_logprob": result.final_infos[i]["generated_cumulative_logprob"],
            }
            if "assistant_prefill" in result.final_infos[i]:
                response_info["assistant_prefill"] = result.final_infos[i]["assistant_prefill"]
            infos_batch.append({
                "reasoning": self._reasoning_info_entry(
                    result.reasoning_infos[i], result.reasoning_outputs[i], add_reasoning_info
                ),
                "response": response_info,
            })
        return result.reasoning_prompts, result.final_outputs, infos_batch

    def calculate_tokens_proba(self, messages, tokens_of_interest, continue_last_assistant_message=True, **kwargs):
        prompts, probs, infos = self.calculate_tokens_proba_batch([messages], [tokens_of_interest], continue_last_assistant_message=continue_last_assistant_message, **kwargs)
        return prompts[0], probs[0], infos[0]

    def calculate_tokens_proba_batch(self, messages, tokens_of_interest, continue_last_assistant_message=True,
                                     enable_thinking=False, add_reasoning_truncing_prompt=True,
                                     add_reasoning_info=True, generation_config=None, **kwargs):
        if self._resolve_dispatch(enable_thinking):
            result = self.backend.calculate_tokens_proba_batch(
                messages, tokens_of_interest, continue_last_assistant_message=continue_last_assistant_message, **kwargs
            )
            return validate_batch_result(
                result, len(messages), method="calculate_tokens_proba"
            )
        self._require_two_pass_continuation()
        # hybrid + True, or reasoning + True: two-pass emulated reasoning.
        # The strategy owns the reasoning+continuation flow; this layer merges
        # outer **kwargs with phase-specific overrides (overrides win), supplies
        # the continuation primitive + extra args (tokens_of_interest is threaded
        # via continuation_phase_kwargs), then shapes the info.
        generation_config = self.backend.generation_config if generation_config is None else generation_config
        reasoning_phase_kwargs = {
            **kwargs,
            'continue_last_assistant_message': continue_last_assistant_message,
            'include_stop_str_in_output': True,
            'return_tokens': False,
            'enable_thinking': True,
            'skip_special_tokens': False,
        }
        continuation_phase_kwargs = {
            **kwargs,
            'tokens_of_interest': tokens_of_interest,
            'continue_last_assistant_message': continue_last_assistant_message,
        }
        result = self._reasoning.run(
            messages,
            reasoning_fn=self.backend.generate_batch,
            continuation_fn=self.backend.calculate_tokens_proba_batch,
            generation_config=generation_config,
            reasoning_phase_kwargs=reasoning_phase_kwargs,
            continuation_phase_kwargs=continuation_phase_kwargs,
            add_reasoning_truncing_prompt=add_reasoning_truncing_prompt,
        )
        infos_batch = []
        for i in range(len(messages)):
            response_info = {
                "generated_len": result.final_infos[i]["generated_len"],
                "generated_token": result.final_infos[i]["generated_token"],
            }
            for key in (
                "prompt_len", "assistant_prefill",
                "candidate_surface_form_aggregation", "candidate_score_semantics",
                "candidate_surface_coverage", "top_logprobs_count",
            ):
                if key in result.final_infos[i]:
                    response_info[key] = result.final_infos[i][key]
            infos_batch.append({
                "reasoning": self._reasoning_info_entry(
                    result.reasoning_infos[i], result.reasoning_outputs[i], add_reasoning_info
                ),
                "response": response_info,
            })
        return result.reasoning_prompts, result.final_outputs, infos_batch

    def calculate_logsoftmax(self, messages, continue_last_assistant_message=True, log_only_last=True):
        return self.backend.calculate_logsoftmax(messages, continue_last_assistant_message=continue_last_assistant_message, log_only_last=log_only_last)

    def calculate_logsoftmax_batch(self, messages, continue_last_assistant_message=True, log_only_last=True):
        return self.backend.calculate_logsoftmax_batch(messages, continue_last_assistant_message=continue_last_assistant_message, log_only_last=log_only_last)

    # --- proxy to backend ---

    @property
    def generation_config(self):
        return self.backend.generation_config

    @property
    def tokenizer(self):
        return self.backend.tokenizer

    def apply_model_prompt(self, messages, continue_last_assistant_message=True, add_think_token=False):
        return self.backend.apply_model_prompt(messages, continue_last_assistant_message=continue_last_assistant_message, add_think_token=add_think_token)

    def count_tokens_for_prompt(self, prompt):
        return self.backend.count_tokens_for_prompt(prompt)

    def count_tokens_for_messages(self, messages, *, continue_last_assistant_message=True, add_think_token=False):
        return self.backend.count_tokens_for_messages(messages, continue_last_assistant_message=continue_last_assistant_message, add_think_token=add_think_token)

    def add_stop_strings(self, stop_strings):
        self.backend.add_stop_strings(stop_strings)

    def reset_stop_strings(self):
        self.backend.reset_stop_strings()

    def support_method(self, method):
        return self.backend.support_method(method)

    def get_params(self):
        rc = self.reasoning_config
        params = dict(self.backend.get_params())
        params['_llmtf_reasoning'] = {
            'model_kind': rc.model_kind.value,
            'configured_max_new_tokens_reasoning': rc.configured_max_new_tokens_reasoning,
            'max_new_tokens_reasoning': rc.max_new_tokens_reasoning,
            'min_new_tokens_reasoning': rc.min_new_tokens_reasoning,
            'format': {
                'think_close': rc.fmt.think_close,
                'truncation_prompt': rc.fmt.truncation_prompt,
                'answer_separator': rc.fmt.answer_separator,
                'end_thinking_token_id': rc.fmt.end_thinking_token_id,
            },
        }
        params['_llmtf_continuation'] = self.continuation_config.to_dict()
        return params

    def get_model_context_len(self):
        return self.backend.get_model_context_len()
