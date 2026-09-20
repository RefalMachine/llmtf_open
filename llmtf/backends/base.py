from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, TypedDict
from llmtf.base import Base
from llmtf.continuation import ContinuationConfig


class BackendCapability(str, Enum):
    GENERATE = "generate"
    TOKEN_PROBABILITY = "calculate_tokens_proba"
    LOGSOFTMAX = "calculate_logsoftmax"


class GenerationInfo(TypedDict):
    prompt_len: int
    generated_len: List[int]
    generated_cumulative_logprob: Optional[List[float]]


class TokenProbabilityInfo(TypedDict, total=False):
    prompt_len: int
    generated_len: int
    generated_cumulative_logprob: Optional[float]
    generated_token: str


class BatchResult(NamedTuple):
    prompts: List[Any]
    outputs: List[Any]
    infos: List[Dict[str, Any]]


class BackendBatchError(RuntimeError):
    def __init__(self, failures: Dict[int, BaseException]):
        self.failures = dict(sorted(failures.items()))
        indexes = ", ".join(str(index) for index in self.failures)
        details = "; ".join(
            f"{index}: {type(error).__name__}: {error}"
            for index, error in self.failures.items()
        )
        super().__init__(f"Backend batch failed at index(es) {indexes}: {details}")


class BackendRequestError(RuntimeError):
    def __init__(self, method, url, status_code=None, message="request failed"):
        self.method = method
        self.url = url
        self.status_code = status_code
        super().__init__(f"{method} {url}: {message} (status={status_code})")


def validate_batch_result(result, expected_size: int, *, method: str) -> BatchResult:
    if not isinstance(result, (tuple, list)) or len(result) != 3:
        raise TypeError(f"{method} backend result must be a 3-tuple")
    prompts, outputs, infos = result
    lengths = tuple(len(part) for part in (prompts, outputs, infos))
    if lengths != (expected_size, expected_size, expected_size):
        raise ValueError(
            f"{method} backend result is misaligned: expected {expected_size}, "
            f"got prompts/outputs/infos={lengths}"
        )
    for index, info in enumerate(infos):
        if not isinstance(info, dict):
            raise TypeError(f"{method} info[{index}] must be a dict")
        if "generated_len" not in info:
            raise ValueError(f"{method} info[{index}] lacks generated_len")
    return BatchResult(list(prompts), list(outputs), list(infos))


def normalize_stop_token_ids(value):
    if value is None:
        return None
    if isinstance(value, int):
        return [value]
    if isinstance(value, (list, tuple)):
        if not all(isinstance(item, int) for item in value):
            raise TypeError("stop token ids must contain only integers")
        return list(value)
    raise TypeError("stop token ids must be int, list[int], or None")


class Backend(Base):
    """Abstract interface for a model backend (primitive provider).

    A backend is responsible ONLY for obtaining tokens/outputs from a model
    deployed in some form (in-process weights, vLLM engine, HTTP server).
    It receives *messages* and returns (prompts, outputs, infos) tuples.

    Reasoning orchestration (two-pass emulated flow, message manipulation)
    lives on the LLM level, NOT here. The backend learns about thinking only
    insofar as it needs to render the chat template (enable_thinking flag) and,
    in the future, to surface a native reasoning_content field in infos.

    Concrete backends: HFBackend, VLLMBackend, APIBackend.
    """

    def __init__(self, **kwargs):
        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise TypeError(f"Unknown {type(self).__name__} option(s): {unknown}")
        super().__init__(**kwargs)
        self.continuation_config = ContinuationConfig()

    def configure_continuation(self, config):
        if not isinstance(config, ContinuationConfig):
            raise TypeError("config must be a ContinuationConfig")
        self.continuation_config = config

    @property
    def capabilities(self):
        result = set()
        for capability in BackendCapability:
            if self.support_method(capability.value):
                result.add(capability)
        return frozenset(result)

    @property
    def supports_assistant_continuation(self):
        """Whether a final assistant message can be continued as one turn.

        Local backends render the prompt themselves and therefore support it.
        Remote backends override this property from their negotiated profile.
        """
        return True

    @abstractmethod
    def support_method(self, method) -> bool:
        """Return True if the backend supports the given scoring method.

        method in {'generate', 'calculate_tokens_proba', 'calculate_logsoftmax'}.
        """

    @abstractmethod
    def from_pretrained(self, model_dir, *, conversation_template_path="auto",
                        is_foundational=False, **kwargs):
        """Load model weights / probe server and finalize backend state."""

    # --- primitives (messages in, (prompts, outputs, infos) out) ---

    @abstractmethod
    def generate_batch(self, messages_batch, *, generation_config=None,
                       continue_last_assistant_message=True, return_tokens=False,
                       include_stop_str_in_output=False, skip_special_tokens=True,
                       enable_thinking=False, **kwargs):
        """Generate continuations for a batch of message lists.

        Returns (prompts, outputs, infos). enable_thinking here is a RENDER
        flag (passed to apply_chat_template / chat_template_kwargs), not a
        reasoning-orchestration flag.
        """

    @abstractmethod
    def calculate_tokens_proba_batch(self, messages_batch, tokens_of_interest, *,
                                     continue_last_assistant_message=True, **kwargs):
        """Return next-token probabilities for tokens_of_interest.

        Returns (prompts, probs, infos).
        """

    def calculate_logsoftmax_batch(self, messages_batch, *, continue_last_assistant_message=True,
                                   log_only_last=True, **kwargs):
        """PPL scoring. HF-only; default raises NotImplementedError."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support calculate_logsoftmax "
            f"(PPL is HF-only)"
        )

    @abstractmethod
    def count_tokens_for_messages(self, messages, *, continue_last_assistant_message=True,
                                  add_think_token=False):
        """Return token count after applying the chat template, or ``None``.

        ``None`` means that an API exposes no preflight token counter.  It is
        not an estimate and must not be converted to zero.  Local backends
        always return an integer.
        """

    # --- config / introspection ---

    @abstractmethod
    def get_model_context_len(self) -> int:
        """Return the deployed model context length (engine ceiling / HF config
        position embeddings / server max_model_len), possibly overridden via
        --model_context_len. This is the deployment property, NOT a per-task
        prompt budget."""

    @abstractmethod
    def get_params(self) -> dict:
        """Return a serializable dict of backend parameters for logging."""

    @abstractmethod
    def add_stop_strings(self, stop_strings):
        """Add stop strings to the active generation config."""

    @abstractmethod
    def reset_stop_strings(self):
        """Reset stop strings / eos token ids to the base configuration."""

    def apply_model_prompt(self, *args, **kwargs):
        raise NotImplementedError(
            f"{type(self).__name__} does not expose local chat-template rendering"
        )

    def count_tokens_for_prompt(self, *args, **kwargs):
        raise NotImplementedError(
            f"{type(self).__name__} does not expose a local tokenizer"
        )

    # generation_config: a sampling-only GenerationConfig (temp/top_p/
    # max_new_tokens/stop_strings/eos). Set by the concrete backend in
    # from_pretrained/_load_model. Reasoning fields (max_new_tokens_reasoning,
    # truncing_prompt, end_thinking_token_id) are NOT stored here; they live on
    # LLM.reasoning_config (Layer 2). Accessed as a plain attribute.
