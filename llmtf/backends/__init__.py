"""Lazy backend exports for dependency-isolated installation profiles."""

from llmtf.backends.base import (
    Backend,
    BackendBatchError,
    BackendCapability,
    BackendRequestError,
    BatchResult,
)

__all__ = [
    'Backend', 'BackendBatchError', 'BackendCapability', 'BackendRequestError',
    'BatchResult', 'HFBackend', 'VLLMBackend', 'APIBackend',
]


def __getattr__(name):
    if name == 'HFBackend':
        try:
            from llmtf.backends.hf import HFBackend
        except ImportError as exc:
            raise ImportError(
                "HFBackend requires the HF installation profile (torch, "
                "transformers and peft). Use docker/Dockerfile.hf or the vLLM profile."
            ) from exc
        return HFBackend
    if name == 'VLLMBackend':
        try:
            from llmtf.backends.vllm import VLLMBackend
        except ImportError as exc:
            raise ImportError(
                "VLLMBackend requires the vLLM installation profile. Use "
                "docker/Dockerfile.vllm."
            ) from exc
        return VLLMBackend
    if name == 'APIBackend':
        from llmtf.backends.api import APIBackend
        return APIBackend
    raise AttributeError(name)
