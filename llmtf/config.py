"""Framework-owned lightweight sampling configuration.

The API client must not depend on transformers merely to carry sampling
parameters.  Local backends may still expose ``transformers.GenerationConfig``;
the orchestration layer intentionally relies only on this structural subset.
"""

from dataclasses import asdict, dataclass
import json
from typing import Any, Dict, List, Optional, Union


DEFAULT_VLLM_GPU_MEMORY_UTILIZATION = 0.92


@dataclass
class SamplingConfig:
    repetition_penalty: float = 1.0
    presence_penalty: float = 0.0
    temperature: float = 0.1
    top_p: float = 0.9
    top_k: int = 40
    max_new_tokens: int = 64
    max_length: Optional[int] = None
    do_sample: bool = True
    num_return_sequences: int = 1
    num_beams: int = 1
    eos_token_id: Optional[Union[int, List[int]]] = None
    stop_strings: Optional[List[str]] = None
    pad_token_id: Optional[int] = None
    bos_token_id: Optional[int] = None

    def __post_init__(self):
        if self.stop_strings is None:
            self.stop_strings = []

    @classmethod
    def from_dict(cls, values: Dict[str, Any]):
        known = cls.__dataclass_fields__
        return cls(**{key: value for key, value in values.items() if key in known})

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json_string(self, use_diff=False) -> str:
        del use_diff
        return json.dumps(self.to_dict())


def config_to_dict(config) -> Dict[str, Any]:
    """Serialize either SamplingConfig or a transformers-style config."""
    if config is None:
        return {}
    if hasattr(config, "to_dict"):
        return config.to_dict()
    if hasattr(config, "to_json_string"):
        return json.loads(config.to_json_string(use_diff=True))
    return {
        key: value for key, value in vars(config).items()
        if not key.startswith("_")
    }
