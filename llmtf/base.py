import abc
from abc import abstractmethod
import logging
from typing import Dict, List, Tuple
import copy
import os
from datasets import load_dataset, Dataset
from tqdm import tqdm
from llmtf.metrics import mean
from llmtf.utils import normalize_message_roles


class PromptTooLongError(ValueError):
    """A task's irreducible zero-shot prompt exceeds its prompt budget."""


def ensure_prompt_fits(token_count, max_prompt_len, task_name="task"):
    """Fail closed when a prompt cannot fit instead of claiming truncation."""
    if token_count is not None and token_count > max_prompt_len:
        raise PromptTooLongError(
            f"{task_name} zero-shot prompt has {token_count} tokens, exceeding "
            f"the available prompt budget {max_prompt_len}; the task does not "
            "define a semantics-preserving truncation policy"
        )


def distribute_sample_limit(total, bucket_count):
    """Distribute an upper bound across buckets without exceeding it."""
    if not isinstance(total, int) or total < 0:
        raise ValueError("sample limit must be a non-negative integer")
    if not isinstance(bucket_count, int) or bucket_count <= 0:
        raise ValueError("bucket_count must be a positive integer")
    base, remainder = divmod(total, bucket_count)
    return [base + (index < remainder) for index in range(bucket_count)]

class Base(abc.ABC):
    def __init__(self, **kwargs):
        self.backend_logger = None

    @classmethod
    def name(cls):
        """Return name of class in lower case"""
        return cls.__name__.lower()

    @property
    def logger(self):
        if self.backend_logger is None:
            self.init_logger()
        return self.backend_logger

    def init_logger(self):
        self.backend_logger = logging.getLogger(__name__ + '.' + self.name())
        self.backend_logger.setLevel(logging.INFO)
        handler_name = 'stream_handler.' + __name__ + '.' + self.name()
        if handler_name in [h.name for h in self.backend_logger.handlers]:
            return
        ch = logging.StreamHandler()
        ch.set_name(handler_name)
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter('%(levelname)s: %(asctime)s: %(name)s: %(message)s')
        ch.setFormatter(formatter)
        self.backend_logger.addHandler(ch)

class Task(Base):
    ALLOW_BOOTSTRAPPING = True
    
    def __init__(self, name_suffix=None, **kwargs):
        super().__init__(**kwargs)
        self.method_additional_args = {}
        if not hasattr(self, '_max_task_new_tokens'):
            self._max_task_new_tokens = None
        self.additional_stop_strings = []
        self.name_suffix = ''
        if name_suffix is not None:
            self.name_suffix = f'_{name_suffix}'

    def run_name(self):
        return self.task_name() + self.name_suffix
    
    def get_aggregation_details(self) -> Dict:
        """
        Возвращает детальные результаты агрегации метрик, если они доступны.
        По умолчанию возвращает пустой словарь.
        Задачи могут переопределить этот метод для сохранения детальной информации.
        """
        return {}
    
    @property
    def max_task_new_tokens(self):
        if self._max_task_new_tokens is None:
            self.logger.error('self._max_task_new_tokens is None. Every task must set _max_task_new_tokens parameter')
            raise Exception('self._max_task_new_tokens is None. Every task must set _max_task_new_tokens parameter')
        return self._max_task_new_tokens

    @abstractmethod
    def evaluate(self, **kwargs) -> Dict:
        pass

    @abstractmethod
    def aggregation(self, **kwargs) -> Dict:
        pass

    @abstractmethod
    def load_dataset(self, **kwargs) -> Tuple[List[Dict], List[Dict]]:
        pass

    def leaderboard_aggregation(self, metrics: Dict) -> float:
        return mean([metrics[m] for m in metrics])

    def require_model_method(self, model):
        if not model.support_method(self.method):
            backend = getattr(model, 'backend', model)
            raise NotImplementedError(
                f"{type(backend).__name__} does not support task method "
                f"{self.method!r} with the selected configuration"
            )

class BaseLLM(Base):
    """Abstract public contract for an LLM (what evaluator/tasks depend on).

    Concrete implementation lives in llmtf/llm.py as `LLM`, which composes a
    Backend (llmtf/backends/) and runs reasoning orchestration on top.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @abstractmethod
    def generate(self, **kwargs):
        pass

    @abstractmethod
    def generate_batch(self, **kwargs):
        pass

    @abstractmethod
    def calculate_tokens_proba(self, **kwargs):
        pass

    @abstractmethod
    def calculate_tokens_proba_batch(self, **kwargs):
        pass

    def calculate_logsoftmax(self, **kwargs):
        raise NotImplementedError("calculate_logsoftmax is an optional HF capability")

    def calculate_logsoftmax_batch(self, **kwargs):
        raise NotImplementedError("calculate_logsoftmax is an optional HF capability")

    @abstractmethod
    def support_method(self, **kwargs):
        pass

    def apply_model_prompt(self, **kwargs):
        """Debug-only: render chat template to a string.

        Optional: supported only on local backends (HF/vLLM). The API
        backend has no local chat template and raises NotImplementedError.
        Prefer count_tokens_for_messages for token-counting; an API backend
        may return None when the server exposes no counter.
        """
        raise NotImplementedError(
            "apply_model_prompt is supported only on local backends (HF/vLLM) "
            "that render the chat template in-process; the API backend exposes "
            "no chat-template rendering. Use count_tokens_for_messages directly "
            "and handle an unavailable count (None)."
        )

    def count_tokens_for_prompt(self, **kwargs):
        """Debug-only: tokenize a bare string.

        Optional: supported only on local backends (HF/vLLM). The API
        backend has no local tokenizer and raises NotImplementedError.
        """
        raise NotImplementedError(
            "count_tokens_for_prompt is supported only on local backends (HF/vLLM) "
            "that have a local tokenizer; the API backend has no local tokenizer. "
            "Use count_tokens_for_messages on messages."
        )

    @abstractmethod
    def add_stop_strings(self, stop_strings):
        pass

    @abstractmethod
    def reset_stop_strings(self):
        pass

    @abstractmethod
    def count_tokens_for_messages(self, **kwargs):
        """Unified token-count contract: messages in, int or None out."""
        pass

    @abstractmethod
    def get_params(self):
        pass

    @abstractmethod
    def get_model_context_len(self):
        """Deployed model context length (engine ceiling / HF config /
        server max_model_len), possibly overridden via --model_context_len."""
        pass

    @property
    @abstractmethod
    def reasoning_config(self):
        """ReasoningConfig (model_kind, max_new_tokens_reasoning, fmt)."""
        pass

class SimpleFewShotHFTask(Task):
    @abstractmethod
    def dataset_args(self) -> Dict:
        pass

    @abstractmethod
    def test_split_name(self) -> str:
        pass

    @abstractmethod
    def prompt_split_name(self) -> str:
        pass

    @abstractmethod
    def create_messages(self, **kwargs) -> List[Dict]:
        pass

    def prompt_dataset_start_idx(self) -> int:
        return 0

    def load_dataset(self, model: BaseLLM, max_prompt_len: int, max_sample_per_dataset: int, few_shot_count: int) -> Tuple[List[Dict], List[Dict]]:
        self.require_model_method(model)

        samples = self._load_dataset(model, max_prompt_len, max_sample_per_dataset, few_shot_count)
        messages = [{'messages': s['messages']} for s in samples]
        samples = [{'sample': s['sample']} for s in samples]

        if self.method == 'calculate_tokens_proba':
            for m in messages:
                m['tokens_of_interest'] = self.choices
        return messages, samples
    
    def _load_dataset(self, model: BaseLLM, max_prompt_len: int, max_sample_per_dataset: int, few_shot_count: int) -> List:
        samples = []
        dataset = load_dataset(**self.dataset_args())
        test_dataset = dataset[self.test_split_name()]
        prompt_dataset = dataset[self.prompt_split_name()]
        
        prompt_dataset_sample_ids = list(range(self.prompt_dataset_start_idx(), min(self.prompt_dataset_start_idx() + few_shot_count, len(prompt_dataset))))
        if self.test_split_name() == self.prompt_split_name():
            prompt_dataset_sample_ids_set = set(prompt_dataset_sample_ids)
            test_dataset_sample_ids = [
                i for i in range(len(test_dataset))
                if i not in prompt_dataset_sample_ids_set
            ][:max_sample_per_dataset]
        else:
            test_dataset_sample_ids = list(
                range(min(max_sample_per_dataset, len(test_dataset)))
            )
            
        test_dataset = test_dataset.select(test_dataset_sample_ids)
        prompt_dataset = prompt_dataset.select(prompt_dataset_sample_ids)
        for sample in tqdm(test_dataset):
            samples.append({'messages': self._prepare_messages(sample, model, max_prompt_len, few_shot_count, prompt_dataset), 'sample': sample})
        return samples
        
    def _prepare_messages(self, sample: Dict, model: BaseLLM, max_prompt_len: int, few_shot_count: int, prompt_dataset: Dataset) -> List:
        k = min(few_shot_count, len(prompt_dataset))

        zero_shot_messages = normalize_message_roles(
            self.create_messages(copy.deepcopy(sample), with_answer=False)
        )
        zero_shot_messages_len = model.count_tokens_for_messages(zero_shot_messages)
        ensure_prompt_fits(
            zero_shot_messages_len, max_prompt_len, self.run_name()
        )

        message_groups = [
            normalize_message_roles(
                self.create_messages(
                    copy.deepcopy(prompt_dataset[i]), with_answer=True
                )
            )
            for i in range(k-1, -1, -1)
        ]
        message_groups.append(zero_shot_messages)
        
        for i in range(k):
            messages = []
            for group in message_groups[i:]:
                messages += group
            few_shot_messages_len = model.count_tokens_for_messages(messages)
            if few_shot_messages_len is None or few_shot_messages_len <= max_prompt_len:
                return messages
        else:
            return zero_shot_messages
