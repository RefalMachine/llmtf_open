import abc
from abc import abstractmethod
import logging
from typing import Dict, List, Tuple
import copy
import os
from datasets import load_dataset, Dataset
from tqdm import tqdm
from llmtf.metrics import mean


os.environ['VLLM_ATTENTION_BACKEND'] = 'XFORMERS'
os.environ['VLLM_USE_V1'] = '0'

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
    def __init__(self, name_suffix=None, **kwargs):
        super().__init__(**kwargs)
        self.method_additional_args = {}
        self._max_new_tokens = None
        self.additional_stop_strings = []
        self.name_suffix = ''
        if name_suffix is not None:
            self.name_suffix = f'_{name_suffix}'

    def run_name(self):
        return self.task_name() + self.name_suffix
    
    @property
    def max_new_tokens(self):
        if self._max_new_tokens is None:
            self.logger.error('self._max_new_tokens is None. Every task must set _max_new_tokens parameter')
            raise Exception('self._max_new_tokens is None. Every task must set _max_new_tokens parameter')
        return self._max_new_tokens

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

class LLM(Base):
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

    @abstractmethod
    def apply_model_prompt(self, **kwargs):
        pass

    @abstractmethod
    def support_method(self, **kwargs):
        pass
    
    @abstractmethod
    def count_tokens_for_prompt(self, **kwargs):
        pass

    @abstractmethod
    def get_params(self):
        pass

    @abstractmethod
    def get_max_model_len(self):
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

    def load_dataset(self, model: LLM, max_len: int, max_sample_per_dataset: int, few_shot_count: int) -> Tuple[List[Dict], List[Dict]]:
        assert model.support_method(self.method)

        samples = self._load_dataset(model, max_len, max_sample_per_dataset, few_shot_count)
        messages = [{'messages': s['messages']} for s in samples]
        samples = [{'sample': s['sample']} for s in samples]

        if self.method == 'calculate_tokens_proba':
            for m in messages:
                m['tokens_of_interest'] = self.choices
        return messages, samples
    
    def _load_dataset(self, model: LLM, max_len: int, max_sample_per_dataset: int, few_shot_count: int) -> List:
        samples = []
        dataset = load_dataset(**self.dataset_args())
        test_dataset = dataset[self.test_split_name()]
        prompt_dataset = dataset[self.prompt_split_name()]
        
        test_dataset_sample_ids = list(range(min(max_sample_per_dataset, len(test_dataset))))
        prompt_dataset_sample_ids = list(range(self.prompt_dataset_start_idx(), min(self.prompt_dataset_start_idx() + few_shot_count, len(prompt_dataset))))
        if self.test_split_name() == self.prompt_split_name():
            prompt_dataset_sample_ids_set = set(prompt_dataset_sample_ids)
            test_dataset_sample_ids = [i for i in test_dataset_sample_ids if i not in prompt_dataset_sample_ids_set]
            
        test_dataset = test_dataset.select(test_dataset_sample_ids)
        prompt_dataset = prompt_dataset.select(prompt_dataset_sample_ids)
        for sample in tqdm(test_dataset):
            samples.append({'messages': self._prepare_messages(sample, model, max_len, few_shot_count, prompt_dataset), 'sample': sample})
        return samples
        
    def _prepare_messages(self, sample: Dict, model: LLM, max_len: int, few_shot_count: int, prompt_dataset: Dataset) -> List:
        k = min(few_shot_count, len(prompt_dataset))

        zero_shot_messages = self.create_messages(copy.deepcopy(sample), with_answer=False)
        zero_shot_messages_len = model.count_tokens_for_prompt(model.apply_model_prompt(zero_shot_messages))
        if zero_shot_messages_len >= max_len:
            self.logger.warning(f'WARNING: sample zero-shot len {zero_shot_messages_len} greater then {max_len}. Will be truncated.')

        message_groups = [self.create_messages(copy.deepcopy(prompt_dataset[i]), with_answer=True) for i in range(k-1, -1, -1)]
        message_groups.append(zero_shot_messages)
        
        for i in range(k):
            messages = []
            for group in message_groups[i:]:
                messages += group
            few_shot_messages_len = model.count_tokens_for_prompt(model.apply_model_prompt(messages))
            if few_shot_messages_len < max_len:
                return messages
        else:
            return zero_shot_messages
