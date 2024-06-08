import abc
from abc import abstractmethod
import logging
from typing import Dict, List, Tuple
import copy
import os

os.environ['VLLM_ATTENTION_BACKEND'] = 'XFORMERS'

logger = logging.getLogger('llmtf')
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(levelname)s: %(asctime)s: %(name)s: %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


class Task(abc.ABC):
    def __init__(self):
        self.backend_logger = None
        self._max_new_tokens = None
        self.additional_stop_tokens = []

    @property
    def logger(self):
        if self.backend_logger is None:
            self.init_logger()
        return self.backend_logger

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

    def init_logger(self):
        self.backend_logger = logging.getLogger(__name__ + '.' + self.name)

class LLM(abc.ABC):
    def __init__(self):
        self.backend_logger = None

    @property
    def logger(self):
        if self.backend_logger is None:
            self.init_logger()
        return self.backend_logger

    def init_logger(self):
        self.backend_logger = logging.getLogger(__name__ + '.llm')

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