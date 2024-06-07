import abc
from abc import abstractmethod
import logging
from typing import Dict, List, Tuple
import copy
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class Task(abc.ABC):
    def __init__(self):
        self.init_logger()
        self.additional_stop_tokens = []

    @property
    def logger(self):
        return self.backend_logger

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
        logging.basicConfig(level=logging.INFO)
        self.backend_logger = logging.getLogger(__name__ + '/' + self.name)