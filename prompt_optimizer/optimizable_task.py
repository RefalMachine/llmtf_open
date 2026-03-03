import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from typing import List, Dict, Tuple
from abc import abstractmethod
from datasets import load_dataset

from llmtf.base import Task

from .utils import Message

class OptimizableTask():
    def __init__(self):
        assert hasattr(self, "task_description"), "Optimizable task has no `task_description` attribute"

    @abstractmethod
    def load_samples(
        self,
        max_sample_per_dataset: int=100000000
    ) -> Tuple[List[Dict], List[Dict]]:
        pass

    @abstractmethod
    def evaluate_pred(
        self,
        sample: Dict,
        y_pred: str
    ) -> float:
        pass

    @abstractmethod
    def fill_prompt(
        self,
        prompt: List[Message],
        sample: Dict,
        with_answer: bool
    ) -> List[Message]:
        pass


class SimpleOptimizableTask(OptimizableTask):    
    def load_samples(self):
        dataset = load_dataset(**self.dataset_args())
        return dataset[self.prompt_split_name()]

    def evaluate_pred(self, sample, y_pred, metrics):
        score = 0
        n = len(metrics.keys())
        for k, v in metrics.items():
            if k in ["f1-macro", "f1-micro", "mcc"] and isinstance(v, tuple):
                if len(v) == 2:
                    score += v[0] == v[1]
                else:
                    tp = sum(v[0].values())
                    fn = sum(v[1].values())
                    fp = sum(v[2].values())
                    score += 2 * tp / (2 * tp + fn + fp) if 2 * tp + fn + fp > 0 else 0
            elif isinstance(v, float):
                score += v
            else:
                n -= 1
        score /= n
        return score
