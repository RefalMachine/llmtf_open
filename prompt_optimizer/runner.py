import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from typing import List, Tuple
from tqdm import tqdm
import torch.multiprocessing as mp

from llmtf.base import LLM

from .utils import Message
from .optimizable_task import OptimizableTask

class Runner:
    def __init__(
        self,
        models: List[LLM],
        task: OptimizableTask,
        batch_size: int
    ):
        self.models = models
        self.task = task
        self.batch_size = batch_size

    def run_all(
        self,
        messages: List[List[Message]]
    ) -> List[List[Tuple]]:
        with mp.Manager() as manager:
            results_dict = manager.dict()
            processes = []
            
            def run_model_wrapper(idx, model, messages, results_dict):
                result = self.run_model(model, messages)
                results_dict[idx] = result
            
            for idx, model in enumerate(self.models):
                p = mp.Process(
                    target=run_model_wrapper,
                    args=(idx, model, messages, results_dict)
                )
                processes.append(p)
                p.start()
            
            for p in processes:
                p.join()
            
            ordered_results = [results_dict[i] for i in range(len(self.models))]
            return ordered_results
    
    def run_model(
        self,
        model: LLM,
        messages: List[List[Message]],
        **kwargs
    ):
        model.add_stop_strings(self.task.additional_stop_strings)
        saved_max_new_tokens = model.generation_config.max_new_tokens
        model.generation_config.max_new_tokens = self.task._max_task_new_tokens
        prompts, y_preds, infos = [], [], []
        for i in tqdm(range(0, len(messages), self.batch_size), desc=f'Generating'):
            messages_batch = messages[i:i+self.batch_size]
            if self.task.method == 'generate':
                _prompts, _y_preds, _infos = model.generate_batch(
                    messages_batch, **kwargs, **self.task.method_additional_args)
            elif self.task.method == 'calculate_tokens_proba':
                _prompts, _y_preds, _infos = model.calculate_tokens_proba_batch(
                    messages_batch, **kwargs, **self.task.method_additional_args)
            prompts += _prompts
            y_preds += _y_preds
            infos += _infos
        model.reset_stop_strings()
        model.generation_config.max_new_tokens = saved_max_new_tokens
        return prompts, y_preds, infos
