import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from typing import List, Dict, Tuple, Optional
from tqdm import tqdm

from llmtf.base import LLM

from .optimizable_task import OptimizableTask
from .runner import Runner
from .solution_database import Solution, SolutionDatabase
from .sampler import Sampler
from .evaluator import Evaluator
from .mutator import Mutator
from .utils import Message, OptimizerLogger

class Optimizer():
    def __init__(
        self,
        task: OptimizableTask,
        models: List[LLM],
        sampler: Sampler,
        evaluator: Evaluator,
        mutator: Mutator,
        model_names: Optional[List[str]]=None,
        solution_database: Optional[SolutionDatabase]=None,
        batch_size: int=1,
        few_shot_count: int=0,
        max_sample_per_dataset: int=100000000,
        log_dir="./eval_logs"
    ):
        self.task = task
        self.models = models
        if model_names:
            assert len(models) == len(model_names), "lists of models and model names must have the same length"
            self.model_names = model_names
        else:
            self.model_names = list(map(lambda x: f"model №{x+1}", list(range(len(models)))))
        self.solution_database = solution_database if solution_database else SolutionDatabase()
        self.sampler = sampler
        self.sampler.set_solution_database(self.solution_database)
        self.runner = Runner(
            models,
            task,
            batch_size
        )
        self.evaluator = evaluator
        self.mutator = mutator
        self.mutator.set_task_description(self.task.task_description)
        self.few_shot_count = few_shot_count
        self.train_samples, self.test_samples = self.task.load_samples(max_sample_per_dataset)
        self.logger = OptimizerLogger(
            log_dir,
            self.task.run_name().replace('/', '_')
        )
        

    def run_root_solution(self, initial_prompt_template: List[Message]) -> Solution:
        root_solution = self.run_prompt(initial_prompt_template, None, "iter_0", "train")
        self.solution_database.add_root(root_solution)
        return root_solution
    
    #TODO: calculate token prob choices
    def run_prompt(
        self,
        prompt_template: List[Message],
        parent: Optional[Solution],
        suffix: str,
        split: str
    ) -> Solution:
        samples = self.train_samples.select(range(self.few_shot_count, len(self.train_samples))) if split == "train" else self.test_samples
        few_shot_samples = self.train_samples.select(range(self.few_shot_count))
        
        few_shot_messages = []
        for sample in few_shot_samples:
            few_shot_messages += self.task.fill_prompt(prompt_template, sample, True)
        messages = [few_shot_messages + self.task.fill_prompt(prompt_template, sample, False) \
                    for sample in samples]
        all_preds = self.runner.run_all(messages)
        
        score, feedback = self.evaluator.evaluate(
            messages, samples, all_preds, self.model_names, suffix)
        if parent:
            parent = [parent]
        return Solution(prompt_template, score, feedback, parent)
    
    def run(
            self,
            score_target: float=float('inf'),
            max_iterations: int=100,
            ) -> Solution:
        with self.logger as logger:
            for i in tqdm(range(max_iterations), desc="Optimization"):
                try:
                    solution = self.sampler.sample()
                    new_prompt = self.mutator.mutate(solution)
                    new_solution = self.run_prompt(new_prompt, solution, f"iter_{i+1}", "train")

                    logger.log_iteration(i, new_solution.score, new_solution.prompt)
                    self.solution_database.add_solution(new_solution)
    
                    if self.solution_database.best_solution.score >= score_target:
                        break
                except Exception as e:
                    logger.log_error(i, e)
        return self.solution_database.best_solution
