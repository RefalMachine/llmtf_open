from typing import Optional, Iterable, List
import logging
import random
import bisect
from .utils import Message, OrientedGraphNode, SortedList

logger = logging.getLogger(__name__)

class Solution(OrientedGraphNode):
    _id_counter: int = 0

    def __init__(
            self,
            prompt: List[Message],
            score: float,
            feedback: str,
            parents: Optional[Iterable]=None,
        ):
        super().__init__(parents, None) # evolution graph is acyclic
        self.prompt = prompt
        self.score = score
        self.feedback = feedback


class SolutionDatabase:
    def __init__(
            self,
            root_solution: Optional[Solution]=None,
            ):
        solutions = SortedList()
        if root_solution:
            self.root_solution = root_solution
            self.best_solution = root_solution
            solutions.add(root_solution, (root_solution.score, root_solution.id))
        self.solutions = solutions

    def add_root(self, root_solution: Solution) -> None:
        self.root_solution = root_solution
        self.best_solution = root_solution
        self.solutions.add(root_solution, (root_solution.score, root_solution.id))
    
    def add_solution(self, solution: Solution) -> None:        
        self.solutions.add(solution, (solution.score, solution.id))
        logger.info(f"added solution (id: {solution.id})")
        if self.best_solution.score < solution.score:
            self.best_solution = solution
            logger.info(f"it is a new best solution! (score: {solution.score})")
