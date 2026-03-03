from typing import Optional
from abc import ABC, abstractmethod
import logging
from .solution_database import Solution, SolutionDatabase
import random
import bisect

logger = logging.getLogger(__name__)

class Sampler(ABC):
    def __init__(
        self,
        solution_database: Optional[SolutionDatabase]=None
    ):
        self.solution_database = solution_database

    def set_solution_database(self, solution_database: SolutionDatabase):
        self.solution_database = solution_database
    
    @abstractmethod
    def sample(self) -> Solution:
        pass


class PowerLawSampler(Sampler):
    def __init__(
        self,
        alpha: float=1.0,
        solution_database: Optional[SolutionDatabase]=None
    ):
        super().__init__(solution_database)
        self.alpha = alpha

    def sample(self):
        solutions = self.solution_database.solutions.items
        n = len(solutions)
        if n == 1:
            return solutions[0]
        
        cdf = [0] * n
        total = 0
        
        for i in range(n):
            total += (n - 1) ** (-self.alpha)
            cdf[i] = total
        
        r = random.uniform(0, total)
        idx = bisect.bisect_left(cdf, r)
        
        logger.info(f"sampled solution (id: {solutions[idx].id})")
        return solutions[idx]
