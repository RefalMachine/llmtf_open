import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Any
import logging
import json

from llmtf.metrics import mean

from .utils import Message, prompt_to_str, EvalLogger
from .optimizable_task import OptimizableTask

logger = logging.getLogger(__name__)

class Evaluator(ABC):
    def __init__(self, task: OptimizableTask):
        self.task = task
    
    @abstractmethod
    def evaluate(
        self,
        prompts: List[Message],
        samples: List[Dict],
        all_preds: List[List[str]],
        model_names: List[str],
        suffix: str
    ) -> Tuple[float, str]:
        pass


class FeedbackGenerator(ABC):
    @abstractmethod
    def get_feedback(
        self,
        prompt: List[Message],
        sample: Dict,
        y_pred: str
    ) -> str:
        pass

    @abstractmethod
    def aggrigate_feedback(
        self,
        feedbacks: List[str]
    ) -> str:
        pass


FEEDBACK_PROMPTS = {
    "ru": """Промпт:
\"\"\"
{prompt}
\"\"\"
Сгенерированный ответ:
\"\"\"
{y_pred}
\"\"\"
Правильный ответ:
\"\"\"
{y_gold}
\"\"\"
""",
    "en": """Prompt:
\"\"\"
{prompt}
\"\"\"
Generated response:
\"\"\"
{y_pred}
\"\"\"
Correct response:
\"\"\"
{y_gold}
\"\"\"
""",
}

class SimpleFeedbackGenerator(FeedbackGenerator):
    def __init__(self, lang="ru"):
        self.feedback_prompt = FEEDBACK_PROMPTS[lang]
    
    def get_feedback(self, prompt, sample, y_pred):
        y_gold = self.task.get_answer(sample)
        prompt = prompt_to_str(prompt)
        return self.feedback_prompt.format(prompt=prompt, y_pred=y_pred, y_gold=y_gold)

    def aggrigate_feedback(self, feedbacks):
        return "\n\n".join(feedbacks)


class PseudoGradEval(Evaluator):
    def __init__(
        self,
        task: OptimizableTask,
        n_worst: int=5,
        feedback_generator: FeedbackGenerator=SimpleFeedbackGenerator(),
        log_dir="./eval_logs"
    ):
        self.task = task
        self.n_worst = n_worst
        self.feedback_generator = feedback_generator
        self.feedback_generator.task = task
        self.log_dir = log_dir
    
    def evaluate(
        self,
        prompts,
        samples,
        all_preds,
        model_names,
        suffix
    ):
        scores = []
        sample_scores = []
        per_sample_data = [[] for _ in samples]
    
        for model_preds, model_name in zip(all_preds, model_names):
            prompts_m, y_preds_m, infos_m = model_preds
    
            metrics = [self.task.evaluate(sample, y_pred) for sample, y_pred in zip(samples, y_preds_m)]
            model_sample_scores = [
                self.task.evaluate_pred(sample, y_pred, sample_metrics)
                for sample, y_pred, sample_metrics in zip(samples, y_preds_m, metrics)
            ]
            sample_scores.append(model_sample_scores)
    
            for i, (score, y_pred, prompt, info) in enumerate(zip(model_sample_scores, y_preds_m, prompts_m, infos_m)):
                per_sample_data[i].append((score, y_pred, prompt, info))
    
            with EvalLogger(self.log_dir, self.task.run_name(), suffix, model_name) as eval_logger:
                for sample, prompt, y_pred, info, metric, score in \
                        zip(samples, prompts_m, y_preds_m, infos_m, metrics, model_sample_scores):
                    eval_logger.log_sample(sample, prompt, y_pred, info, metric, score)
    
                metrics_res = {}
                for metric in metrics[0].keys():
                    agg_result = self.task.aggregation()[metric]([m[metric] for m in metrics])
                    if isinstance(agg_result, tuple) and len(agg_result) == 2:
                        metrics_res[metric] = agg_result[0]
                    else:
                        metrics_res[metric] = agg_result
    
                leaderboard_score = self.task.leaderboard_aggregation(metrics_res)
                scores.append(leaderboard_score)
                eval_logger.log_results(metrics_res, leaderboard_score)
    
        score = sum(scores) / len(scores)
        mean_sample_scores = [sum(scores_per_sample) / len(scores_per_sample) for scores_per_sample in zip(*sample_scores)]
    
        indices = list(range(len(samples)))
        indices.sort(key=lambda i: mean_sample_scores[i])
        worst_indices = indices[:self.n_worst]
    
        worst_samples_data = []
        for i in worst_indices:
            worst_for_sample = min(per_sample_data[i], key=lambda x: x[0]) 
            score_worst, y_pred_worst, prompt_worst, info_worst = worst_for_sample
            worst_samples_data.append((prompt_worst, samples[i], y_pred_worst))
    
        feedback = self.feedback_generator.aggrigate_feedback([
            self.feedback_generator.get_feedback(prompt, sample, y_pred)
            for prompt, sample, y_pred in worst_samples_data
        ])
    
        return (score, feedback)
