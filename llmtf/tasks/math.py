from math_verify import LatexExtractionConfig, parse, verify
from latex2sympy2_extended import NormalizationConfig
from llmtf.base import Task, SimpleFewShotHFTask, LLM
from tqdm import tqdm
from typing import Dict, List, Tuple
from datasets import load_dataset, Dataset
from llmtf.metrics import mean

def accuracy_reward(completion: str, solution: str) -> float:
    """Reward function that checks if the completion matches the ground truth."""
    # parse the gold solution (assumed to always succeed)
    gold_parsed = parse(solution, extraction_mode="first_match")

    # parse the model’s completion with the same LaTeX extraction settings
    answer_parsed = parse(
        completion,
        extraction_config=[
            LatexExtractionConfig(
                normalization_config=NormalizationConfig(
                    nits=False,
                    malformed_operators=False,
                    basic_latex=True,
                    boxed="all",
                    units=True,
                )
            )
        ],
        extraction_mode="first_match",
    )

    # verify and return binary reward; on error, print and give 0.0
    try:
        return float(verify(gold_parsed, answer_parsed))
    except Exception as e:
        print(f"verify failed: {e}, answer: {answer_parsed}, gold: {gold_parsed}")
        return 0.0
    

class DOoM(SimpleFewShotHFTask):
    def __init__(self, domain, system_prompt, instruction, max_new_tokens=30000, **kwargs):
        super().__init__(**kwargs)
        self.domain = domain
        assert self.domain in ['math', 'phys']
        self._max_new_tokens = max_new_tokens
        self.system_prompt = system_prompt
        self.instruction = instruction
        self.method = 'generate'

    def evaluate(self, sample, y_pred) -> Dict:
        key = 'answer' if self.domain == 'phys' else 'short answer'
        if type(y_pred) == str:
            acc = accuracy_reward(y_pred, sample[key])
        elif type(y_pred) == list:
            acc = mean([accuracy_reward(p, sample[key]) for p in y_pred])
            
        return {"acc": acc}

    def name(self) -> str:
        return f"DOoM_{self.domain}"

    def aggregation(self) -> Dict:
        return {"acc": mean}

    def dataset_args(self) -> Dict:
        if self.domain == 'math':
            return {'path': 'Vikhrmodels/russian_math'}
        elif self.domain == 'phys':
            return {'path': 'Vikhrmodels/russian_physics'}

    def test_split_name(self) -> str:
        return 'train'
    
    def prompt_split_name(self) -> str:
        return 'train'
    
    def _load_dataset(self, model: LLM, max_len: int, max_sample_per_dataset: int, few_shot_count: int) -> List:
        samples = []
        dataset = load_dataset(**self.dataset_args())
        test_dataset = dataset[self.test_split_name()]
        test_dataset = test_dataset.select(range(min(max_sample_per_dataset, len(test_dataset))))
        for sample in tqdm(test_dataset):
            samples.append({'messages': self._prepare_messages(sample, model, max_len, few_shot_count, []), 'sample': sample})
        return samples

    def create_messages(self, sample, with_answer):
        messages = []
        instruction_user = self.instruction.replace('{task}', sample['task'].strip())

        messages.append({'role': 'system', 'content': self.system_prompt})
        messages.append({'role': 'user', 'content': instruction_user})

        return messages
    
class TMath(SimpleFewShotHFTask):
    def __init__(self, system_prompt, instruction, max_new_tokens=30000, **kwargs):
        super().__init__(**kwargs)
        self._max_new_tokens = max_new_tokens
        self.system_prompt = system_prompt
        self.instruction = instruction
        self.method = 'generate'

    def evaluate(self, sample, y_pred) -> Dict:
        key = 'verifiable_answer'
        if type(y_pred) == str:
            acc = accuracy_reward(y_pred, sample[key])
        elif type(y_pred) == list:
            acc = mean([accuracy_reward(p, sample[key]) for p in y_pred])
        return {"acc": acc}

    def name(self) -> str:
        return f"T-math"

    def aggregation(self) -> Dict:
        return {"acc": mean}

    def dataset_args(self) -> Dict:
        return {'path': 't-tech/T-math'}

    def test_split_name(self) -> str:
        return 'train'
    
    def prompt_split_name(self) -> str:
        return 'train'
    
    def _load_dataset(self, model: LLM, max_len: int, max_sample_per_dataset: int, few_shot_count: int) -> List:
        samples = []
        dataset = load_dataset(**self.dataset_args())
        test_dataset = dataset[self.test_split_name()]
        test_dataset = test_dataset.select(range(min(max_sample_per_dataset, len(test_dataset))))
        for sample in tqdm(test_dataset):
            samples.append({'messages': self._prepare_messages(sample, model, max_len, few_shot_count, []), 'sample': sample})
        return samples

    def create_messages(self, sample, with_answer):
        messages = []
        instruction_user = self.instruction.replace('{task}', sample['question'].strip())

        messages.append({'role': 'system', 'content': self.system_prompt})
        messages.append({'role': 'user', 'content': instruction_user})

        return messages