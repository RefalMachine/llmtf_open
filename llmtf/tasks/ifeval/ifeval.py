import json
from typing import Dict, List, Tuple
from collections import defaultdict
import re
from datasets import load_dataset, Dataset
from tqdm import tqdm
import os
from llmtf.base import SimpleFewShotHFTask, LLM
from llmtf.metrics import mean
from pathlib import Path
import sys
import copy
import nltk

from . import ru_instructions_registry
from . import en_instructions_registry

class RuIFEvalTask(SimpleFewShotHFTask):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.method = 'generate'
        self._max_task_new_tokens = 1024  # Allow for longer responses
        self.instruction_registry = ru_instructions_registry
        try: nltk.data.find('tokenizers/punkt_tab')
        except LookupError: nltk.download('punkt_tab')

    def task_name(self) -> str:
        return 'ifeval/ruIFEval'

    def dataset_args(self) -> Dict:
        return {'path': str(Path(__file__).parent / 'ifeval_data' / 'ruIFEval_v0.1.jsonl')}

    def test_split_name(self) -> str:
        return 'test'

    def prompt_split_name(self) -> str:
        return 'prompt'

    def create_messages(self, sample, with_answer=None) -> List[Dict]:
        """Format the prompt with instruction and response."""
        prompt = sample['prompt']
        messages = [{'role': 'user', 'content': prompt}]
        return messages

    def evaluate(self, sample, y_pred) -> Dict:
        if type(y_pred) == str:
            y_pred = [y_pred]
        """Evaluate response using ruIFEval's instruction checking logic."""
        # Get instruction IDs and kwargs from sample
        instruction_ids = sample['instruction_id_list']
        kwargs_list = sample['kwargs']
        
        # Initialize metrics and tracking structures
        metrics = {
            'prompt_level_accuracy': 0,
            'instruction_accuracy': {'instruction_followed': [], 'instruction_id': []},
            'category_accuracy': {'category_followed': {}, 'category_total': {}}
        }

        # Check each instruction
        is_following_list = []
        for idx, instruction_id in enumerate(instruction_ids):
            instruction_cls = self.instruction_registry.INSTRUCTION_DICT[instruction_id]
            instruction = instruction_cls(instruction_id)
            instruction.build_description(**kwargs_list[idx])
            args = instruction.get_instruction_args()
            if args and "prompt" in args:
                instruction.build_description(prompt=sample['prompt'])

            # Check if instruction is followed
            is_following = mean([p.strip() and instruction.check_following(p) for p in y_pred])
            is_following_list.append(is_following > 0.5)
            
            # Track individual instruction results
            metrics['instruction_accuracy']['instruction_followed'].append(is_following)
            metrics['instruction_accuracy']['instruction_id'].append(instruction_id)

            # Track category results
            category = instruction_id.split(':')[0] if ':' in instruction_id else 'other'
            if category not in metrics['category_accuracy']['category_total']:
                metrics['category_accuracy']['category_total'][category] = 0

            if category not in metrics['category_accuracy']['category_followed']:
                metrics['category_accuracy']['category_followed'][category] = 0

            metrics['category_accuracy']['category_total'][category] += 1
            if is_following:
                metrics['category_accuracy']['category_followed'][category] += float(is_following)

        # Calculate aggregate metrics
        metrics['prompt_level_accuracy'] = all(is_following_list)
        metrics['instruction_level_accuracy'] = copy.deepcopy(metrics['instruction_accuracy'])
        return metrics
    
    def aggregation(self) -> Dict:
        """Define how to aggregate metrics across samples."""
        return {
            'prompt_level_accuracy': mean,
            'instruction_level_accuracy': self._aggregate_instruction_level_accuracy,
            'instruction_accuracy': self._aggregate_instruction_accuracy,
            'category_accuracy': self._aggregate_category_accuracy
        }

    def leaderboard_aggregation(self, metrics: Dict) -> float:
        return metrics['prompt_level_accuracy']
    
    def _aggregate_instruction_accuracy(self, results):
        """Calculate accuracy per instruction type."""
        instruction_stats = defaultdict(lambda: {'followed': 0, 'total': 0})
        for result in results:
            # Zip instruction IDs with their follow status (0/1)
            for instr_id, is_followed in zip(result['instruction_id'], result['instruction_followed']):
                instruction_stats[instr_id]['followed'] += is_followed
                instruction_stats[instr_id]['total'] += 1
        
        return {k: v['followed']/v['total'] for k, v in instruction_stats.items()}

    def _aggregate_category_accuracy(self, results):
        """Calculate accuracy per instruction category."""
        category_stats = defaultdict(lambda: {'followed': 0, 'total': 0})
        for result in results:
            for cat, count in result['category_followed'].items():
                category_stats[cat]['followed'] += count
            for cat, count in result['category_total'].items():
                category_stats[cat]['total'] += count
        
        return {k: v['followed']/v['total'] for k, v in category_stats.items()}

    def _aggregate_instruction_level_accuracy(self, results):
        """Calculate overall instruction-level accuracy."""
        total_correct = 0
        total_instructions = 0
        for result in results:
            total_correct += sum(result['instruction_followed'])
            total_instructions += len(result['instruction_followed'])
        return total_correct / total_instructions if total_instructions > 0 else 0

    def _load_dataset(self, model: LLM, max_len: int, max_sample_per_dataset: int, few_shot_count: int) -> Tuple[List[Dict], List[Dict]]:
        """Load and prepare ruIFEval dataset from JSONL files."""
        data_path = Path(self.dataset_args()['path'])
        
        samples = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                sample = json.loads(line)
                samples.append(sample)
                if len(samples) >= max_sample_per_dataset:
                    break
        
        # Split into few-shot prompts and test samples
        test_samples = samples[:max_sample_per_dataset]
        
        # Prepare messages for model
        samples = []
        for sample in test_samples:
            samples.append({
                'messages': self.create_messages(sample),
                'sample': sample
            })
            
        return samples

class EnIFEvalTask(RuIFEvalTask):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.method = 'generate'
        self._max_new_tokens = 1024
        self.instruction_registry = en_instructions_registry

    def task_name(self) -> str:
        return 'ifeval/enIFEval'

    def dataset_args(self) -> Dict:
        return {'path': str(Path(__file__).parent / 'ifeval_data' / 'enIFEval.jsonl')}