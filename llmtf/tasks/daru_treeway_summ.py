from llmtf.base import Task, SimpleFewShotHFTask, LLM
from sklearn.metrics import matthews_corrcoef
from tqdm import tqdm
from typing import Dict, List, Tuple
from datasets import load_dataset, Dataset
import copy
from llmtf.metrics import mean, rouge1, rouge2, r_precision
import pandas as pd
import numpy as np
from sklearn.metrics import average_precision_score

class DaruTreewayAbstractive(SimpleFewShotHFTask):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.method = 'generate'
        self._max_new_tokens = 512
        #self.additional_stop_strings.append('\n')
        #self.additional_stop_strings.append('\n\n')

    def evaluate(self, sample, y_pred) -> Dict:
        y_true = ' '.join(sample['summary_sents'])
        return {"rouge1": rouge1(y_true, y_pred).fmeasure, "rouge2": rouge2(y_true, y_pred).fmeasure}

    @classmethod
    def name(cls):
        return 'daru/treewayabstractive'

    def aggregation(self) -> Dict:
        return {"rouge1": mean, "rouge2": mean}

    def dataset_args(self) -> Dict:
        return {'path': 'dichspace/daru_treeway_eval'}

    def test_split_name(self) -> str:
        return 'test'

    def prompt_split_name(self) -> str:
        return 'prompt'

    def create_messages(self, sample, with_answer):
        messages = []
        instruction_user = 'Напиши краткую аннотацию на русском языке к следующему тексту.\nТекст: {text}'
        instruction_bot = 'TL;DR: {summary}'
        instruction_bot_incomplete = 'TL;DR:'

        summary = ' '.join(sample['summary_sents'])
        text = ' '.join(sample['src_sents'])

        bot_content = instruction_bot.replace('{summary}', summary) if with_answer else instruction_bot_incomplete

        messages.append({'role': 'user', 'content': instruction_user.replace('{text}', text)})
        messages.append({'role': 'bot', 'content': bot_content})

        return messages

    def prompt_dataset_start_idx(self) -> int:
        return 0
    
    def get_answer(self, sample):
        return ' ' + ' '.join(sample['summary_sents'])

class DaruTreewayExtractive(Task):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.method = 'calculate_logsoftmax'
        self._max_new_tokens = 1

    @classmethod
    def name(cls):
        return 'daru/treewayextractive'

    def evaluate(self, sample, y_pred) -> Dict:
        return {'r-prec': {'label': sample['label'], 'group_id': sample['group_id'], 'pred': y_pred}}

    def aggregation(self) -> Dict:
        def map_score(data: List[Dict]):
            df = pd.DataFrame()
            group_idx = [d['group_id'] for d in data]
            labels = [d['label'] for d in data]
            scores = [[t[1] for t in d['pred'][-1]['tokens']] for d in data]

            df['id'] = group_idx
            df['labels'] = labels
            df['scores'] = scores

            score = []
            for i, group in df.groupby('id'):    
                min_len = min([len(s) for s in group['scores']])
                scores_group = [np.mean(s[:min_len]) for s in group['scores']]
                group['score'] = scores_group
                ap = r_precision(group['labels'], group['score'])
                score.append(ap)
            score = np.mean(score)
            return score
        return {'r-prec': map_score}

    def dataset_args(self) -> Dict:
        return {'path': 'dichspace/daru_treeway_eval'}

    def test_split_name(self) -> str:
        return 'test'

    def prompt_split_name(self) -> str:
        return 'prompt'

    def load_dataset(self, model: LLM, max_len: int, max_sample_per_dataset: int, few_shot_count: int) -> Tuple[List[Dict], List[Dict]]:
        assert model.support_method(self.method)

        samples = self._load_dataset(model, max_len, max_sample_per_dataset)
        messages = [{'messages': s['messages']} for s in samples]
        samples = [{'sample': s['sample']} for s in samples]

        return messages, samples

    def _load_dataset(self, model: LLM, max_len: int, max_sample_per_dataset: int) -> List:
        samples = []
        dataset = load_dataset(**self.dataset_args())
        test_dataset = dataset[self.test_split_name()]

        hardcoded_max_dataset_len = min(500, len(test_dataset))
        test_dataset = test_dataset.select(range(min(max_sample_per_dataset, hardcoded_max_dataset_len)))
        for i, sample in tqdm(enumerate(test_dataset)):
            messages_group, samples_group = self._prepare_messages(sample, model, max_len)
            for j in range(len(messages_group)):
                samples_group[j]['group_id'] = i
                samples_group[j]['label'] = j in sample['extractive_summary']
                samples.append({'messages': messages_group[j], 'sample': samples_group[j]})
        return samples
        
    def _prepare_messages(self, sample: Dict, model: LLM, max_len: int) -> List:
        messages_group = []
        samples_group = []
        for sentence in sample['src_sents']:
            s = copy.deepcopy(sample)
            s['sentence'] = sentence
            zero_shot_messages = self.create_messages(s)
            zero_shot_messages_len = model.count_tokens_for_prompt(model.apply_model_prompt(zero_shot_messages))
            if zero_shot_messages_len >= max_len:
                self.logger.warning(f'WARNING: sample zero-shot len {zero_shot_messages_len} greater then {max_len}. Will be truncated.')

            messages_group.append(zero_shot_messages)
            samples_group.append(s)

        return messages_group, samples_group

    def create_messages(self, sample: Dict):
        user_content = 'Твоя задача определить наиболее важные предложения.\nТекст: ' + ' '.join(sample['src_sents'])
        bot_content = 'Наиболее важные предложения статьи: ' + sample['sentence']
        messages = [{'role': 'user', 'content': user_content}, {'role': 'bot', 'content': bot_content}]
        return messages