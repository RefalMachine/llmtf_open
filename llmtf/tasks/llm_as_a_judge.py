from llmtf.base import Task, LLM
from typing import List, Dict, Tuple
import codecs
import json
import copy
import pandas as pd
import numpy as np
from evalica import bradley_terry, Winner, pairwise_frame
from scipy.special import expit

def read_json(file_name):
    with open(file_name, encoding="utf-8") as r:
        return json.load(r)
    
def swap(data, key1, key2):
    tmp = data[key1]
    data[key1] = data[key2]
    data[key2] = tmp

class LLMAsJudge(Task):
    def __init__(self, model_outputs: Dict, references_outputs: List[Dict], custom_instruction=None, len_control=True, pen_only_model=False, **kwargs):
        super().__init__(**kwargs)
        self.model_outputs = model_outputs
        self.references_outputs = references_outputs
        self.custom_instruction = custom_instruction
        self.method = 'calculate_tokens_proba'
        self.len_control = len_control
        self.pen_only_model = pen_only_model
        self._max_new_tokens = 1

    @property
    def choices(self) -> List:
        return ["m", "M"]
    
    @classmethod
    def name(cls):
        return 'llm_as_judge'

    def _calculate_score(self, data, model_name):
        # taken from https://github.com/VikhrModels/ru_llm_arena/blob/master/show_result.py
        df = pd.DataFrame(data, columns=['model_a', 'model_b', 'winner', 'answer_len_delta'])
        df['winner'] = df['winner'].map({
                1: Winner.X,
                0: Winner.Y
        })

        result = bradley_terry(
            df['model_a'],
            df['model_b'],
            df['winner'],
            weights=df['answer_len_delta'] * 2,
            tolerance=1e-8
        )
        
        df = pairwise_frame(result.scores)
        np.fill_diagonal(df.values, np.nan)
        return df.loc[model_name].mean()

    def _confident_score_mean(self, results: Dict) -> Dict:
        full_hash =  ['|'.join([str(r['id']), r['model_name'], r['reference_model_name']]) for r in results]
        model_hash = ['|'.join([str(r['id']), r['model_name']]) for r in results]
        reference_hash = ['|'.join([str(r['id']), r['reference_model_name']]) for r in results]

        df = pd.DataFrame()
        df['p'] = [r['p'] for r in results]
        df['full_hash'] = full_hash
        df['model_hash'] = model_hash
        df['reference_hash'] = reference_hash
        
        df['model_name'] = [r['model_name'] for r in results]
        df['reference_model_name'] = [r['reference_model_name'] for r in results]

        df['model_len'] = [r['lens']['model'] for r in results]
        df['reference_len'] = [r['lens']['reference'] for r in results]
        
        answer_len_deltas = {}
        for ref_model_name, group in df.groupby('reference_model_name'):
            answer_len_deltas[ref_model_name] = (group['reference_len'] - group['model_len']).std()

        data = []
        for _, group in df.groupby('model_hash'):
            for _, subgroup in group.groupby('full_hash'):
                assert subgroup.shape[0] == 2
                pred = int(subgroup['p'].mean() >= 0.5)
                
                normalized_answer_delta_weight = 0.5
                if self.len_control and (pred or not self.pen_only_model):
                    answers_length_deltas_std = answer_len_deltas[subgroup['reference_model_name'].iloc[0]]
                    answer_length_delta = (subgroup['reference_len'] - subgroup['model_len']).iloc[0]
                    if answer_length_delta != 0: # same model as ref
                        normalized_answer_delta_weight = expit(answer_length_delta / answers_length_deltas_std)

                data.append([subgroup['model_name'].tolist()[0], subgroup['reference_model_name'].tolist()[0], pred, normalized_answer_delta_weight])
                
        score = self._calculate_score(data, df['model_name'][0])
        return score

    def aggregation(self) -> Dict:
        return {"score": self._confident_score_mean}

    def evaluate(self, sample, y_pred) -> Dict:
        model_proba = float(y_pred[sample['model_label']])
        model_mask_mapping = lambda x: 'model' if x == sample['model_label'] else 'reference'
        lens = {model_mask_mapping('m'): len(sample['output_m']), model_mask_mapping('M'): len(sample['output_M'])}
        #y_pred = sorted([pair for pair in y_pred.items()], key=lambda x: -x[1])[0][0]
        return {"score": {'p': model_proba ,'id': sample['id'], 'model_name': sample['model_name'], 'reference_model_name': sample['reference_model_name'], 'lens': lens}}

    def load_dataset(self, model: LLM, max_len: int, max_sample_per_dataset: int, few_shot_count: int) -> Tuple[List[Dict], List[Dict]]:
        self.logger.info(f'Ignoring few_shot_count for {self.name()}')
        assert few_shot_count == 0

        model_name = self.model_outputs['model_name']
        model_outputs_path = self.model_outputs['path']
        model_outputs = read_json(model_outputs_path)[:max_sample_per_dataset]
        for i, o in enumerate(model_outputs):
            o['id'] = i

        samples = []
        messages = []
        for reference_outputs in self.references_outputs:
            reference_model_name =  reference_outputs['model_name']
            reference_outputs_path = reference_outputs['path']
            reference_outputs = read_json(reference_outputs_path)[:max_sample_per_dataset]
            assert len(reference_outputs) == len(model_outputs)
            for i, o in enumerate(reference_outputs):
                assert o['instruction'] == model_outputs[i]['instruction']
                o['id'] = i

            for i in range(len(model_outputs)):
                sample_direct = self._create_sample(model_outputs[i], reference_outputs[i], model_name, reference_model_name)
                samples.append({'sample': sample_direct})
                messages.append({'messages': self._prepare_messages(sample_direct, model, max_len)})
                
                sample_reverse = self._reverse_sample(sample_direct)
                samples.append({'sample': sample_reverse})
                messages.append({'messages': self._prepare_messages(sample_reverse, model, max_len)})

        for m in messages:
            m['tokens_of_interest'] = self.choices
        return messages, samples

    def _create_sample(self, model_output, reference_output, model_name, reference_model_name):
        sample = {
            'id': model_output['id'],
            'model_name': model_name,
            'reference_model_name': reference_model_name,
            'instruction': model_output['instruction'],
            'output_m': model_output['output'],
            'output_M': reference_output['output'],
            'model_label': 'm'
        }
        return sample
    
    def _reverse_sample(self, sample):
        sample_reverse = copy.deepcopy(sample)
        sample_reverse['model_label'] = 'M'
        swap(sample_reverse, 'output_m', 'output_M')
        return sample_reverse
    
    def _prepare_messages(self, sample: Dict, model: LLM, max_len: int):
        zero_shot_messages = self.create_messages(copy.deepcopy(sample), with_answer=False)
        zero_shot_messages_len = model.count_tokens_for_prompt(model.apply_model_prompt(zero_shot_messages))
        if zero_shot_messages_len >= max_len:
            self.logger.warning(f'WARNING: sample zero-shot len {zero_shot_messages_len} greater then {max_len}. Will be truncated.')
        return zero_shot_messages

    def create_messages(self, sample: Dict, with_answer=False):
        instruction_system = 'You are a helpful assistant, that ranks models by the quality of their answers.'
        instruction_user = '''I want you to create a leaderboard of different of large language models. To do so, I will give you the instructions (prompts) given to the models, and the responses of two models. Please rank the models based on which responses would be preferred by humans. All inputs and outputs should be python dictionaries.

The main language of the these models is **Russian** and you are required to evaluate not only the overall quality of the answers, but also their grammar and coherence.

**Prompt**:
{
    "instruction": "{instruction}",
}

**Models outputs**:
[
    {
        "model": "m",
        "answer": """{output_m}"""
    },
    {
        "model": "M",
        "answer": """{output_M}"""
    }
]

Now please rank the models by the quality of their answers, so that the model with rank 1 has the best output. Then return a list of the model names and ranks, i.e., produce the following output:
[
    {"rank': 1, "model_name": <model-name-winner>},
    {"rank': 2, "model_name": <model-name-loser>}
]

Please provide the ranking that the majority of humans would give.'''
        instruction_bot = '''[
    {"rank': 1, "model_name": "'''
        user_content = instruction_user.replace('{instruction}', sample['instruction']).replace('{output_m}', sample['output_m']).replace('{output_M}', sample['output_M'])
        messages = [
            {'role': 'system', 'content': instruction_system},
            {'role': 'user', 'content': user_content},
            {'role': 'assistant', 'content': instruction_bot}
        ]
        return messages



        






                

