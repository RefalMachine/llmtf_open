from llmtf.base import Task, LLM
from typing import List, Dict, Tuple
import codecs
import json
import copy
import pandas as pd
import numpy as np
from evalica import bradley_terry, Winner, pairwise_frame
from scipy.special import expit
from typing import List, Dict, Tuple
import copy
from tqdm import tqdm
from functools import partial
from scipy.optimize import minimize
from math import log
from random import randint
import re

tqdm.pandas()

STYLE_CONTROL_ELEMENTS = [
    "len_answer",
    "header_count",
    "list_count",
    "bold_count",
    "code_blocks_count"
]

DIFF_MASK = np.array([1.0, -1.0], dtype=np.float64)

def count_style_elements(markdown_text):
    def remove_pattern(answer, pattern):
        blocks = pattern.findall(answer)
        for block in blocks:
            answer = answer.replace(block, "")
        return answer

    len_answer = len(markdown_text)
    code_count = len(re.findall(r"```[^`]+```", markdown_text))
    code_pattern = re.compile("```([^`]+)```")
    markdown_text = remove_pattern(markdown_text, code_pattern)
    markdown_text = markdown_text.replace("```", "")

    mono_count = len(re.findall(r"`[^`]+`", markdown_text))
    mono_pattern = re.compile("`([^`]+)`")
    markdown_text = remove_pattern(markdown_text, mono_pattern)
    counters = {
        f"len_answer": len_answer,
        f"header_count": {
            "h1": len(re.findall(r"^#{1}\s", markdown_text, re.MULTILINE)),
            "h2": len(re.findall(r"^#{2}\s", markdown_text, re.MULTILINE)),
            "h3": len(re.findall(r"^#{3}\s", markdown_text, re.MULTILINE)),
            "h4": len(re.findall(r"^#{4}\s", markdown_text, re.MULTILINE)),
            "h5": len(re.findall(r"^#{5}\s", markdown_text, re.MULTILINE)),
            "h6": len(re.findall(r"^#{6}\s", markdown_text, re.MULTILINE)),
        },
        f"list_count": {
            "ordered": len(re.findall(r"^\s*\d+\.\s", markdown_text, re.MULTILINE)),
            "unordered": len(re.findall(r"^\s*[-*+]\s", markdown_text, re.MULTILINE)),
        },
        f"bold_count": {
            "**": len(re.findall(r"\*\*[^*\n]+\*\*", markdown_text)),
            "__": len(re.findall(r"__[^_\n]+__", markdown_text)),
        },
        f"code_blocks_count": {
            "`": mono_count,
            "```": code_count,
        },
    }
    return counters


def extract_style_feature(x, feature):
    val = x[feature]
    if isinstance(val, int):
        return val
    else:
        return sum(val.values())


def get_element_counts(text):
    style_elements = count_style_elements(text)
    el_counts = []
    for feature in style_elements:
        el_counts.append(extract_style_feature(style_elements, feature))
    return el_counts


def calculate_style(
    model_a: pd.Series,
    model_b: pd.Series,
    style_elements: list[str]=STYLE_CONTROL_ELEMENTS
):
    n_features = len(style_elements)
    n_battles = model_a.shape[0]
    style_matrix = np.zeros(shape=(2*n_features, n_battles))
    for idx, element in enumerate(style_elements):
        style_matrix[idx, :] = np.array([el[idx] for el in model_a])
    for idx, element in enumerate(style_elements):
        style_matrix[n_features + idx, :] = np.array([el[idx] for el in model_b])
    style_diff = (style_matrix[:n_features] - style_matrix[n_features]).astype(float)
    style_sum = (style_matrix[:n_features] + style_matrix[n_features]).astype(float)

    style_diff /= style_sum

    style_mean = np.mean(style_diff, axis=1)
    style_std = np.std(style_diff, axis=1)
    features = ((style_diff - style_mean[:, np.newaxis]) / style_std[:, np.newaxis]).T

    return features


def get_matchups_models(model_a: pd.Series, model_b: pd.Series):
    n_rows = len(model_a)
    assert len(model_b) == n_rows
    model_indices, models = pd.factorize(pd.concat([model_a, model_b]))
    matchups = np.column_stack([model_indices[:n_rows], model_indices[n_rows:]])
    return matchups, models.to_list()


def contextual_bt_loss_and_grad(
    params,
    n_competitors,
    matchups,
    features,
    outcomes,
    alpha=1.0,
    reg=1.0,
    half_reg=0.5,
):
    reg_loss = half_reg * np.inner(params, params)

    ratings = params[:n_competitors]
    feature_params = params[n_competitors:]

    matchup_ratings = ratings[matchups]

    bt_logits = alpha * (matchup_ratings[:, 0] - matchup_ratings[:, 1])
    context_logits = np.dot(features, feature_params)
    probs = expit(bt_logits + context_logits)
    loss = (
        -((np.log(probs) * outcomes + np.log(1.0 - probs) * (1.0 - outcomes))).sum()
        + reg_loss
    )

    error = outcomes - probs
    grad = reg * params
    matchups_grads = -alpha * error
    np.add.at(
        grad[:n_competitors], matchups[:, [0, 1]], matchups_grads[:, None] * DIFF_MASK
    )
    grad[n_competitors:] -= np.dot(features.T, error)

    return loss, grad


def fit_contextual_bt(
    matchups,
    features,
    outcomes,
    models,
    idxs=None,
    alpha=log(10.0),
    reg=0.5,
    tol=1e-6,
):
    n_features = features.shape[1]
    n_models = len(models)
    initial_params = np.zeros(n_models + n_features, dtype=np.float64)
    half_reg = reg / 2.0

    if idxs is not None:
        matchups, features, outcomes = matchups[idxs], features[idxs], outcomes[idxs]

    result = minimize(
        fun=contextual_bt_loss_and_grad,
        x0=initial_params,
        args=(n_models, matchups, features, outcomes, alpha, reg, half_reg),
        jac=True,
        method="L-BFGS-B",
        options={"disp": False, "maxiter": 100, "gtol": tol},
    )
    return result["x"]


def compute_style_control(
    df: pd.DataFrame,
    alpha=log(10.0), reg=0.5, tol=1e-6
):
    features = calculate_style(df.model_a_style, df.model_b_style)
    matchups, models = get_matchups_models(df.model_a, df.model_b)
    outcomes = df.winner.values
    params = fit_contextual_bt(
        matchups,
        features,
        outcomes,
        models=models,
        alpha=alpha,
        reg=reg,
        tol=tol,
    )
    ratings = params[: len(models)]
    return ratings, models


def read_json(file_name):
    with open(file_name, encoding="utf-8") as r:
        return json.load(r)

def get_results_from_file(sourse_files):
    results = []
    if isinstance(sourse_files, str):
        sourse_files = [sourse_files]
    if isinstance(sourse_files, list):
        for file in sourse_files:
            results += read_json(file)
    return results

def swap(data, key1, key2):
    tmp = data[key1]
    data[key1] = data[key2]
    data[key2] = tmp


def scale_and_offset(
    ratings,
    models,
    baseline_model='',
    baseline_rating=50,
):
    """convert ratings from the natural scale to the Elo rating scale with an anchored baseline"""
    scaled_ratings = 50 * (1 + ratings)
    if baseline_model and baseline_model in models:
        baseline_idx = models.index(baseline_model)
        scaled_ratings += baseline_rating - scaled_ratings[..., [baseline_idx]]
    return scaled_ratings

def scale_to_elo(
    ratings,
    models,
    baseline_model='',
    baseline_rating=1000, # Стандартная начальная точка для Elo
    scale_factor=400.0,
):
    """
    Преобразует сырые рейтинги Брэдли-Терри в шкалу Elo.
    Сырые рейтинги должны быть получены с alpha=log(10).
    """
    # Преобразуем сырые рейтинги в шкалу Elo
    elo_ratings = scale_factor * ratings
    
    # Якорим одну из моделей для стабильности шкалы
    if baseline_model and baseline_model in models:
        baseline_idx = models.index(baseline_model)
        # Смещаем все рейтинги так, чтобы у базовой модели был рейтинг baseline_rating
        offset = baseline_rating - elo_ratings[baseline_idx]
        elo_ratings += offset
        
    return elo_ratings

def confident_score_mean(results: List[Dict], model_name=None) -> Dict:
    '''
    На вход подаются
    results = [
        {
            'p': <model_proba_i>,
            'id': <id_i>,
            'model_name': <model_name_i>,
            'reference_model_name': <reference_model_name_i>,
            'style': {
                'model':      [x, x, x, x, x],
                'reference':  [x, x, x, x, x], # [ len_answer, header_count, list_count, bold_count, code_blocks_count ]
            }
        } for i in range(2*n*m) # на каждый instruction было по 2 sample
    ]
    '''
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

    df['model_style'] = [np.array(r['styles']['model']) for r in results]
    df['reference_style'] = [np.array(r['styles']['reference']) for r in results]

    data = []
    for _, group in df.groupby('model_hash'):
        for _, subgroup in group.groupby('full_hash'):
            # assert subgroup.shape[0] == 2
            if (subgroup['model_name'] == subgroup['reference_model_name']).all(): # or (subgroup['model_style'].tolist()[0] == subgroup['reference_style'].tolist()[0]).all():
                continue
            pred = int(subgroup['p'].mean() >= 0.5)
            data.append([
                    subgroup['model_name'].tolist()[0],
                    subgroup['reference_model_name'].tolist()[0], 
                    pred, subgroup['model_style'].tolist()[0], 
                    subgroup['reference_style'].tolist()[0]
                ])
    new_df = pd.DataFrame(data, columns=['model_a', 'model_b', 'winner', 'model_a_style', 'model_b_style'])
    ratings, models = compute_style_control(new_df)
    baseline_model_name = 'gpt-4o-mini'
    elo_ratings = scale_to_elo(ratings, models, baseline_model=baseline_model_name)

    model_rating_dict = dict(zip(models, elo_ratings))
    # Округляем для красивого вывода
    model_rating_dict_rounded = {k: round(v) for k, v in model_rating_dict.items()}
    
    print(' >> ELO RATINGS:\n', json.dumps(model_rating_dict_rounded, indent=4, ensure_ascii=False))
    if model_name is not None:
        return model_rating_dict[model_name]
    return model_rating_dict

    
class LLMAsJudgeStyleControl(Task):
    def __init__(self, model_outputs: Dict, references_outputs: List[Dict], previous_battles_path: str='', custom_instruction=None, **kwargs):
        super().__init__(**kwargs)
        self.model_outputs = model_outputs
        '''
        model_outputs = {
            'model_name': <name> -- имя `главной` модели
            'path': <path> -- путь до ответов данной модели
        }
        '''
        self.references_outputs = references_outputs
        '''
        reference_outputs = [dict, dict, ..., dict], всего с первой моделью сравниваются m=len(reference_outputs) моделей
        reference_outputs[i] = {
            'model_name': <name_i_model> -- имя сравниваемой модели
            'path': <path_i_model> -- путь до ответов данной модели
        }
        '''
        self.custom_instruction = custom_instruction
        self.method = 'calculate_tokens_proba'
        self.previous_battles_path = previous_battles_path
        self._max_new_tokens = 1

    @property
    def choices(self) -> List:
        return ["m", "M"]

    def task_name(self):
        return 'llm_as_judge'

    def _confident_score_mean(self, results: Dict) -> Dict:
        '''
        На вход подаются
        results = [
            {
                'p': <model_proba_i>,
                'id': <id_i>,
                'model_name': <model_name_i>,
                'reference_model_name': <reference_model_name_i>,
                'style': {
                    'model':      [x, x, x, x, x],
                    'reference':  [x, x, x, x, x], # [ len_answer, header_count, list_count, bold_count, code_blocks_count ]
                }
            } for i in range(2*n*m) # на каждый instruction было по 2 sample
        ]
        '''
        
        if self.previous_battles_path:
            results += get_results_from_file(self.previous_battles_path)

        model_rating_dict = confident_score_mean(results)
        return model_rating_dict[self.model_outputs['model_name']]


    def aggregation(self) -> Dict:
        return {"score": self._confident_score_mean}


    def evaluate(self, sample, y_pred) -> Dict:
        model_proba = float(y_pred[sample['model_label']])
        model_mask_mapping = lambda x: 'model' if x == sample['model_label'] else 'reference'
        styles = {
            model_mask_mapping('m'): get_element_counts(sample['output_m']),
            model_mask_mapping('M'): get_element_counts(sample['output_M'])
        }
        '''
        styles = {
            'model':      [x, x, x, x, x],
            'reference':  [x, x, x, x, x],
        }, where the elements of the array correspond to:
            [ len_answer, header_count, list_count, bold_count, code_blocks_count ]
        '''
        return {
            "score": {
                'p': model_proba,
                'id': sample['id'],
                'model_name': sample['model_name'],
                'reference_model_name': sample['reference_model_name'],
                'styles': styles
            }
        }


    def load_dataset(
        self,
        model: LLM,
        max_len: int,
        max_sample_per_dataset: int,
        few_shot_count: int
    ) -> Tuple[List[Dict], List[Dict]]:
        self.logger.info(f'Ignoring few_shot_count for {self.name()}')
        assert few_shot_count == 0
        '''
        n=len(model_outputs) -- число вопросов
        m=len(reference_model_outputs) -- число моделей в пайплайне

        model_outputs и reference_outputs[i] заменяется на их ответ из json-файла:
        model_outputs[i] = {
            'id': i
            'instruction' : <instruction_i>
            ...
        }
        reference_outputs = [
            {
                {
                    'id' = i
                    'instruction' : <instruction_i>
                    ...
                } for i in range(n)
            } for j in range(m)
        ]
        '''
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

            for i in range(len(model_outputs)): # по всем заданиям
                sample_direct = self._create_sample(model_outputs[i], reference_outputs[i], model_name, reference_model_name)
                samples.append({'sample': sample_direct})
                messages.append({'messages': self._prepare_messages(sample_direct, model, max_len)})

                sample_reverse = self._reverse_sample(sample_direct)
                samples.append({'sample': sample_reverse})
                messages.append({'messages': self._prepare_messages(sample_reverse, model, max_len)})

        for m in messages:
            m['tokens_of_interest'] = self.choices
        '''
        samples = [
            {
                'id': model_output['id'],
                'model_name': model_name,
                'reference_model_name': reference_model_name,
                'instruction': model_output['instruction'],
                'output_m': model_output['output'],
                'output_M': reference_output['output'],
                'model_label': 'm'
            },
            {
                'id': model_output['id'],
                'model_name': model_name,
                'reference_model_name': reference_model_name,
                'instruction': model_output['instruction'],
                'output_m': reference_output['output'],
                'output_M': model_output['output'],
                'model_label': 'M'
            } for i in range(n)
        ]
        messages = [
            [
                {'role': 'system', 'content': instruction_system},
                {'role': 'user', 'content': user_content_m_M},
                {'role': 'assistant', 'content': instruction_bot}
                'tokens_of_interest': ['m', 'M']
            ],
            [
                {'role': 'system', 'content': instruction_system},
                {'role': 'user', 'content': user_content_M_m},
                {'role': 'assistant', 'content': instruction_bot}
                'tokens_of_interest': ['m', 'M']
            ]
            for i in range(n)
        ]
        '''
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