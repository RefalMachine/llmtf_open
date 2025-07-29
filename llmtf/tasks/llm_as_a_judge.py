from llmtf.base import Task, LLM
from typing import List, Dict, Tuple
import codecs
import json
import copy
import pandas as pd
import numpy as np
from scipy.special import expit, softplus
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

def bootstrap_elo_ratings(df, n_bootstrap=1000, confidence_level=0.95, alpha=log(10.0), reg=0.5, tol=1e-6):
    """
    Вычисляет Elo рейтинги с доверительными интервалами используя bootstrap.
    
    Args:
        df: DataFrame с результатами боев
        n_bootstrap: количество bootstrap итераций
        confidence_level: уровень доверия для интервалов (по умолчанию 95%)
        
    Returns:
        dict: словарь с рейтингами, std и доверительными интервалами для каждой модели
    """
    n_battles = len(df)
    
    # Получаем базовые рейтинги и модели
    base_ratings, models = compute_style_control_with_ties(df, alpha, reg, tol)
    baseline_model = 'gpt-4o-mini'
    base_elo = scale_to_elo(base_ratings, models, baseline_model=baseline_model)
    
    # Bootstrap
    bootstrap_ratings = []
    
    for _ in tqdm(range(n_bootstrap), desc="Bootstrap iterations"):
        # Создаем bootstrap выборку с заменой
        bootstrap_indices = np.random.choice(n_battles, size=n_battles, replace=True)
        bootstrap_df = df.iloc[bootstrap_indices]
        
        try:
            # Вычисляем рейтинги для bootstrap выборки
            ratings, boot_models = compute_style_control_with_ties(bootstrap_df, alpha, reg, tol)
            
            # Преобразуем в Elo шкалу
            elo_ratings = scale_to_elo(ratings, boot_models, baseline_model=baseline_model)
            
            # Сохраняем рейтинги в том же порядке, что и исходные модели
            aligned_ratings = []
            for model in models:
                if model in boot_models:
                    idx = boot_models.index(model)
                    aligned_ratings.append(elo_ratings[idx])
                else:
                    # Если модель не попала в bootstrap выборку, используем NaN
                    aligned_ratings.append(np.nan)
            
            bootstrap_ratings.append(aligned_ratings)
            
        except Exception as e:
            # В случае ошибки (например, недостаточно данных) пропускаем итерацию
            continue
    
    # Преобразуем в numpy массив
    bootstrap_ratings = np.array(bootstrap_ratings)
    
    # Вычисляем статистики
    results = {}
    alpha = (1 - confidence_level) / 2
    
    for i, model in enumerate(models):
        # Убираем NaN значения для данной модели
        model_ratings = bootstrap_ratings[:, i]
        model_ratings = model_ratings[~np.isnan(model_ratings)]
        
        if len(model_ratings) > 0:
            results[model] = {
                'rating': round(base_elo[i]),
                'std': round(np.std(model_ratings)),
                'ci_lower': round(np.percentile(model_ratings, alpha * 100)),
                'ci_upper': round(np.percentile(model_ratings, (1 - alpha) * 100)),
                'n_samples': len(model_ratings)
            }
        else:
            results[model] = {
                'rating': round(base_elo[i]),
                'std': np.nan,
                'ci_lower': np.nan,
                'ci_upper': np.nan,
                'n_samples': 0
            }
    
    return results


def confident_score_mean_with_ties_and_ci(results: List[Dict], model_name=None, n_bootstrap=1000, confidence_level=0.95) -> Dict:
    """
    Расширенная версия confident_score_mean_with_ties с доверительными интервалами.
    """
    battle_results = {}

    for r in results:
        if not r or r.get('outcome') == 'invalid':
            continue

        battle_key = (r['id'], (r['model_name'], r['reference_model_name']))

        if battle_key not in battle_results:
            battle_results[battle_key] = []

        outcome_code = -1
        if r['outcome'] == 'tie':
            outcome_code = 2
        elif r['outcome'] == 'model_wins':
            outcome_code = 0
        elif r['outcome'] == 'reference_wins':
            outcome_code = 1
        
        battle_results[battle_key].append({
            "outcome": outcome_code,
            "style_a": r['styles']['model'],
            "style_b": r['styles']['reference'],
        })

    data = []
    for (item_id, models_tuple), outcomes_list in battle_results.items():
        if not outcomes_list:
            continue
        
        codes = [o['outcome'] for o in outcomes_list if o['outcome'] != -1]
        if not codes:
            continue

        if len(set(codes)) > 1:
            final_outcome = 2
        else:
            final_outcome = codes[0]
        
        data.append([
            models_tuple[0], models_tuple[1], final_outcome,
            outcomes_list[0]['style_a'], outcomes_list[0]['style_b']
        ])

    if not data:
        print(" >> No valid battles to compute ratings.")
        return {} if model_name is None else {'rating': 0, 'std': 0, 'ci_lower': 0, 'ci_upper': 0}
    
    new_df = pd.DataFrame(data, columns=['model_a', 'model_b', 'winner', 'model_a_style', 'model_b_style'])
    
    # Получаем рейтинги с доверительными интервалами
    ratings_with_ci = bootstrap_elo_ratings(new_df, n_bootstrap=n_bootstrap, confidence_level=confidence_level)
    
    # Выводим результаты
    #print(' >> ELO RATINGS with confidence intervals:')
    #for model, stats in sorted(ratings_with_ci.items(), key=lambda x: x[1]['rating'], reverse=True):
    #    print(f"   {model}: {stats['rating']} ± {stats['std']} (95% CI: [{stats['ci_lower']}, {stats['ci_upper']}])")
    
    if model_name is not None:
        return ratings_with_ci.get(model_name, {'rating': 0, 'std': 0, 'ci_lower': 0, 'ci_upper': 0})
    return ratings_with_ci

def contextual_bt_rao_kupper_loss_and_grad(
    params, n_competitors, matchups, features, outcomes, alpha=1.0, reg=1.0, half_reg=0.5
):
    ratings = params[:n_competitors]
    feature_params = params[n_competitors:-1]
    logit_theta = params[-1]

    # Используем softplus для стабильного вычисления log(p_tie) и log(1-p_tie)
    log_prob_tie = -softplus(-logit_theta) # log(sigmoid(logit_theta))
    log_prob_not_tie = -softplus(logit_theta) # log(1 - sigmoid(logit_theta))

    matchup_ratings = ratings[matchups]
    bt_logits = alpha * (matchup_ratings[:, 0] - matchup_ratings[:, 1])
    context_logits = np.dot(features, feature_params)
    logits = bt_logits + context_logits

    log_probs_A_if_not_tie = -softplus(-logits) # log(sigmoid(logits))
    log_probs_B_if_not_tie = -softplus(logits)  # log(1 - sigmoid(logits))

    log_probs = np.stack([
        log_prob_not_tie + log_probs_A_if_not_tie,
        log_prob_not_tie + log_probs_B_if_not_tie,
        np.full_like(logits, log_prob_tie)
    ], axis=1)

    loss = -np.sum(log_probs[np.arange(len(outcomes)), outcomes]) + (half_reg * np.inner(params, params))

    # Для градиента нам нужны вероятности, а не их логарифмы
    probs_A_if_not_tie = expit(logits)
    prob_tie = expit(logit_theta)
    
    error_conditional = np.zeros_like(logits)
    win_A_mask, win_B_mask = (outcomes == 0), (outcomes == 1)
    error_conditional[win_A_mask] = 1.0 - probs_A_if_not_tie[win_A_mask]
    error_conditional[win_B_mask] = -probs_A_if_not_tie[win_B_mask]
    
    error_tie = np.sum(outcomes == 2) - len(outcomes) * prob_tie
    
    grad = reg * params
    matchups_grads = -alpha * error_conditional
    np.add.at(grad[:n_competitors], matchups[:, [0, 1]], matchups_grads[:, None] * DIFF_MASK)
    grad[n_competitors:-1] -= np.dot(features.T, error_conditional)
    grad[-1] -= error_tie

    return loss, grad

def fit_contextual_bt_with_ties(matchups, features, outcomes, models, alpha=log(10.0), reg=0.5, tol=1e-6):
    n_features, n_models = features.shape[1], len(models)
    initial_params = np.zeros(n_models + n_features + 1, dtype=np.float64)
    result = minimize(
        fun=contextual_bt_rao_kupper_loss_and_grad,
        x0=initial_params,
        args=(n_models, matchups, features, outcomes, alpha, reg, reg / 2.0),
        jac=True, method="L-BFGS-B",
        options={"disp": False, "maxiter": 100, "gtol": tol},
    )
    return result["x"]


def compute_style_control_with_ties(df: pd.DataFrame, alpha=log(10.0), reg=0.5, tol=1e-6):
    features = calculate_style(df.model_a_style, df.model_b_style)
    matchups, models = get_matchups_models(df.model_a, df.model_b)
    outcomes = df.winner.values
    params = fit_contextual_bt_with_ties(matchups, features, outcomes, models, alpha, reg, tol)
    
    ratings = params[:len(models)]
    return ratings, models

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
        self.n_bootstrap = 100
        self.confidence_level = 0.95

    @property
    def choices(self) -> List:
        return ["m", "M", "T"]

    def task_name(self):
        return 'llm_as_judge'

    def leaderboard_aggregation(self, metrics: Dict) -> float:
        return metrics['score']['rating']

    def _confident_score_mean(self, results: Dict) -> Dict:
        """
        Возвращает словарь с рейтингом и статистиками неопределенности.
        """
        if self.previous_battles_path:
            results += get_results_from_file(self.previous_battles_path)

        model_stats = confident_score_mean_with_ties_and_ci(
            results, 
            model_name=self.model_outputs['model_name'],
            n_bootstrap=self.n_bootstrap,
            confidence_level=self.confidence_level
        )
        
        # Возвращаем полную статистику для главной модели
        return model_stats


    def aggregation(self) -> Dict:
        return {"score": self._confident_score_mean}


    def evaluate(self, sample, y_pred) -> Dict:
        # y_pred это словарь вида {'m': p1, 'M': p2, 'T': p3}
        
        # Находим токен с максимальной вероятностью
        winner_token = max(y_pred, key=y_pred.get)
        
        outcome = "invalid" # На случай непредвиденных ошибок
        if winner_token == 'T':
            outcome = "tie"
        # sample['model_label'] это 'm' или 'M' в зависимости от того, 
        # какой ответ был от "главной" модели в этой конкретной паре.
        elif winner_token == sample['model_label']:
            outcome = "model_wins"
        else:
            outcome = "reference_wins"
            
        model_mask_mapping = lambda x: 'model' if x == sample['model_label'] else 'reference'
        styles = {
            model_mask_mapping('m'): get_element_counts(sample['output_m']),
            model_mask_mapping('M'): get_element_counts(sample['output_M'])
        }

        answers = {
            model_mask_mapping('m'): sample['output_m'],
            model_mask_mapping('M'): sample['output_M']
        }
        
        # Возвращаем тот же формат, что и в решении с 'generate'
        return {
            "score": {
                'outcome': outcome,
                'id': sample['id'],
                'model_name': sample['model_name'],
                'reference_model_name': sample['reference_model_name'],
                'styles': styles,
                'answers': answers
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
        instruction_system = 'You are a meticulous and impartial assistant designed to evaluate the quality of language model responses. Your task is to follow a strict, step-by-step evaluation procedure to determine the superior response.'
        instruction_user = '''I will provide you with a prompt and two responses from models 'm' and 'M'. Your task is to determine which model provided a better response by following a strict evaluation procedure. The primary language of the content is Russian.

**Prompt**:
{instruction}

**Model 'm' Response**:
"""{output_m}"""

**Model 'M' Response**:
"""{output_M}"""

---
**Evaluation Procedure**

To make your decision, you must evaluate the responses sequentially according to three criteria, in order of priority:

**Step 1: Core Answer Quality (Highest Priority - Weight 3)**
*   **Relevance and Completeness**: How accurately and thoroughly does the response address the user's prompt?
*   **Factual Accuracy and Helpfulness**: Are the provided facts correct? Does the response effectively solve the user's task or answer their question?
*   **IMPORTANT**: At this stage, you **must ignore** all aspects of grammar, spelling, style, and formatting. Focus solely on the substance and content of the answer.

**Step 2: Language Quality (Secondary Priority - Weight 2)**
*   **Grammar and Spelling**: Is the text free of spelling, punctuation, or grammatical errors?
*   **Fluency and Naturalness**: Does the text read smoothly and naturally in Russian? Is the word choice appropriate?

**Step 3: Style and Structure (Lowest Priority - Weight 1)**
*   **Readability and Formatting**: Is the text easy to read? Does it use paragraphs, lists, or bolding to improve clarity?
*   **Clarity of Exposition**: Is the thought process presented in a clear and logical manner?

---
**Decision-Making Rules**

Use a hierarchical approach based on your evaluation:

1.  **First, compare the responses based on 'Core Answer Quality' (Step 1)**. If one response is significantly better in substance, relevance, and accuracy, choose it immediately, even if it has flaws in language or style.
2.  **If the responses are roughly equal in 'Core Answer Quality'**, use **'Language Quality' (Step 2)** as the tie-breaker. Choose the response that is more grammatically correct and fluent.
3.  **If the responses are equal in both 'Core Answer Quality' and 'Language Quality'**, use **'Style and Structure' (Step 3)** as the final deciding factor. Choose the response that is better structured and clearer.
4.  **If the responses remain tied after all three steps**, you must consider it a tie.

---
**Your Response**

Which response is better? Respond with **only a single character** and no other text or explanation.

- If model 'm' is clearly better according to the rules above, respond with: `m`
- If model 'M' is clearly better according to the rules above, respond with: `M`
- If the responses are of equivalent quality or the choice is a toss-up, respond with: `T`'''

        user_content = instruction_user.format(
            instruction=sample['instruction'],
            output_m=sample['output_m'],
            output_M=sample['output_M']
        )
        
        # Мы не даем 'assistant' контента, чтобы модель сгенерировала 'm', 'M' или 'T' как первый токен.
        messages = [
            {'role': 'system', 'content': instruction_system},
            {'role': 'user', 'content': user_content},
        ]
        return messages