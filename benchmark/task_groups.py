import json
from pathlib import Path

#TODO:
#{'name': 'rublimp', 'params': {'dataset_names': 'russiannlp/rublimp-(classify) russiannlp/rublimp-(choice)', 'few_shot_count': 5, 'name_suffix': 'few_shot'}},

task_groups_knowledge = [
    {'name': 'nlpcoreteam_mmlu_ru_zero_shot', 'params': {'dataset_names': 'nlpcoreteam/rummlu', 'few_shot_count': 0, 'name_suffix': 'zero_shot'}},
    {'name': 'nlpcoreteam_mmlu_en_zero_shot', 'params': {'dataset_names': 'nlpcoreteam/enmmlu', 'few_shot_count': 0, 'name_suffix': 'zero_shot'}},
    {'name': 'shlepa', 'params': {'dataset_names': 'shlepa/moviesmc shlepa/musicmc shlepa/lawmc shlepa/booksmc'}},
]

task_groups_skills = [
    {'name': 'translation', 'params': {'dataset_names': 'darumeru/flores_ru_en darumeru/flores_en_ru'}},
    {'name': 'summarization', 'params': {'dataset_names': 'daru/treewayabstractive ilyagusev/gazeta', 'max_sample_per_dataset': 1000}},
    {'name': 'sentiment', 'params': {'dataset_names': 'ruopinionne ruopinionne_simple', 'max_sample_per_dataset': 1000}},
    {'name': 'rag', 'params': {'dataset_names': 'rusbeirrag/rubqqa rusbeirrag/rus_tydiqa rusbeirrag/sberquadqa rusbeirrag/rus_xquadqa', 'max_sample_per_dataset': 500, 'max_len': 12000}},
    {'name': 'rag_data_first', 'params': {'dataset_names': 'rusbeirrag/rubqqa_data_first rusbeirrag/rus_tydiqa_data_first rusbeirrag/sberquadqa_data_first rusbeirrag/rus_xquadqa_data_first', 'max_sample_per_dataset': 500, 'max_len': 12000}},
    {'name': 'ner_dict', 'params': {'dataset_names': 'MalakhovIlya/NEREL-(dict) nerel-ds/NEREL-BIO-(dict) Mykes/patient_queries_ner-(dict)', 'few_shot_count': 3, 'name_suffix': 'few_shot', 'max_len': 8000}, 'think': False},
    {'name': 'ner_json', 'params': {'dataset_names': 'MalakhovIlya/NEREL-(json) nerel-ds/NEREL-BIO-(json) Mykes/patient_queries_ner-(json)', 'few_shot_count': 3, 'name_suffix': 'few_shot', 'max_len': 8000}, 'think': False},
    {'name': 'ner_in-place', 'params': {'dataset_names': 'MalakhovIlya/NEREL-(in-place) nerel-ds/NEREL-BIO-(in-place) Mykes/patient_queries_ner-(in-place)', 'few_shot_count': 3, 'name_suffix': 'few_shot', 'max_len': 8000}, 'think': False},
]

task_groups_ifeval = [
    {'name': 'ruifeval', 'params': {'dataset_names': 'ruifeval', 'few_shot_count': 0}},
    {'name': 'enifeval', 'params': {'dataset_names': 'enifeval', 'few_shot_count': 0}},
]

task_groups_long = []

with open(str(Path(__file__).parent.parent / 'llmtf' / 'tasks' / 'libra' / 'libra_config.json'), "r", encoding="utf-8") as f:
    libra_tasks = list(json.load(f).keys())
for task in libra_tasks:
    task_groups_long.append({'name': 'libra_' + task, 'params': {'dataset_names': 'libra/' + task, 'max_sample_per_dataset': 1000, 'few_shot_count': 0, 'max_len': 32000}})

task_groups_knowledge_few_shot = [
    {'name': 'nlpcoreteam_mmlu_ru_few_shot', 'params': {'dataset_names': 'nlpcoreteam/rummlu', 'few_shot_count': 5, 'name_suffix': 'few_shot'}},
    {'name': 'nlpcoreteam_mmlu_en_few_shot', 'params': {'dataset_names': 'nlpcoreteam/enmmlu', 'few_shot_count': 5, 'name_suffix': 'few_shot'}}
]
task_groups_math_no_think = [
    {'name': 'doom_math_no_think', 'params': {'dataset_names': 'doom/math', 'few_shot_count': 0, 'max_len': 32000, 'name_suffix': 'no_think'}},
    {'name': 'doom_phys_no_think', 'params': {'dataset_names': 'doom/phys', 'few_shot_count': 0, 'max_len': 32000, 'name_suffix': 'no_think'}},
    {'name': 't-bank_t-math_no_think', 'params': {'dataset_names': 't-bank/t-math', 'few_shot_count': 0, 'max_len': 32000, 'name_suffix': 'no_think'}}
]

task_groups_math_think = [
    {'name': 'doom_math', 'params': {'dataset_names': 'doom/math', 'few_shot_count': 0, 'max_len': 32000, 'name_suffix': 'think', 'max_new_tokens_reasoning': 16000}, 'think': True},
    {'name': 'doom_phys', 'params': {'dataset_names': 'doom/phys', 'few_shot_count': 0, 'max_len': 32000, 'name_suffix': 'think', 'max_new_tokens_reasoning': 16000}, 'think': True},
    {'name': 't-bank_t-math', 'params': {'dataset_names': 't-bank/t-math', 'few_shot_count': 0, 'max_len': 32000, 'name_suffix': 'think', 'max_new_tokens_reasoning': 16000}, 'think': True}
]

task_groups = task_groups_knowledge + task_groups_skills + task_groups_ifeval + task_groups_knowledge_few_shot + task_groups_math_no_think + task_groups_long #+ task_groups_math_no_think
