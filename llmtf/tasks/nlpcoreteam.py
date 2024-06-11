from llmtf.base import Task, LLM
from llmtf.metrics import mean
from typing import Dict, List, Tuple
from datasets import DatasetDict, load_dataset, Dataset
import pandas as pd
from tqdm import tqdm
import copy
from multiprocessing import Pool
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Data preparation taken from https://github.com/NLP-Core-Team/mmlu_ru
SUBCATEGORIES = {
    "abstract_algebra": ["math"],
    "anatomy": ["health"],
    "astronomy": ["physics"],
    "business_ethics": ["business"],
    "clinical_knowledge": ["health"],
    "college_biology": ["biology"],
    "college_chemistry": ["chemistry"],
    "college_computer_science": ["computer science"],
    "college_mathematics": ["math"],
    "college_medicine": ["health"],
    "college_physics": ["physics"],
    "computer_security": ["computer science"],
    "conceptual_physics": ["physics"],
    "econometrics": ["economics"],
    "electrical_engineering": ["engineering"],
    "elementary_mathematics": ["math"],
    "formal_logic": ["philosophy"],
    "global_facts": ["other"],
    "high_school_biology": ["biology"],
    "high_school_chemistry": ["chemistry"],
    "high_school_computer_science": ["computer science"],
    "high_school_european_history": ["history"],
    "high_school_geography": ["geography"],
    "high_school_government_and_politics": ["politics"],
    "high_school_macroeconomics": ["economics"],
    "high_school_mathematics": ["math"],
    "high_school_microeconomics": ["economics"],
    "high_school_physics": ["physics"],
    "high_school_psychology": ["psychology"],
    "high_school_statistics": ["math"],
    "high_school_us_history": ["history"],
    "high_school_world_history": ["history"],
    "human_aging": ["health"],
    "human_sexuality": ["culture"],
    "international_law": ["law"],
    "jurisprudence": ["law"],
    "logical_fallacies": ["philosophy"],
    "machine_learning": ["computer science"],
    "management": ["business"],
    "marketing": ["business"],
    "medical_genetics": ["health"],
    "miscellaneous": ["other"],
    "moral_disputes": ["philosophy"],
    "moral_scenarios": ["philosophy"],
    "nutrition": ["health"],
    "philosophy": ["philosophy"],
    "prehistory": ["history"],
    "professional_accounting": ["other"],
    "professional_law": ["law"],
    "professional_medicine": ["health"],
    "professional_psychology": ["psychology"],
    "public_relations": ["politics"],
    "security_studies": ["politics"],
    "sociology": ["culture"],
    "us_foreign_policy": ["politics"],
    "virology": ["health"],
    "world_religions": ["philosophy"],
}

CATEGORIES = {
    "STEM": ["physics", "chemistry", "biology", "computer science", "math", "engineering"],
    "humanities": ["history", "philosophy", "law"],
    "social sciences": ["politics", "culture", "economics", "geography", "psychology"],
    "other (business, health, misc.)": ["other", "business", "health"],
}

# in the form to fit the prompt headline
SUBCATEGORIES_EN2RU = {
    "abstract_algebra": "абстрактной_алгебре",
    "anatomy": "анатомии",
    "astronomy": "астрономии",
    "business_ethics": "деловой_этике",
    "clinical_knowledge": "медицинским_знаниям",
    "college_biology": "биологии_в_вузе",
    "college_chemistry": "химии_в_вузе",
    "college_computer_science": "компьютерным_наукам_в_вузе",
    "college_mathematics": "математике_в_вузе",
    "college_medicine": "медицине_в_вузе",
    "college_physics": "физике_в_вузе",
    "computer_security": "компьютерной_безопасности",
    "conceptual_physics": "теоретической_физике",
    "econometrics": "эконометрике",
    "electrical_engineering": "электротехнике",
    "elementary_mathematics": "элементарной_математике",
    "formal_logic": "формальной_логике",
    "global_facts": "фактам_о_мире",
    "high_school_biology": "биологии_в_старшей_школе",
    "high_school_chemistry": "химии_в_старшей_школе",
    "high_school_computer_science": "информатике_в_старшей_школе",
    "high_school_european_history": "истории_Европы_в_старшей_школе",
    "high_school_geography": "географии_в_старшей_школе",
    "high_school_government_and_politics": "государству_и_политике_в_старшей_школе",
    "high_school_macroeconomics": "макроэкономике_в_старшей_школе",
    "high_school_mathematics": "математике_в_старшей_школе",
    "high_school_microeconomics": "микроэкономике_в_старшей_школе",
    "high_school_physics": "физике_в_старшей_школе",
    "high_school_psychology": "психологии_в_старшей_школе",
    "high_school_statistics": "статистике_в_старшей_школе",
    "high_school_us_history": "истории_США_в_старшей_школе",
    "high_school_world_history": "всемирной_истории_в_старшей_школе",
    "human_aging": "старению_человека",
    "human_sexuality": "человеческой_сексуальности",
    "international_law": "международному_праву",
    "jurisprudence": "юриспруденции",
    "logical_fallacies": "логическим_ошибкам",
    "machine_learning": "машинному_обучению",
    "management": "менеджменту",
    "marketing": "маркетингу",
    "medical_genetics": "медицинской_генетике",
    "miscellaneous": "разным_темам",
    "moral_disputes": "нравственным_спорам",
    "moral_scenarios": "нравственным_сценариям",
    "nutrition": "правильному_питанию",
    "philosophy": "философии",
    "prehistory": "доисторической_эпохе",
    "professional_accounting": "профессиональному_бухгалтерскому_учету",
    "professional_law": "профессиональному_праву",
    "professional_medicine": "профессиональной_медицине",
    "professional_psychology": "профессиональной_психологии",
    "public_relations": "связям_с_общественностью",
    "security_studies": "исследованиям_в_области_безопасности",
    "sociology": "социологии",
    "us_foreign_policy": "внешней_политике_США",
    "virology": "вирусологии",
    "world_religions": "мировым_религиям",
}

LANGUAGE_CONFIG: Dict[str, Dict[str, str]] = {
    "en": {
        "headline_prefix": "The following are multiple choice questions (with answers) about",
        "answer_prefix": "Answer:",
    },
    "ru": {
        "headline_prefix": "Ниже приведены вопросы с множественным выбором (с ответами) по",
        "answer_prefix": "Ответ:",
    },
}

def load_dataset_single(subject):
    return load_dataset(MMLU.NLPCORE_HF_PATH, name=subject, download_mode='reuse_dataset_if_exists')

def load_dataset_multiprocessing(subjects, num_proc=12):
    with Pool(processes=num_proc) as pool:
        datasets = [ds for ds in pool.map(load_dataset_single, subjects)]
    return datasets

class MMLU(Task):
    NLPCORE_HF_PATH = 'NLPCoreTeam/mmlu_ru'
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.method = 'calculate_tokens_proba'
        self._max_new_tokens = 64

    @property
    def choices(self):
        return ["A", "B", "C", "D"]

    def _per_category_mean(self, results: Dict) -> Dict:
        subjects = set([res['subject'] for res in results])
        assert len(subjects) == 57
        metric_per_subject = {}
        for subject in subjects:
            metric_per_subject[subject] = mean([res['val'] for res in results if res['subject'] == subject])

        category_to_main_category = {value: key for key, sublist in CATEGORIES.items() for value in sublist}
        subcategories2categories = {key: category_to_main_category[value[0]] for key, value in SUBCATEGORIES.items()}
        subjects = sorted(list(subjects))

        df = pd.DataFrame()
        df['subject'] = subjects
        df['metric'] = [metric_per_subject[s] for s in subjects]
        self.logger.info(df.groupby('subject').mean())
        df['subject'] = df['subject'].apply(lambda x: subcategories2categories[x])
        df = df.groupby('subject').mean()
        self.logger.info(df)

        return float(df.mean())

    def aggregation(self) -> Dict:
        return {"acc": self._per_category_mean}

    def evaluate(self, sample, y_pred) -> Dict:
        y_true = sample['answer']
        y_pred = sorted([pair for pair in y_pred.items()], key=lambda x: -x[1])[0][0]
        res = y_true == y_pred
        return {'acc': {'val' : res, 'subject': sample['subject']}}

    def load_dataset(self, model: LLM, max_len: int, max_sample_per_dataset: int, few_shot_count: int) -> Tuple[List[Dict], List[Dict]]:
        messages = []
        samples = []
        subjects = list(SUBCATEGORIES.keys())
        max_samples_per_subject = max_sample_per_dataset // len(subjects) + 1
        subject_datasets = load_dataset_multiprocessing(subjects, 12) #TODO: to params
        for i, dataset in enumerate(tqdm(subject_datasets)):
            subject = subjects[i]

            dataset_test = dataset['test']
            dataset_dev = dataset['dev']

            subject_samples = self._load_dataset(subject, dataset_test, dataset_dev, model, max_len, max_samples_per_subject, few_shot_count)

            subject_messages = [{'messages': s['messages']} for s in subject_samples]
            subject_samples = [{'sample': s['sample']} for s in subject_samples]

            messages += subject_messages
            samples += subject_samples

        for m in messages:
            m['tokens_of_interest'] = self.choices

        return messages, samples


    def _load_dataset(self, subject: str, dataset_test: Dataset, dataset_dev: Dataset, model: LLM, max_len: int, max_sample_per_dataset: int, few_shot_count: int):
        assert model.support_method(self.method)
        samples = []
        dataset_test = dataset_test.select(range(min(max_sample_per_dataset, len(dataset_test))))
        for sample in dataset_test:
            samples.append(self._prepare_messages(subject, sample, model, max_len, few_shot_count, dataset_dev))
        return samples

    def _prepare_messages(self, subject: str, sample: Dict, model: LLM, max_len: int, few_shot_count: int, few_shot_samples: Dataset) -> List:
        k = min(few_shot_count, len(few_shot_samples))
        int2str = few_shot_samples.features['answer'].int2str
        
        zero_shot_messages_with_headline = self._create_messages(subject, sample, int2str, add_headline=True, add_answer=False)
        zero_shot_messages_with_headline_len = model.count_tokens_for_prompt(model.apply_model_prompt(zero_shot_messages_with_headline))
        if zero_shot_messages_with_headline_len >= max_len:
            self.logger.warning(f'WARNING: sample zero-shot len {zero_shot_messages_with_headline_len} greater then {max_len}. Will be truncated.')

        zero_shot_messages_without_headline = self._create_messages(subject, sample, int2str, add_headline=False, add_answer=False)
        messages = copy.deepcopy(zero_shot_messages_without_headline)
        for i in range(k):
            if i == 0:
                few_shot_messages = self._create_messages(subject, few_shot_samples[i], int2str, add_headline=True, add_answer=True)
                _messages = few_shot_messages + messages
                few_shot_messages_len = model.count_tokens_for_prompt(model.apply_model_prompt(_messages))
            else:
                few_shot_messages = self._create_messages(subject, few_shot_samples[i], int2str, add_headline=False, add_answer=True)
                _messages = messages[:-2] + few_shot_messages + messages[-2:]
                few_shot_messages_len = model.count_tokens_for_prompt(model.apply_model_prompt(_messages))

            if few_shot_messages_len >= max_len:
                break

            messages = _messages

        sample['answer'] = int2str(sample['answer'])
        sample['subject'] = subject

        return {'messages': messages, 'sample': sample}

    def _create_messages(self, subject, sample, int2str, add_headline=True, add_answer=False):
        q_key = f'question_{self.lang}'
        choice_key = f'choices_{self.lang}'

        headline_prefix = LANGUAGE_CONFIG[self.lang]['headline_prefix']
        headline_postfix = self._get_pretty_subject(subject=subject, lang=self.lang)
        headline = f"{headline_prefix} {headline_postfix}.\n\n"

        answer_prefix = LANGUAGE_CONFIG[self.lang]['answer_prefix'].rstrip()

        q = sample[q_key]
        options = sample[choice_key]
        a = int2str(sample['answer'])

        lettered_options = [f"{x}. {y}" for x, y in zip(["A", "B", "C", "D"], options)]
        q_with_lettered_options = "\n".join([q] + lettered_options)
        if add_headline:
            q_with_lettered_options = headline + q_with_lettered_options

        answer = f'{answer_prefix}'
        if add_answer:
            answer += f' {a}'
            
        return [{'role': 'user', 'content': q_with_lettered_options}, {'role': 'bot', 'content': answer}]
        
    def _format_subject(self, subject: str) -> str:
        l = subject.split("_")
        s = ""
        for entry in l:
            s += " " + entry
        return s.strip()

    def _get_pretty_subject(self, subject: str, lang: str) -> str:
        return self._format_subject({
            "en": subject,
            "ru": SUBCATEGORIES_EN2RU[subject],  # predefined map
        }[self.lang])

            
class ruMMLU(MMLU):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.lang = 'ru'

    @classmethod
    def name(cls):
        return 'nlpcoreteam/ruMMLU'

class enMMLU(MMLU):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.lang = 'en'

    @classmethod
    def name(cls):
        return 'nlpcoreteam/enMMLU'