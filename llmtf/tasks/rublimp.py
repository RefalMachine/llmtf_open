from llmtf.base import SimpleFewShotHFTask, LLM
from typing import List, Dict, Tuple
from tqdm import tqdm
from datasets import load_dataset, Dataset
from llmtf.metrics import mean, f1_macro_score

RUBLIMP_SLICES_ALL = ['add_new_suffix', 'add_verb_prefix', 'adposition_government', 'anaphor_agreement_gender', 'anaphor_agreement_number', 'change_declension_ending', 'change_declension_ending_has_dep', 'change_duration_aspect', 'change_repetition_aspect', 'change_verb_conjugation', 'change_verb_prefixes_order', 'clause_subj_predicate_agreement_gender', 'clause_subj_predicate_agreement_number', 'clause_subj_predicate_agreement_person', 'conj_verb_tense', 'deontic_imperative_aspect', 'external_possessor', 'floating_quantifier_agreement_case', 'floating_quantifier_agreement_gender', 'floating_quantifier_agreement_number', 'genitive_subj_predicate_agreement_gender', 'genitive_subj_predicate_agreement_number', 'genitive_subj_predicate_agreement_person', 'indefinite_pronoun_to_negative', 'negative_concord', 'negative_pronoun_to_indefinite', 'nominalization_case', 'noun_subj_predicate_agreement_gender', 'noun_subj_predicate_agreement_number', 'noun_subj_predicate_agreement_person', 'np_agreement_case', 'np_agreement_gender', 'np_agreement_number', 'single_verb_tense', 'subj_predicate_agreement_gender_attractor', 'subj_predicate_agreement_number_attractor', 'tense_marker', 'transitive_verb', 'transitive_verb_iobject', 'transitive_verb_object', 'transitive_verb_passive', 'transitive_verb_subject', 'verb_acc_object', 'verb_gen_object', 'verb_ins_object']
RUBLIMP_SLICES_TEST = ['add_new_suffix', 'add_verb_prefix', 'adposition_government', 'anaphor_agreement_gender']
RUBLIMP_CLASSIFY_INSTRUCTION_USER = """Ниже приведено предложение, в котором может содержаться морфологическая, синтаксическая или семантическая ошибка. Определи, есть ли в нём ошибка. Ответом должно служить одно число: 0 или 1, где:
0 - предложение верно
1 - предложение содержит ошибку
Предложение: "{}"
"""

RUBLIMP_CLASSIFY_INSTRUCTION_BOT = "Ответ: {}"

class RuBlimpClassify(SimpleFewShotHFTask):
    DATASET_PATH = 'RussianNLP/rublimp'

    def __init__(
        self,
        dataset_slices: List[str] = RUBLIMP_SLICES_ALL,
        instruction_user: str = RUBLIMP_CLASSIFY_INSTRUCTION_USER,
        instruction_bot: str = RUBLIMP_CLASSIFY_INSTRUCTION_BOT,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.dataset_slices = dataset_slices if type(dataset_slices) is list else [dataset_slices]
        self.instruction_user = instruction_user
        self.instruction_bot = instruction_bot
        self.method = "generate"
        self._max_task_new_tokens = 1

    def task_name(self):
        return 'russiannlp/rublimp-(classify)'

    def test_split_name(self) -> str:
        return 'train'

    def prompt_split_name(self) -> str:
        return 'train'
    
    def dataset_args(self):
        for dataset_slice in self.dataset_slices:
            yield {
                "path": self.DATASET_PATH,
                "name": dataset_slice,
            }

    def create_messages(self, sample, with_answer: bool):
        messages = []
        correct = sample['correct']
        instruction_user = self.instruction_user.format(
            sample["source_sentence"] if correct else sample["target_sentence"])
        instruction_bot = self.instruction_bot.format(0 if correct else 1)
        instruction_bot_incomplete = self.instruction_bot.format("")

        bot_content = instruction_bot if with_answer else instruction_bot_incomplete

        messages.append({'role': 'user', 'content': instruction_user})
        messages.append({'role': 'bot', 'content': bot_content})
        return messages

    def load_dataset(self, model: LLM, max_len: int, max_sample_per_dataset: int, few_shot_count: int) -> Tuple[List[Dict], List[Dict]]:
        assert model.support_method(self.method)

        samples = []
        dataset_args_generator = self.dataset_args()
        for _ in self.dataset_slices:
            samples += self._load_dataset(model, max_len, max_sample_per_dataset, few_shot_count, next(dataset_args_generator))
        messages = [{'messages': s['messages']} for s in samples]
        samples = [{'sample': s['sample']} for s in samples]

        return messages, samples
    
    def _load_dataset(self, model: LLM, max_len: int, max_sample_per_dataset: int, few_shot_count: int, dataset_args) -> List:
        samples = []
        dataset = load_dataset(**dataset_args)
        test_dataset = dataset[self.test_split_name()]
        prompt_dataset = dataset[self.prompt_split_name()]
        
        test_dataset_sample_ids = list(range(min(max_sample_per_dataset, len(test_dataset))))
        prompt_dataset_sample_ids = list(range(self.prompt_dataset_start_idx(), min(self.prompt_dataset_start_idx() + few_shot_count, len(prompt_dataset))))
        if self.test_split_name() == self.prompt_split_name():
            prompt_dataset_sample_ids_set = set(prompt_dataset_sample_ids)
            test_dataset_sample_ids = [i for i in test_dataset_sample_ids if i not in prompt_dataset_sample_ids_set]
            
        test_dataset = test_dataset.select(test_dataset_sample_ids)
        prompt_dataset = prompt_dataset.select(prompt_dataset_sample_ids)
        def add_correct(example, idx):
            example['correct'] = bool(idx % 2)
            return example
        prompt_dataset = prompt_dataset.map(add_correct, with_indices=True)
        for sample in tqdm(test_dataset):
            corrcet_sample = sample.copy()
            incorrect_sample = sample.copy()
            corrcet_sample['correct'] = True
            incorrect_sample['correct'] = False
            samples.append({'messages': self._prepare_messages(corrcet_sample, model, max_len, few_shot_count, prompt_dataset), 'sample': corrcet_sample})
            samples.append({'messages': self._prepare_messages(incorrect_sample, model, max_len, few_shot_count, prompt_dataset), 'sample': incorrect_sample})
        return samples
    
    def evaluate(self, sample, y_pred) -> Dict:
        y_true = '0' if sample['correct'] else '1'
        y_pred = y_pred.strip()[0]
        return {'acc': y_true == y_pred, 'f1_macro': (y_true, y_pred)}

    def aggregation(self) -> Dict:
        return {"acc": mean, "f1_macro": f1_macro_score}


RUBLIMP_CHOICE_INSTRUCTION_USER = """Перед тобой два предложения, в одном из них содержится морфологическая, синтаксическая или семантическая ошибка. Напиши только номер предложения с ошибкой и ничего более.
1. {}
2. {}"""

RUBLIMP_CHOICE_INSTRUCTION_BOT = "Номер предложения с ошибкой: {}"

class RuBlimpChoice(SimpleFewShotHFTask):
    DATASET_PATH = 'RussianNLP/rublimp'

    def __init__(
        self,
        dataset_slices: List[str] = RUBLIMP_SLICES_ALL,
        instruction_user: str = RUBLIMP_CHOICE_INSTRUCTION_USER,
        instruction_bot: str = RUBLIMP_CHOICE_INSTRUCTION_BOT,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.dataset_slices = dataset_slices if type(dataset_slices) is list else [dataset_slices]
        self.instruction_user = instruction_user
        self.instruction_bot = instruction_bot
        self.method = "generate"
        self._max_task_new_tokens = 1

    def task_name(self):
        return 'russiannlp/rublimp-(choice)'

    def test_split_name(self) -> str:
        return 'train'

    def prompt_split_name(self) -> str:
        return 'train'
    
    def dataset_args(self):
        for dataset_slice in self.dataset_slices:
            yield {
                "path": self.DATASET_PATH,
                "name": dataset_slice,
            }

    def create_messages(self, sample, with_answer: bool):
        messages = []
        first_sentance = sample["source_sentence"]
        second_sentance = sample["target_sentence"]
        swap = sample['swap']
        if swap:
            tmp = first_sentance
            first_sentance = second_sentance
            second_sentance = tmp
        
        instruction_user = self.instruction_user.format(first_sentance, second_sentance)
        instruction_bot = self.instruction_bot.format(1 if swap else 2)
        instruction_bot_incomplete = self.instruction_bot.format("")

        bot_content = instruction_bot if with_answer else instruction_bot_incomplete

        messages.append({'role': 'user', 'content': instruction_user})
        messages.append({'role': 'bot', 'content': bot_content})
        return messages
    
    def load_dataset(self, model: LLM, max_len: int, max_sample_per_dataset: int, few_shot_count: int) -> Tuple[List[Dict], List[Dict]]:
        assert model.support_method(self.method)

        samples = []
        dataset_args_generator = self.dataset_args()
        for _ in self.dataset_slices:
            samples += self._load_dataset(model, max_len, max_sample_per_dataset, few_shot_count, next(dataset_args_generator))
        messages = [{'messages': s['messages']} for s in samples]
        samples = [{'sample': s['sample']} for s in samples]

        return messages, samples
    
    def _load_dataset(self, model: LLM, max_len: int, max_sample_per_dataset: int, few_shot_count: int, dataset_args) -> List:
        samples = []
        dataset = load_dataset(**dataset_args)
        test_dataset = dataset[self.test_split_name()]
        prompt_dataset = dataset[self.prompt_split_name()]
        
        test_dataset_sample_ids = list(range(min(max_sample_per_dataset, len(test_dataset))))
        prompt_dataset_sample_ids = list(range(self.prompt_dataset_start_idx(), min(self.prompt_dataset_start_idx() + few_shot_count, len(prompt_dataset))))
        if self.test_split_name() == self.prompt_split_name():
            prompt_dataset_sample_ids_set = set(prompt_dataset_sample_ids)
            test_dataset_sample_ids = [i for i in test_dataset_sample_ids if i not in prompt_dataset_sample_ids_set]
            
        test_dataset = test_dataset.select(test_dataset_sample_ids)
        prompt_dataset = prompt_dataset.select(prompt_dataset_sample_ids)
        def add_swap(example, idx):
            example['swap'] = bool(idx % 2)
            return example
        prompt_dataset = prompt_dataset.map(add_swap, with_indices=True)
        for i, sample in enumerate(tqdm(test_dataset)):
            sample = sample.copy()
            sample['swap'] = bool(i % 2)
            samples.append({'messages': self._prepare_messages(sample, model, max_len, few_shot_count, prompt_dataset), 'sample': sample})
        return samples

    def evaluate(self, sample, y_pred) -> Dict:
        y_true = '1' if sample['swap'] else '2'
        y_pred = y_pred.strip()[0]
        return {'acc': y_true == y_pred, 'f1_macro': (y_true, y_pred)}

    def aggregation(self) -> Dict:
        return {"acc": mean, "f1_macro": f1_macro_score}
