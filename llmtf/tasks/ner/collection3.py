from typing import List, Dict
from abc import ABC
import re
from tqdm import tqdm
from datasets import load_dataset
import json
from .ner_abc import NerDictAbc, NerJsonAbc, NerInPlaceAbc
from llmtf.base import LLM

def check_sample(sample):
    tags = sample["ner_tags"]
    prev_base = ""
    for tag in tags:
        tag_b = tag % 2 == 1 if tag != 0 else True
        tag_base = (tag + 1) // 2

        # remove samples where I-TAGs appearing without B-TAG prior with their B_TAG counterparts
        if not tag_b and prev_base != tag_base:
            return False
        
        prev_base = tag_base
    return True

NO_SPACE_AFTER = set(list(".,!?:;»)") + ["..."])
NO_SPACE_BEFORE = set("(«")

def join_tokens(tokens):
    text = ""
    prev_token = ""
    for token in tokens:
        if token not in NO_SPACE_AFTER and prev_token not in NO_SPACE_BEFORE:
            text += " "
        text += token
        prev_token = token
    return text[1:]

def remove_sp(sample):
    for i, token in enumerate(sample["tokens"]):
        if token in ["[", "]"]:
            del sample["tokens"][i]
            del sample["ner_tags"][i]

def process_sample(sample):
    remove_sp(sample)
    query = join_tokens(sample["tokens"])
    return {"query": query, "tokens": sample["tokens"], "tags": sample["ner_tags"]}

ALL_TAGS = ["PER", "ORG", "LOC"]
IDX_TO_TAGS = ["O", "PER", "ORG", "LOC"]

TAG_DESCRIPTIONS = {
    "PER": "человек (с именем)",
    "ORG": "организация",
    "LOC": "местоположение",
}

class Collection3Abc(ABC):
    DATASET_PATH = "RCC-MSU/collection3"
    DATASET_SLICE = "main"

    def __init__(
        self,
        instruction: str,
        tags=ALL_TAGS,
        tag_descriptions: Dict[str, str]=TAG_DESCRIPTIONS,
        **kwargs
    ):
        self.TAGS = tags
        self.instruction = instruction.replace("{tags}", "\n".join(f"{tag} - {description}." for tag, description in tag_descriptions.items() if tag in self.TAGS))

    def dataset_args(self) -> Dict[str, str]:
        return {"path": self.DATASET_PATH, "name": self.DATASET_SLICE, "trust_remote_code": True}
    
    def test_split_name(self) -> str:
        return 'test'

    def prompt_split_name(self) -> str:
        return 'validation'
    
    def _load_dataset(self, model: LLM, max_len: int, max_sample_per_dataset: int, few_shot_count: int) -> List:
        samples = []
        dataset = load_dataset(**self.dataset_args())
        test_dataset = dataset[self.test_split_name()].filter(check_sample).map(process_sample)
        prompt_dataset = dataset[self.prompt_split_name()].filter(check_sample).map(process_sample)
        
        test_dataset_sample_ids = list(range(min(max_sample_per_dataset, len(test_dataset))))
        prompt_dataset_sample_ids = list(range(self.prompt_dataset_start_idx(), min(self.prompt_dataset_start_idx() + few_shot_count, len(prompt_dataset))))

        test_dataset = test_dataset.select(test_dataset_sample_ids)
        prompt_dataset = prompt_dataset.select(prompt_dataset_sample_ids)
        for sample in tqdm(test_dataset):
            samples.append({'messages': self._prepare_messages(sample, model, max_len, few_shot_count, prompt_dataset), 'sample': sample})
        return samples


COLLECTION3_DICT_INSTRUCTION = """Извлеки из заданного ниже текста все именованные сущности всех представленных ниже классов.
Сущности могут быть представлены целым словом или последовательностей слов, разделенных пробелами.
Оставь сущности в том виде, в каком они даны в тексте, не изменяй и не склоняй их.

**Классы**
{tags}

**Формат вывода**
Для каждого класса: "класс: [сущность, ..., сущность]". Вместо "класс" используй соответствующие классы, представленные выше. Сущности каждого класса выведи на отдельной строке.
Если сущностей соответствующего класса в тексте нет, выведи на соответствующей строке "класс: []".

класс: [сущность, сущность, ... сущность]
...
класс: []
класс: [сущность, ... сущность]

**Текст**
{text}
"""

class Collection3Dict(Collection3Abc, NerDictAbc):
    def __init__(
        self,
        instruction: str = COLLECTION3_DICT_INSTRUCTION,
        **kwargs
    ):
        Collection3Abc.__init__(self, instruction=instruction, **kwargs)
        NerDictAbc.__init__(self, **kwargs)
        self._max_task_new_tokens = 75

    def task_name(self) -> str:
        return "RCC-MSU/collection3-(dict)"

    def get_answer(self, sample) -> Dict[str, List[str]]:
        tagged_tokens = {tag: [] for tag in self.TAGS}
        for token, tag in zip(sample["tokens"], sample["tags"]):
            tag_b = tag % 2 == 1 if tag != 0 else True
            tag_base = IDX_TO_TAGS[(tag + 1) // 2]

            if tag_base not in self.TAGS:
                continue
            if not tag_b:
                tagged_tokens[tag_base][-1] += " " + token
            else:
                tagged_tokens[tag_base].append(token)
        if 'O' in tagged_tokens.keys():
            del tagged_tokens['O']
        return tagged_tokens
    
    def get_answer_str(self, sample) -> str:
        answer = self.get_answer(sample)
        answer_str = ""
        for tag, tokens in answer.items():
            answer_str += f"{tag}: [" + ', '.join(tokens) + "]\n"
        return answer_str


COLLECTION3_JSON_INSTRUCTION = """Извлеки из заданного ниже текста все именованные сущности всех представленных ниже классов.
Сущности могут быть представлены целым словом или последовательностей слов, разделенных пробелами.
Оставь сущности в том виде, в каком они даны в тексте, не изменяй и не склоняй их.

**Классы**
{tags}

**Формат вывода (только json)**
```json
[["класс", "сущность"], ["класс", "сущность"], ...]
```

**Текст**
{text}
"""

class Collection3Json(Collection3Abc, NerJsonAbc):
    def __init__(
        self,
        instruction: str = COLLECTION3_JSON_INSTRUCTION,
        **kwargs
    ):
        Collection3Abc.__init__(self, instruction=instruction, **kwargs)
        NerJsonAbc.__init__(self, **kwargs)
        self._max_task_new_tokens = 100

    def task_name(self) -> str:
        return 'RCC-MSU/collection3-(json)'

    def get_answer(self, sample) -> List[str]:
        tagged_tokens = []
        last_tag_idx = {}
        i = 0
        for token, tag in zip(sample["tokens"], sample["tags"]):
            tag_b = tag % 2 == 1 if tag != 0 else True
            tag_base = IDX_TO_TAGS[(tag + 1) // 2]

            if tag_base not in self.TAGS:
                continue
            if tag_b:
                tagged_tokens.append([tag_base, token])
                last_tag_idx[tag_base] = i
                i += 1
            else:
                tagged_tokens[last_tag_idx[tag_base]][1] += " " + token
        return tagged_tokens
    
    def get_answer_str(self, sample) -> str:
        answer = self.get_answer(sample)
        answer_str = '```json\n' + json.dumps(answer, ensure_ascii=False, indent=4).strip() + '\n```'
        return answer_str


COLLECTION3_IN_PLACE_INSTRUCTION = """Твоя задача точно повторить заданный ниже текст, помечая все именованные сущности всех представленных ниже классов.
Сущности могут быть представлены целым словом или последовательностей слов.
Оставь сущности в том виде, в каком они даны в тексте, не изменяй и не склоняй их.

**Классы**
{tags}

**Формат вывода**
слово слово ... слово <класс>сущность</класс> слово ... слово

**Текст**
{text}
"""

class Collection3InPlace(Collection3Abc, NerInPlaceAbc):
    def __init__(
        self,
        instruction: str = COLLECTION3_IN_PLACE_INSTRUCTION,
        **kwargs
    ):
        Collection3Abc.__init__(self, instruction=instruction, **kwargs)
        NerInPlaceAbc.__init__(self, **kwargs)
        self._max_task_new_tokens = 256

    def task_name(self) -> str:
        return 'RCC-MSU/collection3-(in-place)'
    
    def get_answer(self, sample) -> List[str]:
        tagged_tokens = []
        last_tag_idx = {}
        i = 0
        for token, tag in zip(sample["tokens"], sample["tags"]):
            tag_b = tag % 2 == 1 if tag != 0 else True
            tag_base = IDX_TO_TAGS[(tag + 1) // 2]

            if tag_base not in self.TAGS:
                continue
            if tag_b:
                tagged_tokens.append([tag_base, token])
                last_tag_idx[tag_base] = i
                i += 1
            else:
                tagged_tokens[last_tag_idx[tag_base]][1] += " " + token
        return tagged_tokens

    def get_answer_str(self, sample) -> str:
        new_tokens = []
        entity_tokens = []
        prev_tag_base = ""
        for token, tag in zip(sample["tokens"], sample["tags"]):
            tag_b = tag % 2 == 1 if tag != 0 else True
            tag_base = IDX_TO_TAGS[(tag + 1) // 2]
    
            if tag_b and entity_tokens:
                new_tokens.append(f"<{prev_tag_base}>{' '.join(entity_tokens)}</{prev_tag_base}>")
                entity_tokens = []
            if tag_base not in self.TAGS:
                new_tokens.append(token)
            elif tag_b:
                entity_tokens = [token]
            else:
                entity_tokens.append(token)
            prev_tag_base = tag_base
        if len(entity_tokens) > 0:
            new_tokens.append(f"<{tag_base}>{' '.join(entity_tokens)}</{tag_base}>")
        return join_tokens(new_tokens)

    def check_text(self, sample, gen_pred: str):
        text_pred = re.sub(r"<\w+>|</\w+>", "", gen_pred)
        tokens_pred = re.findall(r"\d+.\d+|[\w]+|\.{3}|[.,!?:;()\[\]«»]", text_pred)
        for token_gold, token_pred in zip(sample["tokens"], tokens_pred):
            if token_gold != token_pred:
                return False
        return True
