from typing import List, Dict
from abc import ABC
import re
from tqdm import tqdm
from datasets import load_dataset
import json
from .ner_abc import NerDictAbc, NerJsonAbc, NerInPlaceAbc
from llmtf.base import LLM

def check_sample(sample):
    # remove samples with [ or ]
    return not("[" in sample["query"] or "]" in sample["query"])

def process_tags(merged_ner: str) -> List[str]:
    tags = merged_ner.split(',')
    prev_base = ""
    for i, tag in enumerate(tags):
        mod_idx = tag.find('-') + 1
        tag_mod, tag_base = tag[:mod_idx], tag[mod_idx:]

        # replace I-TAGs appearing without B-TAG prior with their B_TAG counterparts
        if tag_mod == "I-" \
            and prev_base != tag_base:
            tags[i] = "B-" + tag_base
        # join adjecent entities with same base tags
        elif tag_mod == "B-" and prev_base == tag_base:
            tags[i] = "I-" + tag_base
        
        prev_base = tag_base
    return tags

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

def process_sample(sample):
    tags = process_tags(sample["merged_ner"])
    # same splitting result as with razdel
    tokens = re.findall(r"\d+.\d+|[\w]+|\.{3}|[.,!?:;()\[\]«»]", sample["query"])
    query = join_tokens(tokens)
    return {"query": query, "tokens": tokens, "tags": tags}

# CHILD tag was removed
ALL_TAGS = ["SIM", "SUBW", "GEN", "SPEC", "CHILD"]
DEFAULT_TAGS = ["SIM", "SUBW", "GEN", "SPEC"]

TAG_DESCRIPTIONS = {
    "SIM": "симптомы",
    "SUBW": "станция метро",
    "GEN": "пол",
    "SPEC": "специальность врача",
    "CHILD": "упоминание ребенка"
}

class PatientQueriesNerAbc(ABC):
    DATASET_PATH = "Mykes/patient_queries_ner"

    def __init__(
        self,
        instruction: str,
        tags=DEFAULT_TAGS,
        tag_descriptions: Dict[str, str]=TAG_DESCRIPTIONS,
        **kwargs
    ):
        self.TAGS = tags
        self.instruction = instruction.replace("{tags}", "\n".join(f"{tag} - {description}." for tag, description in tag_descriptions.items() if tag in self.TAGS))

    def dataset_args(self) -> Dict[str, str]:
        return {"path": self.DATASET_PATH}
    
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


PATIENT_QUERIES_NER_DICT_INSTRUCTION = """Извлеки из заданного ниже текста все именованные сущности всех представленных ниже классов.
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

class PatientQueriesNerDict(PatientQueriesNerAbc, NerDictAbc):
    def __init__(
        self,
        instruction: str = PATIENT_QUERIES_NER_DICT_INSTRUCTION,
        **kwargs
    ):
        PatientQueriesNerAbc.__init__(self, instruction=instruction, **kwargs)
        NerDictAbc.__init__(self, **kwargs)
        self._max_task_new_tokens = 75

    def task_name(self) -> str:
        return "Mykes/patient_queries_ner (dict)"

    def get_answer(self, sample) -> Dict[str, List[str]]:
        tagged_tokens = {tag: [] for tag in self.TAGS}
        for token, tag in zip(sample["tokens"], sample["tags"]):
            mod_idx = tag.find('-') + 1
            tag_mod, tag_base = tag[:mod_idx], tag[mod_idx:]

            if tag_base not in self.TAGS:
                continue
            if tag_mod == 'I-':
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


PATIENT_QUERIES_NER_JSON_INSTRUCTION = """Извлеки из заданного ниже текста все именованные сущности всех представленных ниже классов.
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

class PatientQueriesNerJson(PatientQueriesNerAbc, NerJsonAbc):
    def __init__(
        self,
        instruction: str = PATIENT_QUERIES_NER_JSON_INSTRUCTION,
        **kwargs
    ):
        PatientQueriesNerAbc.__init__(self, instruction=instruction, **kwargs)
        NerJsonAbc.__init__(self, **kwargs)
        self._max_task_new_tokens = 100

    def task_name(self) -> str:
        return 'Mykes/patient_queries_ner (json)'

    def get_answer(self, sample) -> List[str]:
        tagged_tokens = []
        last_tag_idx = {}
        i = 0
        for token, tag in zip(sample["tokens"], sample["tags"]):
            mod_idx = tag.find('-') + 1
            tag_mod, tag_base = tag[:mod_idx], tag[mod_idx:]

            if tag_base not in self.TAGS:
                continue
            if tag_mod == 'B-':
                tagged_tokens.append([tag_base, token])
                last_tag_idx[tag_base] = i
                i += 1
            elif tag_mod == 'I-':
                tagged_tokens[last_tag_idx[tag_base]][1] += " " + token
            else:
                tagged_tokens.append([tag_base, token])
                i += 1
        return tagged_tokens
    
    def get_answer_str(self, sample) -> str:
        answer = self.get_answer(sample)
        answer_str = '```json\n' + json.dumps(answer, ensure_ascii=False, indent=4).strip() + '\n```'
        return answer_str


PATIENT_QUERIES_NER_IN_PLACE_INSTRUCTION = """Твоя задача точно повторить заданный ниже текст, помечая все именованные сущности всех представленных ниже классов.
Сущности могут быть представлены целым словом или последовательностей слов.
Оставь сущности в том виде, в каком они даны в тексте, не изменяй и не склоняй их.

**Классы**
{tags}

**Формат вывода**
слово слово ... слово <класс>сущность</класс> слово ... слово

**Текст**
{text}
"""

class PatientQueriesNerInPlace(PatientQueriesNerAbc, NerInPlaceAbc):
    def __init__(
        self,
        instruction: str = PATIENT_QUERIES_NER_IN_PLACE_INSTRUCTION,
        **kwargs
    ):
        PatientQueriesNerAbc.__init__(self, instruction=instruction, **kwargs)
        NerInPlaceAbc.__init__(self, **kwargs)
        self._max_task_new_tokens = 256

    def task_name(self) -> str:
        return 'Mykes/patient_queries_ner (in place)'
    
    def get_answer(self, sample) -> List[str]:
        tagged_tokens = []
        last_tag_idx = {}
        i = 0
        for token, tag in zip(sample["tokens"], sample["tags"]):
            mod_idx = tag.find('-') + 1
            tag_mod, tag_base = tag[:mod_idx], tag[mod_idx:]

            if tag_base not in self.TAGS:
                continue
            if tag_mod == 'B-':
                tagged_tokens.append([tag_base, token])
                last_tag_idx[tag_base] = i
                i += 1
            elif tag_mod == 'I-':
                tagged_tokens[last_tag_idx[tag_base]][1] += " " + token
            else:
                tagged_tokens.append([tag_base, token])
                i += 1
        return tagged_tokens

    def get_answer_str(self, sample) -> str:
        new_tokens = []
        entity_tokens = []
        prev_tag_base = ""
        for token, tag in zip(sample["tokens"], sample["tags"]):
            mod_idx = tag.find('-') + 1
            tag_mod, tag_base = tag[:mod_idx], tag[mod_idx:]
    
            if not tag_mod == "I-" and entity_tokens:
                new_tokens.append(f"<{prev_tag_base}>{' '.join(entity_tokens)}</{prev_tag_base}>")
                entity_tokens = []
            if tag_base not in self.TAGS:
                new_tokens.append(token)
            elif tag_mod == "B-":
                entity_tokens = [token]
            elif tag_mod == "I-":
                entity_tokens.append(token)
            else:
                new_tokens.append(f"<{tag_base}>{token}</{tag_base}>]")
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
