from typing import Dict, List
from abc import ABC
import re
from itertools import chain
from tqdm import tqdm
import json
from datasets import Dataset, load_dataset
from llmtf.base import LLM 
from .ner_abc import NerDictAbc, NerJsonAbc, NerInPlaceAbc

ALL_TAGS_NEREL = [
    "AGE", "AWARD", "CITY", "COUNTRY", "CRIME", "DATE", "DISEASE", "DISTRICT",
    "EVENT", "FACILITY", "FAMILY", "IDEOLOGY", "LANGUAGE", "LAW", "LOCATION",
    "MONEY", "NATIONALITY", "NUMBER", "ORDINAL", "ORGANIZATION", "PENALTY",
    "PERCENT", "PERSON", "PRODUCT", "PROFESSION", "RELIGION", "STATE_OR_PROVINCE",
    "TIME", "WORK_OF_ART"
]

REMOVE_TAGS = ["EVENT", "LAW"]
DEFAULT_TAGS_NEREL = [item for item in ALL_TAGS_NEREL if item not in REMOVE_TAGS]

TAG_DESCRIPTIONS_NEREL = {
    "AGE": "возраст",
    "AWARD": "награда / орден / медаль",
    "CITY": "город",
    "COUNTRY": "страна",
    "CRIME": "преступление",
    "DATE": "дата",
    "DISEASE": "болезнь",
    "DISTRICT": "район города",
    "EVENT": "событие",
    "FACILITY": "здание / учреждение",
    "FAMILY": "семья с конкретной фамилией",
    "IDEOLOGY": "идеология",
    "LANGUAGE": "язык",
    "LAW": "закон",
    "LOCATION": "местоположение",
    "MONEY": "деньги",
    "NATIONALITY": "национальность",
    "NUMBER": "число",
    "ORDINAL": "порядковый номер",
    "ORGANIZATION": "организация",
    "PENALTY": "наказание за преступление",
    "PERCENT": "проценты",
    "PERSON": "конкретный человек с ФИО",
    "PROFESSION": "профессия",
    "RELIGION": "религия",
    "STATE_OR_PROVINCE": "штат или конкретная область / субьект / округ",
    "TIME": "время",
    "WORK_OF_ART": "произведение искусства"
}

def process_entities_nerel(sample):
    entities = []
    for entity in sample["entities"]:
        match = re.match(r"(?P<id>\w+)\s(?P<tag>\w+)\s(?P<b_e_pairs>\d+\s\d+(?:;\d+\s\d+)*)\s(?P<text>.+)", entity)
        tag = match.group("tag")
        b_e_pairs = match.group("b_e_pairs").split(';')
        text = match.group("text")
        for b_e_pair in b_e_pairs:
            begin, end = map(int, b_e_pair.split())
            entities.append(
                {
                    "tag": tag,
                    "begin": begin,
                    "end": end,
                    "text": text
                }
            )
    entities.sort(key=lambda x: (x["begin"], -x["end"]))
    return entities

def process_sample(process_entities_func, sample, do_split, split_by = "\n", skip_empty=True):
    entities = process_entities_func(sample)

    if do_split:
        sentences = sample["text"].split(split_by)
    else:
        sentences = [sample["text"]]
    samples = []
    i = 0
    prev_len, current_len = (0, 0)
    entity_count = len(entities)
    for sentence in sentences:
        sentence_entities = []
        prev_len = current_len
        current_len += len(sentence) + 1
        while i < entity_count and prev_len <= entities[i]["begin"] < current_len:
            entities[i]["begin"] -= prev_len
            entities[i]["end"] -= prev_len
            sentence_entities.append(entities[i])
            i += 1
        if len(sentence) > 5 and (not skip_empty or sentence_entities):
            samples.append({
                "query": sentence,
                "entities": sentence_entities
            })
        prev_len = current_len
    return samples

def process_sample_nerel(sample, do_split):
    return process_sample(process_entities_nerel, sample, do_split)
    
def process_dataset(proccess_func, dataset, do_split):
    return Dataset.from_list(
        list(chain.from_iterable(
                proccess_func(sample, do_split) for sample in dataset
        )))

class NestedNerAbc(ABC):
    DATASET_PATH = "MalakhovIlya/NEREL"
    DATASET_SLICE = "data"
    
    def __init__(
        self,
        instruction: str,
        tags: List[str]=DEFAULT_TAGS_NEREL,
        tag_descriptions: Dict[str, str]=TAG_DESCRIPTIONS_NEREL,
        do_split=True,
        **kwargs
    ):
        self.TAGS = tags
        self.instruction = instruction.replace("{tags}", "\n".join(f"{tag} - {description}." for tag, description in tag_descriptions.items() if tag in self.TAGS))
        self.do_split = do_split

    def dataset_args(self) -> Dict[str, str]:
        return {"path": self.DATASET_PATH, "name": self.DATASET_SLICE}
    
    def test_split_name(self) -> str:
        return 'test'

    def prompt_split_name(self) -> str:
        return 'train'
    
    def _load_dataset(self, model: LLM, max_len: int, max_sample_per_dataset: int, few_shot_count: int) -> List:
        samples = []
        dataset = load_dataset(**self.dataset_args())
        test_dataset = process_dataset(process_sample_nerel, dataset[self.test_split_name()], self.do_split)
        prompt_dataset = process_dataset(process_sample_nerel, dataset[self.prompt_split_name()], self.do_split)
        
        test_dataset_sample_ids = list(range(min(max_sample_per_dataset, len(test_dataset))))
        prompt_dataset_sample_ids = list(range(self.prompt_dataset_start_idx(), min(self.prompt_dataset_start_idx() + few_shot_count, len(prompt_dataset))))

        test_dataset = test_dataset.select(test_dataset_sample_ids)
        prompt_dataset = prompt_dataset.select(prompt_dataset_sample_ids)
        for sample in tqdm(test_dataset):
            samples.append({'messages': self._prepare_messages(sample, model, max_len, few_shot_count, prompt_dataset), 'sample': sample})
        return samples


NEREL_DICT_INSTRUCTION = """Извлеки из заданного ниже текста все вложенные именованные сущности всех представленных ниже классов.
Сущности могут быть представлены только целым словом, окружённым пробелами или знаками препинания, либо непрерывной последовательностью целых слов, разделённых пробелами.
Оставь сущности в том виде, в каком они даны в тексте, не изменяй и не склоняй их.

**Классы**

{tags}

Требуемый формат для каждого класса: "Класс: ["сущность", ..., "сущность"]". Вместо "Класс" используй соответствующие классы, представленные выше. Сущности каждого класса выведи на отдельной строке.
Если сущностей соответствующего класса в тексте нет, выведи на соответствующей строке "Класс: []".

Теперь извлеки вложенные именованные сущности для следующего текста.

**Текст**

{text}
"""

class NestedNerDict(NestedNerAbc, NerDictAbc):
    def __init__(
        self,
        instruction: str=NEREL_DICT_INSTRUCTION,
        **kwargs
    ):
        NestedNerAbc.__init__(self, instruction=instruction, **kwargs)
        NerDictAbc.__init__(self, **kwargs)
        self._max_new_tokens = 75

    def task_name(self) -> str:
        return "MalakhovIlya/NEREL (dict)"

    def get_answer(self, sample) -> Dict[str, List[str]]:
        tagged_tokens = {tag: [] for tag in self.TAGS}
        for entity in sample["entities"]:
            if entity["tag"] in self.TAGS:
                tagged_tokens[entity["tag"]].append(entity["text"])
        return tagged_tokens
    
    def get_answer_str(self, sample) -> str:
        answer = self.get_answer(sample)
        answer_str = ""
        for tag, tokens in answer.items():
            answer_str += f"{tag}: [" + ', '.join(tokens) + "]\n"
        return answer_str

NEREL_JSON_INSTRUCTION = """Извлеки из заданного ниже текста все вложенные именованные сущности всех представленных ниже классов.
Сущности могут быть представлены только целым словом, окружённым пробелами или знаками препинания, либо непрерывной последовательностью целых слов, разделённых пробелами.
Оставь сущности в том виде, в каком они даны в тексте, не изменяй и не склоняй их.

**Классы**
{tags}

Требуемый формат вывода (только json):
```json
[["тип", "сущность"], ["тип", "сущность"], ...]
```

Теперь извлеки вложенные именованные сущности для следующего текста.

**Текст**

{text}
"""

class NestedNerJson(NestedNerAbc, NerJsonAbc):
    def __init__(
        self,
        instruction: str=NEREL_JSON_INSTRUCTION,
        **kwargs
    ):
        NestedNerAbc.__init__(self, instruction=instruction, **kwargs)
        NerJsonAbc.__init__(self, **kwargs)
        self._max_new_tokens = 100

    def task_name(self) -> str:
        return "MalakhovIlya/NEREL (json)"

    def get_answer(self, sample) -> Dict[str, List[str]]:
        tagged_tokens = []
        for entity in sample["entities"]:
            if entity["tag"] in self.TAGS:
                tagged_tokens.append([entity["tag"], entity["text"]])
        return tagged_tokens
    
    def get_answer_str(self, sample) -> str:
        answer = self.get_answer(sample)
        answer_str = '```json\n' + json.dumps(answer, ensure_ascii=False, indent=4).strip() + '\n```'
        return answer_str


def get_answer_str_nested(sample) -> str:
    events = []
    for entity in sample["entities"]:
        events.append(("begin", entity["begin"], entity["end"], entity["tag"]))
        events.append(("end", entity["end"], entity["begin"], entity["tag"]))
    events.sort(key=lambda x: (x[1], 0 if x[0] == "end" else 1, -x[2]))

    text = sample["query"]
    tagged_text = ""
    prev_len = 0
    new_len = 0
    for event in events:
        new_len = event[1]
        tagged_text += text[prev_len:new_len]
        prev_len = new_len
        if event[0] == "begin":
            tagged_text += f"<{event[3]}>"
        else:
            tagged_text += f"</{event[3]}>"
    return tagged_text

def extract_answer_nested(text):
    entities = []
    stack = []
    pos = 0
    
    while pos < len(text):
        open_match = re.search(r'<(\w+)>', text[pos:])
        if not open_match:
            break
            
        open_tag = open_match.group(1)
        open_start = pos + open_match.start()
        open_end = pos + open_match.end()
        
        tag_stack = 1
        search_pos = open_end
        found_closing = False
        
        while tag_stack > 0 and search_pos < len(text):
            next_open = re.search(r'<(\w+)>', text[search_pos:])
            next_close = re.search(r'</(\w+)>', text[search_pos:])
            
            open_pos = next_open.start() if next_open else float('inf')
            close_pos = next_close.start() if next_close else float('inf')
            
            if open_pos < close_pos:
                tag_stack += 1
                search_pos += next_open.end()
            else:
                if next_close is None:
                    break
                    
                tag_stack -= 1
                if tag_stack == 0:
                    close_tag = next_close.group(1)
                    close_start = search_pos + next_close.start()
                    close_end = search_pos + next_close.end()
                    
                    if close_tag == open_tag:
                        inner_text = text[open_end:close_start]
                        clean_text = re.sub(r'<\/?\w+>', '', inner_text).strip()
                        
                        entities.append([open_tag, clean_text])
                        
                        nested_entities = extract_answer_nested(inner_text)
                        entities.extend(nested_entities)
                        
                        pos = close_end
                        found_closing = True
                        break
                search_pos += next_close.end()
        
        if not found_closing:
            pos = open_end
    
    return entities

NEREL_IN_PLACE_INSTRUCTION = """Твоя задача точно повторить заданный ниже текст, помечая все именованные сущности всех представленных ниже классов.
Сущности могут быть представлены только целым словом, окружённым пробелами или знаками препинания, либо непрерывной последовательностью целых слов, разделённых пробелами.
Оставь сущности в том виде, в каком они даны в тексте, не изменяй и не склоняй их.

**Классы**

{tags}

Требуемый формат вывода:

слово слово ... слово <тип>сущность</тип> слово ... слово

Теперь извлеки вложенные именованные сущности для следующего текста.

**Текст**

{text}
"""

class NestedNerInPlace(NestedNerAbc, NerInPlaceAbc):
    def __init__(
        self,
        instruction: str=NEREL_IN_PLACE_INSTRUCTION,
        **kwargs
    ):
        NestedNerAbc.__init__(self, instruction=instruction, **kwargs)
        NerInPlaceAbc.__init__(self, **kwargs)
        self._max_new_tokens = 256

    def task_name(self) -> str:
        return "MalakhovIlya/NEREL (in-place)"

    def get_answer(self, sample) -> Dict[str, List[str]]:
        tagged_tokens = []
        for entity in sample["entities"]:
            if entity["tag"] in self.TAGS:
                tagged_tokens.append([entity["tag"], entity["text"]])
        return tagged_tokens

    def get_answer_str(self, sample) -> str:
        return get_answer_str_nested(sample)

    def extract_answer(self, gen_pred):
        return extract_answer_nested(gen_pred)

    def check_text(self, sample, gen_pred: str):
        text_pred = re.sub(r"<\w+>|</\w+>", "", gen_pred)
        return text_pred.strip() == sample["query"].strip()


# simplified version of NEREL-BIO with less entity types

ALL_TAGS_NEREL_BIO = ["FINDING", "DISO", "INJURY_POISONING", "PHYS", "DEVICE", "LABPROC", "ANATOMY", "CHEM"]

TAG_DESCRIPTIONS_NEREL_BIO = {
    "FINDING": "признаки, симптомы и отклонения, обнаруженные в ходе обследования.",
    "DISO": "расстройства, заболевания, синдромы",
    "INJURY_POISONING": "травмы, отравления и другие повреждения.",
    "PHYS": "физиологические показатели, процессы, явления",
    "DEVICE": "медицинские приборы, оборудование, инструменты.",
    "LABPROC": "лабораторные исследования, анализы и диагностические процедуры.",
    "ANATOMY": "анатомические структуры",
    "CHEM": "химические вещества/лекарства"
}

def process_entities_nerel_bio(sample):
    entities = []
    for i in range(len(sample["span_entity_types"])):
        entities.append(
            {
                "tag": sample["span_entity_types"][i],
                "begin": sample["span_entity_start_chars"][i],
                "end": sample["span_entity_end_chars"][i],
                "text": sample["span_entity_surface"][i]
            }
        )
    entities.sort(key=lambda x: x["begin"])
    return entities

def process_sample_nerel_bio(sample, do_split):
    return process_sample(process_entities_nerel_bio, sample, do_split)

class NerelBioAbc(ABC):
    DATASET_PATH = "RefalMachine/nerel-bio-simple"
    
    def __init__(
        self,
        instruction: str,
        tags: List[str]=ALL_TAGS_NEREL_BIO,
        tag_descriptions: Dict[str, str]=TAG_DESCRIPTIONS_NEREL_BIO,
        do_split=True,
        **kwargs
    ):
        self.TAGS = tags
        self.instruction = instruction.replace("{tags}", "\n".join(f"{tag} - {description}." for tag, description in tag_descriptions.items() if tag in self.TAGS))
        self.do_split = do_split

    def dataset_args(self) -> Dict[str, str]:
        return {"path": self.DATASET_PATH}
    
    def test_split_name(self) -> str:
        return 'test'

    def prompt_split_name(self) -> str:
        return 'validation'
    
    def _load_dataset(self, model: LLM, max_len: int, max_sample_per_dataset: int, few_shot_count: int) -> List:
        samples = []
        dataset = load_dataset(**self.dataset_args())
        test_dataset = process_dataset(process_sample_nerel_bio, dataset[self.test_split_name()], self.do_split)
        prompt_dataset = process_dataset(process_sample_nerel_bio, dataset[self.prompt_split_name()], self.do_split)
        
        test_dataset_sample_ids = list(range(min(max_sample_per_dataset, len(test_dataset))))
        prompt_dataset_sample_ids = list(range(self.prompt_dataset_start_idx(), min(self.prompt_dataset_start_idx() + few_shot_count, len(prompt_dataset))))

        test_dataset = test_dataset.select(test_dataset_sample_ids)
        prompt_dataset = prompt_dataset.select(prompt_dataset_sample_ids)
        for sample in tqdm(test_dataset):
            samples.append({'messages': self._prepare_messages(sample, model, max_len, few_shot_count, prompt_dataset), 'sample': sample})
        return samples


class NerelBioDict(NerelBioAbc, NerDictAbc):
    def __init__(
        self,
        instruction: str=NEREL_DICT_INSTRUCTION,
        **kwargs
    ):
        NerelBioAbc.__init__(self, instruction=instruction, **kwargs)
        NerDictAbc.__init__(self, **kwargs)
        self._max_new_tokens = 150

    def task_name(self) -> str:
        return "nerel-ds/NEREL-BIO (dict)"

    def get_answer(self, sample) -> Dict[str, List[str]]:
        tagged_tokens = {tag: [] for tag in self.TAGS}
        for entity in sample["entities"]:
            if entity["tag"] in self.TAGS:
                tagged_tokens[entity["tag"]].append(entity["text"])
        return tagged_tokens
    
    def get_answer_str(self, sample) -> str:
        answer = self.get_answer(sample)
        answer_str = ""
        for tag, tokens in answer.items():
            answer_str += f"{tag}: [" + ', '.join(tokens) + "]\n"
        return answer_str


class NerelBioJson(NerelBioAbc, NerJsonAbc):
    def __init__(
        self,
        instruction: str=NEREL_JSON_INSTRUCTION,
        **kwargs
    ):
        NerelBioAbc.__init__(self, instruction=instruction, **kwargs)
        NerJsonAbc.__init__(self, **kwargs)
        self._max_new_tokens = 200

    def task_name(self) -> str:
        return "nerel-ds/NEREL-BIO (json)"

    def get_answer(self, sample) -> Dict[str, List[str]]:
        tagged_tokens = []
        for entity in sample["entities"]:
            if entity["tag"] in self.TAGS:
                tagged_tokens.append([entity["tag"], entity["text"]])
        return tagged_tokens
    
    def get_answer_str(self, sample) -> str:
        answer = self.get_answer(sample)
        answer_str = '```json\n' + json.dumps(answer, ensure_ascii=False, indent=4).strip() + '\n```'


class NerelBioInPlace(NerelBioAbc, NerInPlaceAbc):
    def __init__(
        self,
        instruction: str=NEREL_IN_PLACE_INSTRUCTION,
        **kwargs
    ):
        NerelBioAbc.__init__(self, instruction=instruction, **kwargs)
        NerInPlaceAbc.__init__(self, **kwargs)
        self._max_new_tokens = 512

    def task_name(self) -> str:
        return "nerel-ds/NEREL-BIO (in-place)"

    def get_answer(self, sample) -> Dict[str, List[str]]:
        tagged_tokens = []
        for entity in sample["entities"]:
            if entity["tag"] in self.TAGS:
                tagged_tokens.append([entity["tag"], entity["text"]])
        return tagged_tokens

    def get_answer_str(self, sample) -> str:
        return get_answer_str_nested(sample)

    def extract_answer(self, gen_pred):
        return extract_answer_nested(gen_pred)

    def check_text(self, sample, gen_pred: str):
        text_pred = re.sub(r"<\w+>|</\w+>", "", gen_pred)
        return text_pred.strip() == sample["query"].strip()