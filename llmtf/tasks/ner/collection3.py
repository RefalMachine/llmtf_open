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
    
    def _load_dataset(self, model: LLM, max_prompt_len: int, max_sample_per_dataset: int, few_shot_count: int) -> List:
        samples = []
        dataset = load_dataset(**self.dataset_args())
        test_dataset = dataset[self.test_split_name()].filter(check_sample).map(process_sample)
        prompt_dataset = dataset[self.prompt_split_name()].filter(check_sample).map(process_sample)
        
        test_dataset_sample_ids = list(range(min(max_sample_per_dataset, len(test_dataset))))
        prompt_dataset_sample_ids = list(range(self.prompt_dataset_start_idx(), min(self.prompt_dataset_start_idx() + few_shot_count, len(prompt_dataset))))

        test_dataset = test_dataset.select(test_dataset_sample_ids)
        prompt_dataset = prompt_dataset.select(prompt_dataset_sample_ids)
        for sample in tqdm(test_dataset):
            samples.append({'messages': self._prepare_messages(sample, model, max_prompt_len, few_shot_count, prompt_dataset), 'sample': sample})
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
{text}"""

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

    def get_gold_entities(self, sample) -> Dict[str, List[str]]:
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
        answer = self.get_gold_entities(sample)
        answer_str = ""
        for tag, tokens in answer.items():
            answer_str += f"{tag}: [" + ', '.join(tokens) + "]\n"
        return answer_str


COLLECTION3_JSON_INSTRUCTION = """Ты — эксперт по извлечению именованных сущностей из русскоязычных текстов. Твоя задача — извлечь ВСЕ именованные сущности строго определённых классов: PER (персона по имени), ORG (организация), LOC (географическое местоположение).

**ВАЖНО:**
- Извлекай ТОЛЬКО те сущности, которые явно указаны в тексте и относятся к одному из классов.
- НЕ извлекай местоимения (например: "он", "его", "меня", "я", "они" и т.п.), даже если они ссылаются на человека.
- НЕ извлекай абстрактные понятия, процессы, объекты, награды, должности, временные метки или общие категории (например: "проекты", "суд", "регионы", "октябре 2012г.", "1988г.", "орден Дружбы", "кандидат", "гендиректор", "деятельность" и т.п.).
- Сущности должны быть представлены целым словом или последовательностью слов, как они даны в тексте. Не изменяй написание, не склоняй, не добавляй и не удаляй символы.
- Особое внимание удели именам собственным: если имя содержит инициалы, сохраняй оригинальное написание с пробелами вокруг точки (например: "И . Слюняев", а не "И. Слюняев"). Сохраняй все пробелы, кавычки, тире и другие символы в точности, как они встречаются в тексте.
- Извлекай сущности строго в порядке их появления в тексте.
- Если подходящих сущностей нет — верни пустой список: `[]`.

**Классы:**
- **PER** — конкретный человек, упомянутый по имени, фамилии, отчеству или инициалам (например: "Д . Кудрявцев", "Иван Иванов", "А . Усманов"). Должно быть имя собственное. Не извлекай местоимения, даже если они отсылают к человеку. Инициалы должны быть записаны с пробелами до и после точки.
- **ORG** — официальное название организации, компании, учреждения, партии, комиссии и т.п., употреблённое как имя собственное (например: "Федеральной конкурсной комиссии", "Госкомиссии по радиочастотам ( ГКРЧ )", "Союза журналистов"). Извлекай ТОЛЬКО те формы, которые используются в тексте как часть официального названия. Не извлекай общие словосочетания вроде "правительственная комиссия по транспорту", если они не являются закреплённым именем собственным. Если название организации включает топоним (например, "Социал - демократической партии России"), разделяй его: часть — как ORG ("Социал - демократической партии"), часть — как LOC ("России"), если топоним употреблён как самостоятельное географическое название.
- **LOC** — географическое местоположение: страна, город, регион, район, край и т.п., указанное как имя собственное (например: "Москва", "Татарстан", "России"). Даже если топоним склоняется и входит в состав названия организации, он всё равно может быть извлечён как LOC, если является самостоятельным географическим объектом и отделён от ORG по смыслу или синтаксису. Например, в "Союз журналистов России" — "Союза журналистов" (ORG), "России" (LOC). Однако если топоним является неотъемлемой частью официального названия и не выделен — предпочтение отдавай ORG. При сомнении: если топоним может быть заменён на "этой страны", "этого региона" и т.п. — извлекай как LOC.

**Особые правила:**
- Разделяй составные сущности: если в тексте встречается "Социал - демократической партии России", извлекай "Социал - демократической партии" как ORG и "России" как LOC.
- Не извлекай названия наград, орденов, медалей, титулов, должностей — это не относится к PER, ORG или LOC.
- Если в тексте встречается официальное сокращение в скобках (например, "ГКРЧ"), включай его в состав ORG, если оно прикреплено к названию, и сохраняй все пробелы вокруг скобок и внутри (например: "Госкомиссии по радиочастотам ( ГКРЧ )").
- Не извлекай сущности, которые являются частью цитаты, если они не являются именованными сущностями по смыслу (например, " не считает, что выходит на рынок " — не содержит сущностей).
- В случае, если топоним употреблён в родительном падеже как часть составного названия (например, "России" в "Союз журналистов России"), и при этом он склоняется — он всё равно может быть извлечён как LOC, если является самостоятельным географическим объектом.

**Формат вывода:**
Верни ТОЛЬКО корректный JSON-массив в следующем формате:
```json
[
    [
        "класс",
        "текст сущности"
    ],
    [
        "класс",
        "текст сущности"
    ]
]
```
Соблюдай точное написание, включая:
- пробелы до и после точек в инициалах (например: "А . Усманов", а не "А. Усманов"),
- кавычки и их расположение (включая пробелы после открывающей и до закрывающей кавычки, если они есть),
- тире, скобки и другие символы — как в оригинальном тексте.
Не изменяй регистр, не исправляй орфографию, не добавляй или не удаляй слова.
Извлекай сущности строго в порядке их появления в тексте.
Никаких пояснений, комментариев или дополнительного текста быть не должно.

**Текст:**
{text}"""

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

    def get_gold_entities(self, sample) -> List[str]:
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
        answer = self.get_gold_entities(sample)
        answer_str = '```json\n' + json.dumps(answer, ensure_ascii=False, indent=4).strip() + '\n```'
        return answer_str


COLLECTION3_IN_PLACE_INSTRUCTION = """Ты — эксперт по извлечению именованных сущностей из русскоязычных текстов. Твоя задача — точно повторить заданный текст, пометив все именованные сущности соответствующими тегами. Не изменяй и не склоняй сущности — оставляй их в том виде, в каком они встречаются в тексте.

**Классы сущностей:**
- **PER** — персона (имя, фамилия, отчество, инициалы, полное имя, включая обращения вроде "Джону Бассу").  
  ❌ **Не выделяй** как PER: местоимения ("он", "его", "их"), звания, должности без имени ("президент", "гендиректор", "заместитель главы миссии"), даже если они упоминаются рядом с именем. Эти слова — **не сущности**.
- **ORG** — организация (официальные названия компаний, СМИ, агентств, международных объединений, государственных учреждений, включая аббревиатуры и названия в кавычках).  
  ✅ Примеры: `<ORG>ИТАР - ТАСС</ORG>`, `<ORG>ИД</ORG>`, `<ORG>" Коммерсантъ "</ORG>`.  
  ❌ **Не выделяй** как ORG: описательные слова, не являющиеся частью официального названия (например, "пресс - служба", "издательского дома", "региональное представительство").
- **LOC** — местоположение (страны, города, регионы, географические объекты, включая упоминания в косвенных падежах).  
  ✅ Примеры: `<LOC>США</LOC>`, `<LOC>Грузии</LOC>`, `<LOC>Белого дома</LOC>`, `<LOC>Ирак</LOC>`.  
  ❌ **Не выделяй** как LOC: общие слова вроде "региональном", "посольстве", "миссии" — только если они не являются частью официального топонима.

**Критически важные правила:**
1. **Помечай все вхождения именованных сущностей**, даже повторяющиеся.
2. **Выделяй ТОЛЬКО те фрагменты, которые являются именованными сущностями.**  
   - ❌ Не выделяй:  
     - должности и звания без имени: "президент", "гендиректор", "заместитель главы", "посол", "глава миссии".  
     - местоимения: "он", "его", "их", "их слова".  
     - временные метки: "в мае", "2012г.", "19 июня".  
     - описательные термины: "пресс - служба", "региональное представительство", "издательского дома" — даже если рядом есть сущность.
3. **Сущность должна быть выделена целиком и точно**, как она записана в тексте:  
   - Сохраняй пробелы до/после кавычек: `" Коммерсантъ "` → `<ORG>" Коммерсантъ "</ORG>`.  
   - Сохраняй дефисы, точки, скобки: `Д . Кудрявцев`, `(ИД)`, `пресс - служба`.  
   - Не "исправляй" опечатки или форматирование.
4. **Не разбивай составные сущности.**  
   - Например: `<LOC>Белого дома</LOC>` — корректно, так как это устойчивое географическое название.  
   - Но: `пресс - служба Белого дома` → только `<LOC>Белого дома</LOC>`, а `пресс - служба` — не выделяется.
5. **Если аббревиатура и полное название используются вместе**, выделяй только именованные части:  
   - Пример: `издательского дома (ИД) " Коммерсантъ "` → выделяй: `(<ORG>ИД</ORG>) " <ORG>Коммерсантъ</ORG> "`.  
   - ❌ Не выделяй "издательского дома" — это описательная фраза, не входящая в официальное название.
6. **Используй строго следующие теги:** `<PER>`, `<ORG>`, `<LOC>`. Никаких других тегов, вариаций или пояснений.
7. **Формат вывода:**  
   Повтори исходный текст **дословно**, вставляя теги в формате `<класс>сущность</класс>` вокруг соответствующих фрагментов.  
   - Никаких дополнительных предложений, комментариев, списков или изменений структуры текста.  
   - Теги должны обрамлять **только сущность**, без лишних слов.

**Примеры корректной разметки:**
- `" Коммерсантъ "` → `<ORG>" Коммерсантъ "</ORG>`
- `Д . Кудрявцев` → `<PER>Д . Кудрявцев</PER>`
- `пресс - служба Белого дома` → `пресс - служба <LOC>Белого дома</LOC>`
- `в США` → `в <LOC>США</LOC>`
- `гендиректор ИД " Коммерсантъ "` → `гендиректор <ORG>ИД " Коммерсантъ "</ORG>`

**Текст для обработки:**
{text}"""

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
    
    def get_gold_entities(self, sample) -> List[str]:
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
