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


COLLECTION3_JSON_INSTRUCTION = """Ты — эксперт по извлечению именованных сущностей из русскоязычных текстов. Твоя задача — извлечь ВСЕ именованные сущности из текста, строго относящиеся к следующим классам:
- **PER** — конкретный человек, указанного по имени, фамилии, отчеству или инициалам (например, "Иван Иванов", "А. Сидоров", "Д. Кудрявцев"). Местоимения ("он", "она", "я", "меня", "его", "её") НЕ являются сущностями класса PER, даже если они относятся к человеку. Извлекается только явно указанное имя или инициалы с фамилией.
- **ORG** — официальное, полное название организации, компании, учреждения, холдинга и т.п., включая аббревиатуры и юридические формы (например, "ООО Ромашка", "Правительство РФ", "Издательский дом Коммерсантъ"). Общие слова вроде "суд", "министерство", "проекты", "холдинг" без конкретного названия НЕ относятся к ORG. Также не следует извлекать названия орденов, медалей, государственных наград — они НЕ являются организациями.
- **LOC** — конкретное географическое местоположение: страна, город, регион, субъект РФ, улица и т.п. (например, "Москва", "Тверская область", "Алтайского края"). Слова вроде "регион", "внутри", "на рынке", "в стране" без указания конкретного места НЕ являются LOC.

**Важные правила:**
1. Извлекай ТОЛЬКО те сущности, которые явно названы в тексте и однозначно относятся к одному из классов.
2. Не извлекай:
   - местоимения (например, "он", "его", "меня", "её");
   - временные обозначения (например, "2006г.", "мая 2012г.", "октябре 2012г.");
   - абстрактные понятия ("проекты", "деятельность", "суд", "министерство", "холдинг", "орган", "комиссия" без уточнения конкретного названия);
   - должности и титулы без привязки к конкретному человеку ("премьер-министра", "гендиректора", "сенатор");
   - государственные награды, ордена, медали (например, "орден Дружбы", "За заслуги перед Отечеством") — они не являются ORG или LOC;
   - части составных названий, если они не являются самостоятельными официальными наименованиями (например, "Федеральной службы по надзору в сфере связи" — извлекать только если это полное официальное название; если в тексте указано сокращённо или фрагментарно — извлекать только ту часть, что соответствует официальному названию).
3. Сохраняй ТОЧНОЕ написание сущности из текста, включая пробелы, дефисы, пробелы вокруг дефисов и точки: например, если в тексте написано "Д . Кудрявцев" или "И . Слюняев", то извлекай именно так, а не "Д. Кудрявцев".
4. Если сущность встречается несколько раз — извлеки каждое вхождение в порядке появления в тексте.
5. Никакие другие слова, не относящиеся к PER, ORG, LOC, извлекать не нужно.
6. Не объединяй сущности, не добавляй пояснения, не изменяй регистр и формат.
7. Обрати особое внимание: если в тексте встречается название организации, содержащее кавычки, скобки или аббревиатуру, извлекай его целиком, как оно написано, включая все знаки препинания и пробелы вокруг них.
8. Не извлекай сущности, которые не являются именованными — например, "деятельность", "работа", "проекты", "внутри", "в пользу", "в составе" и т.п.

**Ключевые уточнения на основе типичных ошибок:**
- **Местоимения НЕ являются сущностями PER.** Даже если они отсылают к человеку, они не должны извлекаться.
- **Составные названия организаций:** извлекай только те части, которые являются официальными наименованиями. Например, "Федеральная конкурсная комиссия по телерадиовещанию" — извлекать как "Федеральной конкурсной комиссии", если именно так она названа в тексте и это соответствует её официальному сокращённому наименованию.
- **Географические названия:** если в тексте встречается "России", "Москвы", "Татарстана" — это LOC, даже если они в родительном падеже.
- **Организации, включающие географические названия:** если организация состоит из двух частей — названия типа "Союз", "Партия", "Комиссия" и географического наименования — извлекай только часть с названием организации, а географическое — отдельно как LOC, если оно явно выделено. Например, в "Социал - демократической партии России" извлекаются: "Социал - демократической партии" как ORG и "России" как LOC.
- **Государственные награды:** НЕ извлекать ни при каких условиях. Даже если они содержат слова "орден", "медаль", "за заслуги" — это не ORG и не PER.
- **Пробелы вокруг дефисов и точек:** критически важны. Если в тексте написано "А . Усманов", "С . Ситников", "Социал - демократической", то извлекать нужно именно с пробелами.
- **Кавычки и скобки:** извлекай вместе с сущностью, если они являются частью её написания в тексте, но только если они непосредственно окружают название организации. Например, "Издательский дом \" Коммерсантъ \"" — извлекать как "Издательский дом \" Коммерсантъ \"", без лишних пробелов снаружи, если они не входят в официальное название.

**Формат вывода (только JSON, без дополнительного текста):**
```json
[["класс", "сущность"], ["класс", "сущность"], ...]
```

**Текст для анализа:**
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


COLLECTION3_IN_PLACE_INSTRUCTION = """Ты — эксперт по извлечению именованных сущностей из русскоязычных текстов. Твоя задача — точно повторить заданный текст, пометив все именованные сущности соответствующими тегами. Не изменяй, не склоняй и не переформулируй сущности — оставляй их в том виде, в каком они встречаются в тексте.

**Классы сущностей:**
- PER — персона (человек с именем, фамилией, инициалами или их сочетанием)
- ORG — организация (компании, учреждения, агентства, СМИ, структуры, объединения)
- LOC — местоположение (страны, города, регионы, географические объекты, официальные названия территорий)

**Важные правила:**
1. Помечай сущности только тогда, когда они явно относятся к одному из классов. Не выделяй обычные существительные, глаголы, прилагательные или служебные слова.
2. Если организация состоит из нескольких слов (включая аббревиатуры и названия в кавычках), выделяй её целиком как ORG.
3. Географические названия, включая страны, города и официальные топонимы (например, "Белый дом", "Госдепартамент США"), помечай как LOC. Обрати особое внимание: "США" — это LOC, а не часть ORG, если не входит в название организации как её структурный элемент (но даже в этом случае "США" как страна — отдельная сущность LOC).
4. ФИО, имена с инициалами, фамилии в контексте человека — это PER. Даже если указаны только инициалы и фамилия (например, "Д. Кудрявцев") — это PER.
5. Не разбивай сущности на части. Если сущность состоит из нескольких слов (например, "ИД \"Коммерсантъ\""), выделяй её целиком.
6. Не используй тег <tags> — используй только строго определённые: <PER>, <ORG>, <LOC>. Никаких других тегов быть не должно.
7. Сохраняй все знаки препинания, пробелы и форматирование оригинального текста.

**Формат вывода:**
Повтори исходный текст дословно, вставляя теги вида <класс>сущность</класс> вокруг соответствующих сущностей. Все остальные слова оставь без изменений.

**Пример правильной разметки:**
Исходный текст: "Президент США Барак Обама посетил Грузию."
Разметка: "Президент <LOC>США</LOC> <PER>Барак Обама</PER> посетил <LOC>Грузию</LOC>."

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
