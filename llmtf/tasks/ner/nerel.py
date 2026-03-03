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

REMOVE_TAGS = []
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
        return {"path": self.DATASET_PATH, "name": self.DATASET_SLICE, "trust_remote_code": True}
    
    def test_split_name(self) -> str:
        return 'test'

    def prompt_split_name(self) -> str:
        return 'train'
    
    def _load_dataset(self, model: LLM, max_prompt_len: int, max_sample_per_dataset: int, few_shot_count: int) -> List:
        samples = []
        dataset = load_dataset(**self.dataset_args())
        test_dataset = process_dataset(process_sample_nerel, dataset[self.test_split_name()], self.do_split)
        prompt_dataset = process_dataset(process_sample_nerel, dataset[self.prompt_split_name()], self.do_split)
        
        test_dataset_sample_ids = list(range(min(max_sample_per_dataset, len(test_dataset))))
        prompt_dataset_sample_ids = list(range(self.prompt_dataset_start_idx(), min(self.prompt_dataset_start_idx() + few_shot_count, len(prompt_dataset))))

        test_dataset = test_dataset.select(test_dataset_sample_ids)
        prompt_dataset = prompt_dataset.select(prompt_dataset_sample_ids)
        for sample in tqdm(test_dataset):
            samples.append({'messages': self._prepare_messages(sample, model, max_prompt_len, few_shot_count, prompt_dataset), 'sample': sample})
        return samples


NEREL_DICT_INSTRUCTION = """Извлеки из заданного ниже текста все вложенные именованные сущности всех представленных ниже классов.
Сущности могут быть представлены только целым словом, окружённым пробелами или знаками препинания, либо непрерывной последовательностью целых слов, разделённых пробелами.
Оставь сущности в том виде, в каком они даны в тексте, не изменяй и не склоняй их, иначе тебе будет выставлен штраф 100$.

## Что такое вложенные сущности
Сущности могут быть вложены друг в друга. Например:
- "Мэр Москвы Сергей Собянин" содержит:
  - ["PROFESSION", "Мэр Москвы"] — внешняя сущность
  - ["CITY", "Москвы"] — вложенная внутрь PROFESSION
  - ["PERSON", "Сергей Собянин"] — отдельная сущность
- "Московский драматический театр им. М.Н. Ермоловой" содержит:
  - ["ORGANIZATION", "Московский драматический театр им. М.Н. Ермоловой"]
  - ["CITY", "Московский"] — прилагательное от названия города
  - ["PERSON", "М.Н. Ермоловой"] — имя человека внутри названия

Категории сущностей (Tags):

*ЛЮДИ И РОЛИ*
- PERSON: Имена людей (ФИО, псевдонимы).
- PROFESSION: Должности, звания, титулы (включая указание места работы).
- FAMILY: Фамилии, обозначающие семью/династию (не конкр. человека).
- NATIONALITY: Национальности, религиозные или политические группы, этнонимы.

*ОРГАНИЗАЦИИ И МЕСТА*
- ORGANIZATION: Компании, партии, музыкальные группы, госструктуры.
- FACILITY: Здания, инфраструктура (стадионы, мосты, музеи как здания).
- COUNTRY: Страны (и прилагательные, если относятся к огранам власти или организациям: "российская" власть).
- CITY: Города (и прилагательные: "московский").
- STATE_OR_PROVINCE: Области, штаты, края, округа.
- DISTRICT: Районы внутри города/области.
- LOCATION: Географические объекты (реки, горы, континенты) и места, не попавшие в другие категории.

*СОБЫТИЯ И ЗАКОН*
- EVENT: События (конференции, войны, спортивные матчи, фестивали).
- CRIME: Преступления (убийство, кража, коррупция).
- LAW: Названия законов, кодексов.
- PENALTY: Наказания (штраф, тюремный срок).
- DISEASE: Болезни, вирусы, синдромы.

*ЧИСЛА И ВРЕМЯ*
- DATE: Даты (полные или частичные).
- TIME: Время суток.
- AGE: Возраст.
- NUMBER: Числа (не являющиеся датами/деньгами).
- ORDINAL: Порядковые числительные (первый, 2-й).
- PERCENT: Проценты.
- MONEY: Денежные суммы.

*ПРОЧЕЕ*
- WORK_OF_ART: Книги, фильмы, песни, картины.
- PRODUCT: Продукты, товары, оружие, техника.
- AWARD: Медали, ордена, премии.
- LANGUAGE: Языки.
- RELIGION: Религии, конфессии.
- IDEOLOGY: Идеологии (коммунизм, демократия).

**Требуемый формат вывода**
Для каждого класса: "Класс: [сущность, ..., сущность]". Вместо "Класс" используй соответствующие классы, представленные выше. Сущности каждого класса выведи на отдельной строке.
Если сущностей соответствующего класса в тексте нет, выведи на соответствующей строке "Класс: []".

Набор классов в примере ниже может отличаться от классов задания.
**Пример**
Будущий ученый тайно покинул дом 15 декабря 1730 года и вскоре он догнал торговый обоз, шедший в Москву.
->
DISTRICT: []
CITY: [Москву]
STATE_OR_PROVINCE: []
COUNTRY: []
PERSON: []
PROFESSION: [ученый]
DATE: [15 декабря 1730 года]

Теперь извлеки вложенные именованные сущности для следующего текста.
**Текст**
{text}"""

class NestedNerDict(NestedNerAbc, NerDictAbc):
    def __init__(
        self,
        instruction: str=NEREL_DICT_INSTRUCTION,
        **kwargs
    ):
        NestedNerAbc.__init__(self, instruction=instruction, **kwargs)
        NerDictAbc.__init__(self, **kwargs)
        self._max_task_new_tokens = 1800

    def task_name(self) -> str:
        return "MalakhovIlya/NEREL-(dict)"

    def get_gold_entities(self, sample) -> Dict[str, List[str]]:
        tagged_tokens = {tag: [] for tag in self.TAGS}
        for entity in sample["entities"]:
            if entity["tag"] in self.TAGS:
                tagged_tokens[entity["tag"]].append(entity["text"])
        return tagged_tokens
    
    def get_answer_str(self, sample) -> str:
        answer = self.get_gold_entities(sample)
        answer_str = ""
        for tag, tokens in answer.items():
            answer_str += f"{tag}: [" + ', '.join(tokens) + "]\n"
        return answer_str

NEREL_JSON_INSTRUCTION = '''Ты — эксперт по извлечению именованных сущностей из русскоязычных текстов. Твоя задача — найти ВСЕ именованные сущности в тексте, включая ВЛОЖЕННЫЕ (когда одна сущность полностью содержится внутри другой). Если одна и та же сущность встречается несколько раз, извлеки каждое вхождение в порядке их появления в тексте.

Категории сущностей (теги):

*ЛЮДИ И РОЛИ*
- PERSON: Имена и фамилии людей (включая инициалы, сокращённые формы, обращения: "Сешнс", "Рахимов", "Дэвид Саутеру").
- PROFESSION: Должности, звания, титулы (включая составные: "президент Башкирии", "глава пресс-службы", "гендиректор ОАО").
- FAMILY: Фамилии как обозначение династии или семьи (если явно указано).
- NATIONALITY: Национальности, этнонимы, религиозные или политические группы (например: "русская", "мусульмане", "демократы", "республиканцы").

*ОРГАНИЗАЦИИ И МЕСТА*
- ORGANIZATION: Организации, компании, партии, СМИ, комитеты, учреждения (например: "Единая Россия", "Los Angeles Times", "юридический комитет").
- FACILITY: Здания, сооружения, инфраструктура (стадионы, мосты, музеи как объекты).
- COUNTRY: Названия стран и прилагательные, относящиеся к государствам (например: "США", "российская власть").
- CITY: Названия городов и прилагательные от них (например: "Уфы", "московский").
- STATE_OR_PROVINCE: Регионы, штаты, области, республики (например: "Алабама", "Башкирия").
- DISTRICT: Районы, округи внутри городов или регионов.
- LOCATION: Природные и географические объекты (горы, реки, континенты), а также абстрактные места (например: "федеральный центр").

*СОБЫТИЯ И ЗАКОН*
- EVENT: События, процессы, действия (например: "отставка", "выборы", "подписал указ", "назначение").
- CRIME: Преступления (вымогательство, коррупция).
- LAW: Названия законов, кодексов.
- PENALTY: Виды наказаний.
- DISEASE: Болезни, вирусы.

*ЧИСЛА И ВРЕМЯ*
- DATE: Даты (полные или частичные: "вчера", "6 июня", "в июне", "с 1997 года").
- TIME: Время суток.
- AGE: Возраст (например: "42-летний").
- NUMBER: Числовые значения (не даты и не деньги).
- ORDINAL: Порядковые числительные ("первый", "2-й").
- PERCENT: Проценты.
- MONEY: Денежные суммы.

*ПРОЧЕЕ*
- WORK_OF_ART: Книги, фильмы, газеты, журналы, СМИ (например: "Московский комсомолец", "Ъ").
- PRODUCT: Товары, техника, оружие.
- AWARD: Награды, премии.
- LANGUAGE: Языки.
- RELIGION: Религии.
- IDEOLOGY: Идеологии, политические направления (например: "консерватор", "демократ", "коммунизм").

ВАЖНЫЕ ПРАВИЛА:
1. Извлекай ВСЕ сущности, включая вложенные. Например, в "Президент Башкирии Муртаза Рахимов" должно быть:
   - ["PROFESSION", "Президент Башкирии"]
   - ["STATE_OR_PROVINCE", "Башкирии"]
   - ["PERSON", "Муртаза Рахимов"]
2. Не изменяй текст сущности — копируй его дословно, включая склонения, регистр и пунктуацию.
3. Сохраняй порядок сущностей строго по их первому появлению в тексте. Если сущность начинается раньше — она должна быть раньше в списке.
4. Не объединяй сущности. Каждая должна быть отдельным элементом.
5. Извлекай даже короткие формы: "Сешнс", "Хабирова", "Ъ", "РФ".
6. Прилагательные, образованные от географических названий, могут быть отдельными сущностями (например: "московский" → CITY, "башкирский" → STATE_OR_PROVINCE).
7. Политические и идеологические термины (республиканец, демократ, консерватор) — это IDEOLOGY, если не указано как должность.
8. СМИ и газеты — WORK_OF_ART, если не указано как организация (но в примерах "Ъ" и "Los Angeles Times" — ORGANIZATION).
9. Действия и процессы (уволил, отставка, назначение, покинуть) — это EVENT.
10. Должности, содержащие географические названия, могут включать вложенные сущности (например: "глава башкирской администрации" → PROFESSION + STATE_OR_PROVINCE).

Формат вывода:
Только JSON-список списков: [["TAG", "текст сущности"], ...]
Без пояснений, без комментариев, без дополнительного текста.

Текст для обработки:
{text}'''

class NestedNerJson(NestedNerAbc, NerJsonAbc):
    def __init__(
        self,
        instruction: str=NEREL_JSON_INSTRUCTION,
        **kwargs
    ):
        NestedNerAbc.__init__(self, instruction=instruction, **kwargs)
        NerJsonAbc.__init__(self, **kwargs)
        self._max_task_new_tokens = 1800

    def task_name(self) -> str:
        return "MalakhovIlya/NEREL-(json)"

    def get_gold_entities(self, sample) -> Dict[str, List[str]]:
        tagged_tokens = []
        for entity in sample["entities"]:
            if entity["tag"] in self.TAGS:
                tagged_tokens.append([entity["tag"], entity["text"]])
        return tagged_tokens
    
    def get_answer_str(self, sample) -> str:
        answer = self.get_gold_entities(sample)
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

NEREL_IN_PLACE_INSTRUCTION = """Твоя задача — экспертно разметить русскоязычный текст, точно выделив все именованные сущности (включая вложенные) согласно заданным категориям. Сохрани исходный текст полностью, не изменяя ни одного слова, не склоняя и не перефразируя. Размечай каждую сущность с помощью тегов в формате `<ТИП>сущность</ТИП>`. Если сущности вложены (одна находится внутри другой), размечай все уровни вложенности, используя вложенные теги.

## Категории сущностей

*ЛЮДИ И РОЛИ*
- **PERSON**: Полные или частичные имена конкретных людей (включая отчества, фамилии, инициалы).
- **PROFESSION**: Должности, профессии, звания, титулы, роли (например: президент, министр, генсек, вице-президент). Включай указание места работы, если оно часть должности.
- **FAMILY**: Фамилии, обозначающие династию или семью (не конкретного человека).
- **NATIONALITY**: Национальности, этнонимы, религиозные или политические группы (например: россиянин, мусульмане, демократы).

*ОРГАНИЗАЦИИ И МЕСТА*
- **ORGANIZATION**: Компании, государственные структуры, политические партии, СМИ, международные организации (например: Газпром, ООН, Коммерсант).
- **FACILITY**: Здания, сооружения, объекты инфраструктуры (например: штаб-квартира, Штокмановское месторождение).
- **COUNTRY**: Названия стран и прилагательные от них, если относятся к государству (например: Россия, российская власть).
- **CITY**: Названия городов и прилагательные от них (например: Москва, московский).
- **STATE_OR_PROVINCE**: Области, края, штаты, округа.
- **DISTRICT**: Районы внутри города или региона.
- **LOCATION**: Географические объекты (реки, горы, континенты), не вошедшие в другие категории.

*СОБЫТИЯ И ЗАКОН*
- **EVENT**: Конкретные события (саммиты, встречи, заседания, выборы).
- **CRIME**: Преступления (взятка, убийство).
- **LAW**: Названия законов, кодексов.
- **PENALTY**: Виды наказаний (штраф, тюремный срок).
- **DISEASE**: Болезни, вирусы, синдромы.

*ЧИСЛА И ВРЕМЯ*
- **DATE**: Даты, временные периоды, месяцы, годы, временные указатели (сегодня, вчера, в декабре).
- **TIME**: Время суток (14:00, утром).
- **AGE**: Возраст (45 лет, не младше 45).
- **NUMBER**: Числа, не относящиеся к датам или деньгам (шесть стран, 15-летний).
- **ORDINAL**: Порядковые числительные (первый, 2-й).
- **PERCENT**: Проценты (50%, на 75%).
- **MONEY**: Денежные суммы (100 долларов).

*ПРОЧЕЕ*
- **WORK_OF_ART**: Книги, фильмы, песни, картины.
- **PRODUCT**: Товары, техника, оружие, бренды (включая названия компаний, если они употребляются как продукт).
- **AWARD**: Награды, премии, ордена.
- **LANGUAGE**: Названия языков.
- **RELIGION**: Религии и конфессии.
- **IDEOLOGY**: Политические и социальные идеологии.

## Важные правила
1. **Не пропускай сущности**, даже если они короткие (например: "генсек", "министр", "россиянин").
2. **Размечай вложенность**: если должность включает организацию (например, "министр энергетики РФ"), размечай и должность, и страну внутри.
3. **Прилагательные от географических названий** (московский, иранский) — относи к соответствующей категории (CITY, COUNTRY).
4. **СМИ и газеты** (например, "Ъ", "Коммерсант") — размечай как **ORGANIZATION** или **PRODUCT**, в зависимости от контекста употребления.
5. **События** (заседание, встреча, саммит) — размечай как **EVENT**, особенно если они связаны с датой или местом.
6. **Совет директоров "Газпрома"** — размечай как **ORGANIZATION**, а "директора" — как **PROFESSION**.
7. **Не создавай ложных сущностей**: не размечай слова, не относящиеся к категориям (например, "объемы", "влияние").
8. **Следи за согласованием тегов**: все открывающие теги должны иметь закрывающие, без пересечений.

## Пример правильной разметки
Текст: Будущий ученый покинул дом 15 декабря 1730 года и направился в Москву.
Разметка: Будущий <PROFESSION>ученый</PROFESSION> покинул дом <DATE>15 декабря 1730 года</DATE> и направился в <CITY>Москву</CITY>.

Текст: Мэр Москвы Сергей Собянин открыл музей.
Разметка: <PROFESSION>Мэр <CITY>Москвы</CITY></PROFESSION> <PERSON>Сергей Собянин</PERSON> открыл <FACILITY>музей</FACILITY>.

Текст: Вице-президент Иранской национальной газовой компании Ходжатолла Ганимифард.
Разметка: <PROFESSION><PROFESSION>Вице-президент</PROFESSION> <ORGANIZATION><COUNTRY>Иранской</COUNTRY> национальной газовой компании</ORGANIZATION></PROFESSION> <PERSON>Ходжатолла Ганимифард</PERSON>.

# Текст для разметки
{text}"""

class NestedNerInPlace(NestedNerAbc, NerInPlaceAbc):
    def __init__(
        self,
        instruction: str=NEREL_IN_PLACE_INSTRUCTION,
        **kwargs
    ):
        NestedNerAbc.__init__(self, instruction=instruction, **kwargs)
        NerInPlaceAbc.__init__(self, **kwargs)
        self._max_task_new_tokens = 2200

    def task_name(self) -> str:
        return "MalakhovIlya/NEREL-(in-place)"

    def get_gold_entities(self, sample) -> Dict[str, List[str]]:
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
        return text_pred.replace("\n", "").replace(" ", "") == sample["query"].replace("\n", "").replace(" ", "")


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
    
    def _load_dataset(self, model: LLM, max_prompt_len: int, max_sample_per_dataset: int, few_shot_count: int) -> List:
        samples = []
        dataset = load_dataset(**self.dataset_args())
        test_dataset = process_dataset(process_sample_nerel_bio, dataset[self.test_split_name()], self.do_split)
        prompt_dataset = process_dataset(process_sample_nerel_bio, dataset[self.prompt_split_name()], self.do_split)
        
        test_dataset_sample_ids = list(range(min(max_sample_per_dataset, len(test_dataset))))
        prompt_dataset_sample_ids = list(range(self.prompt_dataset_start_idx(), min(self.prompt_dataset_start_idx() + few_shot_count, len(prompt_dataset))))

        test_dataset = test_dataset.select(test_dataset_sample_ids)
        prompt_dataset = prompt_dataset.select(prompt_dataset_sample_ids)
        for sample in tqdm(test_dataset):
            samples.append({'messages': self._prepare_messages(sample, model, max_prompt_len, few_shot_count, prompt_dataset), 'sample': sample})
        return samples


NEREL_BIO_DICT_INSTRUCTION = """Ты — эксперт по извлечению биомедицинских сущностей. В тексте ниже найди все именованные сущности следующих классов.

**Классы**
{tags}

**Ключевые правила**
1. Извлекай ВСЕ вхождения сущностей, включая дубликаты и части составных терминов, сохраняя порядок встречаемости
2. Сохраняй оригинальный регистр и форму слов
3. Для перекрывающихся сущностей ("распространенность кариеса"→PHYS, "кариеса"→DISO) извлекай обе
4. Строго соблюдай типизацию: 
   - "кариес" → DISO, 
   - "индекс КПУ" → PHYS, 
   - "зубочелюстных" → ANATOMY

**Требуемый формат вывода**
Для каждого класса: "Класс: [сущность, ..., сущность]". Вместо "Класс" используй соответствующие классы, представленные выше. Сущности каждого класса выведи на отдельной строке.
Если сущностей соответствующего класса в тексте нет, выведи на соответствующей строке "Класс: []".

**Текст**
{text}"""


class NerelBioDict(NerelBioAbc, NerDictAbc):
    def __init__(
        self,
        instruction: str=NEREL_BIO_DICT_INSTRUCTION,
        **kwargs
    ):
        NerelBioAbc.__init__(self, instruction=instruction, **kwargs)
        NerDictAbc.__init__(self, **kwargs)
        self._max_task_new_tokens = 1500

    def task_name(self) -> str:
        return "nerel-ds/NEREL-BIO-(dict)"

    def get_gold_entities(self, sample) -> Dict[str, List[str]]:
        tagged_tokens = {tag: [] for tag in self.TAGS}
        for entity in sample["entities"]:
            if entity["tag"] in self.TAGS:
                tagged_tokens[entity["tag"]].append(entity["text"])
        return tagged_tokens
    
    def get_answer_str(self, sample) -> str:
        answer = self.get_gold_entities(sample)
        answer_str = ""
        for tag, tokens in answer.items():
            answer_str += f"{tag}: [" + ', '.join(tokens) + "]\n"
        return answer_str


NEREL_BIO_JSON_INSTRUCTION = """Ты — эксперт по извлечению биомедицинских сущностей. В тексте ниже найди все именованные сущности следующих классов.

**Классы**
{tags}

**Ключевые правила**
1. Извлекай ВСЕ вхождения сущностей, включая дубликаты и части составных терминов, сохраняя порядок встречаемости
2. Сохраняй оригинальный регистр и форму слов
3. Для перекрывающихся сущностей ("распространенность кариеса"→PHYS, "кариеса"→DISO) извлекай обе
4. Строго соблюдай типизацию: 
   - "кариес" → DISO, 
   - "индекс КПУ" → PHYS, 
   - "зубочелюстных" → ANATOMY

**Требуемый формат вывода (только json)**
```json
[["тип", "сущность"], ["тип", "сущность"], ...]
```

**Текст**
{text}"""


class NerelBioJson(NerelBioAbc, NerJsonAbc):
    def __init__(
        self,
        instruction: str=NEREL_BIO_JSON_INSTRUCTION,
        **kwargs
    ):
        NerelBioAbc.__init__(self, instruction=instruction, **kwargs)
        NerJsonAbc.__init__(self, **kwargs)
        self._max_task_new_tokens = 1500

    def task_name(self) -> str:
        return "nerel-ds/NEREL-BIO-(json)"

    def get_gold_entities(self, sample) -> Dict[str, List[str]]:
        tagged_tokens = []
        for entity in sample["entities"]:
            if entity["tag"] in self.TAGS:
                tagged_tokens.append([entity["tag"], entity["text"]])
        return tagged_tokens
    
    def get_answer_str(self, sample) -> str:
        answer = self.get_gold_entities(sample)
        answer_str = '```json\n' + json.dumps(answer, ensure_ascii=False, indent=4).strip() + '\n```'
        return answer_str


NEREL_BIO_IN_PLACE_INSTRUCTION = """Ты — эксперт по извлечению именованных биомедицинских сущностей из русскоязычных текстов. Твоя задача — точно и полно извлечь ВСЕ вхождения сущностей в тексте, строго соблюдая классы и правила.

**Классы сущностей**
FINDING — признаки, симптомы и отклонения, обнаруженные в ходе обследования.
DISO — расстройства, заболевания, синдромы.
INJURY_POISONING — травмы, отравления и другие повреждения, включая хирургические вмешательства.
PHYS — физиологические показатели, процессы, явления.
DEVICE — медицинские приборы, оборудование, инструменты.
LABPROC — лабораторные исследования, анализы и диагностические процедуры.
ANATOMY — анатомические структуры.
CHEM — химические вещества и лекарства.

**Ключевые правила**
1. Извлекай ВСЕ вхождения сущностей, включая дубликаты и части составных терминов. Сохраняй порядок их появления в тексте.
2. Сохраняй оригинальный регистр, форму слов и пунктуацию.
3. Для перекрывающихся сущностей (например, "распространенность кариеса" → PHYS, "кариеса" → DISO) извлекай обе сущности, используя вложенные теги при необходимости.
4. Строго соблюдай типизацию:
   - "кариес" → DISO
   - "индекс КПУ" → PHYS
   - "зубочелюстных" → ANATOMY
5. Не добавляй сущности, которых нет в тексте, и не объединяй несвязанные слова в одну сущность без оснований.
6. Не генерируй пояснений, комментариев или списков вне формата. Работай строго с текстом как с последовательностью слов.
7. Используй вложенные теги для вложенных сущностей (например: <DISO>острая <DISO>водянистая <DISO>диарея</DISO></DISO></DISO>).
8. Учитывай контекст: например, "рецидив" — это часть заболевания (DISO), а не процесс; "уменьшить частоту осложнений" — FINDING, так как описывает клинический эффект.
9. Диагностические и визуализационные методы (КТ, МРТ, ФБС) — LABPROC, даже если они включают название устройства.
10. Хирургические операции и вмешательства — INJURY_POISONING; хирургические устройства (анастомоз) — DEVICE.
11. Аббревиатуры (ТМ, РСТ, МСКТ, МРТ, ФБС) — извлекай как сущности того же типа, что и их полные формы.

**Требуемый формат вывода**
Верни исходный текст, в котором каждая именованная сущность обрамлена тегами вида <тип>сущность</тип>. Для вложенных сущностей используй вложенные теги. Не изменяй структуру текста, не пропускай слова, не добавляй пояснения.

**Текст**
{text}"""


class NerelBioInPlace(NerelBioAbc, NerInPlaceAbc):
    def __init__(
        self,
        instruction: str=NEREL_BIO_IN_PLACE_INSTRUCTION,
        **kwargs
    ):
        NerelBioAbc.__init__(self, instruction=instruction, **kwargs)
        NerInPlaceAbc.__init__(self, **kwargs)
        self._max_task_new_tokens = 2000

    def task_name(self) -> str:
        return "nerel-ds/NEREL-BIO-(in-place)"

    def get_gold_entities(self, sample) -> Dict[str, List[str]]:
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
        return text_pred.replace("\n", "").replace(" ", "") == sample["query"].replace("\n", "").replace(" ", "")