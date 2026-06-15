from typing import List, Dict
from abc import ABC
from tqdm import tqdm
from datasets import load_dataset
import json
from .ner_abc import NerDictAbc, NerJsonAbc, NerInPlaceAbc, get_gold_entities_bio_dict, get_gold_entities_bio_list, get_answer_str_bio_in_place
from llmtf.base import LLM

def process_sample(sample):
    sample["query"] = sample.pop("text")
    sample["tokens"] = json.loads(sample["tokens"])
    sample["tags"] = json.loads(sample.pop("ner_tags"))
    return sample

class PiiAbc(ABC):
    DATASET_PATH = "redmadrobot-rnd/pii_benchmark"
    TAGS = [
        "FIRST_NAME", "LAST_NAME", "MIDDLE_NAME",
        "COUNTRY", "REGION", "DISTRICT", "CITY", "STREET", "HOUSE",
        "EMAIL", "PHONE", "URL", "IP_ADDRESS",
        "PASSPORT", "INN", "SNILS", "OMS", "CREDIT_CARD", "DRIVER_LICENSE", "MILITARY_ID", "BIRTH_CERTIFICATE"
    ]

    def __init__(
        self,
        instruction: str,
        **kwargs
    ):
        self.instruction = instruction

    def dataset_args(self) -> Dict[str, str]:
        return {"path": self.DATASET_PATH}
    
    def test_split_name(self) -> str:
        return 'test'

    def prompt_split_name(self) -> str:
        return 'test'

    def _load_dataset(self, model: LLM, max_prompt_len: int, max_sample_per_dataset: int, few_shot_count: int) -> List:
        samples = []
        dataset = load_dataset(**self.dataset_args())
        test_dataset = dataset[self.test_split_name()].map(process_sample, remove_columns=["text", "ner_tags"])
        prompt_dataset = test_dataset

        # issue: can break if few_shot_count >= len(test_dataset)
        test_dataset_sample_ids = list(range(min(few_shot_count + max_sample_per_dataset, len(test_dataset))))
        prompt_dataset_sample_ids = list(range(min(few_shot_count, len(prompt_dataset))))

        test_dataset = test_dataset.select(test_dataset_sample_ids)
        prompt_dataset = prompt_dataset.select(prompt_dataset_sample_ids)
        for sample in tqdm(test_dataset):
            samples.append({'messages': self._prepare_messages(sample, model, max_prompt_len, few_shot_count, prompt_dataset), 'sample': sample})
        return samples
        

PII_DICT_INSTRUCTION = """Извлеки из заданного ниже текста все именованные сущности всех представленных ниже классов.
Сущности могут быть представлены целым словом или последовательностей слов, разделенных пробелами.
Оставь сущности в том виде, в каком они даны в тексте, не изменяй и не склоняй их.

**Классы**
1. Имена людей: FIRST_NAME, LAST_NAME, MIDDLE_NAME — включая имена, фамилии и отчества. Обрабатываются написания как кириллицей, так и латиницей, в любом регистре (например, "Александр", "Воронцов", "Аркадьевич").

2. Адреса и локации: COUNTRY, REGION, DISTRICT, CITY, STREET, HOUSE — полная иерархия российских адресов: страна, область/край, район, город, улица, номер дома. Примеры: "Португалия", "Иркутская область", "Центральный район", "Белогорск", "бульвар Королёва", "строение 7".

3. Контактные данные и сетевые идентификаторы: EMAIL, PHONE, URL, IP_ADDRESS — структурированные идентификаторы в разных реальных форматах. Примеры: "ivan.petrov@example.org", "8-800-555-35-35", "https://mysite.online", "192.168.1.1" (IPv4) и "2001:db8::ff00:42:8329" (IPv6).

4. Российские идентификационные номера: PASSPORT, INN, SNILS, OMS, CREDIT_CARD, DRIVER_LICENSE, MILITARY_ID, BIRTH_CERTIFICATE — номера документов в каноническом виде. Примеры: паспорт "6301 234567", ИНН "123456789012", СНИЛС "123-456-789 00", полис ОМС "1234 5678 9012 3456", номер карты "1234-5678-9012-3456", водительское удостоверение "99 99 999999", военный билет "БА 123456", свидетельство о рождении "VII - ЮЗ 654321".

**Формат вывода**
Для каждого класса: "КЛАСС: [СУЩНОСТЬ, ..., СУЩНОСТЬ]". Вместо КЛАСС используй соответствующие классы, представленные выше. Сущности каждого класса выведи на отдельной строке.
Если сущностей соответствующего класса в тексте нет, выведи на соответствующей строке "КЛАСС: []".

КЛАСС: [СУЩНОСТЬ, СУЩНОСТЬ, ... СУЩНОСТЬ]
...
КЛАСС: []
КЛАСС: [СУЩНОСТЬ, ... СУЩНОСТЬ]

**Текст**
{text}"""


class PiiDict(PiiAbc, NerDictAbc):
    def __init__(
        self,
        instruction: str=PII_DICT_INSTRUCTION,
        **kwargs
    ):
        PiiAbc.__init__(self, instruction=instruction, **kwargs)
        NerDictAbc.__init__(self, **kwargs)
        self._max_task_new_tokens = 100

    def task_name(self) -> str:
        return "redmadrobot-rnd/pii_benchmark-(dict)"

    def get_gold_entities(self, sample) -> Dict[str, List[str]]:
        return get_gold_entities_bio_dict(self, sample)


PII_JSON_INSTRUCTION = """Извлеки из заданного ниже текста все именованные сущности всех представленных ниже классов.
Сущности могут быть представлены целым словом или последовательностей слов, разделенных пробелами.
Оставь сущности в том виде, в каком они даны в тексте, не изменяй и не склоняй их.

**Классы**
1. Имена людей: FIRST_NAME, LAST_NAME, MIDDLE_NAME — включая имена, фамилии и отчества. Обрабатываются написания как кириллицей, так и латиницей, в любом регистре (например, "Александр", "Воронцов", "Аркадьевич").

2. Адреса и локации: COUNTRY, REGION, DISTRICT, CITY, STREET, HOUSE — полная иерархия российских адресов: страна, область/край, район, город, улица, номер дома. Примеры: "Португалия", "Иркутская область", "Центральный район", "Белогорск", "бульвар Королёва", "строение 7".

3. Контактные данные и сетевые идентификаторы: EMAIL, PHONE, URL, IP_ADDRESS — структурированные идентификаторы в разных реальных форматах. Примеры: "ivan.petrov@example.org", "8-800-555-35-35", "https://mysite.online", "192.168.1.1" (IPv4) и "2001:db8::ff00:42:8329" (IPv6).

4. Российские идентификационные номера: PASSPORT, INN, SNILS, OMS, CREDIT_CARD, DRIVER_LICENSE, MILITARY_ID, BIRTH_CERTIFICATE — номера документов в каноническом виде. Примеры: паспорт "6301 234567", ИНН "123456789012", СНИЛС "123-456-789 00", полис ОМС "1234 5678 9012 3456", номер карты "1234-5678-9012-3456", водительское удостоверение "99 99 999999", военный билет "БА 123456", свидетельство о рождении "VII - ЮЗ 654321".

**Формат вывода (только JSON, без дополнительного текста):**
```json
[["КЛАСС", "СУЩНОСТЬ"], ["КЛАСС", "СУЩНОСТЬ"], ...]
```

**Текст**
{text}"""


class PiiJson(PiiAbc, NerJsonAbc):
    def __init__(
        self,
        instruction: str=PII_JSON_INSTRUCTION,
        **kwargs
    ):
        PiiAbc.__init__(self, instruction=instruction, **kwargs)
        NerJsonAbc.__init__(self, **kwargs)
        self._max_task_new_tokens = 100

    def task_name(self):
        return "redmadrobot-rnd/pii_benchmark-(json)"

    def get_gold_entities(self, sample) -> List[str]:
        return get_gold_entities_bio_list(self, sample)


PII_IN_PLACE_INSTRUCTION = """Ты — эксперт по извлечению именованных сущностей из русскоязычных текстов. Твоя задача — точно повторить заданный текст, пометив все именованные сущности соответствующими тегами. Не изменяй, не склоняй и не переформулируй сущности — оставляй их в том виде, в каком они встречаются в тексте.

**Классы**
1. Имена людей: FIRST_NAME, LAST_NAME, MIDDLE_NAME — включая имена, фамилии и отчества. Обрабатываются написания как кириллицей, так и латиницей, в любом регистре (например, "Александр", "Воронцов", "Аркадьевич").

2. Адреса и локации: COUNTRY, REGION, DISTRICT, CITY, STREET, HOUSE — полная иерархия российских адресов: страна, область/край, район, город, улица, номер дома. Примеры: "Португалия", "Иркутская область", "Центральный район", "Белогорск", "бульвар Королёва", "строение 7".

3. Контактные данные и сетевые идентификаторы: EMAIL, PHONE, URL, IP_ADDRESS — структурированные идентификаторы в разных реальных форматах. Примеры: "ivan.petrov@example.org", "8-800-555-35-35", "https://mysite.online", "192.168.1.1" (IPv4) и "2001:db8::ff00:42:8329" (IPv6).

4. Российские идентификационные номера: PASSPORT, INN, SNILS, OMS, CREDIT_CARD, DRIVER_LICENSE, MILITARY_ID, BIRTH_CERTIFICATE — номера документов в каноническом виде. Примеры: паспорт "6301 234567", ИНН "123456789012", СНИЛС "123-456-789 00", полис ОМС "1234 5678 9012 3456", номер карты "1234-5678-9012-3456", водительское удостоверение "99 99 999999", военный билет "БА 123456", свидетельство о рождении "VII - ЮЗ 654321".

**Формат вывода:**
Повтори исходный текст дословно, вставляя теги вида <КЛАСС>СУЩНОСТЬ</КЛАСС> вокруг соответствующих сущностей. Все остальные слова оставь без изменений.

**Текст**
{text}"""


class PiiInPlace(PiiAbc, NerInPlaceAbc):
    def __init__(
        self,
        instruction: str=PII_IN_PLACE_INSTRUCTION,
        **kwargs
    ):
        PiiAbc.__init__(self, instruction=instruction, **kwargs)
        NerInPlaceAbc.__init__(self, **kwargs)
        self._max_task_new_tokens = 256

    def task_name(self) -> str:
        return "redmadrobot-rnd/pii_benchmark-(in-place)"

    def get_gold_entities(self, sample) -> List[str]:
        return get_gold_entities_bio_list(self, sample)
    
    def get_answer_str(self, sample) -> str:
        return get_answer_str_bio_in_place(self, sample)
