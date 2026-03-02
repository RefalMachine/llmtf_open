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
{text}"""

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
        return "Mykes/patient_queries_ner-(dict)"

    def get_gold_entities(self, sample) -> Dict[str, List[str]]:
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
        answer = self.get_gold_entities(sample)
        answer_str = ""
        for tag, tokens in answer.items():
            answer_str += f"{tag}: [" + ', '.join(tokens) + "]\n"
        return answer_str


PATIENT_QUERIES_NER_JSON_INSTRUCTION = """Ты — эксперт по извлечению именованных сущностей из русскоязычных текстов. Твоя задача — извлечь ВСЕ именованные сущности из текста, строго следуя указанным классам и формату вывода.

**Классы сущностей:**
- PER — человек (с именем или прямым указанием на лицо, например, «Иван», «доктор Петров»). Обычные местоимения или фразы вроде «мой ребенок», «я», «он» НЕ являются PER, если не указано конкретное имя.
- ORG — организация (медицинские учреждения, компании и т.п.).
- LOC — географическое местоположение (города, районы, улицы и т.п.).
- SIM — симптом или проявление болезни (например, «болит горло», «головная боль», «пятна на коже», «одышка»).
- SPEC — специальность врача или медицинская специализация (например, «психолог», «отоларинголог», «кардиолог»).
- GEN — указание на пол или возрастную группу (например, «мужчине», «женщине», «детскому»).
- SUBW — район или часть города, упомянутая как ориентир (например, «в районе Арбатской» → «Арбатской»).

**Критически важные правила:**
1. Извлекай ТОЛЬКО те сущности, которые явно относятся к одному из указанных классов. Никаких вымышленных или неоднозначных трактовок.
2. Никогда не извлекай местоимения, притяжательные конструкции или обобщённые обращения («я», «он», «она», «мой», «свой», «ребёнок», «ребёнку», «дочь», «моя дочь», «мой ребенок» и т.п.) как PER. Даже если они упоминаются с возрастом или описанием — это НЕ PER и НЕ GEN, если не указано конкретное имя.
3. Класс GEN используется ТОЛЬКО для явных указаний на пол («мужской», «женский») или возрастные категории («подросток», «младенец», «доношенный»), но НЕ для фраз вроде «мой ребенок», «14 лет», «в 10 лет» — такие конструкции НЕ являются GEN.
4. Класс SIM — только для описания симптомов, жалоб или проявлений болезни. Извлекай целостные фразы, отражающие симптом в контексте его проявления (например, «стал очень капризным», «часто плачет», «увеличились молочные железы»). Не разбивай симптом на части. Не включай общие слова вроде «патология», «лечение», «помощь», «проблемы» — они НЕ являются симптомами. Также НЕ извлекай вопросы вроде «что может быть причиной» — это не симптомы.
5. Класс SPEC — только для названий медицинских специальностей врачей («кардиолог», «гинеколог»). Извлекай в той форме, в которой они даны (например, «детского гинеколога» → SPEC). Не включай слова, не относящиеся к специальности (например, «лечение», «облегчение болей» — это НЕ SPEC).
6. Класс SUBW — только для названий районов, станций метро, упомянутых как ориентиры (например, «смоленская» в контексте «рядом со станцией метро смоленская»). Не включай полные описания вроде «станция метро смоленская» — извлекай только название ориентира.
7. Если сущность встречается несколько раз — извлеки все вхождения в порядке появления.
8. Сохраняй точную форму сущности из текста — не изменяй, не склоняй, не дополняй.
9. Никогда не добавляй сущности, которых нет в тексте, и не выдумывай классы.
10. Строго соблюдай формат вывода: список списков в JSON, где каждый элемент — [«ТИП», «текст сущности»].

**Особое внимание:**
- Фразы вроде «мой ребенок», «моя дочь» — НЕ являются PER, НЕ являются GEN, НЕ извлекаются вообще.
- Возрастные указания в скобках или в составе фразы («14 лет», «в 10 лет») — НЕ являются GEN.
- Слова «помощь», «проблемы», «патология», «лечение» — НЕ являются SIM.
- Симптомы должны быть описанием состояния или поведения: «стал капризным», «часто плачет», «увеличились молочные железы», «стали болезненными» — это SIM.
- Не извлекай отдельно прилагательные (например, «капризный»), если они не являются частью устойчивого симптомного выражения. Лучше извлекать глагольные или описательные конструкции, отражающие изменение состояния.
- Если симптом выражен составной фразой, извлекай её целиком, но без лишних слов (например, «часто плачет без причины» — допустимо, «плачет без причины» — тоже допустимо, но не «плачет» отдельно).
- Никаких сущностей, кроме строго определённых в классах. Никаких домыслов.

**Формат вывода:**
```json
[["ТИП", "текст сущности"], ["ТИП", "текст сущности"], ...]
```

**Текст для анализа:**
{text}"""

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
        return 'Mykes/patient_queries_ner-(json)'

    def get_gold_entities(self, sample) -> List[str]:
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
        answer = self.get_gold_entities(sample)
        answer_str = '```json\n' + json.dumps(answer, ensure_ascii=False, indent=4).strip() + '\n```'
        return answer_str


PATIENT_QUERIES_NER_IN_PLACE_INSTRUCTION = """Ты — эксперт по обнаружению именованных сущностей в русскоязычных текстах, специализирующийся на медицинских и повседневных описаниях состояния здоровья. Твоя задача — **дословно воспроизвести исходный текст**, пометив **только те фрагменты, которые однозначно соответствуют одному из заданных классов сущностей**, в точном соответствии с правилами. Не изменяй, не склоняй и не переформулируй слова — сохраняй оригинальное написание, пунктуацию и порядок слов.

### Классы сущностей:
- **PER** — конкретное имя или обращение к человеку (например, "Иван", "доктор Петров").
  ❌ Не размечай: "мой сын", "ребёнок", "дочь", "врач", "специалист", "мужчина", "женщина" — если это не имя собственное или официальное обращение с именем.
- **ORG** — организация (например, "поликлиника №5", "медицинский центр Линия жизни").
- **LOC** — географическое местоположение: город, район, улица, станция метро и т.п. (например, "Москва", "метро Смоленская", "Арбат").
  ⚠️ Если название совпадает с анатомической зоной (например, "смоленская", "арбатская"), но используется как топоним — это LOC.
- **SPEC** — профессия или специальность врача/специалиста (например, "детский психолог", "кардиолог", "отоларинголог").
  ✅ Разрешены: "педиатр", "невролог", "психотерапевт" — как узкие специальности.
  ❌ Не размечай: "врач", "специалист", "помощь специалиста", "консультация врача" — без указания специальности это не SPEC.
- **GEN** — указание на пол или возраст **только если явно обозначает медицинскую характеристику пациента** (например, "мужчина 45 лет", "девочка 3 лет").
  ⚠️ Размечай **только в формате "существительное + возраст"**, где возраст выражает клинический признак.
  ❌ Не размечай: "ребёнок", "мужчина", "женщина", "дочь", "сын" — если они используются без возраста или с притяжательными местоимениями ("мой ребёнок", "у дочери").
- **SIM** — симптом, жалоба или клиническое проявление:
  - физические или эмоциональные состояния ("ломота", "раздражительность", "плохо спит");
  - описания боли, зуда, отека и т.п. ("опухла кисть", "болит голова");
  - фразы, выражающие жалобу целиком ("часто плачет без причины", "проблемы с речью").
  ✅ Размечай **целые устойчивые фразы**, если они вместе образуют симптом.
  ✅ Включай глаголы действия, если они являются частью описания симптома ("стал капризным", "увеличились железы", "поднялась температура").
  ❌ Не включай в тег SIM вводные конструкции, вопросы, риторические фразы ("что может быть", "нужно ли обращаться", "как понять", "что делать").
- **SUBW** — анатомическая часть тела, орган или область ("рука", "голова", "подмышка").
  ⚠️ Не размечай топонимы, даже если они совпадают с названиями частей тела.

### Ключевые правила:
1. **Не размечай обобщённые слова.** Такие слова, как "врач", "специалист", "помощь", "приём", "цена", "поведение", "настроение", "причина", "можно", "подскажите", "как понять", "что происходит", "нужно ли" — **не являются сущностями** и не должны быть помечены.
2. **Точный охват для SIM.** Помечай **только конкретные проявления симптомов**, включая глаголы действия, если они описывают изменение состояния.
   - ✅ Правильно: `<SIM>стал очень капризным</SIM>`, `<SIM>поднялась температура до 39</SIM>`, `<SIM>увеличились молочные железы</SIM>`
   - ❌ Неправильно: `<SIM>что может быть</SIM>`, `<SIM>нужно ли обращаться к врачу</SIM>`
3. **Не разбивай сущности.** Если сущность состоит из нескольких слов — помечай её целиком (например, `<SIM>часто плачет без причины</SIM>`).
4. **Не дублируй теги.** Одна и та же фраза не должна быть помечена дважды.
5. **Не изменяй текст.** Сохраняй исходное написание, включая регистр, орфографию и пунктуацию. Не добавляй и не удаляй слова.
6. **Избегай ложных срабатываний.**
   - Слово "ребёнок" — ❌ **не является GEN**, если не используется в конструкции "ребёнок X лет" как характеристика пациента.
   - Слово "специалист" — ❌ **не является SPEC**, если не указано, кем он является.
   - Слово "хирург" — ✅ **является SPEC**, так как это специальность.
7. **Приоритет — точность.** Лучше пропустить сомнительную сущность, чем пометить её ошибочно.
8. **Не помечай вводные и риторические вопросы.** Фразы вроде "что может быть", "как помочь", "нужно ли обращаться к врачу", "как понять, что происходит" — ❌ **не являются симптомами** и не должны быть помечены как SIM.
9. **Не помечай местоимения и родственные формы.** Слова "ребёнок", "дочь", "сын", "мужчина", "женщина" в родительном падеже или с притяжательными местоимениями ("моего ребенка", "у дочери") — ❌ **не являются GEN или PER**, даже если указан возраст, если конструкция не описывает пациента напрямую.
10. **Сосредоточься на симптомах.** Основной фокус — на выявлении **реальных клинических проявлений**, а не на описании поведения, вопросах или общих рассуждениях.
11. **Не выносить симптом за пределы предложения.** Не дублируй симптомы из одного предложения в другое, даже если они повторяются.
12. **Не размечай отдельные прилагательные.** Такие слова, как "капризный", "зеленые", "сухой", не должны быть помечены отдельно, если не входят в устойчивое словосочетание, описывающее симптом целиком.

### Формат вывода:
Повтори текст дословно. Вставляй теги в формате `<тип>сущность</тип>` только вокруг фрагментов, которые **однозначно** относятся к одному из классов. Ничего не пропускай, ничего не добавляй.

**Текст для обработки:**
{text}"""

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
        return 'Mykes/patient_queries_ner-(in-place)'
    
    def get_gold_entities(self, sample) -> List[str]:
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
