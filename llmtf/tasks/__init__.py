from . import (
    darumeru,
    nlpcoreteam,
    rucola,
    shlepa,
    daru_treeway_summ,
    habrqa,
    ruopinionne,
    ruparam,
    rublimp,
    translation,
    ifeval,
    libra,
    math,
    rag,
    ner
)
from pathlib import Path
import json

########################################
# All tasks
########################################

ruopinionne_default_instruction = '''Твоя задача состоит в том, чтобы проанализировать текст и извлечь из него выражения мнений, представленные в виде кортежа мнений, состоящих из 4 основных составляющих:
1. Источник мнения: автор, именованная сущность текста (подстрока исходного текста), либо NULL. Key = Source;
2. Объект мнения: именованная сущность в тексте (подстрока исходного текста). Key = Target;
3. Тональность: положительная/негативная (POS/NEG). Key = Polarity;
4. Языковое выражение: аргумент, на основании которого принята результирующая тональность (одна или несколько неразрывных подстрок исходного текста). Key = Expression;

Если источник мнения отсутствует, то Source = NULL. Если источником мнения является автор, то Source = AUTHOR. В прочих случаях поле Source должно полностью совпадать с подстрокой исходного текста. Поля Target, Expression всегда совпадают с подстроками текста.
Ответ необходимо представить в виде json списка, каждый элемент которого является кортежем мнений. Каждый кортеж мнений это словарь, состоящий из четырех значений: Source, Target, Polarity, Expression.
Для извлечённых Source, Target, Polarity, Expression должно быть справедливо утверждение: "На основании выражения Expression можно сказать, что Source имеет Polarity отношение к Target".

Ниже представлены примеры выполнения задачи:

***Текст***
Он посмотрел на Премьер-Министра и сказал: "тебе пора переставать валять дурака".

***Ответ***
[{"Source": "NULL", "Target": "Премьер-Министра", "Polarity": "NEG", "Expression": ["пора переставать валять дурака"]}]

***Текст***
Также Владимира Машкова поздравил его учитель Олег Табаков: Он безусловный талант и настоящий хулиган.

***Ответ***
[{"Source": "Олег Табаков", "Target": "Владимира Машкова", "Polarity": "POS", "Expression": ["безусловный талант"]}, {"Source": "Олег Табаков", "Target": "Владимира Машкова", "Polarity": "POS", "Expression": ["настоящий хулиган"]}, {"Source": "Олег Табаков", "Target": "Владимира Машкова", "Polarity": "POS", "Expression": ["поздравил"]}]

***Текст***
Административный суд Кёльна снял с последнего альбома немецкой индастриал-метал группы Rammstein «Liebe ist für alle da» все ограничения на реализацию.

***Ответ***
[{"Source": "Административный суд Кёльна", "Target": "Rammstein", "Polarity": "POS", "Expression": ["снял", "ограничения на реализацию"]}]

Проанализируй таким же образом следующий текст. Внимательно следи за тем, чтобы содержимое Source, Target, Expression полностью копировалось из текста без каких либо изменений.

***Текст***
{text}'''
ruopinionne_default_instruction_short = '''***Текст***\n{text}'''

ruopinionne_simple_instruction = '''Твоя задача состоит в том, чтобы проанализировать текст и извлечь из него выражения мнений, представленные в виде кортежа мнений, состоящих из 3 основных составляющих:
1. Источник мнения: автор, именованная сущность текста (подстрока исходного текста), либо NULL. Key = Source;
2. Объект мнения: именованная сущность в тексте (подстрока исходного текста). Key = Target;
3. Тональность: положительная/негативная (POS/NEG). Key = Polarity;

Если источник мнения отсутствует, то Source = NULL. Если источником мнения является автор, то Source = AUTHOR. В прочих случаях поле Source должно полностью совпадать с подстрокой исходного текста. Поля Target всегда совпадают с подстроками текста.
Ответ необходимо представить в виде json списка, каждый элемент которого является кортежем мнений. Каждый кортеж мнений это словарь, состоящий из трех значений: Source, Target, Polarity.
Для извлечённых Source, Target, Polarity должно быть справедливо утверждение: "Source имеет Polarity отношение к Target".

Ниже представлены примеры выполнения задачи:

***Текст***
Он посмотрел на Премьер-Министра и сказал: "тебе пора переставать валять дурака".

***Ответ***
[{"Source": "NULL", "Target": "Премьер-Министра", "Polarity": "NEG"}]

***Текст***
Также Владимира Машкова поздравил его учитель Олег Табаков: Он безусловный талант и настоящий хулиган.

***Ответ***
[{"Source": "Олег Табаков", "Target": "Владимира Машкова", "Polarity": "POS"}]

***Текст***
Административный суд Кёльна снял с последнего альбома немецкой индастриал-метал группы Rammstein «Liebe ist für alle da» все ограничения на реализацию.

***Ответ***
[{"Source": "Административный суд Кёльна", "Target": "Rammstein", "Polarity": "POS"}]

Проанализируй таким же образом следующий текст. Внимательно следи за тем, чтобы содержимое Source, Target полностью копировалось из текста без каких либо изменений.

***Текст***
{text}'''
ruparam_default_instruction = 'Какое из двух предложений является правильным и грамматичным с точки зрения русского языка?\nПредложение 1. {sent_lhs}\nПредложение 2. {sent_rhs}\nОтветь только одной цифрой 1 или 2, ничего не добавляя.'

doom_system = '''Ты эксперт в математике и физике. Ты внимательно и подробно размышляешь и безошибочно решаешь задачи.'''
doom_instruction = '''Реши следующую задачу эффективно и ясно. Последняя строка твоего ответа должна иметь следующий формат:
'Таким образом, окончательный ответ: $\\boxed{ОТВЕТ}$.' (без кавычек), где ОТВЕТ - это просто окончательное число или выражение, решающее задачу.
Думай шаг за шагом перед ответом.

**Задача**
{task}
'''.strip()

# outdated
nerel_default_prompt = '''Извлеки из заданного ниже текста все вложенные именованные сущности всех представленных ниже классов.
Сущности могут быть представлены только целым словом, окружённым пробелами или знаками препинания, либо непрерывной последовательностью целых слов, разделённых пробелами.
Оставь сущности в том виде, в каком они даны в тексте, не изменяй и не склоняй их, иначе тебе будет выставлен штраф 100$.

**Классы**
DISTRICT - район города.
CITY - город.
STATE_OR_PROVINCE - штат или конкретная область / субьект / округ.
COUNTRY - страна.
PERSON - конкретный человек с ФИО.
PROFESSION - профессия.
DATE - дата.

Требуемый формат для каждого класса: "Класс: ["сущность", ..., "сущность"]". Вместо "Класс" используй соответствующие классы, представленные выше. Сущности каждого класса выведи на отдельной строке.
Если сущностей соответствующего класса в тексте нет, выведи на соответствующей строке "Класс: []".

**Пример**
Будущий ученый тайно покинул дом 15 декабря 1730 года и вскоре он догнал торговый обоз, шедший в Москву.
->
DISTRICT: []
CITY: ["Москву"]
STATE_OR_PROVINCE: []
COUNTRY: []
PERSON: []
PROFESSION: ["ученый"]
DATE: ["15 декабря 1730 года"]

Теперь извлеки вложенные именованные сущности для следующего текста.
**Текст**
{text}'''

nerel_bio_default_prompt = '''Ты — эксперт по извлечению биомедицинских сущностей. В тексте ниже найди все именованные сущности типов:
- **DISO** (расстройства, заболевания, синдромы)
- **PHYS** (физиологические показатели, процессы, явления)
- **ANATOMY** (анатомические структуры)
- **CHEM** (химические вещества/лекарства)
- **SPECIES** (биологические виды)

**Ключевые правила:**
1. Извлекай ВСЕ вхождения сущностей, включая дубликаты и части составных терминов, сохраняя порядок встречаемости
2. Сохраняй оригинальный регистр и форму слов
3. Для перекрывающихся сущностей ("распространенность кариеса"→PHYS, "кариеса"→DISO) извлекай обе
4. Строго соблюдай типизацию: 
   - "кариес" → DISO, 
   - "индекс КПУ" → PHYS, 
   - "зубочелюстных" → ANATOMY

**Формат вывода (ТОЛЬКО JSON):**
```json
[["тип", "сущность"], ["тип", "сущность"], ...]

#Text:
{text}'''

rusbeir_rag_task_first = '''Ты полезный вопросно-ответный ассистент, который кратно и точно отвечает на вопросы пользователя, без объяснений, без markdown разметки, только максимально краткий ответ.
Для помощи тебе в ответах на вопросы будет использоваться поисковая система, которая будет возвращать некоторое количество **сегментов** с возможно релевантной информацией. Сегментов может не быть и они не обязательно содержат полезную для ответа информацию. 
Порядок сегментов случайный и не отражает степень полезности для ответа. 

**Результат работы поисковой системы:**
{segments}

**Запрос пользователя:**
{question}
'''.strip()

rusbeir_rag_data_first = '''**Результат работы поисковой системы:**
{segments}

**Инструкция**
Ты полезный вопросно-ответный ассистент, который кратно и точно отвечает на вопросы пользователя, без объяснений, без markdown разметки, только максимально краткий ответ.
Для помощи тебе в ответах на вопросы будет использоваться поисковая система, которая будет возвращать некоторое количество **сегментов** с возможно релевантной информацией. Сегментов может не быть и они не обязательно содержат полезную для ответа информацию. 
Порядок сегментов случайный и не отражает степень полезности для ответа. 

**Запрос пользователя:**
{question}
'''.strip()

# REGISTRY
TASK_REGISTRY = {
    'darumeru/multiq': {'class': darumeru.MultiQ},
    'darumeru/parus': {'class': darumeru.PARus},
    'darumeru/rcb': {'class': darumeru.RCB},
    'darumeru/rummlu': {'class': darumeru.ruMMLU},
    'darumeru/ruopenbookqa': {'class': darumeru.ruOpenBookQA},
    'darumeru/ruworldtree': {'class': darumeru.ruWorldTree},
    'darumeru/rwsd': {'class': darumeru.RWSD},
    'darumeru/use': {'class': darumeru.USE},
    'nlpcoreteam/rummlu': {'class': nlpcoreteam.ruMMLU},
    'nlpcoreteam/enmmlu': {'class': nlpcoreteam.enMMLU},
    'russiannlp/rucola_custom': {'class': rucola.RuColaCustomTask},
    'shlepa/moviesmc': {'class': shlepa.ShlepaSmallMMLU, 'params': {'dataset_name': 'Vikhrmodels/movie_mc'}},
    'shlepa/musicmc': {'class': shlepa.ShlepaSmallMMLU, 'params': {'dataset_name': 'Vikhrmodels/music_mc'}},
    'shlepa/lawmc': {'class': shlepa.ShlepaSmallMMLU, 'params': {'dataset_name': 'Vikhrmodels/law_mc'}},
    'shlepa/booksmc': {'class': shlepa.ShlepaSmallMMLU, 'params': {'dataset_name': 'Vikhrmodels/books_mc'}},
    'daru/treewayabstractive': {'class': daru_treeway_summ.DaruTreewayAbstractive},
    'daru/treewayextractive': {'class': daru_treeway_summ.DaruTreewayExtractive},
    'ilyagusev/gazeta': {'class': daru_treeway_summ.Gazeta},
    'darumeru/cp_sent_ru': {'class': darumeru.CopyText, 'params': {'subtask': 'sent', 'lang': 'ru'}},
    'darumeru/cp_sent_en': {'class': darumeru.CopyText, 'params': {'subtask': 'sent', 'lang': 'en'}},
    'darumeru/cp_para_ru': {'class': darumeru.CopyText, 'params': {'subtask': 'para', 'lang': 'ru'}},
    'darumeru/cp_para_en': {'class': darumeru.CopyText, 'params': {'subtask': 'para', 'lang': 'en'}},
    'darumeru/cp_doc_ru': {'class': darumeru.CopyText, 'params': {'subtask': 'doc', 'lang': 'ru'}},
    'darumeru/ruscibench_grnti_ru': {'class': darumeru.ruSciBenchGRNTIRu},
    'darumeru/flores_ru_en': {'class': translation.DaruFlores, 'params': {'input_lang': 'ru'}},
    'darumeru/flores_en_ru': {'class': translation.DaruFlores, 'params': {'input_lang': 'en'}},
    'vikhrmodels/habr_qa_sbs': {'class': habrqa.HabrQASbS},
    'ruopinionne': {'class': ruopinionne.RuOpinionNE, 'params': {'instruction': ruopinionne_default_instruction, 'short_instruction': ruopinionne_default_instruction_short}},
    'ruopinionne_simple': {'class': ruopinionne.RuOpinionNESimple, 'params': {'instruction': ruopinionne_simple_instruction, 'short_instruction': ruopinionne_default_instruction_short}},
    'ruparam': {'class': ruparam.RuParam, 'params': {'instruction': ruparam_default_instruction}},
    'russiannlp/rublimp-(classify)': {'class': rublimp.RuBlimpClassify, 'params': {}},
    'russiannlp/rublimp-(choice)': {'class': rublimp.RuBlimpChoice, 'params': {}},
    'MalakhovIlya/NEREL-(dict)': {'class': ner.NestedNerDict, 'params': {}},
    'MalakhovIlya/NEREL-(json)': {'class': ner.NestedNerJson, 'params': {}},
    'MalakhovIlya/NEREL-(in-place)': {'class': ner.NestedNerInPlace, 'params': {}},
    'nerel-ds/NEREL-BIO-(dict)': {'class': ner.NerelBioDict, 'params': {}},
    'nerel-ds/NEREL-BIO-(json)': {'class': ner.NerelBioJson, 'params': {}},
    'nerel-ds/NEREL-BIO-(in-place)': {'class': ner.NerelBioInPlace, 'params': {}},
    'Mykes/patient_queries_ner-(dict)': {'class': ner.PatientQueriesNerDict, 'params': {}},
    'Mykes/patient_queries_ner-(json)': {'class': ner.PatientQueriesNerJson, 'params': {}},
    'Mykes/patient_queries_ner-(in-place)': {'class': ner.PatientQueriesNerInPlace, 'params': {}},
    'RCC-MSU/collection3-(dict)': {'class': ner.Collection3Dict, 'params': {}},
    'RCC-MSU/collection3-(json)': {'class': ner.Collection3Json, 'params': {}},
    'RCC-MSU/collection3-(in-place)': {'class': ner.Collection3InPlace, 'params': {}},
    'ruifeval':  {
        'class': ifeval.RuIFEvalTask
    },
    'enifeval':  {
        'class': ifeval.EnIFEvalTask
    },
    'doom/math': {
        'class': math.DOoM,
        'params': {'domain': 'math', 'system_prompt': doom_system, 'instruction': doom_instruction, 'max_new_tokens': 30000}
    },
    'doom/phys': {
        'class': math.DOoM,
        'params': {'domain': 'phys', 'system_prompt': doom_system, 'instruction': doom_instruction, 'max_new_tokens': 30000}
    },
    't-bank/t-math': {
        'class': math.TMath,
        'params': {'system_prompt': doom_system, 'instruction': doom_instruction, 'max_new_tokens': 30000}
    },
    'rusbeirrag/rubqqa': {
        'class': rag.RusbeirRag,
        'params': {'instruction': rusbeir_rag_task_first, 'dataset': 'bearberry/rubqqa'}
    },
    'rusbeirrag/rus_tydiqa': {
        'class': rag.RusbeirRag,
        'params': {'instruction': rusbeir_rag_task_first, 'dataset': 'bearberry/rus_tydiqa'}
    },
    'rusbeirrag/sberquadqa': {
        'class': rag.RusbeirRag,
        'params': {'instruction': rusbeir_rag_task_first, 'dataset': 'bearberry/sberquadqa'}
    },
    'rusbeirrag/rus_xquadqa': {
        'class': rag.RusbeirRag,
        'params': {'instruction': rusbeir_rag_task_first, 'dataset': 'bearberry/rus_xquadqa'}
    },
    'rusbeirrag/rubqqa_data_first': {
        'class': rag.RusbeirRag,
        'params': {'instruction': rusbeir_rag_data_first, 'dataset': 'bearberry/rubqqa', 'name_suffix': 'data_first'}
    },
    'rusbeirrag/rus_tydiqa_data_first': {
        'class': rag.RusbeirRag,
        'params': {'instruction': rusbeir_rag_data_first, 'dataset': 'bearberry/rus_tydiqa', 'name_suffix': 'data_first'}
    },
    'rusbeirrag/sberquadqa_data_first': {
        'class': rag.RusbeirRag,
        'params': {'instruction': rusbeir_rag_data_first, 'dataset': 'bearberry/sberquadqa', 'name_suffix': 'data_first'}
    },
    'rusbeirrag/rus_xquadqa_data_first': {
        'class': rag.RusbeirRag,
        'params': {'instruction': rusbeir_rag_data_first, 'dataset': 'bearberry/rus_xquadqa', 'name_suffix': 'data_first'}
    }
}

# LIBRA
with open(str(Path(__file__).parent / 'libra' / 'libra_config.json'), "r", encoding="utf-8") as f:
    libra_tasks = list(json.load(f).keys())

for task in libra_tasks:
    TASK_REGISTRY['libra/'+ task] = {
        'class': libra.LibraTask,
        'params': {'dataset_slice': task}
    }