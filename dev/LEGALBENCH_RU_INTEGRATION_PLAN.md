# LegalBench-RU: исследование и план интеграции в llmtf

Дата исходного плана: 2026-09-29. Обновление 2026-09-30: интеграционный код
реализован; фактические проверки и ограничения описаны в
[validation report](LEGALBENCH_RU_VALIDATION_REPORT.md). Ниже сохранён исходный
план, а не перечень незавершённых работ.

Уточнение scope от 2026-10-02: вместо отдельных LegalBench-RU YAML используются
общие `benchmark/llmtf_legal_foundational.yaml` и
`benchmark/llmtf_legal_instruct.yaml`, включающие также `shlepa/lawmc`.
Обычный отчёт показывает отдельные значения режимов без категорий и парных
дельт. Упомянутый ниже `benchmark/legalbench_ru_report.py` перенесён в
`dev/tools/legalbench_ru_report.py` и остаётся необязательной replay-утилитой.
Пути и этапы исходного плана ниже сохранены для истории; актуальная инструкция —
[юридический бенчмарк](../docs/legal_benchmark.md), фикс Shlepa —
[отчёт](SHLEPA_FEW_SHOT_FIX_REPORT.md).

Рекомендация: добавить LegalBench-RU как отдельный экспериментальный набор
задач с `method='generate'`, используя текущие `SimpleFewShotHFTask`, `LLM` и
`Evaluator`. Обоснованные изменения scoring и invocation допустимы при явном
версионировании и измерении их влияния; предложения приведены в разделе 10.
Требования после обсуждения: весь исходный корпус используется без фильтрации
по upstream public/holdout; выделяется собственный few-shot pool, поддержка
базовых моделей входит в первую поставку. Инструменты передаются исходным
текстовым prompt. Первая поставка — closed-протокол с 0/1/5-shot, затем контекстные условия с
проверкой покрытия и парным сравнением. Общий task/eval v2 из
[плана рефакторинга](TASK_EVAL_GRAPH_REFACTOR_PLAN_v2.md) для этого не требуется:
каждый пример требует одного логического вызова модели и детерминированного
скоринга. Framework reasoning при необходимости сам выполняет две фазы.

## 1. Источники и воспроизводимость исследования

Исследованы реальные файлы, а не только карточка датасета. Сеть использовалась
через прокси `http://127.0.0.1:8118`. Зафиксированы:

| Источник | Revision |
|---|---|
| [HF dataset](https://huggingface.co/datasets/AlsKozlov/legalbench-ru/tree/d6e21bfee9a2842ea3eb5b2e97a5c8f1186d41f1) | `d6e21bfee9a2842ea3eb5b2e97a5c8f1186d41f1` |
| [GitHub code/data](https://github.com/AlsKozlov/legalbench-ru/tree/090163977d0e9a58a5887a38ec144f95cb37e464) | `090163977d0e9a58a5887a38ec144f95cb37e464` |
| llmtf, HEAD при исследовании | `d36543888b6cc3865cf3a584b5c1bda0b0455567` |

В llmtf также прочитаны текущие незакоммиченные планы в `dev/`; они описывают
будущую архитектуру, не действующий API. Существовавшие изменения пользователя
не являются частью этой интеграции.

Контрольные SHA-256:

```text
legalbench_ru.jsonl
4c63ca94fb87787e98f46f2ed013e41cef0c455c73943d4f850127062d3bca2a
holdout_questions_only.jsonl
f6d21020d2e55f0d6eaa28624ecee4e0782d075323c19c6117c0e61c5e9cf6c5
tool_catalog.json
a8fd548eb204d4172706fca7637b304f5d4692a811a37a6f10fe89904bf970d0
score.py
1e25d9de7ab20543a6ca383eb8e209cef3ad8a0a7fb8986a1974a48b0080c42c
models.py
381ff3d8f1fe248b272121e08447ee2990b0af480cf59cfaf5bad59dc380f7a8
```

Оба JSONL из HF побайтово совпали с `publish/` в указанном GitHub commit.
Все 846 записей сопоставлены по `(task, id)` с `tasks/*/test.jsonl`:
значения исходных полей совпали. HF export добавляет метаданные и `split`.

## 2. Фактический состав данных

В основном JSONL 847 объектов: один служебный объект `_canary` и 846 примеров.
HF показывает физический split `train`; исходные метки `public`/`holdout`
хранятся в поле каждой записи. Таблицы ниже описывают upstream snapshot;
эти метки не управляют нашим разделением на demonstrations/evaluation.

| Трек | Public | Holdout | Всего |
|---|---:|---:|---:|
| knowledge | 247 | 64 | 311 |
| reasoning | 282 | 66 | 348 |
| tool-use | 145 | 42 | 187 |
| **Всего** | **674** | **172** | **846** |

| Тип ответа | Public | Holdout | Всего | JSON-тип `answer` |
|---|---:|---:|---:|---|
| norm_citation | 135 | 35 | 170 | list |
| extraction | 183 | 42 | 225 | string |
| binary | 145 | 40 | 185 | string |
| multiple_choice | 66 | 13 | 79 | string |
| tool_call | 145 | 42 | 187 | object |

В исходниках 78 задач, восемь значений `reasoning_type`, 33 значения `domain`:
16 доменов знания/рассуждения и 17 названий источников для tool-use. Поэтому
«16 доменов» из README нельзя использовать как размер множества поля `domain`.
`temporal` — условие подачи контекста; отдельного значения `track=temporal`
нет. Поддержанный кодом тип `open` в текущих данных не представлен.

Каталог содержит 79 инструментов 17 серверов. Среди 187 tool-примеров
17 имеют `answer.tool=null`. Это задачи выбора инструмента и аргументов:
реальных вызовов MCP, запросов к реестрам и многошаговых эпизодов здесь нет.

Все 846 примеров и все 78 metadata-файлов имеют `needs_expert_review=true`.
Это статус авторской разметки; техническая проверка скорера не подтверждает
юридическую корректность ответов. В исследовании она не проверялась.

`id` не глобально уникален: три значения встречаются в двух разных задачах.
Использовать составной ключ `(task, id)`, не удалять такие строки как дубли.

Holdout-ответы **уже опубликованы в основном JSONL**. Файл
`holdout_questions_only.jsonl` содержит 172 записи без gold и без
`norm_text`/`distractor_text`/`temporal_text`. Нельзя называть локальный замер
на основном файле приватным holdout или считать файл вопросов самостоятельной
размеченной задачей. Интеграция берёт все 846 записей и выделяет собственный
фиксированный demonstration pool; остальные записи образуют evaluation set
(раздел 6). Upstream split сохраняется только как metadata/provenance.

## 3. Устройство upstream-кода

| Файл | Реальная роль | Решение для llmtf |
|---|---|---|
| [score.py](https://github.com/AlsKozlov/legalbench-ru/blob/090163977d0e9a58a5887a38ec144f95cb37e464/score.py) | Prompts, парсеры, scorers, загрузка каталогов, runner, агрегация | Выделить чистые функции протокола; зафиксировать версию |
| [models.py](https://github.com/AlsKozlov/legalbench-ru/blob/090163977d0e9a58a5887a38ec144f95cb37e464/models.py) | Stub и синхронный OpenAI-compatible клиент через urllib | Использовать только как reference; inference выполняет `LLM` |
| `tasks/*/{meta.json,test.jsonl}` | Метаданные и исходные примеры | Сверять с HF snapshot; не загружать весь GitHub при каждом запуске |
| `tool_catalog.json` | Полный текстовый каталог для prompt | Поставлять pinned ресурс и проверять hash/порядок |
| `validate.py`, `qa.py`, `coverage.py` | Ручная валидация, QA-сигналы, статистика | Reference-аудит плюс собственные contract tests |
| `schema/*.json` | Описательные JSON Schema | Адаптировать: схемы отстают от данных |
| `publish/` | HF export, лицензия и карточка | Источник атрибуции и контрольных файлов |

`score.py` читает GitHub `tasks/`, не HF JSONL. В его CLI нет фильтра по
логическому split; обычный запуск оценивает все 846 записей. `--limit` применяется
отдельно к каждой из 78 задач. В llmtf лимит применяется к зарегистрированному
dataset id, поэтому численно одинаковые лимиты не означают одинаковую выборку.

В `models.py` отправляется один `user` message, `temperature=0`; при HTTP 400
есть повтор без temperature. Лимит ответа не задаётся. Не переносить этот
клиент и его retry: llmtf уже управляет backend, таймаутами, batch alignment,
reasoning и provenance. Новый явный answer budget записывать как отличие
runtime-протокола, не обещать побитовое совпадение живых ответов.

`validate.py` не исполняет приложенные JSON Schema. Он допускает tool_call,
но schema для instance не допускает объект в `answer` и не описывает `split`;
task schema не включает tool-use в соответствующие enum. Прошедшая ручная
валидация не означает соответствия экспортного файла этим схемам.

Лицензирование явно описано в
[publish/LICENSE](https://github.com/AlsKozlov/legalbench-ru/blob/090163977d0e9a58a5887a38ec144f95cb37e464/publish/LICENSE):
данные CC BY 4.0, код Apache 2.0. При переносе сохранить attribution, сведения
об изменениях, текст применимой лицензии и notices. Canary исключить из
evaluation, сохранив его наличие в manifest. Few-shot использует выделенный
pool согласно требованию пользователя; ответы evaluation set не используются
как демонстрации. Обновления весов модели не входят в эту интеграцию.
Полные вопросы и gold не копировать в tests.

## 4. Скоринг: что необходимо сохранить и что нельзя скрыто исправлять

| Тип | Поведение текущего `score.py` |
|---|---|
| extraction | После lowercase, `ё→е`, схлопывания пробелов ищет gold или `accept` как **подстроку** ответа |
| binary | Берёт первое отдельное «да»/«нет» и сравнивает с gold |
| multiple_choice | Берёт первую отдельную заглавную латинскую A–D; затем fallback по тексту варианта |
| norm_citation | Извлекает пары «акт, статья», считает set F1; части/пункты не оцениваются отдельно |
| tool_call | Неверный tool: 0; верный tool: 0.6 + 0.4 × доля совпавших gold-аргументов; при отсутствии gold-args: 1 |
| open | Доля gold-термов, встретившихся в ответе; пока нет соответствующих примеров |

Название AST-match из карточки не описывает строгий AST exact match:
server prefix может игнорироваться, лишние аргументы не штрафуются, значения
сравниваются через нормализованный `str`. Для отрицательных tool-примеров
работает эвристика явного отказа.

Проверенные на реальных функциях контрпримеры (все synthetic):

- `answer='10'`, output `210` → extraction score 1.
- Gold `right.lookup`, prediction `wrong.lookup` с верными args → score 1;
  дополнительный аргумент не снижает результат.
- Gold tool=null, prediction с ненулевым tool и `null` внутри args → score 1.
- Число в поле predicted `tool` при положительном gold → `AttributeError`.
- Валидный JSON со строкой, содержащей закрывающую фигурную скобку, может
  не распознаться: счётчик скобок не учитывает JSON string escaping.
- `ст. 1 ГК РФ; ст. 2 ТК РФ` разбирается как две статьи ГК: статья привязывается
  к предшествующему коду, если он есть.

Предлагаемый reference scorer id: `legalbench_ru_upstream_0901639_v1`.
Сохранить численные правила для входов, которые upstream обрабатывает;
выделить общий валидатор модельного формата, чтобы неверный тип JSON-поля
получал 0 и `parse_status`, а не обрушал всю задачу. Это явный совместимый
по корректным входам repair; записать его в change ledger и scorer version.
Не ловить произвольные ошибки программы как «неправильный ответ».

Улучшенный строгий JSON parser, границы чисел, проверка полного server.tool,
правильное определение отказа и новый citation parser — отдельный corrected scorer.
Сначала сравнить оба scorer на одинаковых сохранённых outputs и описать
разницу. Автоматически пересчитывать старый baseline новым scorer нельзя.
Слабые места reference scorer явно отражать в документации экспериментального
набора. Рекомендация после обсуждения: включить исправления доказанных ошибок
уже в первую поставку; сохранять reference score для сравнения (раздел 10).

Upstream overall — micro mean **округлённых до трёх знаков** per-instance
scores; промежуточные task means вычисляются до округления. Для основного
reference score сохранить `round(raw_score, 3)` перед общим mean, а raw score
оставить в sample details. Balanced accuracy бинарных ответов — диагностика,
не дополнительное слагаемое leaderboard score.

## 5. Режимы и сопоставимые выборки

| Условие | Необходимое поле | Public | Holdout | Всего |
|---|---|---:|---:|---:|
| closed | question, optional context/choices | 674 | 172 | 846 |
| grounded, reasoning | norm_text | 168 | 37 | 205 |
| distractor, reasoning | distractor_text | 156 | 33 | 189 |
| temporal, reasoning | temporal_text | 7 | 1 | 8 |

В upstream при отсутствии нужного поля prompt деградирует до closed.
Предлагаемая интеграция контекстных режимов использует **eligible-only**:
фильтрация до sample limit, отсутствие поля у выбранного примера — ошибка.
Это явно другой selection protocol относительно запуска upstream по всем
348 reasoning-примерам; идентичность baseline заявлять только на совпавших ids.

Все 189 distractor-примеров имеют norm_text. Все восемь temporal-примеров
также имеют norm_text, но не distractor_text. Общего набора для всех трёх
контекстных полей нет. Рекомендуемый отчёт:

1. Closed score на всём нашем evaluation set, micro по примерам.
2. `grounded − closed` на 205 eligible ids.
3. `distractor − grounded` на 189 eligible ids.
4. `temporal − grounded` и `temporal − closed` на 8 eligible ids.

Эти размеры сохраняются, если demonstrations выбраны без norm/distractor/
temporal fields, как предлагается в разделе 6. Фактические counts фиксируются
после проверки split manifest. В каждой паре режимов demonstration ids,
их порядок и тексты одинаковы: меняется только контекст текущего вопроса.

Дельта — mean индивидуальных разностей на пересечении **заранее заданных**
eligible ids; значения из разных выборок напрямую не вычитать. Проверять
совпадение dataset/model/scorer/prompt versions и generation settings;
отличаться должен режим контекста. При лимите или неполном покрытии сообщать
фактические N и недостающие ids. Полный paired report не строить по случайно
совпавшим остаткам после ошибок. Temporal N=7/8 явно показывать в отчёте.

`reject` сейчас меняет только инструкцию, но оставляет прежний gold/scorer.
Отказ «Недостаточно данных» обычно даёт 0; метрики корректности отказа нет.
В первую поставку этот режим не включать как измерение abstention. Позднее
возможен отдельный диагностический `reject` с исходным gold-score и refusal
rate, без заявления о правильности отказа. Полноценная rejection-задача
требует разметки достаточности контекста и собственного versioned протокола.

`grounded` — подача уже готовой нормы. Retriever, embeddings, векторная база
и judge-модель для этой интеграции не нужны.

## 6. Предлагаемый интерфейс в llmtf

### Реестр и выбор задач

Один класс `LegalBenchRU(SimpleFewShotHFTask)` с фиксированными registry
конфигурациями `mode`, `split_manifest`, `dataset_revision`, `protocol_version`.
Количество demonstrations передаётся штатным `few_shot_count`.
Предлагаемые ids (будущий API):

```text
legalbench_ru/closed
legalbench_ru/grounded
legalbench_ru/distractor
legalbench_ru/temporal
legalbench_ru/upstream_all_zero_shot
```

Смысл каждого id:

| Dataset id | Что получает модель | Что измеряем и на какой выборке |
|---|---|---|
| `legalbench_ru/closed` | Исходный вопрос и предусмотренные им context, choices или текстовый каталог инструментов; дополнительная норма не подставляется | Основная оценка на всём нашем evaluation set: предварительно 816 примеров из knowledge, reasoning и tool-use |
| `legalbench_ru/grounded` | Тот же вопрос с подходящей нормой из `norm_text` | Применение предоставленной нормы; 205 eligible reasoning-примеров. Разница с closed на тех же ids показывает пользу контекста |
| `legalbench_ru/distractor` | Тот же вопрос с подставной неверной нормой из `distractor_text` | Устойчивость к вводящему в заблуждение контексту; 189 eligible reasoning-примеров. Gold остаётся прежним |
| `legalbench_ru/temporal` | Тот же вопрос с устаревшей редакцией из `temporal_text`, без предупреждения об устаревании | Устойчивость к устаревшему контексту; 8 eligible reasoning-примеров. Gold соответствует актуальной версии по разметке snapshot, а не автоматически на дату запуска |
| `legalbench_ru/upstream_all_zero_shot` | Исходные closed-промпты всех 846 примеров без demonstrations | Необязательная reference-диагностика для сверки с upstream: исходный scorer и его агрегация, включая зарезервированные в нашем split демонстрации |

Первые четыре id поддерживают 0/1/5-shot из нашего фиксированного pool.
Для парных сравнений состав, порядок и текст demonstrations одинаковы;
меняется только дополнительный контекст текущего вопроса. Размеры 816 и
205/189/8 предполагают выделение 30 demonstrations без контекстных полей;
окончательные counts определяет проверенный split manifest.

Tool-use входит в `closed` с исходным текстовым каталогом; остальные три
режима относятся к reasoning-примерам с соответствующим полем нормы.
Отсутствие поля не превращает grounded/distractor/temporal в closed.
Closed не означает отсутствие фактов из `context` или каталога tools.

Сравнивать режимы на одинаковых ids: grounded с closed на 205 примерах,
distractor с grounded на 189, temporal с grounded/closed на 8. Общий closed
score на 816 примерах не вычитать из score контекстного subset.
`upstream_all_zero_shot` требует `few_shot_count=0`; его общий score на 846
примерах не сравнивать напрямую с основным на 816. Для сравнения scorers
использовать одни и те же сохранённые outputs и ids. Этот reference id
не обещает точного воспроизведения опубликованных live baselines моделей.

Отдельные 78 классов/registry ids не нужны:
исходная `task` сохраняется в каждой записи и в агрегатах. Все task_name и
run_name должны различать mode/selection/protocol; 0/1/5-shot выводятся в
разные output_dir и с явными name_suffix `0shot`/`1shot`/`5shot`, присутствуют
в fingerprint. Учесть замену `/` на `_` в logger.

### Собственное разделение для few-shot

Все 846 записей рассматриваются как единый исходный корпус. Предлагаемый
стартовый manifest: **30 demonstration records + 816 evaluation records**.
Это проект размера, а не уже утверждённый список ids. Выделение постоянное:
даже при `few_shot_count=0` demonstrations не возвращаются в основной test,
поэтому разница между 0/1/5-shot не смешивается со сменой оцениваемых вопросов.

| Группа выбора demonstrations | Pool | Ограничения |
|---|---:|---|
| norm_citation | 5 | Разные акты/задачи; одиночная и множественная ссылка |
| binary | 5 | Обе метки, например 3 Да / 2 Нет |
| multiple_choice | 5 | Все A–D представлены, несколько задач |
| extraction + knowledge | 5 | Короткие ответы/правила, разные формы ответа |
| extraction + reasoning | 5 | Расчёт/применение/вывод, разнообразные формы |
| tool_call | 5 | Четыре положительных вызова разных серверов и один отказ |

Выбор demonstration bucket зависит только от `answer_type` и, для extraction,
от `track` текущего примера. Внутри bucket один заранее фиксированный порядок:
1-shot использует префикс из одного примера, 5-shot — весь bucket. Не выбирать
демонстрации по gold или результатам модели. `few_shot_count` от 0 до 5
поддерживается; больше 5 — явная ошибка до inference, пока pool не расширен
с новой версией split. По одному pool на каждую из 78 задач не требуется.

Pool выбирать из строк без `norm_text`, `distractor_text`, `temporal_text`,
чтобы сохранить малые контекстные cohorts для оценки. Проверка snapshot
подтвердила достаточное число кандидатов: 170 citations, binary 26 Да/16 Нет,
MC A/B/C/D = 13/12/11/11, extraction knowledge/reasoning = 141/54,
tool positives/negatives = 170/17. Это counts кандидатов до duplicate review.

В L0 построить группы близких/производных вопросов по нормализованным
question/context, исходной task и ручной проверке совпадений. Варианты одного
кейса и его режимы контекста не разносить по demo/eval. Совпадение только номера
статьи или bare id не является достаточным основанием объединять вопросы.
Предпочесть 30 подходящих singleton cases без близких аналогов в evaluation;
если необходим целый кластер, зарезервировать его полностью и явно пересчитать
N. Не обещать 816 до фиксации manifest и не удалять evaluation примеры по
результатам ответов модели.

Кандидатов ранжировать воспроизводимо (seed 555 и стабильные composite keys),
проверить разнообразие/разметку pool, затем сохранить exact ids и порядок в
`split_manifest.json`. У reviewed demonstrations проверяется пригодность
короткого gold для показа модели; сомнительные candidates заменяются до freeze.
Изменение pool — новая версия протокола, без подбора под leaderboard.
Одинаковые pool и evaluation ids использовать для Base и Instruct.

Важная текущая особенность: `datasets_names='all'` разворачивается во **весь**
`TASK_REGISTRY` и для normal evaluation, и для PPL. Простое добавление ids
изменит существующие запуски. Перед регистрацией добавить небольшой общий
признак `include_in_all` с default true и общий resolver; новым experimental
ids выставить false. Явный выбор всегда доступен. Применить resolver к обоим
evaluation paths и проверить сохранение прежнего состава `all`. Если к этому
моменту task/eval v2 уже предоставляет такой механизм, использовать его.

Поставить YAML `benchmark/legalbench_ru_instruct.yaml` и
`benchmark/legalbench_ru_foundational.yaml` с одинаковой selection и явными
few-shot settings. Base по умолчанию 5-shot; для Instruct — 0-shot и отдельный
5-shot контроль. Не включать экспериментальные результаты
в средний балл существующих instruct/foundational suites. Для generic report
использовать отдельный suite/category; paired deltas показывать отдельно,
не усреднять повторные измерения одного примера как независимые benchmarks.
Сейчас `Evaluator.create_report` усредняет все найденные totals в output_dir.
Поэтому closed и дополнительные условия запускать в отдельных каталогах;
offline report принимает несколько каталогов и объявляет primary score явно.
Даже внутри LegalBench среднее closed/grounded/distractor/temporal из generic
report не является итоговым баллом этого benchmark.

### Загрузка и промпты

- `dataset_args()` описывает pinned источник. Переопределить `_load_dataset`:
  получить исходный JSONL через `huggingface_hub` и читать stdlib `json`.
  `answer` имеет string/list/object; не полагаться на автоматическое приведение
  общего столбца Arrow или на Parquet viewer. `trust_remote_code` не требуется.
- Проверить SHA-256, служебную строку, обязательные поля и типы по answer_type,
  допустимые splits/tracks, уникальность `(task,id)`, структуру каталога и
  соответствие gold tool/args каталогу. Отсутствующий gold — data error.
- Исключить весь demonstration pool и отфильтровать eligible mode до лимита.
  Upstream public/holdout не участвует в фильтрации. Сохранить порядок pinned
  export и selected keys. Для small matrix использовать заранее сохранённый
  manifest, иначе первые восемь строк покрывают только citation-задачи.
- Реализовать собственный selection layer и split methods для логических
  `demonstrations`/`evaluation`. Не использовать generic выбор первых k строк.
- `create_messages(sample, with_answer=False)` возвращает один `user` message
  с prompt, эквивалентным upstream `build_prompt`. При `with_answer=True`
  добавляет `assistant` с каноническим коротким gold: буква/Да/Нет/строка,
  список норм через `; ` или компактный tool JSON. Никаких gold explanations.
- Few-shot message list: k пар `user/assistant` из своего bucket, затем
  текущий `user`. Текст каждого user сохраняет upstream context/options/
  tool-каталог. Демонстрации всегда строятся в closed, одинаково для всех
  режимов текущего вопроса. Не добавлять system instruction или task prefill.
- Использовать белый список prompt-полей. У текущего evaluation примера
  `answer`, `accept`, `gold_norm`, `explanation`, review metadata и canary
  не входят в prompt. У demonstration только answer идёт в assistant. Norm fields
  включаются только в соответствующем режиме; отрицательные/устаревшие нормы
  не маркируются иначе, чем в исходном протоколе.
- Проверять context budget общими средствами llmtf для полного k-shot prompt.
  Текущий `_prepare_messages` может уменьшать число demonstrations; для этой
  задачи переопределить сборку, чтобы фиксированные 5-shot не стали скрыто
  2-shot. При переполнении — явная ошибка; меньший k выбирается как отдельный
  run. Не обрезать каталог, норму или answer демонстрации. Если API не умеет
  считать токены, сохранять все k и unknown count; context error остаётся error.

Для HF/local-vLLM Base использовать `is_foundational=True` и явную
`default_foundational.json`: общая conversation template рендерит те же
user/assistant пары и начало ответа. Проверить stop strings и компактный
одноабзацный gold JSON. Шаблон модели является частью provenance.
В API `is_foundational` сейчас восстанавливает stops, но сам prompt рендерит
сервер. Нужен deployment с тем же foundational template и проверка фактически
отрендеренного многоходового prompt; одного клиентского flag недостаточно.

### Inference и метрики

`method='generate'` для всех пяти типов ответа. Замена binary/MC на token
probabilities меняет измеряемый протокол, поэтому возможна только отдельной
будущей конфигурацией. PPL не является метрикой LegalBench-RU.

Начальный `_max_task_new_tokens=512` — предлагаемый проверяемый budget;
зафиксировать его до baseline, проверить finish reasons и tool JSON. Если
smoke выявит систематическую нехватку, поменять до фиксации протокола и
перезапустить затронутые проверки. `temperature=0`, одна последовательность,
repetition penalty 1, presence penalty 0. Thinking off — исходная reference
конфигурация; thinking on — отдельный прогон через существующий LLM dispatcher.
Scorer получает только answer continuation, не reasoning trace.

`evaluate` возвращает стабильный ключ `score` с записью: composite id,
raw/rounded score, parser status, task/track/domain/answer_type и binary gold/
prediction при необходимости. `aggregation()['score']` возвращает
`(micro_score, details)` — это уже поддерживается `Evaluator` (пример подхода
есть в RuParam). `leaderboard_aggregation` явно берёт только `metrics['score']`.
Details: N, coverage, разрезы по task/track/domain/answer_type, balanced
accuracy, parse failure counts, review status. Все группировки считают mean
по примерам; mean 78 task means не заменяет reference micro.

Текущий `show_results.py` при bootstrap не распаковывает tuple reducer.
Для начальной версии выставить `ALLOW_BOOTSTRAPPING=False` и проверить
обычное чтение totals. При необходимости CI добавить отдельный paired
bootstrap в offline report с seed 555; единица ресемплирования — composite
example со всеми сравниваемыми условиями. До реализации CI их не обещать.

Пустой/неверно оформленный output остаётся в denominator с score 0.
Backend/network/context error или дефект данных делает task failed и даёт
non-zero exit; upstream исключает API errors из mean, это правило не переносить.
Нельзя публиковать partial mean как успешный полный результат.

Для PPL сохранить отсутствие `get_answer`: текущий evaluator отмечает такую
задачу как skipped. В документации и проверках фиксировать это поведение,
не называть его explicit unsupported/non-zero. Если общий task/eval refactor
добавит capability rejection, использовать общий механизм при миграции.

## 7. Provenance, кэш и отчёт

Текущий `build_run_config` хеширует модуль класса и registry init params, но
не автоматически все импортированные scorer/prompt helpers и JSON resources.
Fingerprint вычисляется **до** `load_dataset`; записать hashes после inference
недостаточно для проверки cache hit.

Добавить узкий необязательный task provenance hook в общий builder, например
`get_task_provenance()`. Для прежних tasks отсутствие hook сохраняет прежний
payload. LegalBench возвращает JSON-compatible manifest без секретов:

- dataset repo/revision, filename/hash, code commit, catalog hash;
- prompt/scorer/parser/selection versions и hashes фактических helper-файлов;
- собственный split manifest/hash, mode, missing-context policy, score rounding
  и aggregation rule;
- shot count, bucket mapping, ordered demonstration ids и hashes их текстов,
  answer serialization и conversation template, политика context overflow;
- frozen selection manifest/hash, если используется; answer budget;
- change ledger version и review status исходного snapshot.

Конструктор не выполняет сеть/inference: manifest и локальные helper hashes
доступны до cache check, данные загружаются и сверяются при выполнении.
Для локального data override hash фактического файла вычислять до cache check;
не доверять пути или заявленному пользователем revision вместо содержимого.
Selected count/ids и фактическое покрытие записать в aggregation details.
В sample artifact сохранять ids/порядок demonstrations и requested/effective k;
при фиксированном k их несовпадение считается ошибкой исполнения.

Сохранить стандартные params/sample/total/details и framework run fingerprint.
Output сохранять полностью: upstream обрезает его до 200 символов в отчёте,
чего недостаточно для повторного скоринга citation/tool-call. Реплей scorer
должен работать без загрузки LLM и без сети.

Отдельный offline `benchmark/legalbench_ru_report.py` собирает per-track
таблицу, counts и paired deltas по ключам. Проверяет membership, отсутствие
дубликатов, завершённые totals и совместимые manifests. Refusal/tool-call
diagnostics и review status не влияют на primary score.

Реализуемый scope кэша — переиспользование завершённого total с совпавшим
fingerprint. Текущий evaluator не возобновляет inference с середины sample
array: после прерывания задача перезапускается. Не обещать sample-level resume
и не строить отдельный checkpoint runtime только для этого benchmark.

## 8. План работ и изменяемые файлы

| Этап | Работа | Проверяемый результат |
|---|---|---|
| L0: freeze | Pinned manifest, attribution/licenses, data validator, catalog resource, собственный demo/eval split | Все 846 keys учтены; reviewed pool на 6 buckets, zero overlap, сохранены контекстные cohorts |
| L1: protocol | Чистые prompts/scorers, reference и corrected версии, differential fixtures | Prompt parity; oracle=1, stub=0; replay обоих scorers; явно описанные расхождения |
| L2: closed integration | Task, few-shot assembly, provenance hook, registry opt-in, Instruct/Base YAML, report | 0/1/5-shot с одинаковым eval set; Python API, обе CLI и три runners; прежний `all` сохранён |
| L3: conditions | Eligible grounded/distractor/temporal, offline paired report | Matched cohorts 205/189/8 после split audit; одинаковые demos в парных условиях |
| L4: validation | Offline contracts, Docker smokes, Base/Instruct small matrix, полный evaluation set | Полные artifacts, контекстные budgets few-shot и воспроизводимый validation report |
| L5: дальнейшее развитие | Новые определения метрик, проверенные текстовые расширения prompt, reject-разметка | Отдельные версии протокола; не блокируют L0–L4 |

Предлагаемая раскладка:

```text
llmtf/tasks/legalbench_ru/
    __init__.py, task.py, data.py, prompts.py, scoring.py
    manifest.json, split_manifest.json, tool_catalog.json, NOTICE, LICENSE*
llmtf/tasks/__init__.py                 # registry configurations
llmtf/provenance.py                    # optional manifest hook
llmtf/evaluator.py                     # общий opt-in selection resolver
benchmark/legalbench_ru_instruct.yaml
benchmark/legalbench_ru_foundational.yaml
benchmark/legalbench_ru_report.py
tests/test_legalbench_ru.py
tests/fixtures/legalbench_ru/           # synthetic, небольшие fixtures
docs/legalbench_ru.md
dev/LEGALBENCH_RU_VALIDATION_REPORT.md
```

Точный общий модуль resolver выбрать при реализации рядом с registry; не
дублировать списки исключений в evaluator и CLI. Существующий strict YAML
достаточен: mode/selection/protocol задаются registry params, few_shot_count —
существующий evaluation option. Не добавлять в YAML
неподдерживаемые task kwargs и не использовать backend_kwargs для task config.

Scoring/data preparation используют stdlib и уже имеющийся `huggingface_hub`.
Новые torch/encoder/MCP dependencies не нужны; API profile остаётся CPU-only.
Git submodule и запуск внешнего `score.py` из evaluator не требуются.

## 9. Проверки и критерии приёмки

### Уже выполнено в этом исследовании

В отдельном `/tmp` checkout запущены оригинальные скрипты на Python 3.10.12:

```bash
python3 validate.py
python3 qa.py
python3 score.py --model oracle --json oracle.json
python3 score.py --model stub --json stub.json
```

Результаты: validate — 0 ошибок/0 предупреждений; QA — 3 предупреждения о
межзадачных повторениях id; oracle — 1.0, stub — 0.0 на всех 846 примерах.
Дополнительно вычислены counts/hashes, сверены HF/GitHub записи, проверены
контрпримеры парсеров из раздела 4. Это offline-аудит upstream: реальный
inference и интеграционные тесты llmtf в рамках планирования не запускались.

### Offline gates при реализации

1. Loader: canary не считается sample; смешанные типы answer сохраняются;
   upstream split не фильтрует данные; demo pool исключается до limit при любом k;
   повторяющиеся bare id не теряются; близкие demo/eval дубли проверены;
   wrong hash, неизвестный тип, отсутствие gold/контекста дают data error.
2. Prompt parity: побайтовое сравнение с upstream для всех 846 closed prompts,
   205 grounded, 189 distractor и 8 temporal; каталог полный и в том же порядке.
   Прямые тесты исключения текущего gold/accept/explanation и зависимости от mode.
   Для few-shot проверить k полных user/assistant пар, gold только demos,
   правильный bucket, порядок, 0/1/5 и одинаковые demos в парных режимах.
3. Scorer parity: oracle/stub, частичный norm F1, tool partial credits,
   отказ/неверный формат, число вместо tool, строки со скобками/escaping,
   разные регистры MC, конфликтующие binary слова, numeric substring,
   несколько нормативных актов. Определённые repair outcomes проверяются
   отдельно от случаев точного совпадения upstream.
4. Aggregation: micro vs task macro, округление до mean, consistent metric
   keys, отсутствие изменения primary score при добавлении diagnostics;
   paired joins по composite keys, missing/duplicate ids, N=0/1.
5. Provenance/cache: одинаковая конфигурация даёт cache hit; изменение данных,
   helper, каталога, demo ids/order/answers, k, split/mode/selection/scorer его
   исключает; повторная
   обработка без inference совпадает с totals; секретов в params нет.
6. Integration: явный id работает, прежний `all` сохранён в обоих paths;
   few-shot 0..5 работает, >5 отклоняется, overflow не уменьшает k;
   PPL фиксируется как skipped; generic report читает
   tuple details с отключённым legacy bootstrap. Проверить обе CLI и все три
   runners, не ограничиваться Python API.
7. Failures: transport/context error не превращается в правильный/нулевой
   ответ и успешный total; пустой/неверный ответ модели даёт 0 с сохранением N.

### Реальные проверки после реализации

Сначала inspect environment и GPU passthrough в целевом Docker по AGENTS.md.
Записать image ids, GPU, model/tokenizer revisions и runtime pins. Проверить
dependency-free suite; при наличии pytest запустить его вариант. Даже при
неизменных backends нужны новые task tests и regression checks evaluator /
provenance / selection:

```bash
python3 tests/test_refactor_logic.py
python3 -m compileall -q llmtf evaluate_model.py evaluate_model_api.py benchmark show_results.py dev/tools examples
git diff --check
python3 -m pytest tests/test_refactor_logic.py -q
```

Small matrix: frozen восемь evaluation примеров вне demo pool, представляющих все пять типов,
включая положительный и отрицательный tool-call, multi-norm citation и разные
binary ответы. Один smoke sample перед каждым новым backend path. Модель
`Qwen/Qwen3.5-2B`, HF/local-vLLM/API, hybrid off/on, 0-shot и 5-shot;
включённые cells используют явный end token id. Дополнительные short cohorts
проверяют контекстные режимы и все восемь temporal-примеров. У контекстных cells matched ids
из closed/grounded должны быть сохранены или отдельно прогнаны.

Base — обязательная часть основной матрицы: `Qwen/Qwen3.5-2B-Base`,
foundational, thinking off, 0/1/5-shot на том же наборе восьми evaluation ids,
HF/local-vLLM/API с проверенным серверным template для API. До small matrix
проверить один 5-shot tool-call с полным каталогом и его token budget.
Strict reasoning smoke — отдельно на каждом backend. Не выводить
общую local/API parity из одинакового prompt-builder. Проверить полноту
tool JSON, отсутствие truncation, реальные applied budgets, batch ordering,
число totals и non-zero exit при искусственном backend failure.

Если меняются model/reasoning/backend компоненты, выполнить применимые v4
generate/probability/PPL cells по AGENTS.md. Перед заявлением общей runtime
безопасности собрать и проверить все API/HF/vLLM profiles, включая torch-free
API imports; успех новых generate-задач не заменяет эту матрицу.

После small matrix оценить tokens/time для каждого k и провести полный
evaluation run минимум на Base 5-shot и Instruct 0-shot с одинаковыми eval ids.
При pool ровно 30 singleton cases без контекстных полей:
816 closed + 205 grounded + 189 distractor + 8 temporal = **1218 логических
генераций на configuration**. Если split audit меняет N, пересчитать manifest.
Demonstrations увеличивают prompt tokens, но не число model requests;
thinking on может давать две фазы на генерацию. Tool 5-shot повторяет весь
каталог в каждом из шести user messages: оценить эту стоимость заранее.
Для прямого эффекта few-shot отдельно сравнить 0/5-shot одной и той же модели
на одинаковых ids; сравнение Base 5-shot с Instruct 0-shot не изолирует этот
эффект. Полные runs на каждой backend/thinking/k комбинации не обязательны.

Сравнение с upstream сначала проводить replay: идентичные prompts/outputs →
per-example scores → округление → totals. Опубликованные проценты моделей
из README не являются acceptance target: в карточке недостаточно сведений
о deployment/model revisions и sampling для точного воспроизведения.

Работа завершена, когда pinned данные и протокол доступны через штатные
entry points, собственные demo/eval manifests зафиксированы, Base и Instruct
поддерживают few-shot, контекстные дельты проверяют membership, кэш учитывает
демонстрации и все ресурсы скорера, есть стандартные artifacts и
validation report с фактически выполненными cells. Изменение основного
leaderboard и его весов — отдельное решение после этого выпуска.

## 10. Предложения по улучшению после обсуждения

Уточнение пользователя: scoring и процесс вызова можно улучшать относительно
upstream при обосновании и прозрачном описании. Сохранение исходных дефектов
не является требованием интеграции. Ниже — рекомендуемые решения и эксперименты;
новые метрики ещё не реализованы и не валидированы. По последнему уточнению
способ предъявления tools сохраняется текстовым, а Base/few-shot входит в
основной scope. Ранее предложенный native-tool эксперимент исключён из плана.

### 10.1. Исправления для первой поставки

Предлагаемый id исправленной версии: `legalbench_ru_corrected_v1`.
Upstream scorer остаётся reference; запускать оба на одном raw output.

| Изменение | Основание | Как доказать эффект |
|---|---|---|
| JSON-aware parsing вместо ручного подсчёта скобок | Валидные строки со скобками сейчас теряются | Fixtures с quoting/escaping/nesting; восстановленные outputs пометить parser repair |
| Проверка типов tool/args | Некорректный ответ может обрушить оценку | Число/list вместо tool, args не object → format error и 0 |
| Сравнение полного server.tool | В каталоге реально есть `search_cases` у 3 серверов, `get_case` у 2, `get_corpus_stats` у 3 | Неверный сервер при одинаковом function name → routing score 0 |
| Явный отказ вместо поиска слова null по всему output | Ненулевой tool с null в аргументе получает балл за отказ | Только корректный верхнеуровневый tool=null означает отказ в текстовом JSON-протоколе |
| Числовые границы в extraction | Gold 10 совпадает с 210 | Отдельный matcher чисел и единиц; отрицательные fixtures 10/210, знак, дробь, проценты |
| Корректная привязка статьи к акту | Несколько актов в строке разбираются неверно | Fixtures «акт → статья», «статья → акт», списки/разделители; неоднозначность не скрывать |

Для corrected JSON parsing принимать ровно один объект, при необходимости
обёрнутый одним Markdown code fence; произвольный текст до/после отмечать
отдельным нарушением формата. Два объекта, duplicate keys, NaN/Infinity —
invalid, без выбора «удобного» значения. Если понадобится извлечение объекта
из свободного текста для совместимости, хранить это как отдельную политику.

Числовой matcher не должен автоматически превращать все ответы в числа:
сначала классификация формы gold/accept, затем точное сопоставление с
допустимыми нормализованными значениями. ИНН, номера дел и иные идентификаторы
остаются строками; ведущие нули сохраняются. Единицы, интервалы и русские
числительные требуют явных правил и тестов. Неоднозначные случаи отправляются
в data review; расширять accept после просмотра ошибок конкретной модели нельзя.

Не использовать LLM judge для этих исправлений. Synthetic fixtures и
доказуемые контрпримеры дают воспроизводимый контракт без новой judge-модели.

### 10.2. Сделать tool score интерпретируемым

Сохранять reference `0.6 + 0.4 × argument_hit_rate`, но дополнительно показывать:

- `routing_accuracy`: выбран правильный полный tool или корректный отказ;
- `gold_call_exact`: совпал tool и нормализованный объект gold args целиком;
- `gold_args_recall`: доля совпавших размеченных аргументов при верном tool;
- долю unannotated args, format errors и неизвестных инструментов;
- отдельно positive routing и negative refusal accuracy, а также их mean.

Balanced positive/negative score полезен при 17 отрицательных случаях из 187:
общая accuracy может скрывать систематическое отсутствие отказов. У этих
метрик свои denominators: условный args score показывать вместе с routing
coverage, а совместную call accuracy считать по всем примерам.

`gold_call_exact` измеряет совпадение с аннотацией. Дополнительный легальный
optional аргумент может приводить к корректному вызову, отличному от gold.
Поэтому перед назначением этой метрики primary проверить полноту gold,
добавить утверждённые альтернативные calls там, где они нужны, и отделить
annotation exactness от schema validity. Не называть любой дополнительный
ключ ошибкой схемы. Выполнение инструментов в этот этап не входит.

Для аргументов задать нормализацию по schema/type/field: порядок object keys
не важен, порядок arrays по умолчанию важен, строки не превращаются поголовно
в lowercase, null отличается от отсутствующего аргумента. Даты, названия,
идентификаторы и свободный query имеют разные допустимые преобразования.
У query может быть несколько смыслово допустимых формулировок; без отдельной
разметки это остаётся метрикой совпадения с gold, а не качества поиска.

Для binary/MC в corrected score принимать однозначный ответ и отмечать
несколько конфликтующих ответов как ambiguous. Изменение инструкции на
«ровно одна метка» делать отдельной prompt version. Для citation сохранить
set F1 на уровне «акт, статья» и добавить exact-set diagnostic; переход к
частям/пунктам требует более подробного gold. Общий corrected mean считать
без округления per-sample, округляя только отображение; reference сохраняет
исходный порядок округления. Это отдельная строка change ledger.

### 10.3. Исходный текстовый prompt и допустимые расширения

Основной invocation protocol — `text_catalog_v1`. Полный каталог остаётся
в user message, модель генерирует JSON как обычный текст, scorer разбирает
answer continuation. OpenAI `tools`/`tool_choice`, native tool_calls и
принудительное structured decoding в эту интеграцию не входят.

Сначала сохранить исходные строки `build_prompt`, включая порядок каталога,
названия инструментов, descriptions, context и options. Few-shot добавляет
перед вопросом полные пары user/assistant; оригинальный prompt каждого примера
остаётся прежним. Для tool-call каталог повторяется и в demonstrations, и
в текущем вопросе. Не переносить его в system message и не убирать повторения
скрыто ради экономии контекста.

Допустимое расширение — устранение доказанной ошибки/неясности инструкции
или проверенное уточнение текстового описания аргумента. Каждое оформляется
как отдельная prompt/catalog version с исходным/новым текстом, обоснованием,
fixtures и paired comparison при фиксированных demo/eval ids и scorer.
Не менять формулировки только потому, что определённая модель отвечает лучше.
Общие подсказки про формат допускаются при зафиксированном эксперименте,
но не добавляются по умолчанию поверх исправного upstream prompt.

Текущий каталог содержит только `server`, `tool`, `params`, `desc`: нет типов,
required/optional, enum/defaults. Для расширения scorers или текстового
описания брать эти сведения из проверенных определений инструментов;
не выводить их из единственного gold и не объявлять все params обязательными.
Аудит schemas не блокирует запуск исходного текстового benchmark.
Полный каталог одинаков для всех evaluated rows; не отбирать инструменты
по gold. Raw textual output и parse status сохраняются для повторного скоринга.

### 10.4. Прозрачность изменений и порядок внедрения

Manifest различает независимые оси: `dataset_revision`, `annotation_version`,
`catalog_version`, `prompt_version`, `invocation_protocol`,
`split_manifest_version`, `demonstration_policy_version`, `few_shot_count`,
`scorer_version`, `aggregation_version`. Изменение одной оси не маскировать
общим названием «улучшенный LegalBench». Каждая влияет на fingerprint.

Change ledger содержит для каждого изменения: прежнее/новое правило, причину,
минимальный контрпример, затронутые типы/ids, fixtures, дату/версию и результаты
replay. В отчёте сохраняются обе оценки одного output и число случаев, где
балл вырос/снизился; при сравнении invocation вариантов — ещё format error
rate, coverage, tokens/latency и backend-specific settings.

Разделить эксперименты: сначала scorer A/B на одинаковых outputs без нового
inference; затем prompt/invocation A/B при фиксированном corrected scorer,
выборке, модели, budgets и thinking mode. Методика и synthetic fixtures
фиксируются до сравнения моделей; собственный evaluation set не используют для настройки
парсеров под конкретный leaderboard. Цена запросов и число попыток видны:
автоматический ответный repair/retry после неверного JSON считается отдельным
многошаговым протоколом, а не первой попыткой модели.

Рекомендуемый порядок: **freeze demo/eval → исходные prompts с 0/1/5-shot →
исправленные парсеры и двойной scoring → Base/Instruct validation →
детальные tool metrics и обоснованные текстовые расширения**.
Основной experimental score целесообразно перевести на corrected после
проверок L1/L4; reference score оставить рядом для сопоставления. Новые веса
tool-метрик и смена общего primary score фиксируются отдельным решением после
аудита gold, а не подбираются по лучшему результату выбранной модели.
