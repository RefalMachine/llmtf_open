# Аудит task/eval и интеграции RuBookSum / RuWikiBench

Дата: 2026-09-23. Статус: аудит кода и проектных решений, **не runtime validation**.

## Вывод

Интеграция целесообразна, но исходный план пока недостаточно точен для
реализации воспроизводимого benchmark. Основные проблемы — конфликт двух
execution API, неполное описание протоколов и завышенные ожидания от
вероятностей API. Копирование внешних циклов под новым `PipelineTask` сохранит
значительную часть этих проблем.

Рекомендуется одно ядро вызовов и artifacts, небольшой контракт управляемой
многошаговой задачи и последовательные работающие срезы. Первым — RuBookSum
hierarchical, вторым — Wiki ranking как проверка смешанного generate/probability
исполнения. Массовая миграция встроенных задач и coding environment не должны
быть условием выпуска этих срезов. Поддержка multi-turn остаётся отдельной
целью, а не объявляется достигнутой благодаря суммаризации.

Результат аудита задаёт два согласованных проекта:

- [Task/eval v2](TASK_EVAL_GRAPH_REFACTOR_PLAN_v2.md) — общее ядро и границы миграции;
- [Интеграция v2](EXTERNAL_BENCHMARK_INTEGRATION_PLAN_v2.md) — конкретные протоколы,
  этапы и условия выпуска семи задач.

Уточнение после обсуждения: пользователь разрешает исправлять внешние
benchmarks при сохранении их подхода и воспроизводимости прошлых экспериментов.
Запрет на изменение сабмодулей не является целевой политикой. Рекомендуются
отдельные проверенные commits исправлений и сохранённый historical runner.
Наблюдения ниже описывают исходные snapshots; не всякое необычное поведение
является доказанным дефектом. В частности singleton passthrough требует проверки
намерения метода. Изменения ranking/aggregation/sampling из плана являются
кандидатами нового протокола и не входят автоматически в совместимый перенос.

## 1. Область и доказательства

Изучены `AGENTS.md`, `BACKLOG.md`, task graph plan, integration plan,
task bugfix plan/report/test plan, RuParam fix/data audit, связь с dataset
mirror plan, архитектурная документация и релевантные части исторических
model-refactor plans. Проверены текущие Task/Evaluator/LLM/budget/provenance/
logger/registry/reporting, конфигурация runners и исходники обоих сабмодулей.

Точка отсчёта:

| Компонент | Ревизия |
|---|---|
| LLMTF HEAD | `d36543888b6cc3865cf3a584b5c1bda0b0455567` |
| RuBookSum gitlink и checkout | `a997d2154b343bf01d53ce7d7bf1fa2777fc6070` |
| RuWikiBench gitlink и checkout | `5dbf8d95c2cced4583943639f9a59e3e22a259d4` |

До аудита рабочее дерево уже содержало правку `dev/README.md`, незакоммиченный
integration plan и `results_foundational/`. Они не использовались как новые
runtime evidence и не удалялись. Сабмодули не изменялись.

Выполненные проверки:

- `python3 tests/test_refactor_logic.py`: 40 PASS, 0 failures;
- AST-разбор всех 19 `.py` двух сабмодулей: один SyntaxError в
  `rubooksum/utils.py:84`, незакрытая структура начинается на строке 72;
- из исходников через AST извлечены отдельные методы, без импорта внешних
  пакетов и без запросов к модели: singleton Wiki merge возвращает вход
  `raw source`, число вызовов summarize — 0; Blueprint `merge_pair` в cluster
  mode передаёт список в `summarize_with_blueprint`; метрики Wiki для
  retrieved relevance `[1, 0]` дают NDCG=1 и R-Precision=1 независимо от
  релевантных документов вне пула;
- `pytest` в активном host Python отсутствует;
- repository `compileall` по команде из AGENTS.md и `git diff --check` прошли;
- проверены локальные Markdown-ссылки, парность code fences и trailing
  whitespace в семи затронутых документах: ошибок не обнаружено.

Изолированные проверки методов воспроизводятся извлечением нужного `FunctionDef`
из `ClassDef` через `ast.parse`, компиляцией только этого узла и подстановкой
fake callbacks. Проверялись `WikiGen.hierarchical_merge(['raw source'], fake)`,
`Blueprint.merge_pair('one two', 'three four', word_limit=1,
blueprint=['question', 'question'])` при `mode='cluster'` и
`WikiEvaluater.ndcg/r_precision([1, 0])`. Эти probes подтверждают конкретные
ветви кода, но не заменяют импорт всего upstream package или реальный run.

Полные datasets не скачивались, GPU/backend matrix не выполнялась.
Число 634 подтверждено **описанием** закреплённой
[карточки RuBookSum](https://huggingface.co/datasets/NejimakiTori/literature_sum/blob/972f0646a65a63cb513a2eea276766471e98b9f4/README.md),
а не пересчётом строк. Wiki snapshot и 100 статей взяты из предыдущего плана:
повторное открытие его закреплённой карточки через web-инструмент не удалось.
Новый план поэтому требует отдельного data inventory, а не объявляет эти
числа заново проверенными. Выводы о коде ниже относятся к указанным gitlinks.

## 2. Что подтверждается в текущем LLMTF

| Наблюдение | Основание | Следствие |
|---|---|---|
| Основной dispatch атомарный | `evaluator.py`: `evaluate`, `evaluate_dataset`, whitelist методов | Многошаговый протокол требует нового task execution contract |
| PPL имеет отдельный lifecycle | `evaluate_ppl`, `evaluate_dataset_ppl` | Есть реальное дублирование; его устранение не требует одновременной миграции всех задач |
| Конфиги временно мутируются | `utils.py:MaxLenContext`, task stops в evaluator | Нельзя параллельно исполнять разные configs на одном LLM без изоляции |
| Judge inference находится в aggregation | `tasks/rag/rusbeir_rag.py:llm_judge_accuracy_agg` | Offline reducer должен быть отдельным от ресурсного scoring |
| Digest охватывает модуль класса | `provenance.py:_task_implementation_identity` | Helpers/prompts/preparation/encoders требуют явной dependency identity |
| Cache проверяет total fingerprint | `provenance.py:validate_cache` | Нет проверки целостности полного набора artifacts и нет sample resume |
| Total пишется раньше aggregation details | `evaluator.py:evaluate_dataset` | Нужен completion manifest и публикация total последним |
| Logger открывает samples в `w` | `sample_logger.py:JsonArrayLogger` | JSON array не является журналом checkpoints |
| Registry импортирует модули заранее | `tasks/__init__.py` | Нельзя добавить безусловный SentenceTransformer import в общий путь |
| Reporting создаёт задачи | `show_results.py:extract_task_datas` | Отделение normal report от registry необходимо для optional benchmarks |
| `all` перечисляет весь registry | `evaluator.py:evaluate` | Регистрация дорогих optional tasks требует явной политики default selection |
| Directory report усредняет все totals | `evaluator.py:create_report` | Отсутствия в categories недостаточно: experimental outputs нужно отделить |
| Строгий YAML не знает новых resource/run полей | `benchmark/config.py` | Их propagation во все runners должен быть отдельным этапом |

Завершённые task bugfixes не требуется повторно проектировать. Например,
RuParam уже исправляет неуникальные ids и double-order scoring, а API уже
сохраняет candidate coverage. Аудит не отменяет прошлые runtime reports и
не распространяет их на новые workflow tasks.

## 3. Противоречия и гипотезы планов

| Гипотеза/решение | Оценка | Решение v2 |
|---|---|---|
| Task получает `model` в `execute_batch` | Противоречит graph plan: вызовы скрыты от executor | Task выдаёт requests и принимает keyed responses |
| Статического DAG достаточно | Blueprint ветвится по числу вопросов и длине outputs; фильтрация меняет дерево | Ограниченный controller с сериализуемым состоянием и fan-out |
| Episode с одним Await сохраняет batching | Между книгами — да; внутри одной длинной книги — нет | `AwaitMany` с barrier и независимыми requests, один dispatcher |
| Сначала полный graph/coding/registry refactor | Слишком большая зависимость для двух benchmarks | Интеграционный выпуск и полный task refactor имеют разные gates |
| Resume можно отложить | Противоречит integration plan и стоимости длинных samples | В первом benchmark-выпуске — checkpoints завершённых raw samples |
| `batch_size` решает всю нагрузку | Не ограничивает очередь, CPU encoder, число активных книг и GPU residency | Разделить transport batch и deployment resource limits |
| Имя primitive гарантирует точность probability | API возвращает censored lower bounds | Проверять представимость и достаточность coverage каждого вызова |
| Семь registrations означают готовую интеграцию | Не проверяет CLI, reporting, denominator, offline replay | Отдельные acceptance gates для каждого потребителя |

Идеи graph plan о keyed identities, чистой aggregation, явных ресурсах,
централизованных reasoning/continuation и atomic completion сохраняются.
Публичный graph DSL, tool protocol и полноценный environment scheduler для
этих benchmarks пока не обоснованы.

## 4. Аудит RuBookSum

Источники: [methods](../external_benchmarks/rubooksum/src/methods.py),
[hierarchical](../external_benchmarks/rubooksum/src/hierarchical.py),
[blueprint](../external_benchmarks/rubooksum/src/blueprint.py),
[utils](../external_benchmarks/rubooksum/utils.py),
[metrics](../external_benchmarks/rubooksum/metrics.py).

1. **Snapshot не импортируется.** SyntaxError в клиенте подтверждён AST.
   Поэтому parity с этим commit требует отдельно описанных repair patches.
   Просто назвать будущий режим `reference-2026-03` недостаточно: дата не
   задаёт исполнимую implementation identity.
2. **Chunking описан неполно.** После tokenizer windows 2000/200 decoded text
   дополнительно проверяется по числу символов; при `len(chunk)>chunk_size`
   срезается до последнего пробела. Это не token-budget проверка и не
   корректный лимит 2000 символов. Удаление этого шага меняет входы модели.
3. **Hierarchical — не обычное попарное дерево.** Тройки объединяются; для
   шести элементов вторая тройка суммаризуется с результатом первой как
   контекстом. Хвост из одного/двух элементов переносится, финальные два
   объединяются. Упрощение до binary merge изменит algorithm baseline.
4. **Filtered затрагивает и исходные chunks.** Threshold 0.85 применяется до
   chunk generation и после уровней. Каждый элемент сравнивается со всеми
   предшествующими, включая уже исключённые; это отличается от сравнения
   только с retained set.
5. **Blueprint имеет динамический fan-out.** Каждая непустая строка output
   становится вопросом; число ответов заранее неизвестно. На merge могут
   заново генерироваться вопросы и ответы. Верхняя граница стоимости в
   исходном плане отсутствует.
6. **Cluster blueprint содержит дефект формы данных.** `generate_blueprint`
   возвращает `list[str]`, который передаётся в merge как общий blueprint и
   затем форматируется в prompt. Есть лишнее пересоздание blueprint после
   последнего merge, а KMeans запрашивает минимум 2 кластера даже для 1 вопроса.
7. **Выборка — first eligible, не случайная.** `text` соединяется через newline,
   применяется `cap_chars`, затем счётчик книг. Ошибочные книги занимают лимит,
   но исчезают из metric denominator. Фильтр 80 000 символов существенно
   определяет измеряемый срез; долю исключённых ещё надо измерить.
8. **Метрика — sentence embedding PRF.** ROUGE использует пакет `rouge`;
   имеющийся stemming helper не вызывается в `rouge_L`. Добавить stemming при
   переносе означало бы изменить метрику. Пустые sentence lists не обработаны.

## 5. Аудит RuWikiBench

Источники: [agent](../external_benchmarks/ruwikibench/src/wiki_agent.py),
[generation](../external_benchmarks/ruwikibench/src/wiki_gen.py),
[evaluation](../external_benchmarks/ruwikibench/src/wiki_evaluater.py),
[preparation](../external_benchmarks/ruwikibench/src/wiki_utils.py),
[HTML extraction](../external_benchmarks/ruwikibench/src/wiki_extract.py),
[runner](../external_benchmarks/ruwikibench/src/wiki_bench.py).

1. **Три независимых задачи, не end-to-end агент.** Outline получает sources
   нужной статьи, sections — reference mapping; ranking output не используется.
2. **Sections использует gold text до generation.** `filter_snippets` кодирует
   `page.filtered_outline[section]`, то есть эталонный текст секции, и отбирает
   snippets по threshold 0.6. Это oracle-assisted generation, а не только
   предоставление модели названия секции. Удалять эту подсказку без нового
   протокола тоже нельзя.
3. **Outline hint — не просто embeddings заголовков.** По reference positions
   выбирается embedding первого доступного source для блока заголовка;
   эти embeddings задают центры/число кластеров. Название метода
   `get_header_embeddings` не описывает это точно.
4. **Ranking зависит от полного корпуса.** `top_k=3 * snippets(article)`;
   комментарий про отношение 1:2 не гарантирует фактическое число positives.
   `max_sample_per_dataset=1` не должен превращать retrieval corpus в одну статью.
5. **Relevance — принадлежность статье.** Общий источник двух статей может
   оказаться формально negative. NDCG ideal и R считаются внутри retrieved pool;
   это не full-corpus retrieval recall. Требуются точные названия метрик.
6. **Ranking score нестандартен.** Максимумы YES/NO берутся по top-logprobs
   нескольких сгенерированных позиций, затем применяется `1-p(NO)`, если
   NO больше YES, иначе `p(YES)`. Перейти на один next-token primitive —
   содержательная смена scoring, даже при сохранении букв YES/NO.
7. **Текущий API LLMTF не устраняет censoring.** В
   [APIBackend](../llmtf/backends/api.py) `candidate_ranking_resolved` означает
   наличие хотя бы одного candidate surface. Отсутствующий кандидат получает
   0.0 lower bound. Этого недостаточно для отношения YES/NO и ранжирования
   scores между snippets. `supports_logprobs=True` не доказывает полноту.
8. **Singleton section может не вызвать модель вообще.** `hierarchical_merge`
   возвращает единственный input. Это происходит и в group summary, и в
   section generation. Новый протокол должен явно решить, допускается ли
   extractive passthrough в задаче генерации.
9. **Неполные sections исчезают из оценки.** Пустой ответ и sentinel `-1`
   пропускаются; bootstrap flatten-ит все оставшиеся секции, а не статьи.
   Крупные статьи получают больший вес, dependence секций не учитывается.
10. **Порядок и identity ненадёжны.** `rglob` не сортируется; используются
    original и sanitized titles, positional source numbering, dict order
    embeddings и snippets. Нужен keyed join, а не перенос файловой структуры.
11. **Нормализацию encoder нельзя предполагать.** Cached snippets кодируются
    без явного `normalize_embeddings=True`, затем dot product трактуется как
    cosine; модель может нормализовать внутри себя. Это риск для проверки
    конкретного encoder snapshot, а не доказанный численный дефект BERTA.
12. **Реальные мелкие наборы ломают reference path.** Sections использует
    неинициализированный `result` при числе успешных статей <2; KMeans может
    запросить больше кластеров, чем samples. `openai_utils` содержит helper
    с `re` без импорта; это не доказательство, что helper вызывается каждым run.

## 6. Данные, зависимости и публикация

В деревьях обоих закреплённых commits нет отдельного `LICENSE`. Карточка
RuBookSum объявляет MIT и указывает стороннее происхождение книг и аннотаций.
Это наблюдение о metadata, а не проверка прав на каждый текст или код.
Условия Wiki, переноса prompts и публичного распространения prepared caches
остаются не подтверждены этим аудитом. Требуется зафиксировать основания для
конкретных распространяемых материалов до их включения; разработка общего
ядра и синтетических fixtures от этого не зависит.

Не следует автоматически включать данные в план публичного HF-зеркала.
Существующий `DATASET_MIRROR_PLAN.md` уже отделяет внешние benchmarks этим gate.
Локальный download/cache, публикация данных, публикация outputs и перенос кода
должны рассматриваться отдельно. Наличие сабмодуля не заменяет эту инвентаризацию.

Dependency list исходного плана неполон: Wiki preparation реально использует
`pymorphy3`, dictionaries и NLTK stopwords; clustering — scikit-learn;
HTML parser и версия parser backend влияют на sections. Некоторые пакеты уже
есть в common profile. Нужен diff зависимостей по import graph, а не копия
внешних requirements. Encoder должен быть ресурсом с явным устройством:
неявная загрузка на ту же GPU, где vLLM занимает почти всю память, неприемлема.

## 7. Приоритеты

**До интеграционного кода:** разрешить конфликт executor API; составить
data/protocol inventory; определить oracle semantics, candidate coverage,
denominator и выбранные исправления reference поведения.

**До первого публичного benchmark:** общий invocation path; изоляция configs;
lazy optional imports; checkpoint и atomic completion; метрики offline;
CLI/YAML/reporting; реальные проверки поддерживаемых backend configurations.

**После первого среза:** расширить варианты; подтвердить стоимость и batching;
завершить общий task refactor, coding и judge workstreams по собственным gates.
Совместимость с опубликованными baseline требует отдельного parity report.
Ни новый protocol label, ни успешная компиляция не заменяют такое сравнение.
