# План переработки task/eval: единое исполнение обычных и multi-turn задач

Статус: проект реализации; описанные ниже новые API ещё не реализованы.
Дата: 2026-09-21. Полностью переработан после архитектурного аудита.

Документ задаёт целевое решение, границы первого релиза и порядок реализации.
Текущий код, benchmark-протоколы и инварианты [AGENTS.md](../AGENTS.md)
сохраняют силу до миграции. Изменения модели, данных, prompts и метрик не
следует незаметно включать в архитектурный рефакторинг.

## 1. Мотивация и ожидаемый результат

Главная цель — расширить LLMTF задачами с multi-turn исполнением: agentic,
coding, работой с инструментами и окружением. Например, модель получает
задание, предлагает исправление кода, запускает доступные тесты, видит
результат, исправляет решение и завершает работу. Число обращений к модели
и следующие действия зависят от предыдущих результатов.

Нужно обслуживать такие задачи теми же механизмами, что и обычные benchmarks:
модельными backend-ами, batching, budgets, logging, provenance, scoring и
reporting. Добавление нового протокола не должно требовать отдельного evaluator.

У текущего task/eval слоя есть и непосредственные ограничения:

- `load_dataset()` смешивает загрузку, выборку, few-shot, построение prompts
  и проверку бюджета; возвращает два позиционно связанных списка;
- `Evaluator` дублирует жизненный цикл для PPL;
- некоторые LLM-метрики вызывают модель внутри aggregation ради batching;
- конфигурации answer/reasoning/stops временно меняются на общем model instance;
- reporting создаёт task objects, а bootstrap не задаёт единицу выборки;
- registry загружает задачи заранее и является глобальным mutable объектом.

Решение должно удовлетворять трём требованиям:

1. **Простота:** обычная задача описывается загрузкой, запросом и scoring;
   автор не пишет scheduler и не управляет ресурсами.
2. **Прозрачность:** видны все модельные вызовы, действия, причины остановки,
   параметры, расходы и данные, из которых получилась метрика.
3. **Функциональность:** single-turn, grouped tasks, PPL, LLM judge и bounded
   multi-turn работают через одно ядро исполнения.

Типизированный graph нужен для явных зависимостей между этапами. Сам graph
не решает динамическое исполнение: для него в первом цикле вводится
`Episode` с состоянием и ограничениями. Совместимость с multi-turn проверяется
до массовой миграции существующих задач.

## 2. Основное решение и границы

Внешний контракт:

```text
Task + RunConfig → EvaluationPlan → Evaluator.execute() → Summary + artifacts
```

- **Task** определяет данные, протокол взаимодействия, scoring и aggregation.
- **EvaluationPlan** описывает этапы и их зависимости.
- **Evaluator** исполняет план и владеет ресурсами, batching и artifacts.
- **LLM** сохраняет one-pass/two-pass reasoning orchestration.
- **Backend** исполняет существующие модельные примитивы.
- **Environment adapter** создаёт и обслуживает окружение эпизода.
- **Metric adapter** обслуживает дополнительные вычислительные метрики,
  которым нужен собственный ресурс, например encoder для BERTScore.

Top-level план — DAG. Динамический цикл находится внутри `Episode`; произвольные
обратные рёбра и вложенные эпизоды в первом релизе запрещены. Есть один executor
и один механизм вызова ресурсов для обычных узлов и шагов эпизода.

В первый законченный релиз входят:

- текущие операции generate, token probability и logsoftmax/PPL;
- one-to-many подготовка и grouped scoring;
- явная LLM-as-a-Judge оценка, включая второй модельный ресурс;
- ограниченный multi-turn протокол и одна реальная coding-задача;
- единые aggregation, artifacts, provenance и offline reporting;
- миграция built-in tasks и минимальный публичный task API.

Отложены: distributed execution, общий workflow DSL, автоматический node cache,
resume живого окружения после перезапуска, произвольные process pools,
потоковая обработка всех стадий, автоматическая установка plugins и поддержка
всех вариантов native tool calling. У реального coding-среза допускается
текстовый протокол действий; это явно фиксируется в task protocol.

## 3. Где находятся scoring, judge и aggregation

**LLM-as-a-Judge по смыслу является scoring. При исполнении это явный вызов
модельного ресурса. Aggregation получает уже вычисленные оценки.**

Смысл операции и способ её физического исполнения — разные вещи.
Вычисление per-case метрик можно делать batch-ами без переноса в aggregation.

| Операция | Смысл | Представление в плане |
|---|---|---|
| Exact match, parsing, локальная метрика | Scoring | Обычный Transform |
| BERTScore или другая ресурсная метрика | Scoring | Invoke metric resource |
| LLM-as-a-Judge | Scoring | Build request → Invoke judge → Parse score |
| Проверка решения тестами | Scoring или feedback внутри episode | Invoke environment |
| Среднее, F1 по общим counts, Elo, срезы | Aggregation | Reduce готовых records |

BERTScore здесь — пример будущего adapter, а не заявление о наличии такой
интеграции в текущем репозитории.

Отдельный primitive `JudgeNode` не нужен. Helper `LLMJudgeScorer` строит
обычные Transform/Invoke узлы. Аналогично ресурсная метрика может иметь
`score_batch()`, который вызывает executor через metric adapter.

Пример RAG с двумя метриками:

```text
BuildRequest → Invoke(primary) ─┬→ ROUGE ──────────────────────┐
                              └→ BuildJudgeRequest           │
                                   → Invoke(judge)           │
                                   → ParseJudgeScore ────────┤
                                                           Merge by case_id
                                                                → Aggregate
```

Executor собирает готовые judge requests в совместимые batches. Размер batch
не задаётся внутри task. Judge имеет собственные sampling, context budget,
capabilities, model identity, время и usage; они сохраняются в provenance.

Для pairwise judging сначала формируются comparison cases с идентификаторами
обоих ответов и порядка предъявления. Затем judge оценивает сравнения, а
reducer рассчитывает итоговый рейтинг. Dataset-wide scoring также можно
выполнить явным этапом; потребность во всём наборе не превращает inference
в aggregation.

Обычная `score(case, response)` — вычисление без скрытых model/API calls.
Scorer, которому нужен внешний ресурс, объявляет это при построении плана.
`aggregate(records)` не обращается к моделям, сети и окружениям. Его достаточно
для повторного offline расчёта по сохранённым scoring records.

Текущий `RusbeirRagLLMJudge.llm_judge_accuracy_agg()` переносится именно таким
образом: prompts, parser и правило оценки сначала сохраняются. Для
`LLMAsJudgeStyleControl`, где оцениваемая модель сама играет роль судьи,
не нужно искусственно создавать дополнительный ресурс: binding задаёт task.

## 4. Минимальное графовое ядро

Не следует превращать каждый helper или каждое сообщение в узел. Достаточно
пяти видов узлов:

| Kind | Вход и выход | Назначение |
|---|---|---|
| Source | Нет входа → подготовленные cases | Загрузка и task-defined preparation |
| Transform | Records → records | Request building, parsing, scoring, map/flat-map/filter |
| Invoke | Requests → responses | Вызов именованного model/metric/environment resource |
| Episode | Cases → episode results | Ограниченная динамическая последовательность действий |
| Reduce | Records → group records или task result | Grouped scoring и final aggregation |

Подготовка может быть вынесена из Source в Transform, если промежуточный
результат нужен нескольким стадиям. Это не обязательное усложнение простой задачи.

`EvaluationPlan` содержит task identity, список node descriptors, resource
requirements и ссылку на единственный terminal `AggregationResult`.
Node descriptor содержит:

- уникальный `node_id` и один из пяти kinds;
- именованные inputs: ссылки `(producer_id, output_port)`;
- типы и форму output ports: коллекция records либо task-level value;
- стабильный `handler_id` и JSON-config обработчика;
- для Invoke — имя resource; для Episode — protocol и limits;
- для Transform — cardinality `map | filter | flat_map`;
- для Reduce — grouping key либо task-wide reduction.

Реализации handlers — обычные Python функции/объекты из task package.
Сериализуется descriptor и identity реализации, а не closure или Python source.
Публичный builder для произвольных графов не нужен в первом релизе; templates
используют этот внутренний контракт.

Правила исполнения и валидации:

1. До начала run проверяются уникальность ids, существование ports, отсутствие
   циклов, совместимость типов и объявленные resources.
2. Map сохраняет ключи; filter пишет причину исключения; flat-map создаёт
   уникальные дочерние ids и сохраняет родителя.
3. Transform с несколькими case-level inputs соединяет их по case_id.
   По умолчанию наборы ключей должны совпасть; позиционный zip запрещён.
   Иные join semantics требуют отдельного явно описанного transform.
4. Reduce группы получает все ожидаемые элементы; неполная группа является
   ошибкой, если протокол не определяет иное.
5. Начальное исполнение Source/Transform/Reduce последовательное,
   с материализованными коллекциями. Оптимизация streaming не меняет контракт.
6. Invoke batch-ит готовые requests. Episode использует тот же dispatcher.
   Ветви DAG первоначально могут выполняться последовательно.
7. Не вводятся неиспользуемые flags `deterministic`, `side_effect_free`,
   resource affinity и дополнительные scopes. Retry policy принадлежит
   конкретной операции ресурса, а не угадывается по общему boolean.

## 5. Данные, identity и task API

### 5.1 Cases и результаты

Начальный набор типов:

- `RawCase(raw_id, payload)` — исходная запись;
- `EvaluationCase(case_id, raw_id, group_id, payload, metadata)`;
- `PreparedCases(cases, selection_manifest, expected_groups)`;
- `ScoringRecord(case_id, values, status, details)`;
- `GroupRecord(group_id, member_ids, values, details)`;
- `AggregationResult(metrics, leaderboard_score, details, coverage)`.

Payload и scoring values могут содержать task-specific JSON-структуры:
например, NER counts или label/prediction pairs. Это не произвольные runtime
objects. Итоговые metrics также могут содержать именованные срезы, как в Libra;
`leaderboard_score` — отдельное конечное число. NaN/Infinity не сохраняются
как успешный leaderboard result.

Identity имеет три уровня:

- `raw_id`: dataset snapshot/config/split + проверенный source key либо
  исходная позиция до selection/sort/expansion;
- `case_id`: raw identity + стабильное имя варианта, например orientation;
- `invocation_id`: case_id + node/stage + номер шага и действия внутри episode.

`episode_id` в первом релизе совпадает с case_id: один episode на case.
Повтор ресурса сохраняет invocation_id, но получает новый `attempt`.
Порядок завершения никогда не используется для назначения ids.
Неуникальные upstream ids не считаются достаточной identity.

`expected_groups` задаёт точный состав групп после preparation.
Отдельно считаются selected raw rows, excluded rows, cases, groups и calls.
Для RuParam две ориентации — два cases и одна оцениваемая группа.

### 5.2 API обычной задачи

Схема API; аннотации окончательно закрепляются contract tests на этапе 1:

```python
class SingleTurnTask:
    spec: TaskSpec

    def load_data(self, context): ...
    def prepare_cases(self, data, context) -> PreparedCases: ...
    def build_request(self, case, context) -> ModelRequest: ...
    def score(self, case, response) -> ScoringRecord: ...
    def aggregate(self, records) -> AggregationResult: ...
```

Default preparation создаёт один case на raw row. Default aggregation/helper
может вычислять среднее; задача переопределяет только необходимые методы.
Template строит план автоматически. Дополнительный judge задаётся scorer
helper-ом, который добавляет явные узлы.

`TaskSpec` включает task_id, protocol_version, few-shot policy, request defaults,
resource requirements и reducer descriptor. Требования проверяют не только
operation, но и features: token IDs, reference tokenization, offsets,
assistant continuation, native tool calls.

`PreparationContext` предоставляет task-local RNG, эффективный prompt budget,
capabilities и узкие сервисы token counting/rendering/tokenization, когда они
доступны. Весь mutable model и credentials в него не передаются.
CopyText получает reference tokenization и leading-space metadata; API без
таких возможностей отклоняется до исполнения cases.

Few-shot helper сохраняет текущий выбор demonstrations, их порядок, исключение
из test split и сокращение по бюджету. Отсутствующий token counter остаётся
`None`: нельзя трактовать его как ноль или молча сокращать few-shot.
Переход на локальный RNG сначала воспроизводит прежнюю последовательность seed
555; изменение алгоритма выборки — отдельная protocol migration.

Порядок limit/filter/sort и распределение лимита по subjects принадлежат task
protocol. Например, текущий Libra сначала ограничивает выборку и затем
фильтрует длины. Универсальный helper не должен молча менять этот порядок.

## 6. Контракт модельного вызова

Используется discriminated union, а не один request с произвольными kwargs:

| Request | Существенные поля | Response |
|---|---|---|
| GenerateRequest | messages, invocation config, output options | text, optional token IDs, structured actions |
| TokenProbabilityRequest | messages, candidates, invocation config | candidate scores и coverage diagnostics |
| LogSoftmaxRequest | messages, score scope, optional answer boundary | token scores, offsets, сведения о rendering |

Task строит request с resource name и continuation options. Перед исполнением
executor добавляет invocation_id в request envelope и разрешает effective config.
Response envelope содержит тот же invocation_id, operation, usage, finish
reason, optional rendered prompt и backend diagnostics. Отсутствующий rendered
prompt у API не подменяется вымышленной строкой.

Generate output явно представляет одну или несколько sequences. Обычный
single-turn template первоначально требует одну; несколько допускаются только
при объявленном task policy. Two-pass reasoning по-прежнему поддерживает одну
sequence и отклоняет неподдерживаемый режим.

Probability response сохраняет candidate coverage: отсутствующий в API top-k
кандидат не превращается в доказанную нулевую вероятность. Историческое правило
объединения surface variants остаётся в `llmtf.continuation`.

`EffectiveInvocationConfig` включает sampling, stops, answer/reasoning budgets,
requested/effective thinking и влияющие на output options. Его строит общий
resolver из model defaults, task defaults, явных run overrides и budget policy.
Он возвращает независимый snapshot с immutable вложенными значениями.
Один `frozen=True` вокруг mutable `SamplingConfig` этого не обеспечивает.

Приоритеты сначала воспроизводят текущий CLI/programmatic контракт:
явные CLI значения перекрывают backend JSON, пропущенные — нет; полностью
переданный custom generation config сохраняет его существующий приоритет.
Task stops объединяются с model/template stops по текущему правилу.
Reasoning phase overlays по-прежнему формирует `LLM`.

Это отдельное изменение `LLM`/budget API: effective reasoning config передаётся
явно, без временной мутации общего `reasoning_config`. Backend-ам достаются
копии необходимых runtime config. Старые add/reset stops и MaxLenContext
сохраняются только в последовательном compatibility path до его удаления.

### PPL и обычный logsoftmax

Оба используют LogSoftmaxRequest и общий lifecycle. PPL — scorer по reference
answer, а не отдельный evaluator. Сохраняются exact rendered-prefix boundary,
правило включения пересекающего boundary токена и mean answer-token log
probability; это не exponentiated perplexity и не corpus token-weighted mean.
Treeway extractive сохраняет собственную область и grouped scoring.

Task request builder/адаптер выбирает PPL-режим до подготовки запросов.
Требование — HF logsoftmax; vLLM/API отклоняются до дорогой загрузки, если
несовместимость известна заранее.
Проверка context учитывает полный scored input, включая reference answer;
обрезка эталонного ответа ради размещения в context запрещена.

### Сообщения и действия

Для текущих text tasks сохраняются canonical system/user/assistant и
централизованная обработка assistant prefill. Episode observation не вводит
неподдерживаемую role автоматически.

Текстовый action protocol разбирает ответ в `EnvironmentRequest`; observation
включается в следующий обычный prompt согласно task protocol.
Native tool protocol использует отдельные типизированные tool-call/tool-result
сообщения и явную capability adapter-а. Их нельзя молча преобразовывать в
обычный текст с сохранением заявления о native tool support.

Backend response extraction должен сохранять visible content отдельно от
reasoning и tool calls. Поддержка native tools, включая сочетание с two-pass
reasoning, вводится только с проверенным request/response mapping;
неподдерживаемые сочетания завершаются понятной capability error.

## 7. Multi-turn: один ограниченный Episode

Episode — протокол переходов, не отдельный evaluator и не скрытый Python loop,
который напрямую вызывает модели или shell.

```python
class EpisodeProtocol:
    def start(self, case, context) -> Decision: ...
    def advance(self, state, observation, context) -> Decision: ...

Decision = Await | Finish
# Await(state, request): ModelRequest или EnvironmentRequest
# Finish(result): final answer/artifact refs + termination reason
```

В первом варианте у одного episode не более одного ожидаемого действия.
Несколько tool calls из одного ответа выполняются последовательно с
сохранением их ids. Разные episodes могут прогрессировать независимо.
Внешние handles принадлежат executor, а state — сериализуемые данные протокола.

Алгоритм:

1. Создать episode state через start().
2. Для Await проверить capabilities и оставшиеся limits; назначить invocation_id.
3. Собрать совместимые готовые model requests разных episodes в batch;
   environment requests передать соответствующему adapter.
4. Записать результат и usage; передать observation в advance().
5. Повторять до Finish, лимита или infrastructure failure.
6. При нормальном завершении или исчерпании лимита сохранить финальный snapshot,
   если он нужен scorer-у, затем EpisodeResult и траекторию; гарантировать cleanup.
7. Выполнить final scoring и aggregation тем же DAG executor.

Environment session создаётся лениво при первом environment request.
Adapter получает case-specific immutable setup; результат первого действия
может быть начальным наблюдением. Протокол не хранит container/process handles.
Snapshot переносится в artifact storage до cleanup и имеет content identity;
ссылка на временный workspace не является сохранённым результатом.

Отрицательный результат тестов или невалидное действие может быть обычным
observation, если это предусмотрено протоколом. Ошибка adapter-а, dataset или
инфраструктуры не маскируется под feedback модели.

EpisodeResult содержит termination reason, final output/artifact references,
usage totals, число turns/actions и ссылку на trajectory.
Scorer отдельно определяет корректность результата: завершённый эпизод может
получить ноль; остановленный по лимиту тоже может быть валидным исходом benchmark.

### Первый coding-протокол

Обязательный вертикальный срез:

1. Создать изолированный workspace из фиксированного маленького задания.
2. Модель возвращает structured text action: изменение разрешённого файла,
   запуск доступных тестов либо final.
3. Parser валидирует действие, executor передаёт его окружению.
4. Модель получает ограниченный по размеру feedback и может исправить решение.
5. После final/лимита проверяющий scorer запускает итоговую проверку на
   зафиксированном candidate snapshot и выдаёт test pass rate.
6. Сохраняются действия, patch/snapshot identity, результаты проверок и причина
   завершения. Рабочее окружение закрывается.

Тесты, используемые для финальной оценки, недоступны для изменения из agent
workspace. Если benchmark разделяет visible и hidden tests, hidden results
не попадают в observations. Scoring работает с финальным snapshot даже после
закрытия интерактивной сессии.

## 8. Ресурсы, batching и ограничения

Resources объявляются по именам: `primary`, `judge`, `metric_encoder`,
`coding_env`. Task задаёт требования; RunConfig связывает имена с adapters.
Resources создаются лениво, переиспользуются в пределах run и закрываются
в finally. Общий handle закрывается один раз; env sessions изолированы по episode.

Model resources используют LLM, metric resources — `score_batch` adapter,
environment resources — `open / execute / snapshot / close`.
`MetricRequest` содержит case_id, prediction, references и metric options;
`MetricResponse` — keyed scoring values и resource diagnostics.
`EnvironmentRequest` содержит session/episode identity, action name и
валидируемые JSON arguments. `EnvironmentResponse` содержит observation,
action outcome, artifact refs и diagnostics. Executor оборачивает оба вида
в тот же invocation envelope. Parser error и infrastructure error отличаются
от обычного action outcome, например ненулевого exit code тестов.
Allowed actions, schemas arguments и размер observations определяет adapter
совместно с task protocol; неизвестное действие не исполняется.

GPU allocation остаётся deployment/runner обязанностью. Если primary и judge
не помещаются одновременно, run config задаёт последовательную residency
либо внешний API resource; executor не обещает автоматический GPU scheduler.

Batch compatibility key включает resource identity, operation, effective
config и output/continuation options. Только параметры, которые adapter
явно поддерживает как per-item, могут различаться внутри batch.
Response alignment валидируется до маршрутизации по invocation_id.

Executor управляет очередью готовых вызовов и числом активных episodes.
APIBackend сохраняет transport retries/timeouts и внутреннюю request concurrency.
Новый уровень не повторяет весь batch поверх backend retries.
В первом релизе один LLM resource исполняет один batch за раз; API concurrency
внутри batch и batching между episodes сохраняются.

Ограничения объявляются раздельно:

- invocation: context и answer/reasoning budget, request timeout;
- episode: max_model_turns, max_actions, max_total_tokens, wall-time deadline;
- scoring: отдельные limits judge/final tests;
- run: общий deadline и при необходимости общий расход ресурсов.

Для Episode обязательны конечные turn/action limits и deadline. Token limit
можно не задавать; если он задан как строгий, его проверяемость обязательна.
Model turn — один логический LLM вызов; reasoning phases не увеличивают turn
count, но входят в token/time accounting. Любое Await расходует action budget,
чтобы env-only цикл также был ограничен. Общий run usage включает primary,
judge и resource attempts; расходы на solving и scoring видны отдельно.
У двухфазного LLM учитываются input/output обеих фаз, включая повторный input;
одного размера финального ответа недостаточно для total-token accounting.

Перед вызовом проверяется остаток, а ответ/reasoning ограничиваются остатком
по явной policy. Уменьшение ниже strict reasoning floor запрещено.
Для очередного model turn проверяется весь накопленный context. При его
переполнении применяется только объявленная task policy: остановка либо
версионированное сокращение истории; неявного удаления сообщений нет.
Если точный input/usage counter недоступен, это сохраняется как unknown.
Строгий total-token limit требует проверяемого upper bound/counter; при их
отсутствии такой режим отклоняется. Turn/time limits остаются доступны.
Deadline включает повторы; adapter должен уметь ограничить запрос или
остановить процесс. Cleanup имеет отдельный ограниченный timeout.

Coding environment adapter обязан изолировать candidate execution от host и
соседних episodes: отдельный workspace, ограниченные mounts, CPU/RAM/PID/time,
отключённая по умолчанию сеть, отсутствие host credentials и Docker socket
внутри candidate environment. Исполнение на host не является fallback.
Конкретный механизм изоляции и область доверия фиксируются до реального smoke;
сам факт запуска Docker не считается достаточным доказательством изоляции.
Filesystem effects разрешены только через environment contract, а не scorer.

## 9. Ошибки, retries и воспроизводимость

Три независимые характеристики:

| Уровень | Примеры |
|---|---|
| Execution status | success, failed, cached, skipped |
| Episode termination | final, budget_exhausted, invalid_action |
| Scoring status | valid, invalid_prediction |

Точные причины ошибок сохраняются структурированно с task/case/invocation ids.
Неверный ответ и infrastructure failure не объединяются.

По умолчанию infrastructure failure обязательного case/group/scorer завершает
task без успешного total. Invalid prediction получает task-defined оценку
и остаётся в denominator. Исключение строки допустимо только по объявленной
data policy, с причиной и coverage. Частичные infrastructure results могут
сохраняться для диагностики, но не публикуются как полноценный benchmark score.

При failed task executor прекращает выдачу новых действий, отменяет ожидающие
запросы по возможностям adapter-ов и закрывает все активные sessions. Ошибки
cleanup сохраняются отдельно и не стирают исходную причину сбоя.

Явно запрошенная неподдерживаемая задача/операция даёт non-zero exit code.
Cache hit — cached, а не failed. Optional metric можно пропустить лишь когда
она объявлена optional и не участвует в leaderboard; coverage показывает пропуск.

Environment action, меняющее файлы или внешнее состояние, автоматически не
повторяется. Timeout с неизвестным результатом действия — failed episode,
если adapter не имеет проверенного idempotency/reconciliation contract.
Для model/API retries сохраняются attempts; неопределённый usage не объявляется
нулевым. Judge infrastructure failure не трактуется как отрицательный verdict.

Первый релиз поддерживает offline replay scoring по сохранённым responses и
observations. Он не обещает восстановление живой сессии или повторение
side effects. Task-local RNG и stable ids исключают зависимость dataset
selection от порядка завершения запросов. Изменение batching само по себе
не считается гарантированно численно нейтральным для backend.

## 10. Aggregation, bootstrap и reporting

Reducer descriptor содержит id, version, JSON-config и bootstrap policy:
`disabled | case | group`. Он ссылается на чистую offline реализацию из
установленного task package. Произвольные closures не сериализуются.

- Case bootstrap выбирает независимые case records.
- Group bootstrap выбирает уже завершённые group records либо целые группы,
  сохраняя их multiplicity. Повтор группы — повтор bootstrap draw, а не
  duplicate case в исходной выборке.
- Для RuParam сначала вычисляется pair_correct по двум ориентациям, затем
  пересэмплируются пары. Для treeway единица — статья.
- Structured records и aggregation details обрабатываются единообразно;
  bootstrap не передаёт tuple `(value, details)` как числовую метрику.
- Scorer/judge не запускается повторно внутри bootstrap. Такая оценка
  неопределённости условна относительно уже полученных model/judge outputs.

Normal report читает сохранённые totals без импорта registry и создания tasks.
Пересчёт метрик/bootstrap загружает только нужный reducer. Если package
отсутствует или версия не совпадает, normal report работает, а пересчёт
завершается конкретной ошибкой, без model/environment calls.

Один reducer используется при initial aggregation и offline recomputation.
Dataset-level mean, category weighting и leaderboard semantics сохраняются.
Существующий directory-wide report не меняет молча состав: текущий run summary
и обзор всех completed totals различаются явно.

## 11. Artifacts и cache

Существующие имена основных файлов сохраняются:

- `<task>.jsonl` — JSON array итоговых case records;
- `<task>_params.jsonl`, `<task>_total.jsonl`;
- `<task>_aggregation_details.jsonl`, если нужны детали;
- `evaluation_results.txt`, `evaluation_log.txt`.

Case record сохраняет совместимые поля sample/predict/prompt/metric/info и
добавляет version, ids и statuses. `predict` остаётся newly generated
continuation без assistant prefill. Для episode в нём final output, а полная
история находится в отдельном artifact.

Новые sidecars:

- `<task>_plan.json`: versioned descriptor плана и resource bindings;
- `<task>_events.jsonl`: настоящий JSON Lines с invocation/action событиями;
- `<task>_groups.jsonl`: group records, когда они нужны для offline reduction.

Events содержат ids, порядковый номер внутри episode, request/response либо
artifact refs, usage, timing, attempt и error/termination reason.
Они пишутся и для single-turn, чтобы judge calls не исчезали в общей метрике.
Большие outputs ограничиваются по объявленной policy; обрезка наблюдения,
видимого модели, является частью task protocol и provenance.
Логи не содержат credentials, private source и host-private absolute paths.

Schema version есть у новых records/sidecars; run config schema повышается
от текущей v2. Старые artifacts читаются reporting-ом, но не используются как
совместимый execution cache новой схемы.

Fingerprint включает task/protocol, topology/config, handlers/reducer identity,
model/resource/runtime snapshots, effective invocation policy, dataset identity,
selection/few-shot policy, environment image/setup, limits и seed.
Влияющие на результат output options включаются явно. Batch size и engine
settings в первом релизе консервативно остаются в fingerprint.

Dependency identity: digest task module + явных helper/config dependencies,
а также version/content digest framework evaluation core. Для внешних tasks
нужны manifest/version и digest объявленных dependencies, без сохранения source.
Автоматическое определение всех Python imports не обещается.

Dataset identity — resolved revision/content manifest, включая локальные файлы.
Если source нельзя разрешить без loading, cache проверяется после loading.
Нельзя объявлять reusable cache только по плавающему dataset name или
`revision=main`. Сохраняется digest выбранных raw ids и demonstrations.
Для изменчивого API model alias неизвестная revision обозначается явно;
строгое повторное использование требует закреплённой deployment identity.

Запись результатов:

1. Проверить уникальность artifact names после нормализации task ids и
   исключить двух writers одной задачи.
2. Писать новую попытку во временные файлы, не выдавая её за completed.
3. При публикации force-recalc убрать прежний completion total из активного
   набора, заменить обязательные artifacts, удалить/заменить устаревшие sidecars.
4. Атомарно опубликовать новый total последним. Он содержит completed status,
   fingerprint и manifest обязательных artifacts с digest/counts.
5. Cache hit требует совпадения fingerprint и целостности этого manifest.
   Partial attempts остаются диагностикой и не входят в leaderboard.

Resume отдельных nodes/cases отсутствует. До финального total это одна
незавершённая попытка; event log полезен для анализа, но не означает resume.

## 12. Миграция и порядок реализации

Каждый этап должен иметь самостоятельный проверяемый результат.
Нельзя сначала мигрировать весь registry, а затем впервые проверить multi-turn.

### Этап 0. Зафиксировать поведение и рабочие сценарии

- Зафиксировать исходное состояние текущего working tree и sanitized fixtures;
  один HEAD недостаточен, если task fixes ещё не закоммичены.
- Описать expected prompts, selection, per-case scoring, reducers и totals для
  RuCoLA, Flores/Gazeta, CopyText, RuParam, treeway, PPL, NER, Libra и RAG judge.
- Добавить сценарий coding episode с ошибкой, feedback и исправлением.
- Зафиксировать условия реального environment adapter и видимость тестов.
- Инвентаризировать CLI, benchmark runners, examples, show_results,
  LLMAAJ scripts и prompt_optimizer как потребителей.

Выход: characterization suite и явно выбранный первый multi-turn протокол.
Исторический model-refactor baseline из v4 не заменяет baseline текущих tasks.

### Этап 1. Контракты, общий dispatcher и effective config

- Ввести case/invocation ids, request/response variants, statuses, reducer
  descriptor и artifact schemas из разделов 5–11.
- Реализовать model invocation adapter поверх LLM и общий envelope для
  model/metric/environment requests.
- Выделить budget/config resolver и убрать мутации из нового execution path.
- Определить static preflight до model loading и dynamic capability validation
  после loading/probes; статически неизвестное не считать поддержанным.
- Проверить alignment, output options, continuation, reasoning и failure mapping
  сначала fake resources, затем применимыми backend smokes.

Выход: один явный вызов модели со всем effective config и provenance;
запрос не меняет состояние следующего запроса.

### Этап 2. Минимальный DAG и Episode на fake resources

- Реализовать пять node kinds, graph validation и последовательный executor.
- Добавить queue совместимых model requests и bounded active episodes.
- Реализовать resource lifecycle, event sink и atomic completion.
- Выполнить generation, probability, grouped task, judge branch и coding loop
  на детерминированных adapters.
- Проверить разный порядок завершения, incomplete groups, malformed actions,
  budget exhaustion, timeout и cleanup.
- Показать короткие author-facing examples single-turn и episode.

Выход: multi-turn уже работает через то же ядро, без прямых вызовов модели
из task и без второго evaluator.

### Этап 3. Реальный вертикальный срез до стабилизации API

- Запустить одну обычную задачу и RuParam на новом пути.
- Выполнить реальный coding episode через изолированное окружение: модельный
  ответ, тестовый feedback, следующая итерация и итоговая проверка.
- Выполнить RAG judge с именованным ресурсом и пакетной оценкой.
- Проверить HF/local-vLLM/API там, где они поддерживают выбранный протокол;
  unsupported возможности должны давать явный результат.
- Сравнить latency/число вызовов и проверить отсутствие повторной загрузки
  ресурса на каждом case/turn.
- По результатам уточнить контракты, затем зафиксировать первую версию API.

Выход: реальные multi-turn и judge сценарии подтверждают архитектуру.
Fake tests не заменяют этот gate; отсутствие endpoint/runtime отражается
как незавершённая проверка, а не как подтверждённая поддержка.

### Этап 4. Миграция существующих tasks и reporting

- LegacyTaskAdapter переводит старые payloads/tuples в новый цикл без смены
  scoring. Он не дублирует artifacts, cache и summary lifecycle.
- Adapter для произвольного legacy loader получает только порядок подготовленных
  cases; не выдумывает raw/group identity, которую loader не сохранил.
  Полная identity добавляется при миграции loader.
- Legacy tasks с mutable state выполняются последовательно; скрытые judge
  calls допустимы лишь временно и устраняются явной миграцией этих tasks.
- Мигрировать generate/probability, CopyText, RuParam/treeway, PPL,
  RAG/LLMAAJ, NER и Libra; проверить все оставшиеся registrations.
- Перевести bootstrap и show_results на общий reducer contract.
- Сравнивать старый и новый путь на одинаковых сохранённых model responses.
  Отдельно выполнять live smokes, не требуя побитового равенства stochastic runs.

Выход: built-in tasks используют один execution lifecycle, включая PPL;
aggregation не запускает inference.

### Этап 5. Registry, внешние задачи и завершение

- Ввести instance-level lazy TaskRegistry: task_id, factory, config schema,
  package/version/dependency identity. Конфликт требует explicit override.
- Constructors не загружают datasets/models и не делают network calls.
- Обновить CLI, benchmark runners, LLMAAJ и custom-task examples.
- Показать внешнюю задачу из установленного/примонтированного Python package
  через explicit registration; полноценный directory plugin loader остаётся
  отдельной задачей из BACKLOG.
- Для prompt_optimizer зафиксировать решение: адаптация к общему invocation/
  reducer API либо явно experimental unsupported consumer. Не сохранять
  обещание поддержки сломанного импорта.
- После characterization и runtime gates удалить legacy evaluator path,
  обновить docs и migration guide.

Выход: единственный поддерживаемый task API с рабочими single-turn и episode
examples. Версии схем и protocol changes задокументированы.

## 13. Проверки и критерии завершения

Обязательные contract tests:

- graph ports/types/cycles, duplicate ids, keyed joins и cardinality;
- stable raw/case/invocation identity и сохранение порядка представления;
- few-shot, selection/filters, unavailable counters и limits;
- config precedence, отсутствие утечки stops/reasoning и batch compatibility;
- generate/probability/logsoftmax response validation;
- PPL boundary и CopyText capabilities;
- group completeness, denominator, group bootstrap с повтором одной группы;
- judge batching, parser errors и отсутствие inference в aggregation/reporting;
- episode transitions, loop limits, malformed actions и independent sessions;
- неизвестный usage, deadline, отсутствие повторов side effects и cleanup;
- artifact publication при сбое между файлами, stale totals и cache mismatch;
- lazy registry и localized optional dependency errors.

После runtime-sensitive изменений выполнять dependency-free checks из
AGENTS.md и применимую реальную матрицу из [TEST_PLAN_v4.md](TEST_PLAN_v4.md):
API CPU-only, HF generate/probability/PPL, local vLLM и managed API,
hybrid thinking off/on, strict reasoning, foundational continuation.
Для заявления runtime safety выполнить требования всех трёх Docker profiles.
Coding environment и judge resource добавляются к матрице отдельными cells.

Сравнивать selected ids, demonstrations, messages/rendered prompts, requests,
responses, scoring records, group inputs и totals. Golden replay проверяет
сохранение semantics; live tests проверяют backend integration. Не обещать
общий local/API parity для whitespace-prefill и иных непроверенных возможностей.

Рефакторинг завершён, когда:

- обычная задача остаётся короткой и не требует ручного graph builder;
- реальные bounded multi-turn/coding и judge задачи работают через общее ядро;
- каждый resource call виден в плане или episode events и учтён в budgets;
- reasoning/continuation остаются централизованными в LLM/continuation;
- PPL не имеет отдельного полного evaluator lifecycle;
- aggregation и reporting работают по сохранённым данным без inference;
- все built-in tasks мигрированы, baseline изменения объяснены и проверены;
- failures не превращаются в успешные totals, resources всегда освобождаются;
- API/examples, схемы artifacts и external registration документированы;
- производительность проверена на одинаковых workloads, без случайного
  per-case model loading и утраты batching.

Простота оценивается по тому, сколько task-specific кода нужно для новой задачи
и насколько легко восстановить причину её результата по artifacts.
Количество узлов или универсальность scheduler сами по себе не являются целью.
