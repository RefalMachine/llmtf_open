# Task/eval v2: общее исполнение с поэтапной миграцией

Дата: 2026-09-23. Статус: **предлагаемый план, API не реализован**.
Основание: [аудит](EXTERNAL_BENCHMARK_AUDIT.md).
Заменяет порядок реализации и конфликтующие решения
[v1](TASK_EVAL_GRAPH_REFACTOR_PLAN.md); описание будущих coding/environment
требований v1 сохраняется в пределах отдельного multi-turn этапа.

## 1. Два результата вместо одного большого переключения

**R1 — общий execution contract для внешних benchmarks.** Обычная задача и
многошаговый model/encoder workflow используют общий dispatcher, identities,
budget resolution, artifact store и failure semantics. Существующие tasks
продолжают работать с прежними prompts, выборкой и метриками.

**R2 — завершённая переработка task layer.** Встроенные tasks, PPL, judge,
bounded coding episode и reporting мигрированы; старый lifecycle удалён.
R1 не означает готовность R2. R2 не блокирует последовательный выпуск внешних
задач после их собственных gates.

Не входят в R1: публичный graph DSL, plugin discovery/installation,
distributed scheduler, native tool calling, coding sandbox, live-session
resume, автоматический cache каждого model node. Не вводить эти механизмы
под видом обязательной инфраструктуры суммаризации.

Уточнение scope: внешние benchmarks можно исправлять и выделять их algorithms
в переиспользуемые модули. LLMTF adapter связывает эти модули с общим dispatcher;
сам algorithm не обязан дублироваться внутри `llmtf/tasks`. Изменения сохраняют
подход benchmark, проверяются относительно предыдущих экспериментов и
версионируются по правилам раздела 2.1 integration v2. Архитектурная миграция
сама по себе не разрешает менять prompts, sampling или определение метрики.

## 2. Разделение ответственности

```text
Task: data + protocol + pure scoring/reducer
                    |
             EvaluationPlan
                    |
Evaluator: selection, execution, artifacts, completion
                    |
Dispatcher: keyed batches + resources + budgets
             /                  \
     LLM -> Backend         encoder/metric adapter
```

Task не получает mutable `LLM`, backend или credentials. Она строит запросы,
читает responses и определяет следующий шаг. Модельные обращения, embeddings
динамических текстов и ресурсные метрики видимы executor-у.
Reasoning orchestration остаётся в `LLM`; token surfaces и assistant-prefill —
в `llmtf.continuation`.

Не добавлять backend primitive `pipeline` и не использовать `method='pipeline'`
в `support_method`. У плана есть набор resource requirements, у каждого
invocation — существующая operation. Capability проверяется по operation и
features, а не по имени workflow.

## 3. Минимальные контракты

Использовать идеи v1, реализуя только нужную часть:

- `TaskSpec`: task/protocol identity, defaults, requirements, selection policy,
  reducer descriptor, dependency manifest;
- `RawCase`: immutable dataset identity + исходный row key/index;
- `EvaluationCase`: case id, raw id, group id, payload/artifact references;
- `Invocation`: invocation id, resource, operation, validated input,
  effective config, parent dependency ids;
- `Response`: тот же id, output, status, usage/finish diagnostics;
- `ScoringRecord`, `GroupRecord`, `AggregationResult`: keyed values, coverage,
  конечные metrics, отдельный primary score;
- `RunContext`: task-local RNG, token counting service, read-only capabilities,
  artifact references; не скрытый доступ к модели.

`raw_id` не равен upstream title. `case_id` включает устойчивый variant/section
id; `invocation_id` — raw/case id, stage, semantic position и iteration.
Порядок завершения не участвует в identity. Retry сохраняет invocation id и
меняет attempt. Число выбранных raw samples не равно числу calls.

Сериализуются JSON-compatible state/descriptors и ссылки на большие данные;
handlers идентифицируются именем, версией и digest. Python generators,
closures, model handles и pickle не являются persistence contract.

## 4. DAG и ограниченное динамическое исполнение

Оставить Source/Transform/Invoke/Reduce из v1. Single-turn template строит
минимальный план автоматически; автор простой задачи не пишет graph builder.
Transform/Reduce не выполняют скрытого inference. Все multi-input joins —
по ключам; cardinality и missing/duplicate ids валидируются.

Для динамического fan-out использовать controller внутри предусмотренного
v1 `Episode`. Model-only вариант можно называть `WorkflowTask` на уровне
authoring helper; это **не второй executor**. Контракт решений:

```python
start(case, context) -> Decision
advance(state, observations_by_id, context) -> Decision

Decision = AwaitMany(state, requests) | Finish(result)
```

`AwaitMany` содержит конечную непустую коллекцию независимых requests;
`Await(request)` — helper для одного элемента. Все responses текущего barrier
проверяются и передаются одним keyed набором. Следующий decision не зависит
от порядка их прихода. Нет произвольных обратных рёбер и вложенных episodes.

Это явное изменение v1, где допускался только один ожидающий request:
иначе одна книга с десятками chunks не использует batching. Динамический
Blueprint и model-generated filtering показывают, зачем нужен controller;
публичный универсальный workflow язык из этого не следует.

Executor собирает requests разных активных raw cases, разбивает по совместимости
и исполняет batches. Он ограничивает active cases и число ожидающих requests;
task не создаёт свой thread pool. Для одной модели первоначально исполняется
один batch за раз. Backend сохраняет свою сетевую concurrency и retries.
Повтор целой волны поверх backend retries не добавляется.

Batch key: resource/deployment identity, operation, effective sampling и
reasoning config, continuation/output options. Per-item различия допускаются
только если adapter их поддерживает. Сортировка на batching не меняет
presentation order или membership группы.

## 5. Config, context и стоимость

Resolver строит независимый deep snapshot из model defaults, stage defaults
и явных run overrides; фиксирует requested/effective thinking и answer/reasoning
budgets. Stage answer caps принадлежат protocol. Старый programmatic full
generation-config override сохраняет своё поведение для legacy tasks;
для нового workflow конфликтующий override должен быть явно разрешён
protocol schema либо отклонён. Нельзя silently применить один cap ко всем стадиям.

Каждый фактически сформированный prompt проверяется перед invocation.
Chunk tokenizer и tokenizer оцениваемой модели — разные сущности.
Правило R1: сохранять существующий budget algorithm, но применять его отдельно
к стадии, затем проверять prompt целиком. Не добавлять эвристическое уменьшение
fan-in, output/reasoning budgets или обрезку истории ради размещения prompt.
Новая политика такого сокращения — самостоятельное protocol изменение.

Unknown token count остаётся unknown; endpoint context error остаётся ошибкой.
Строгий общий token cap разрешён только при проверяемом счётчике/верхней границе.
Обе reasoning phases учитываются в usage; если backend не сообщает расход
целиком, total помечается incomplete, а не вычисляется из длины final output.

Предпочтительный окончательный API передаёт immutable invocation reasoning
config явно в LLM. Допустимый промежуточный R1 adapter применяет существующий
`MaxLenContext` последовательно, внутри scope с полным восстановлением
answer/reasoning/stops после success/exception. Он не объявляется immutable
и запрещает перекрывающиеся scopes на одном resource. Tasks не получают
доступ к этому compatibility механизму. Замена adapter на explicit config
не должна менять protocol и вызывающие tasks; её runtime gate отдельный.

У controller конечны step/call limits; у run есть deadline. Превышение
инфраструктурного лимита не создаёт полноценный score. Ограничение числа
вопросов/кластеров, влияющее на ответы, задаёт сам protocol и включает в identity.
Deadline передаётся adapter как оставшийся timeout: простая проверка времени
между неограниченными вызовами не является гарантированным deadline.

## 6. Ресурсы и optional зависимости

Для R1 достаточно model и encoder/metric adapters. Metric adapter предоставляет
явные embedding/score batch operations; один encoder может использоваться
preparation, workflow и scorer, сохраняя отдельные operation configs.
В частности `prompt`, normalize, dtype и max sequence length входят в manifest.

Resource manager создаёт ресурс лениво один раз и закрывает в `finally`.
GPU residency задаётся deployment: remote primary + local encoder, CPU encoder
или проверенное совместное размещение. Не переносить `.cuda()` и
`empty_cache()` из task algorithm в runtime policy.

До регистрации внешних tasks необходима lazy factory как минимум для новых
entries. Ошибка отсутствующего optional пакета относится к запрошенной задаче;
обычные API imports остаются torch-free. Полная instance-level registry migration
относится к R2. Resource overrides и execution limits должны проходить через
одну проверяемую схему Python API, single-model CLIs и всех трёх runners.

У новых experimental entries предлагается `default_enabled=False`: явный id
доступен, но текущий `datasets_names='all'` сохраняет прежний набор встроенных
задач. Это изменение selection code должно иметь regression test; просто
добавить class в существующий dict недостаточно. Static preflight task/config/
operation выполняется до model loading там, где несовместимость известна;
неизвестные свойства проверяются после loading/probes и при каждом response.

## 7. Scoring, coverage и reporting

Scoring может состоять из pure transforms и явных Invoke(metric/judge).
Aggregation получает готовые records и никогда не вызывает inference.
Offline пересчёт aggregation требует только reducer; пересчёт encoder metric
из texts — отдельная scoring операция с ресурсом и новой scorer identity.

Различать:

- invalid model prediction — protocol-defined score, обычно ноль, denominator
  сохраняется;
- infrastructure/data-contract failure — task failed, полноценного total нет;
- predeclared dataset exclusion — причина в selection manifest;
- cache hit — completed ранее, не новая генерация.

Coverage содержит requested limit, available/eligible/selected raw counts,
excluded rows и причины, expected/completed/invalid/failed groups/cases.
Если eligible меньше запрошенного лимита, это нормальная явная coverage,
а не обещание исполнить указанное число records.

Bootstrap unit задаёт reducer. Для books — книга; для Wiki sections — статья
со всеми секциями. Повтор bootstrap draw сохраняет multiplicity группы.
Point estimate — фактическое среднее, CI считается отдельно. Normal report
читает totals без registry/encoder; offline bootstrap загружает только reducer.

## 8. Artifacts, fingerprints и recovery

Сохранить основные имена и pretty JSON formats. Case samples имеют version,
ids, statuses и ссылки на trace; `predict` содержит конечный output. Для sections
это declared structured result, а не выдуманная склеенная статья.

Plan, selection manifest, keyed scoring/group records и events — sidecars.
Большие request/output тексты хранятся один раз по content digest; trace
ссылается на них. Для воспроизводимости development runs сохраняют полные
входы/выходы локально. Режим публикации с сокращёнными текстами явно отражает
ограничения replay и не выдаётся за полный checkpoint.

Три различные identities:

1. preparation: данные, parser/chunker/retriever/encoder inputs и runtime;
2. execution: protocol, selected ids, prompts/dependencies, resources,
   configs, limits, model deployment и runtime;
3. scoring: execution/output identity, scorer/encoder/reducer implementation.

Для R1 completion fingerprint консервативно объединяет все три. Раздельный
rescoring не обещается как автоматический cache hit; это явная offline операция.
Один prepared cache может использоваться несколькими model runs.

**Sample resume — изменение относительно v1, обязательное для R1 benchmarks.**
После завершения всех calls и scoring одного raw sample атомарно сохраняется
checkpoint с fingerprint, ids, counts и hashes нужных artifacts. Resume
использует только валидные complete checkpoints; частичный raw sample
исполняется заново. Утерянный response после отправки API может привести
к повторной оплате: exactly-once remote execution не обещается.

Публикация task:

1. Один writer на нормализованный task artifact id; коллизии запрещены.
2. Новая attempt пишет в отдельный каталог и не имеет completed total.
3. Force-recalc исключает старый total из active reporting; предыдущая attempt
   может сохраняться отдельно для восстановления.
4. Проверяются expected raw/group membership и все обязательные файлы.
5. Final samples/details/manifest публикуются; total с completion status,
   fingerprint и hashes публикуется последним атомарной заменой.

Cache hit проверяет manifest целиком. Events или закрытый JSON array сами
по себе не доказывают completion. Старые schema-v2 totals остаются читаемыми;
их нельзя использовать как checkpoints нового execution path.

## 9. Последовательность работ

| Этап | Результат | Gate |
|---|---|---|
| T0 Characterization | Golden selection/messages/responses/scoring для RuCoLA, RuParam, PPL, treeway, NER, Libra, judge; fake book/wiki | Отделены известные bugfixes от архитектурной миграции |
| T1 Invocation path | Keyed dispatcher, resource lifecycle, config/budget adapter, probability coverage contract | Generate/proba/logsoftmax alignment, exception restoration, capability failure |
| T2 Workflow | Single-turn и bounded AwaitMany, encoder calls, queue limits | Одна книга batch-ит chunks; разные стадии не смешиваются; dynamic fan-out завершается |
| T3 Persistence | Selection/events/checkpoints/completion + pure reducer/report | Kill/restart, corruption, duplicate writer, force-recalc; offline report без models |
| T4 Первый реальный срез | Одна обычная задача и RuBookSum hierarchical через общие сервисы | Backend smokes и CLI/YAML/reporting из integration v2 |
| T5 Проверка обобщаемости | Wiki ranking, затем outline/sections/Blueprint; fake judge branch | Смешанные operations, group denominators, dynamic encoder requests без обхода dispatcher |
| T6 R2 multi-turn | Реальный coding environment и judge через то же ядро | Изоляция, bounded feedback loop, cleanup и hidden-test policy из v1 |
| T7 R2 migration | Legacy adapters, единый PPL lifecycle, built-ins, lazy registry | Golden replay всех registrations, live matrix, удаление старого lifecycle |

T0–T3 допускают небольшие вертикальные PR; не накапливать универсальный
scheduler без runnable task. Внешние задачи выпускаются по мере прохождения
T4/T5 и собственного validation; незавершённость T6/T7 фиксируется честно.

## 10. Критерии простоты и проверки

Новая single-turn задача остаётся короткой: данные → request → score → reducer.
Новая workflow задача описывает state/transitions; не содержит client, executor,
logging/resume loop или code для backend-specific payloads.

Обязательные fake tests проверяют также unknown usage, incomplete candidate
coverage, out-of-order responses, empty/duplicate AwaitMany, exhausted limits,
optional dependency errors, отсутствие encoder/model calls при aggregation.
Benchmark-specific golden fixtures перечислены в integration v2.

После core/backend-sensitive изменений выполняются checks из AGENTS.md и
применимые runtime cells. Перед общим заявлением runtime safety — все три
Docker profiles и реальная матрица по `TEST_PLAN_v4.md`; успешный R1 smoke
не заменяет этот gate. Изменения PPL/continuation/reasoning требуют проверок
соответствующих старых задач, даже если основной feature — суммаризация.

В R2 дополнительно инвентаризировать LLMAAJ, examples, prompt_optimizer и
внешнюю регистрацию. Несовместимые потребители не оставлять молча сломанными:
мигрировать либо явно документировать неподдерживаемый experimental статус.
