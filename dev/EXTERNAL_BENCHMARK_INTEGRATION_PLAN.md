# План интеграции RuBookSum и RuWikiBench

> Архитектурный план интеграции двух external benchmarks в штатный task/evaluator
> путь LLMTF. Это development-документ, а не пользовательская инструкция запуска.

Статус документа: **исторический проект v1; заменён планом v2**.

Результаты проверки исходников: [аудит](EXTERNAL_BENCHMARK_AUDIT.md).
Актуальный проект реализации:
[EXTERNAL_BENCHMARK_INTEGRATION_PLAN_v2.md](EXTERNAL_BENCHMARK_INTEGRATION_PLAN_v2.md).
Текст ниже сохранён для истории; рекомендации v2 имеют приоритет.

Дата ревизии: 2026-09-23.

## 1. Цель

Интегрировать RuBookSum и RuWikiBench так, чтобы они:

1. регистрировались в `llmtf.tasks.TASK_REGISTRY`;
2. запускались через обычные `evaluate_model.py` и `evaluate_model_api.py`;
3. включались в общие benchmark YAML без отдельного OpenAI-клиента и отдельного
   формата результатов;
4. использовали существующие `LLM`, reasoning orchestration и HF/vLLM/API
   backends;
5. сохраняли стандартные params/sample/total artifacts, provenance и cache
   fingerprint;
6. имели явно версионированный evaluation protocol и не меняли опубликованные
   baseline semantics молча;
7. не нарушали torch-free контракт базового API-профиля.

Прямой перенос внешних CLI в `benchmark/` не считается интеграцией. Целевой
результат — штатные LLMTF tasks поверх общего task/evaluator контракта.

## 2. Зафиксированная отправная точка

В `.gitmodules` закреплены два проекта:

| Бенчмарк | Путь | Зафиксированная ревизия |
|---|---|---|
| RuBookSum | `external_benchmarks/rubooksum` | `a997d2154b343bf01d53ce7d7bf1fa2777fc6070` |
| RuWikiBench | `external_benchmarks/ruwikibench` | `5dbf8d95c2cced4583943639f9a59e3e22a259d4` |

Сабмодули были инициализированы 2026-09-23. Их исходники следует использовать
как reference implementation и источник фактического поведения. Они не готовы
к прямому импорту как Python packages.

Актуальные на дату ревизии dataset snapshots:

- `NejimakiTori/literature_sum`, revision
  `972f0646a65a63cb513a2eea276766471e98b9f4`: 634 пары книга–аннотация,
  `text` имеет тип `list[string]`, dataset card объявляет MIT;
- `NejimakiTori/RuWikiBench`, revision
  `ae4c23629fa286c404ce22bb751f94648b5dd550`: 100 статей, HTML и списки
  источников, dataset card объявляет `license: other`.

Dataset revisions выше являются входами будущего канонического протокола. Их
нельзя заменять плавающим `main` без нового protocol version и validation run.

## 3. Почему текущий task contract недостаточен

Текущий `Evaluator` поддерживает три атомарных task methods:

- `generate`;
- `calculate_tokens_proba`;
- `calculate_logsoftmax`.

Для одного sample evaluator формирует один набор messages, выполняет один batch
primitive и передаёт один результат в `Task.evaluate`. Оба внешних бенчмарка
являются графами зависимых модельных вызовов.

### 3.1. RuBookSum

Иерархический метод выполняет:

1. разбиение книги на перекрывающиеся чанки;
2. независимую генерацию аннотации каждого чанка;
3. многоуровневое объединение аннотаций;
4. опциональную embedding-фильтрацию близких промежуточных узлов;
5. оценку итоговой аннотации относительно reference summary.

Blueprint выполняет:

1. генерацию вопросов для каждого чанка;
2. генерацию ответов на вопросы либо кластеризацию и обобщение вопросов;
3. генерацию chunk summaries по blueprint;
4. рекурсивное слияние итогов;
5. оценку итоговой аннотации.

Промежуточные outputs входят в следующие prompts. Их нельзя представить как
набор независимых однопроходных samples.

### 3.2. RuWikiBench

RuWikiBench состоит из трёх логически независимых subbenchmarks.

**Ranking**:

1. генерирует русское и английское описание поискового запроса;
2. выполняет BM25 retrieval;
3. получает probability релевантности каждого найденного snippet;
4. считает NDCG и R-Precision.

**Outline**:

1. получает embeddings source snippets;
2. кластеризует их, опционально используя reference headers как hint;
3. генерирует описание и план каждого кластера;
4. объединяет планы;
5. сравнивает заголовки с reference outline.

**Sections**:

1. использует reference section-to-source mapping;
2. отбирает snippets по embedding similarity;
3. группирует близкие snippets;
4. иерархически суммаризует группы;
5. генерирует текст каждой reference section;
6. считает embedding PRF, ROUGE-L и BLEU.

Outline не потребляет ranking output, а sections не потребляет сгенерированный
outline. Поэтому они должны быть отдельными tasks, а не стадиями одного
неразделимого запуска.

## 4. Состояние внешних snapshots

Reference code нельзя подключать к runtime LLMTF без адаптации.

### 4.1. RuBookSum

- `utils.py` не компилируется: словарь `extra_body` в `LlmCompleter` не закрыт;
- проект использует `sys.path` mutation вместо package imports;
- dataset открывается относительно текущей директории;
- NLTK resources скачиваются при импорте `metrics.py`;
- прямой `AsyncOpenAI` client обходит `APIBackend` и framework retries;
- thinking выключается добавлением текстового суффикса к каждому prompt;
- один model-specific repetition penalty зашит по имени модели;
- per-book exceptions записываются отдельно и пропускаются, после чего total
  считается только по успешным книгам;
- JSONL открывается в append mode без cache identity или защиты от дублей.

### 4.2. RuWikiBench

- проект также не является Python package;
- helper в `openai_utils.py` использует `re` без импорта;
- запуск sections менее чем на двух статьях может обратиться к
  неинициализированному `result`;
- API payload содержит vLLM-specific поля независимо от возможностей endpoint;
- ranking ищет YES/NO только среди ограниченного top-logprobs ответа и может
  получить два нуля;
- ошибки статей пропускаются, а total считается по оставшимся результатам;
- preprocessing пишет BM25, snippets и embeddings внутрь директории проекта;
- повторный запуск дописывает records в существующие outputs.

### 4.3. Метрики

Название BERTScore в обоих проектах не соответствует стандартной библиотеке
`bert_score`. Реализация кодирует предложения SentenceTransformer-моделью и
считает максимальные cosine similarities, затем precision/recall/F1.

Bootstrap-функции называют `mean` медиану bootstrap distribution. Это следует
либо сохранить в отдельном reference-compatible protocol, либо исправить в
новом protocol с явным изменением metric identity.

## 5. Архитектурное решение

### 5.1. Новый `PipelineTask`

Добавить в task layer отдельный контракт для многошаговых задач. Рабочее имя:
`PipelineTask(Task)`.

Минимальные свойства контракта:

```python
class PipelineTask(Task):
    method = "pipeline"
    required_methods = frozenset({"generate"})

    def load_samples(self, *, max_sample_per_dataset):
        ...

    def execute_batch(self, *, model, samples, context):
        ...
```

Точные имена можно скорректировать при реализации, но необходимо сохранить
следующие инварианты:

1. task управляет benchmark algorithm, но не знает concrete backend;
2. каждый модельный вызов идёт через публичные batch primitives `LLM`;
3. pipeline объявляет полный набор требуемых capabilities до загрузки данных;
4. evaluator владеет output directory, logging, cache identity и failure
   semantics;
5. pipeline возвращает по одному финальному result record на исходный sample;
6. внутренние calls сохраняются как структурированный trace;
7. task не создаёт собственный OpenAI client и не читает API credentials.

Добавлять новый backend primitive `pipeline` не нужно. Это orchestration level,
аналогично тому, что reasoning остаётся на уровне `LLM`, а не backends.

### 5.2. Pipeline execution context

Evaluator должен передавать task ограниченный execution context, который:

- вызывает `model.generate_batch`;
- вызывает `model.calculate_tokens_proba_batch`;
- применяет effective thinking mode;
- создаёт stage-local sampling config;
- проверяет prompt budget для конкретной стадии;
- сохраняет prompt/output/info и имя стадии;
- сохраняет исходные индексы при batch failure;
- не раскрывает backend internals.

Pipeline task не должен вручную воспроизводить reasoning dispatch, stop-token
propagation, continuation policy или API payload construction.

### 5.3. Ready-wave batching

Наивное выполнение всей цепочки отдельно для каждой книги или статьи потеряет
основное преимущество локальных backends и API batching. Executor должен
группировать готовые узлы одного типа в waves:

1. собрать все независимые prompts текущего уровня;
2. выполнить один или несколько batches с обычным `batch_size`;
3. вернуть outputs их исходным sample/node ids;
4. построить следующий уровень графа;
5. повторять до завершения samples.

Для RuBookSum это означает batching chunk summaries и merge nodes одного
уровня. Для RuWikiBench — batching query generation, snippet probabilities,
cluster descriptions и независимых sections.

`batch_size` остаётся числом model requests в одной волне. Отдельный внешний
параметр `concurrency` не нужен.

### 5.4. Stage-local context budgeting

Один `MaxLenContext` вокруг всего task неприменим: стадии имеют разные output
budgets, а prompt следующей стадии зависит от предыдущего output.

Перед каждой волной executor должен:

1. клонировать базовый sampling config;
2. установить канонический `max_new_tokens` стадии;
3. вычислить answer/reasoning/prompt budgets;
4. проверить каждый сформированный prompt;
5. восстановить configs после вызова даже при исключении.

Для локальных backends превышение бюджета должно обнаруживаться до generation.
Если API endpoint не предоставляет token counter, сохраняется текущий fail-closed
контракт API: неизвестное число токенов не заменяется нулём, а context error
endpoint остаётся ошибкой.

Автоматически менять group size или обрезать книгу при переполнении нельзя:
это изменяет benchmark protocol. Допустимые политики должны быть явно заданы
конкретной task и записаны в provenance.

### 5.5. Thinking и reasoning

`enable_thinking` должен проходить через штатный `LLM` во всех модельных узлах.
Текстовые суффиксы внешних implementations не переносятся.

Каноническая первоначальная политика: выбранный thinking mode применяется к
каждому модельному вызову pipeline. Если позднее потребуется thinking только на
финальной стадии, это будет отдельный task parameter и отдельный fingerprint.

Следует учитывать, что reasoning на каждом chunk/merge node существенно
увеличивает стоимость и время. Это ожидаемая семантика, а не основание молча
выключать reasoning на промежуточных стадиях.

### 5.6. Logging и resume

Текущий sample logger рассчитан на один prompt и один prediction. Для pipeline
result record нужен дополнительный совместимый блок, например:

```json
{
  "sample_id": "...",
  "predict": "final output",
  "metrics": {},
  "pipeline": {
    "protocol_version": "...",
    "calls": [
      {
        "stage": "chunk_summary",
        "node_id": "...",
        "prompt": "...",
        "output": "...",
        "info": {}
      }
    ]
  }
}
```

Для длинных задач нужен sample-level checkpoint/resume. Завершённый sample
может переиспользоваться только при совпадающем run fingerprint. Частично
завершённый граф можно восстанавливать позднее; для первой версии допустимо
возобновление только на границе целого sample, если это явно документировано.

## 6. Регистрация задач

### 6.1. RuBookSum

Одна task class с фиксированными registry parameters:

| Registry name | Method | Mode |
|---|---|---|
| `rubooksum/hierarchical` | hierarchical | default |
| `rubooksum/hierarchical_filtered` | hierarchical | filtered |
| `rubooksum/blueprint` | blueprint | default |
| `rubooksum/blueprint_cluster` | blueprint | cluster |

Канонические defaults первой версии:

- dataset: `NejimakiTori/literature_sum` с фиксированной revision;
- split: `train`;
- chunk tokenizer: `DeepPavlov/rubert-base-cased` с фиксированной revision;
- chunk size: 2000 tokenizer tokens;
- overlap: 200 tokenizer tokens;
- initial word limit: 500;
- default source-length filter: reference `cap_chars=80000`;
- embedding encoder: `deepvk/USER-bge-m3` с фиксированной revision;
- few-shot: не поддерживается, effective value равен нулю.

`max_sample_per_dataset` применяется после канонического length filter, как
reference `number_of_books`. В sample record обязательно сохраняется исходный
dataset index, чтобы фильтрация была воспроизводимой.

### 6.2. RuWikiBench

| Registry name | Required methods | Основные метрики |
|---|---|---|
| `ruwikibench/ranking` | generate, calculate_tokens_proba | NDCG, R-Precision |
| `ruwikibench/outline` | generate | embedding P/R/F |
| `ruwikibench/sections` | generate | embedding P/R/F, ROUGE-L, BLEU |

Канонические defaults первой версии:

- dataset: `NejimakiTori/RuWikiBench` с фиксированной revision;
- source window: 600 words;
- overlap: 0;
- ranking positive/negative candidates: `YES`/`NO`;
- outline neighbor count: 0;
- outline description mode: true;
- clusterization with reference hint: true;
- embedding encoder: `sergeyzh/BERTA` с фиксированной revision;
- few-shot: не поддерживается, effective value равен нулю.

Полный прогон задаётся списком трёх registry names в benchmark YAML. Composite
alias не должен повторно выполнять общую подготовку данных; shared prepared
cache решает эту задачу.

### 6.3. Task-specific параметры

Для первой версии канонические варианты регистрируются отдельными именами, а
их protocol parameters фиксируются в class/registry config. Не следует сразу
добавлять множество benchmark-specific CLI flags.

Если появится подтверждённый сценарий нестандартных вариантов, добавить общий
строго валидируемый `task_kwargs` mapping в Python API, single-model CLIs и
benchmark YAML. Он должен участвовать в provenance и не разрешать незаметно
переопределять task identity.

## 7. Dataset и preprocessing layer

### 7.1. Общие правила

- Данные загружаются через `datasets` либо из явно заданного локального mirror.
- Dataset revision обязательна и входит в fingerprint.
- Prepared artifacts не пишутся в git submodule.
- Cache location задаётся стандартным cache root или отдельным
  `LLMTF_BENCHMARK_CACHE`.
- Cache key не содержит API credentials или приватные абсолютные пути.
- Cache validation проверяет manifest и content hashes, а не только наличие
  каталога.

### 7.2. RuBookSum

Отдельная предварительная распаковка не нужна. Loader нормализует `text` из
`list[str]` в один документ, сохраняя исходные сегменты в sample metadata.

Chunking является частью evaluation protocol. Tokenizer и его revision должны
быть загружены один раз на task, а не заново для каждой книги.

### 7.3. RuWikiBench

Prepared cache должен содержать:

- нормализованные article ids и titles;
- распарсенный reference outline;
- reference section texts и source mappings;
- source snippets и стабильные `SnippetKey`;
- BM25 corpus/index;
- embeddings snippets по статьям;
- preprocessing manifest.

Рекомендуемый cache key:

```text
dataset_revision
preprocessing_version
window_size
overlap
bm25_config
stemmer_config
encoder_name
encoder_revision
```

Downloader HTML/source content из интернета не нужен для evaluation: dataset
уже содержит HTML и source texts. Исторические downloader-зависимости должны
остаться вне runtime task path.

## 8. Метрики и protocol versioning

### 8.1. Два возможных режима

Необходимо различать:

1. `reference-2026-03` — максимально близкое воспроизведение внешнего кода
   после минимальных исправлений, нужное для сравнения с его baseline;
2. `llmtf-v1` — исправленный и backend-neutral protocol.

Первой публичной task identity рекомендуется сделать `llmtf-v1`. Reference
режим добавляется только при наличии реальной необходимости сверять числа с
публикацией или авторами.

### 8.2. Предлагаемые правила `llmtf-v1`

- RuWiki ranking использует `calculate_tokens_proba_batch`, а не поиск
  кандидатов в top-logprobs сгенерированной последовательности;
- перед запуском выполняется явная проверка представимости YES/NO candidate
  surfaces выбранным backend/tokenizer contract;
- ошибки samples не исчезают из denominator без отчёта;
- total artifact содержит requested/succeeded/failed counts;
- sample mean называется mean;
- confidence interval и его метод записываются явно;
- embedding PRF не называется стандартным BERTScore без поясняющего namespace;
- ROUGE implementation и stemming/tokenization фиксируются;
- пустые predictions и пустые sentence lists имеют явную детерминированную
  политику;
- случайные sampling/clustering операции используют зафиксированный seed.

### 8.3. Сравнимость

Результаты разных protocol versions не объединяются в одной leaderboard серии.
Любое изменение prompts, encoders, chunking, retrieval, filtering, metric
implementation или aggregation требует нового fingerprint; содержательное
изменение требует нового protocol version.

## 9. Provenance и cache identity

Текущий task implementation hash покрывает файл task class, но не покрывает
внешние prompts, preprocessing и encoder weights. Для этих benchmarks в
`run_config.task` необходимо добавить стабильный pipeline manifest:

- protocol name/version;
- external reference repository URL и commit;
- dataset repository/config/split/revision;
- dataset schema/content identity;
- prompt hashes;
- stage graph version;
- stage generation configs;
- chunking/filtering parameters;
- retrieval/BM25 parameters;
- encoder name/revision;
- metric implementations и versions;
- random seeds;
- requested/effective sample count;
- requested/effective thinking policy;
- prepared-cache manifest hash.

Секреты, endpoint credentials и содержимое приватных cache paths в provenance
не попадают.

## 10. Dependencies и Docker

Минимально ожидаемые дополнительные зависимости:

- `sentence-transformers`;
- `razdel`;
- package `rouge`, если требуется reference-compatible ROUGE;
- `bm25s`;
- `PyStemmer`;
- выбранный HTML parser;
- необходимые NLTK data resources для Wiki preprocessing.

Не следует копировать внешние `requirements.txt` целиком. Они содержат старые
pins, дубли model runtime и downloader libraries, не нужные evaluation path.

API image должен оставаться torch-free. Рекомендуемый layout:

1. оставить `api`, `hf` и `vllm` без изменения их базового назначения;
2. добавить optional benchmark dependencies поверх HF runtime либо отдельный
   `longbench` target, который расширяет HF;
3. использовать этот образ и для удалённого API evaluation, поскольку local
   embedding metrics требуют torch/SentenceTransformer;
4. не требовать CUDA для логической корректности, но документировать, что
   CPU-encoder для полного benchmark будет медленным.

`ruwikibench/ranking` теоретически может работать без encoder при готовом BM25
cache, но не следует преждевременно создавать отдельную матрицу installation
вариантов до появления измеримой пользы.

## 11. Лицензирование

В обоих git repositories на закреплённых revisions отсутствует `LICENSE`.
Dataset card RuBookSum объявляет MIT только для dataset. Dataset card
RuWikiBench объявляет `license: other`.

До копирования внешних исходников и prompts в `llmtf/` необходимо:

1. получить явную лицензию upstream либо письменное разрешение;
2. определить условия распространения prompts;
3. сохранить attribution и citation metadata;
4. отдельно проверить допустимость распространения подготовленных caches,
   содержащих source texts или embeddings этих texts.

До решения лицензирования допустимо проектировать framework contract и писать
синтетические contract tests. Публикация перенесённых prompts/data в основном
репозитории не должна происходить неявно.

## 12. План реализации

### Этап 0. Зафиксировать протокол и лицензии

- согласовать `llmtf-v1` против reference-compatible режима;
- получить лицензионное основание для code/prompts;
- закрепить dataset, tokenizer и encoder revisions;
- подготовить machine-readable protocol manifests;
- определить канонические task names и metric names.

Acceptance:

- все immutable inputs перечислены;
- нет плавающих model/dataset revisions;
- ясно, какие результаты сравнимы с внешними baseline;
- перенос внешних материалов разрешён либо исключён из выбранной реализации.

### Этап 1. Общий `PipelineTask`

- добавить контракт pipeline task;
- добавить capability validation для нескольких primitives;
- реализовать ready-wave batching;
- реализовать stage-local generation config и context budgeting;
- расширить logger schema;
- добавить sample-level resume;
- включить pipeline manifest в provenance/cache fingerprint;
- сохранить обычное поведение атомарных tasks без изменений.

Acceptance:

- dependency-free fake pipeline выполняет не менее трёх зависимых waves;
- порядок samples сохраняется при batch execution и ошибках;
- HF/API-like fake backends получают одинаковые logical calls;
- config восстанавливается после success и exception;
- существующий `tests/test_refactor_logic.py` проходит без изменений baseline.

### Этап 2. RuBookSum hierarchical/default

- добавить dataset loader;
- реализовать canonical chunking;
- реализовать chunk-summary и merge graph;
- добавить embedding PRF и ROUGE-L;
- зарегистрировать `rubooksum/hierarchical`;
- добавить synthetic fixture без скачивания полного dataset.

Acceptance:

- one-book fake-model run создаёт стандартные artifacts;
- trace содержит все chunk и merge calls;
- исходный dataset index и фильтрация записаны;
- повторный запуск с тем же fingerprint использует cache;
- изменение chunking или encoder revision вызывает cache mismatch.

### Этап 3. Остальные варианты RuBookSum

- hierarchical filtered;
- blueprint default;
- blueprint cluster;
- детерминированная кластеризация;
- единые metric/aggregation rules между четырьмя tasks.

Acceptance:

- четыре registry names проходят contract tests;
- некорректное сочетание method/mode невозможно зарегистрировать;
- каждая task имеет отдельную identity и output files;
- cluster mode корректно обрабатывает малое число вопросов.

### Этап 4. RuWikiBench preprocessing и ranking

- реализовать загрузку/нормализацию dataset;
- реализовать prepared-cache manifest;
- перенести BM25 preprocessing без downloader path;
- реализовать query generation;
- реализовать YES/NO scoring через token-probability primitive;
- зарегистрировать `ruwikibench/ranking`.

Acceptance:

- cache строится идемпотентно и валидируется по manifest;
- one-article run корректно завершается;
- отсутствие candidate probability является явной ошибкой;
- ranking работает через HF, local vLLM и API model interfaces;
- NDCG/R-Precision сверены на hand-built fixture.

### Этап 5. RuWikiBench outline и sections

- реализовать parsing reference outline/source mappings;
- реализовать embeddings и кластеризацию;
- реализовать outline graph;
- реализовать section selection/grouping/generation graph;
- зарегистрировать две оставшиеся tasks;
- реализовать vector/list metric aggregation и aggregation details.

Acceptance:

- one-article smoke не требует минимум двух samples;
- reference hints явно отражены в task params;
- section failures не исчезают из total;
- каждая metric имеет documented denominator;
- три RuWikiBench tasks переиспользуют один совместимый prepared cache.

### Этап 6. Docker и runtime matrix

- добавить optional dependency profile/target;
- проверить encoder на CPU и CUDA;
- выполнить one-sample HF smoke;
- выполнить one-sample local-vLLM smoke;
- выполнить one-sample API smoke;
- затем выполнить малую фиксированную матрицу всех семи registry tasks;
- сохранить sanitized commands, params и totals в отдельном report.

Acceptance:

- обычные single-model CLIs запускают задачи без внешних CLI;
- benchmark YAML запускает те же registry names;
- API credentials не попадают в artifacts;
- backend differences не меняют task algorithm;
- unsupported capability завершается явной ошибкой с ненулевым exit code.

### Этап 7. Reference parity

- исправить external snapshots в отдельном worktree или upstream branch;
- выполнить одинаковый ограниченный набор samples;
- сравнить prompts, intermediate nodes и metrics;
- классифицировать расхождения как bugfix, intentional protocol change или
  backend numerical difference;
- опубликовать parity report без смешивания protocol versions.

Этот этап обязателен перед заявлением совместимости с внешними baseline, но не
блокирует выпуск явно названного `llmtf-v1` protocol.

## 13. Ожидаемые изменения файлов

Ориентировочное разбиение, которое может быть уточнено после первого contract
prototype:

- `llmtf/base.py` — базовый pipeline task contract;
- новый `llmtf/pipeline.py` — execution context, waves и trace types;
- `llmtf/evaluator.py` — pipeline dispatch, logging, aggregation и resume;
- `llmtf/provenance.py` — pipeline manifest;
- `llmtf/sample_logger.py` — structured multi-call trace;
- `llmtf/tasks/rubooksum.py` — четыре RuBookSum variants;
- `llmtf/tasks/ruwikibench.py` либо package — три RuWikiBench variants;
- `llmtf/tasks/__init__.py` — registry entries;
- `benchmark/config.py` — только если потребуется общий `task_kwargs`;
- `requirements/profiles/` и `docker/` — optional benchmark environment;
- `tests/` — dependency-free contract/golden fixtures;
- `docs/` — пользовательская инструкция после runtime validation.

Не следует помещать generic pipeline scheduler внутрь одной из benchmark task:
он нужен обоим проектам и должен иметь единые logging/failure semantics.

## 14. Проверки

После изменений ядра обязательны стандартные dependency-free checks:

```bash
python3 tests/test_refactor_logic.py
python3 -m compileall -q llmtf evaluate_model.py evaluate_model_api.py benchmark show_results.py dev/tools examples
git diff --check
```

Если доступен pytest:

```bash
python3 -m pytest tests/test_refactor_logic.py -q
```

Дополнительные pipeline tests должны проверять:

- capability negotiation;
- ready-wave batching и stable ordering;
- stage-specific token budgets;
- reasoning enabled/disabled propagation;
- restoration configs после исключения;
- batch failures с исходными indexes;
- sample resume и fingerprint mismatch;
- deterministic dataset selection;
- deterministic clustering;
- empty outputs и частичные failures;
- exact metric aggregation на маленьких hand-built fixtures;
- отсутствие import-time downloads и optional-dependency side effects.

Runtime safety нельзя заявлять только по fake-model или compile tests. После
реализации необходимы реальные HF, local vLLM и API smokes в соответствующем
Docker environment.

## 15. Риски и решения

| Риск | Решение |
|---|---|
| Pipeline API разрастается под два конкретных проекта | Сначала минимальный fake contract и RuBookSum hierarchical; расширять только по фактической необходимости |
| Слишком большие intermediate traces | Сохранять обязательную metadata и configurable full prompt trace, не теряя final reproducibility |
| Reasoning многократно увеличивает стоимость | Явная per-call policy и provenance; никаких скрытых отключений |
| Encoder/runtime ломает API-only profile | Отдельный optional profile поверх HF |
| Изменение метрик даёт несравнимые числа | Protocol versioning и отдельный parity report |
| Частичные ошибки завышают total | Явные requested/succeeded/failed counts и fail-closed default |
| Prepared cache устаревает | Manifest, immutable revisions и content hashes |
| External code исправляется upstream | Submodule commit остаётся частью reference identity; обновление только отдельным решением |
| Лицензия не позволяет перенос prompts/code | Не копировать материалы до разрешения; оставить clean adapter/protocol implementation |

## 16. Решения, которые нужно принять до реализации

1. Требуется ли численная совместимость с опубликованными external baselines,
   или достаточно нового явно названного `llmtf-v1` protocol?
2. Разрешено ли переносить prompts и части исходников в основной repository?
3. Нужно ли сохранять полный текст каждого intermediate prompt/output по
   умолчанию, либо полный trace включается отдельной опцией?
4. Достаточен ли resume на границе книги/статьи для первой версии?
5. Нужны ли нестандартные task parameters в CLI/YAML сразу, или достаточно
   семи фиксированных канонических registry entries?
6. Входят ли эти задачи в leaderboard categories сразу после интеграции, или
   сначала публикуются как standalone experimental tasks?

Рекомендуемые первоначальные ответы: новый `llmtf-v1`, семь фиксированных task
ids, полный trace для development validation, sample-level resume и отсутствие
leaderboard inclusion до завершения runtime/parity report.
