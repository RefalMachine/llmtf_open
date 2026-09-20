# Рефакторинг v4: стабилизация model/reasoning/backend архитектуры

> Development stabilization record. Фактический результат проверки находится
> в `TEST_REPORT_v4.md`, пользовательский контракт — в `../docs/`.

Статус документа: **активный план**.

Дата ревизии: 2026-09-20.

Статус реализации на 2026-09-20: code-level P0/P1 changes из §§4–5
реализованы в рабочем дереве и покрыты dependency-free regression suites.
Централизованный continuation/prefill contract также реализован и описан в
[`docs/assistant_continuation.md`](../docs/assistant_continuation.md) и
[`docs/api_backend.md`](../docs/api_backend.md). Чистая сборка API-образа прошла;
HF/vLLM images импортируются, а Docker passthrough видит RTX 4090. Model load,
реальная HF/vLLM/API матрица и legacy comparison ещё не выполнены; поэтому
runtime parity и завершённость рефакторинга пока не заявляются. Наблюдения ниже
описывают исходное состояние, если конкретный пункт не помечен acceptance
результатами последующей validation.

### Handoff для следующей сессии

1. Не проверять доступность GPU через sandbox-shell: он не видит NVIDIA driver.
2. Использовать существующие `llmtf:hf-cu129` и `llmtf:vllm-cu129` с
   `docker run --gpus all`; CUDA passthrough уже подтверждён.
3. Сначала выполнить one-sample HF и local-vLLM smoke из
   [`TEST_PLAN_v4.md`](TEST_PLAN_v4.md), затем расширять до восьми samples.
4. После локальных smoke поднять vLLM API server и выполнить continuation probe,
   generate и token-probability cells.
5. Не менять continuation contract без нового обсуждения: дальнейшая работа —
   runtime validation и исправление только фактически найденных несовместимостей.

Краткий зафиксированный continuation contract:

- внутри framework используется только роль `assistant`; legacy `bot`
  нормализуется на границе task data;
- `predict` — только вновь сгенерированный текст, без prefill;
- локальные HF/vLLM сохраняют prefill точно через один общий renderer;
- generic API не считается поддерживающим assistant continuation;
- `api_profile=vllm` включает соответствующие transport extensions, а опасный
  whitespace проверяется отдельной prefill policy/probe;
- закрытый API без tokenizer endpoints полностью поддерживает обычную
  user-final генерацию, но не получает выдуманных token counts;
- варианты следующего токена вроде `A`/` A` обрабатываются централизованно.

`REFACTOR_PLAN.md`, `REFACTOR_PLAN_v2.md` и `REFACTOR_PLAN_v3.md` описывают историю решений и уже выполненные архитектурные шаги. Этот файл заменяет их как источник приоритетов для следующего этапа. Краткая operational-сводка находится в [`AGENTS.md`](../AGENTS.md).

## 1. Цель v4

Архитектурное разделение модели, reasoning и backend уже выполнено. Задача v4 — не переписать его ещё раз, а довести новую архитектуру до состояния, в котором:

1. один и тот же режим однозначно задаётся через Python API, single-model CLI и benchmark YAML;
2. HF, vLLM и API получают одинаковые phase-specific параметры reasoning;
3. ошибки не приводят к тихой потере или смещению samples;
4. каждый результат содержит достаточно параметров для воспроизведения;
5. кеш не смешивает несовместимые запуски;
6. добавление нового backend не требует знания неформальных форматов данных;
7. API-only, HF и vLLM доступны как отдельные проверяемые Docker/install profiles;
8. документация и тесты соответствуют фактическому коду.

После v4 текущий model/reasoning refactor можно считать завершённым. Bootstrap, task scaffolding и большие изменения benchmark scheduling остаются отдельным следующим этапом.

## 2. Фактическая отправная точка

### 2.1. Что уже реализовано

- [`llmtf/base.py`](../llmtf/base.py) содержит `BaseLLM`, `Task` и `SimpleFewShotHFTask`.
- [`llmtf/llm.py`](../llmtf/llm.py) содержит единственный concrete `LLM`, который держит `backend` и выполняет reasoning dispatch.
- [`llmtf/reasoning.py`](../llmtf/reasoning.py) содержит `ModelKind`, `ReasoningConfig`, `ReasoningFormat`, `ReasoningResult` и общий `EmulatedReasoningStrategy`.
- [`llmtf/backends/base.py`](../llmtf/backends/base.py) задаёт primitive backend contract.
- [`llmtf/backends/hf.py`](../llmtf/backends/hf.py), [`vllm.py`](../llmtf/backends/vllm.py) и [`api.py`](../llmtf/backends/api.py) реализуют три способа инференса.
- Старые `llmtf/model.py`, `llmtf/models/`, `HFModel*`, `VLLMModel*`, `ApiVLLMModel*` и `LocalHostedLLM` удалены.
- Reasoning-поля больше не хранятся в sampling `generation_config`.
- `count_tokens_for_messages` стал основным контрактом подсчёта токенов.
- `MaxLenContext` вычисляет prompt/answer/reasoning budgets из deployment context length.
- Per-sample output записывается валидным JSON-массивом.
- Pure-logic suite [`tests/test_refactor_logic.py`](../tests/test_refactor_logic.py) проходит без реальных ML-зависимостей.

### 2.2. Текущая стадия

Это **pre-merge stabilization**, а не готовая версия:

- `HEAD` совпадает с `origin/main`;
- новая архитектура и тесты находятся в untracked/unstaged worktree;
- вместе с рефакторингом присутствуют удаления `examples/`, `prompt_optimizer/`, external submodules и mode-only изменения;
- реальная GPU/API матрица после рефакторинга не выполнялась;
- существовавший на момент составления плана корневой `Dockerfile` описывал только полный NGC PyTorch + vLLM образ и устанавливал монолитный `requirements.txt`;
- API-only путь пока не отделён от torch/HF imports, поэтому CPU-only установка ещё не является рабочим поддерживаемым профилем;
- README, benchmark YAML и часть старых планов противоречат коду.

### 2.3. Источники истины

При разборе поведения использовать следующий порядок:

1. исполняемый код;
2. автоматические тесты;
3. [`AGENTS.md`](../AGENTS.md);
4. этот план;
5. v1-v3 plans как историю;
6. [`README.md`](../README.md) и исторический `todo` только как список пользовательских ожиданий.

### 2.4. Модели для validation

Основная проверка v4 выполняется на одной RTX 4090 и двух text-only сценариях:

- hybrid instruct: `Qwen/Qwen3.5-2B`;
- foundational: `Qwen/Qwen3.5-2B-Base`.

Для этого ограниченного эксперимента model revision отдельно не фиксируется и
поддержка `model_revision`/`tokenizer_revision` не входит в scope v4. Ожидается,
что модели не изменятся во время серии запусков. Старый и новый код должны
использовать один общий HF cache или одни и те же заранее скачанные model paths,
чтобы фактические веса не различались внутри эксперимента. Model weights не
должны попадать в image layers.

`Qwen3.5-2B` используется как `model_kind=hybrid`: обязательны отдельные прогоны
с thinking disabled и enabled. `Qwen3.5-2B-Base` проверяется в foundational mode
и не объявляется reasoning-моделью только из-за общей архитектуры семейства.
Мультимодальность в scope v4 не входит; для vLLM server использовать text-only
режим, если установленная версия его поддерживает.

Официальный model card на момент этой ревизии рекомендует latest/main
Transformers и vLLM main/nightly для Qwen3.5. Поэтому текущие draft pins
`transformers==5.9.0` и `vllm==0.21.0` являются гипотезой, а не подтверждённым
совместимым набором. Сначала нужен реальный load/generate probe обеих моделей.
После нахождения рабочего набора нельзя оставлять плавающие `main`/`nightly`:
закрепить immutable git commit или точный wheel URL/hash и записать их в report.

## 3. Инварианты, которые v4 не должен сломать

1. Reasoning orchestration остаётся на уровне `LLM`, а не возвращается в concrete backends.
2. Backend получает messages и возвращает batch result; он не выбирает `model_kind`.
3. `generation_config` остаётся sampling-only. Reasoning budgets и format остаются в `ReasoningConfig`.
4. `model_kind=auto` не возвращается.
5. `model_kind=plain` — one-pass; `hybrid` — переключаемый; `reasoning` — обязательный two-pass.
6. PPL остаётся HF-only до отдельного обоснованного расширения.
7. `VLLM_USE_V1` не возвращается.
8. Legacy model facades и compatibility shims не возвращаются.
9. Существующие пользовательские изменения и удаления не восстанавливаются и не коммитятся автоматически.
10. Специальные model tokens не приводятся в документации буквально; используется имя константы или placeholder.
11. Docker является основным поддерживаемым способом запуска; возможности случайного host-окружения не считаются характеристикой проекта.
12. API credentials никогда не попадают в image layers, Git, shell command logs или result artifacts.

## 4. P0: блокирующие проблемы корректности

### P0.1. Reasoning-настройки не проходят через benchmark

#### Наблюдение

[`benchmark/calculate_benchmark.py`](../benchmark/calculate_benchmark.py), [`calculate_benchmark_api.py`](../benchmark/calculate_benchmark_api.py) и [`calculate_benchmark_existing_api.py`](../benchmark/calculate_benchmark_existing_api.py) знают только старое поле `think`. Они добавляют `--disable_thinking`, но не передают:

- `model_kind`;
- `end_thinking_token_id`;
- `min_new_tokens_reasoning`;
- `model_context_len`;
- backend-specific параметры;
- явный положительный режим thinking.

CLI по умолчанию использует `model_kind=plain`. Поэтому отсутствие `--disable_thinking` не включает two-pass: plain dispatcher только предупреждает и делает one-pass. В существующих YAML нет `think: true`; reasoning-run через benchmark фактически отсутствует.

Дополнительно YAML содержат `max_prompt_len` или `max_len`, но loaders эти поля больше не читают.

#### Целевое поведение

Reasoning mode задаётся один раз и одинаково интерпретируется всеми entry points.

Предлагаемая схема benchmark config:

```yaml
model:
  model_kind: plain
  enable_thinking: false
  model_context_len: null
  max_new_tokens_reasoning: 4096
  min_new_tokens_reasoning: 1024
  end_thinking_token_id: null

defaults:
  few_shot_count: 0
  generation:
    temperature: 0.0

tasks:
  - name: example
    datasets:
      - russiannlp/rucola_custom
    enable_thinking: false
```

Model-wide значения задаются в `model`. Task может переопределять только `enable_thinking` и task/evaluation параметры. Не следует дублировать `model_kind` по каждой задаче одной модели.

#### План исправления

1. Добавить общий parser/normalizer benchmark YAML, используемый всеми тремя runners.
2. Ввести валидируемые секции `model`, `defaults`, `tasks[].generation`, `tasks[].evaluation`.
3. Поддержать legacy `extra_args.think` один переходный цикл:
   - вывести deprecation warning;
   - преобразовать в `enable_thinking`;
   - ошибка, если одновременно заданы оба поля и значения конфликтуют.
4. Добавить в single-model CLI положительный `--enable_thinking`.
5. Временно оставить `--disable_thinking` как deprecated alias для обратной совместимости.
6. Сделать flags mutually exclusive и вычислять одно нормализованное boolean-значение.
7. Default нового интерфейса: `enable_thinking=False`. Reasoning должен включаться явно.
8. Для `model_kind=reasoning` автоматически требовать `enable_thinking=True`; запретить противоречивую конфигурацию до загрузки модели.
9. Передавать все model/reasoning значения в subprocess command.
10. Удалить или реализовать мёртвый `--add_reasoning_tasks`; предпочтительно удалить после проверки внешних scripts.
11. Удалить из YAML неработающие `max_prompt_len`/`max_len`. Не преобразовывать их автоматически в `model_context_len`, потому что это разные сущности.
12. Неизвестные YAML keys должны приводить к понятной ошибке, а не игнорироваться.

#### Тесты

- Unit tests для config normalization: defaults, task override, legacy `think`, конфликт полей, неизвестный key.
- Tests, которые проверяют сформированный subprocess command без запуска модели.
- Отдельные случаи `plain/false`, `hybrid/false`, `hybrid/true`, `reasoning/true`, `reasoning/false`.
- Проверка одинаковых командных параметров local/API/existing-API runners.

#### Acceptance

- Один tiny benchmark реально выполняет two-pass через HF, vLLM и API.
- `_params.jsonl` показывает `model_kind=hybrid`, `enable_thinking=true` и end token id.
- Plain benchmark не пишет warning о попытке включить thinking.
- Legacy `think` даёт deprecation warning, но временно сохраняет поведение.

### P0.2. Phase-specific end token теряется на vLLM и API

#### Наблюдение

`EmulatedReasoningStrategy._build_reasoning_config` создаёт отдельную копию sampling config и записывает в неё reasoning end token.

Однако:

- [`VLLMBackend.generate_batch`](../llmtf/backends/vllm.py) передаёт в `SamplingParams.stop_token_ids` значение из `self.generation_config`, а не из аргумента `generation_config`;
- [`APIBackend.generate`](../llmtf/backends/api.py) также читает stop-token данные из `self.generation_config` и не использует phase-local `eos_token_id`.

Поэтому обязательный `end_thinking_token_id` для strict reasoning mode надёжно применяется только на HF. vLLM/API фактически зависят от текстового stop marker.

#### Целевое поведение

Backend primitive использует только переданный ему effective `generation_config`. Base config применяется лишь когда аргумент равен `None`.

#### План исправления

1. В vLLM брать `stop_token_ids` из effective `generation_config.eos_token_id`.
2. Нормализовать `int | list[int] | None` в одном helper.
3. В API отправлять phase-local ids в поддерживаемое vLLM extension-поле `stop_token_ids`.
4. Не путать `stop_strings`, `eos_token_id` и возможный legacy `stop_token_ids` attribute.
5. Проверить, что reasoning copy не мутирует backend base config.
6. Проверить continuation phase: она должна снова использовать исходные answer stop tokens.
7. Зафиксировать в backend contract, что переданный generation config имеет полный приоритет внутри конкретного primitive call.

#### Тесты

- Fake `SamplingParams`, фиксирующий полученные `stop_token_ids` для reasoning и continuation phases.
- Mock HTTP test, проверяющий JSON payload обеих API phases.
- Тест, что base config не изменился после успешного и аварийного вызова.
- Тест с `end_thinking_token_id=None`: поле не должно искусственно перезаписываться.

#### Acceptance

- HF, vLLM и API останавливают reasoning phase на одном explicit id.
- Вторая фаза не наследует reasoning-only end token.
- Text marker остаётся дополнительной страховкой, а не единственным механизмом vLLM/API.

### P0.3. `backend_kwargs` перезаписывается argparse defaults и скрывает опечатки

#### Наблюдение

В [`evaluate_model.py`](../evaluate_model.py) JSON kwargs объединяются с `explicit`, но `explicit` содержит все argparse defaults. Поэтому JSON для уже экспонированного параметра никогда не работает, если его значение отличается от CLI default.

Например, JSON `gpu_memory_utilization=0.9` перезаписывается значением `0.95`, даже если пользователь не вводил соответствующий CLI flag.

Кроме того, `Backend.__init__(**kwargs)` -> `Base.__init__(**kwargs)` молча игнорирует неизвестные параметры.

#### Целевая семантика

Приоритет:

1. явно введённый CLI option;
2. значение из `--backend_kwargs`;
3. default concrete backend constructor.

Опечатки и параметры чужого backend должны приводить к ошибке до загрузки модели.

#### План исправления

1. Для backend CLI options использовать `default=None` или `argparse.SUPPRESS`, чтобы отличать «не передан» от явного значения.
2. Для boolean options использовать tri-state parsing (`None/True/False`).
3. Парсить `--backend_kwargs` только как JSON object; list/scalar запрещать.
4. Сначала взять JSON, затем наложить только действительно переданные CLI values.
5. Остальные defaults оставить concrete backend constructors.
6. Валидировать keys через сигнатуру выбранного backend или явную schema.
7. Убрать silent sink неизвестных kwargs либо разрешить его только для документированных forward-compatible keys.
8. Исправить API-конструирование так, чтобы `model_context_len` не мог быть передан дважды.
9. Логировать итоговый sanitized backend config без secrets.

#### Тесты

- JSON-only значение применяется.
- Явный CLI value перекрывает JSON.
- Неуказанный CLI default не перекрывает JSON.
- Неизвестный key и key другого backend дают readable error.
- Invalid JSON и non-object JSON дают readable error.
- API key не попадает в logs.

#### Acceptance

- Acceptance-пример из v3 с `calculate_tokens_proba_logprobs_count` реально работает.
- `_params.jsonl` отражает effective constructor values.
- Опечатка в имени параметра завершает запуск до model loading.

### P0.4. Context budget не учитывает фактический режим thinking

#### Наблюдение

[`MaxLenContext`](../llmtf/utils.py) резервирует reasoning budget, если `reasoning_config.is_reasoning=True`. Он не получает фактический `enable_thinking` текущего run.

Следствия:

- `hybrid + enable_thinking=False` теряет до 4096 prompt tokens без причины;
- PPL может резервировать reasoning budget, хотя reasoning phase там нет;
- warning сообщает, что reasoning будет skipped, но решение о dispatch принимается позже в другом объекте;
- при ровно нулевом остатке для prompt trimming использует пограничную логику, которая может завершиться ошибкой вместо уменьшения reasoning budget на минимально необходимую величину.

#### Целевое поведение

Budgeting получает уже нормализованный execution mode:

- plain — reasoning budget 0;
- hybrid/disabled — reasoning budget 0;
- hybrid/enabled — reserve/trim reasoning budget;
- reasoning/enabled — reserve/trim reasoning budget;
- reasoning/disabled — configuration error до dataset loading;
- PPL — reasoning budget 0, поскольку PPL path one-pass.

#### План исправления

1. Ввести маленький immutable `ExecutionMode` или функцию `resolve_reasoning_execution(model_kind, enable_thinking, scoring_method)`.
2. Вызывать её один раз до `MaxLenContext` и dispatch.
3. Передавать в `MaxLenContext` `reasoning_enabled`, а не выводить его повторно из model kind.
4. Убрать дублирование решений между `MaxLenContext` и `_resolve_dispatch` либо заставить оба использовать общий resolver.
5. Явно определить PPL как one-pass/no-reasoning и документировать warning/error для strict reasoning model.
6. Исправить boundary case, чтобы всегда оставался положительный prompt budget или выдавалась точная ошибка о невозможной конфигурации.
7. Сохранять configured и effective reasoning budgets отдельно; временный trim не должен уничтожать исходное значение.

#### Тесты

- Hybrid disabled получает полный prompt remainder.
- Hybrid enabled резервирует upper bound.
- Trim выше floor сохраняет reasoning.
- Trim ниже floor отключает reasoning только для текущей задачи.
- Strict reasoning disabled падает до dataset loading.
- PPL не резервирует reasoning tokens.
- State восстанавливается после exception.

#### Acceptance

- Одинаковый plain execution mode получает одинаковый prompt budget независимо от того, plain это модель или отключённый hybrid.
- Effective budgets записаны в task params.

### P0.5. API batch failure нарушает alignment

#### Наблюдение

[`APIBackend.generate_batch`](../llmtf/backends/api.py) и `calculate_tokens_proba_batch` сохраняют исходные индексы во время thread execution, но затем фильтруют ошибки и возвращают укороченные lists. [`Evaluator`](../llmtf/evaluator.py) связывает outputs с samples по позиции.

Если запрос с индексом 0 упал, output исходного sample 1 будет оценён как output sample 0. Это хуже явного падения, потому что создаёт правдоподобные, но неверные метрики.

#### Целевое поведение

Evaluation integrity важнее partial completion. По умолчанию batch должен fail closed.

#### План исправления

1. Создать `BackendBatchError` с индексами и исходными exceptions.
2. После завершения futures проверять все позиции.
3. Если есть ошибки — не фильтровать результаты, а поднять единое exception.
4. Добавить ограниченные retries только для явно retriable HTTP/network statuses.
5. Добавить request timeout; бесконечное ожидание запрещено.
6. Не превращать context-length server error в пустой prediction: считать это configuration error.
7. Удалить permanent fallback token count = 0. Ошибка `/tokenize` должна либо остановить корректный run, либо использовать явно настроенный local tokenizer.
8. Evaluator должен помечать задачу failed и не писать `_total.jsonl`.

#### Тесты

- Один failed future в середине batch вызывает `BackendBatchError` с правильным индексом.
- Ни один sample не оценивается после partial batch failure.
- Retry сохраняет порядок.
- Context-length error не превращается в metric sample.
- Tokenize failure не кешируется как вечный нулевой размер.

#### Acceptance

- Невозможно получить `_total.jsonl` из неполного API batch.
- Benchmark process возвращает non-zero status при task failure.

### P0.6. Reasoning provenance отсутствует, cache identity недостаточна

#### Наблюдение

`LLM.get_params()` проксирует только backend params. Результаты не фиксируют model kind, thinking mode, reasoning budgets и end token id.

Cache проверяет только существование `<run_name>_total.jsonl`. Plain, hybrid, разные templates, few-shot counts и generation settings могут использовать один файл.

#### Целевое поведение

Каждый run имеет canonical, сериализуемую конфигурацию и fingerprint. Кеш можно использовать только при совпадении fingerprint.

#### План исправления

1. Добавить `LLM.get_reasoning_params()` или расширить `get_params()` additive-полем `_llmtf_reasoning`.
2. Сформировать `run_config` из:
   - model identity и backend class;
   - backend constructor/effective parameters;
   - conversation template identity;
   - model kind и thinking mode;
   - configured/effective reasoning budgets;
   - end token id и truncation behaviour;
   - task name, few-shot count, sample limit, scoring method;
   - effective generation config;
   - framework schema version.
3. Удалить secrets перед сериализацией.
4. Канонически сериализовать config и вычислять стабильный hash.
5. Записывать config/hash в `_params.jsonl` и `_total.jsonl`.
6. При найденном `_total.jsonl`:
   - skip только при совпадающем hash;
   - при несовпадении выдать error с предложением `--force_recalc`, другого output dir или `--name_suffix`;
   - не перезаписывать молча.
7. Сохранить существующие filenames для совместимости на первом этапе.

#### Тесты

- Стабильный hash при одинаковом config и различный при изменении каждого существенного поля.
- Порядок keys не влияет на hash.
- API keys и другие secrets отсутствуют.
- Cache hit/mismatch/force-recalc.
- Configured и effective reasoning budgets различимы.

#### Acceptance

- По одному `_params.jsonl` можно восстановить способ запуска.
- Plain и reasoning run не используют один кеш без явного решения пользователя.

### P0.7. Docker/install profiles созданы как draft, но ещё не валидированы

#### Наблюдение

Старый корневой `Dockerfile` всегда:

- использует GPU-oriented `nvcr.io/nvidia/pytorch:26.02-py3`;
- устанавливает единый `requirements.txt`, включающий HF и vLLM зависимости;
- проверяет `torch`, `flash_attn`, `transformers` и `vllm` одним build step.

Добавлен первый вариант новых профилей:

- [`docker/Dockerfile.api`](../docker/Dockerfile.api);
- [`docker/Dockerfile.hf`](../docker/Dockerfile.hf);
- [`docker/Dockerfile.vllm`](../docker/Dockerfile.vllm);
- [`requirements/profiles/`](../requirements/profiles/);
- [`.dockerignore`](../.dockerignore);
- build/run guide [`docker/README.md`](../docker/README.md).

Новые GPU-профили не используют NGC: HF собирается на публичных CUDA 12.9
devel/runtime images, а vLLM расширяет локально собранный HF image. В HF входят
`flash-attn`, `flash-linear-attention` и `causal-conv1d`, необходимые для
ускоренных Qwen3.5 paths. Torch и vLLM выровнены на CUDA 12.9, в отличие от
исследованного legacy image, где были смешаны cu128 и cu129.

Проверка wheel metadata также показала, что vLLM 0.21 требует
`opencv-python-headless>=4.13`, а этот пакет на Python 3.9+ требует NumPy 2.
Поэтому новые profiles используют общий диапазон `numpy>=2,<2.3`; старое
ограничение `numpy<2` остаётся только в legacy `requirements.txt`
и несовместимо с новым vLLM profile. Pure-logic suite проходит на NumPy 2.2.6,
но реальные task/backend runs всё ещё входят в validation gate.

Профили пока не считаются готовыми: clean builds и реальные GPU checks не
выполнялись, а корневой legacy Dockerfile ещё не удалён. Кроме того, одного
разделения requirements недостаточно, потому что API execution path сейчас
имеет нарушения dependency boundaries:

- [`evaluate_model_api.py`](../evaluate_model_api.py) напрямую импортирует `torch`, хотя не использует его;
- [`llmtf/evaluator.py`](../llmtf/evaluator.py) безусловно импортирует torch для seed setup;
- [`llmtf/backends/__init__.py`](../llmtf/backends/__init__.py) eagerly импортирует все concrete backends;
- удалённый `llmtf/backends/_common.py` смешивал torch, peft, transformers и API dependencies;
- `APIBackend` использует `transformers.GenerationConfig` через общий import layer;
- import всего task registry может транзитивно потребовать зависимости задач, не нужных конкретному API run.

Отдельный blocker — фактическая совместимость draft pins с зафиксированными
Qwen3.5 checkpoints. Успешный импорт библиотек не доказывает, что конкретная
архитектура загружается и использует нужные kernels. Нельзя менять версии только
по формальному номеру: рабочий набор определяется model-level probe, после чего
закрепляется воспроизводимыми версиями.

Работа может выполняться с разных машин. Поэтому наличие или отсутствие пакета на конкретном host не должно попадать в документацию как свойство фреймворка. Поддерживаемая среда определяется Docker target и его проверками.

#### Целевые installation profiles

1. **`api`**
   - CPU-only base image;
   - нет torch, vLLM, CUDA toolkit и GPU runtime requirement;
   - содержит framework core, task/evaluation dependencies и HTTP client;
   - поддерживает `evaluate_model_api.py`, API generate и token probability;
   - PPL и local model loading недоступны и дают понятное capability error.
2. **`hf`**
   - GPU/CUDA PyTorch base;
   - включает всё из logical API/common dependency profile плюс transformers, accelerate, peft и local HF runtime;
   - поддерживает HF generate, token probability и PPL;
   - не обязан содержать vLLM.
3. **`vllm`**
   - расширяет Docker/dependency profile `hf`;
   - добавляет vLLM и необходимые vLLM/flash-attention проверки;
   - поддерживает local vLLM и запуск vLLM OpenAI server;
   - является полным образом для всей validation matrix.

`vllm` должен расширять `hf`, а не копировать его setup. API и HF могут использовать разные base images, но requirements должны иметь явную иерархию общих зависимостей.

#### Реализованная draft-структура

Три явных Dockerfile и разделённые requirements:

```text
docker/
  Dockerfile.api
  Dockerfile.hf
  Dockerfile.vllm
requirements/profiles/
  common.txt
  api.txt
  hf.txt
  constraints-cu129.txt
```

Рекомендуемая зависимость файлов:

```text
api.txt  -> common.txt
hf.txt   -> api.txt + HF runtime
Dockerfile.vllm -> ранее собранный HF image + vLLM runtime
```

Верхнеуровневые `Dockerfile` и `requirements.txt`
пока оставлены как legacy compatibility path. Их дальнейшее удаление или
перенаправление должно быть отдельным решением после валидации новых образов.

Команды сборки:

```bash
docker build -f docker/Dockerfile.api -t llmtf:api .
docker build -f docker/Dockerfile.hf -t llmtf:hf-cu129 .
docker build -f docker/Dockerfile.vllm --build-arg HF_BASE_IMAGE=llmtf:hf-cu129 -t llmtf:vllm-cu129 .
```

#### План исправления

1. Построить import/dependency graph для CLI, core и task modules.
2. Удалить неиспользуемый `torch` import из API CLI.
3. Сделать torch seeding в evaluator optional: импортировать torch только при наличии/local backend необходимости, не ослабляя reproducibility HF/vLLM runs.
4. Разделить `_common.py` на явные backend-specific imports и lightweight shared helpers.
5. Сделать `llmtf.backends` lazy или организовать exports так, чтобы импорт `APIBackend` не импортировал HF/vLLM modules.
6. Убрать обязательный transformers dependency из API core:
   - предпочтительно ввести framework-owned sampling config dataclass/protocol;
   - либо создать лёгкий API config object с тем же структурным контрактом;
   - HF может продолжать адаптировать `transformers.GenerationConfig` на своей границе.
7. Проверить task registry на безусловные torch/HF imports. Опциональные tasks должны регистрироваться по capabilities или выдавать локальную понятную ошибку, а не ломать весь API CLI import.
8. Проверить и при необходимости скорректировать уже разделённые requirements. Для новых profiles сохранить согласованный диапазон `numpy>=2,<2.3` и `datasets<4`; не подмешивать legacy `requirements.txt` с `numpy<2`.
9. Довести три draft Dockerfile до validated build profiles:
   - `api` на CPU Python base;
   - `hf` на согласованном CUDA/PyTorch base;
   - `vllm FROM hf` с vLLM additions.
10. Выполнить HF и vLLM load/generate compatibility probe на
    `Qwen/Qwen3.5-2B`; если draft pins не работают, найти совместимый набор из
    официальных releases/nightly, затем заменить floating dependency точным
    commit/wheel URL и hash.
11. Повторить load probe для `Qwen/Qwen3.5-2B-Base` в foundational mode.
12. Проверить существующие profile-specific build-time smoke checks на чистых сборках.
13. Проверить новую `.dockerignore`, чтобы outputs, model artifacts, credentials и `.git` не попадали в build context без необходимости.
14. Не копировать API credentials в image. Endpoint/key/model задаются только на `docker run`/orchestrator runtime.
15. Документировать mount points для source tree, model weights, HF cache и output directory.
16. Добавить CI или локальный script, который последовательно собирает три targets и выполняет profile tests.

#### Profile tests

**API target:**

- `import llmtf.llm`, `APIBackend`, `Evaluator` проходит;
- `import torch` и `import vllm` отсутствуют/не требуются;
- `evaluate_model_api.py --help` работает;
- mocked HTTP generate/probability tests проходят;
- реальный API smoke использует runtime credentials;
- попытка создать HF/vLLM backend даёт понятное missing-extra/capability сообщение, а не package import crash.

**HF target:**

- torch видит CUDA на GPU host;
- transformers/accelerate/peft imports проходят;
- `VLLMBackend` не требуется для HF smoke;
- tiny HF generate/probability/PPL проходят.

**vLLM target:**

- наследует успешные HF checks;
- vLLM и flash-attention imports проходят;
- tiny local vLLM generate/probability проходят;
- локальный API server стартует и проходит API smoke.

#### Credentials и секреты

1. Предпочтительный источник API key — environment variable/secret mount.
2. CLI `--api_key` следует сделать optional compatibility path; он не должен печататься в generated command logs.
3. Params/provenance sanitizer обязан удалять `api_key`, authorization headers и известные secret-like fields.
4. Тестовые команды в документации используют placeholder variable, а не значение ключа.
5. Внешний endpoint и model name можно сохранять только если это не раскрывает секретную инфраструктуру; по умолчанию поддержать redaction.

#### Acceptance

- Все три Docker profiles собираются из чистого checkout.
- `api` image работает на CPU host без NVIDIA runtime и не содержит обязательных torch/vLLM imports.
- `hf` image выполняет tiny HF matrix на GPU и не требует vLLM.
- `vllm` image строится поверх `hf` и выполняет local vLLM + local API matrix.
- Обе Qwen3.5 модели реально загружаются; точные package/wheel hashes сохранены,
  floating `main`/`nightly` для runtime dependencies в validated profile нет.
- API credentials передаются только runtime и не обнаруживаются в `docker history`, logs или result JSON.
- README содержит отдельные build/run команды для трёх profiles.

## 5. P1: укрепление контрактов и расширяемости

### P1.1. Формализовать batch result и info schema

Сейчас backend contract описывает tuple `(prompts, outputs, infos)`, но shape полей не типизирован. Reasoning напрямую ожидает:

- `prompt_len`;
- `generated_len`, причём для generate это list;
- `generated_cumulative_logprob`;
- для probability continuation — `generated_token`.

Новый backend может удовлетворить ABC и всё равно упасть внутри orchestration.

План:

1. Ввести `TypedDict` или dataclasses для `GenerationInfo`, `TokenProbabilityInfo`, `BatchResult`.
2. Нормализовать `generated_len` до одного документированного типа.
3. Добавить runtime validation на boundary backend -> LLM с понятным exception.
4. Сначала сохранить tuple-compatible adapter, затем мигрировать внутренний код.
5. Описать semantics `prompts`: reasoning phase prompt против continuation prompt.
6. Проверять равенство batch lengths перед возвратом evaluator.

### P1.2. Довести `BaseLLM` и `Backend` contracts до фактического использования

План:

1. Сверить методы, вызываемые evaluator/tasks, с `BaseLLM`.
2. Добавить в contract `add_stop_strings`, `reset_stop_strings` и HF-only logsoftmax semantics либо явно вынести capability protocol.
3. Добавить default `NotImplementedError` для optional debug helpers в `Backend`, чтобы новый backend не давал случайный `AttributeError`.
4. Заменить строковый `support_method` на capabilities enum/set, сохранив compatibility wrapper.
5. Добавить contract tests для minimal fake backend.

### P1.3. Определить поведение `num_return_sequences` в reasoning

Текущая стратегия ожидает один reasoning text, но backends возвращают list при `num_return_sequences > 1`.

Для v4 минимальное безопасное решение:

1. При two-pass reasoning требовать `num_return_sequences == 1`.
2. Проверять до первого backend call.
3. Выдавать понятный `NotImplementedError` с объяснением неоднозначности ветвления continuation.

Поддержка дерева N reasoning branches -> N continuations может быть отдельным feature после стабилизации.

### P1.4. Сделать reasoning format действительно конфигурируемым

`ReasoningFormat` существует, но `_setup_reasoning` фиксирует close marker framework-константой.

План:

1. Разрешить передавать `ReasoningFormat` в Python API.
2. Для CLI дать JSON/file-based format config только после появления второго реального формата; не добавлять множество преждевременных flags.
3. Валидировать непустой close marker и согласованность end token id.
4. Не переносить format logic в backend.
5. Добавить fake-format two-pass test без буквального использования специальных model tokens в документации.

### P1.5. Развязать зависимости backend-модулей

Удалённый `llmtf/backends/_common.py` импортировался через `*` всеми backends и тянул `torch`, `transformers`, `peft`, requests и прочее даже для API-only use.

Блокирующая часть этой работы входит в P0.7, потому что без неё нельзя собрать API-only image. Здесь остаётся последующая структурная чистка после того, как profile boundaries уже подтверждены tests.

План:

1. Заменить wildcard imports явными imports.
2. API backend не должен импортировать torch/peft.
3. HF backend не должен зависеть от vLLM.
4. Вынести только реально общие lightweight helpers в нейтральный module.
5. После этого решить, нужен ли `TokenizerBackendMixin` для общего HF/vLLM кода.
6. Не объединять различающиеся stop-string semantics насильно: оставить override points.
7. Добавить import tests в минимальных dependency profiles.

### P1.6. Честно определить совместимость API backend

Реализован один `APIBackend` с простыми профилями `auto`, `openai` и `vllm`.
Консервативные профили отправляют стандартный OpenAI-compatible payload;
vLLM-only поля включаются только явным профилем. `/v1/models`, `/tokenize` и
`/detokenize` больше не являются общими startup requirements.

Принятый контракт:

1. `auto` безопасно проверяет только optional discovery/token-count endpoints и
   не выводит из их наличия поддержку assistant continuation.
2. `openai` не делает `/tokenize` probe; `vllm` включает известные расширения.
3. Отсутствующий token counter возвращает `None`, не ноль и не эвристику;
   few-shot prompt не урезается скрыто.
4. Assistant-prefill требует объявленного continuation, а `portable` запрещает
   prefill полностью.
5. Provider-native transports, reasoning и count-only endpoints в будущем
   добавляются внутренними adapters, не новыми model facades.
6. Подробный пользовательский контракт находится в `docs/api_backend.md`.

### P1.7. Ошибки должны доходить до process exit status

Сейчас evaluator перехватывает многие exceptions, а API CLI завершает процесс через `os._exit(0)`.

План:

1. Ввести `EvaluationSummary` с succeeded/skipped/failed tasks.
2. На уровне dataset можно собрать несколько ошибок, но финальный CLI exit code должен быть non-zero, если есть failed tasks.
3. Не писать success total для failed task.
4. Удалить `os._exit(0)` после выяснения исходной причины либо вызывать его только с вычисленным status после flush/cleanup.
5. Benchmark workers должны прекращать бесконечный retry для deterministic configuration errors.
6. Разделить retriable resource errors и permanent errors.

## 6. P2: документация и последующий технический долг

### P2.1. README и примеры

[`README.md`](../README.md) сейчас одновременно:

- ссылается на удалённый `llmtf/model.py`;
- показывает удалённый `--max_prompt_len`;
- ниже утверждает, что `--max_prompt_len` удалён;
- описывает старые model classes;
- ссылается на notebook examples, удалённые в worktree.

После стабилизации CLI необходимо переписать README по фактическим smoke-командам. До этого не следует «чинить» команды несколько раз вслед за меняющимся интерфейсом.

### P2.2. Task scaffolding

Пункт `todo` о шаблонном добавлении задач остаётся актуальным. После фиксации `BaseLLM` contract:

1. добавить минимальный `examples/tasks/minimal_task.py`;
2. добавить checklist регистрации и `_max_task_new_tokens`;
3. добавить unit fixture с in-memory dataset;
4. не возвращать большой notebook как единственный источник документации.

### P2.3. Bootstrap rewrite

Bootstrap не относится к model refactor. Вынести в отдельный plan после v4. Требуется определить:

- reproducible random state;
- confidence interval method;
- поведение aggregations, возвращающих details;
- сериализацию bootstrap metadata.

### P2.4. Старые runners и scripts

После успешной матрицы:

- пометить `run_evaluate_singlenode_multigpu.py` deprecated или удалить отдельным commit;
- проверить shell scripts на новые flags;
- решить судьбу `remap_qwen35_checkpoint.py`;
- отдельно решить судьбу `prompt_optimizer`, examples и external submodules.

## 7. План исполнения по этапам

Каждый этап должен быть отдельным reviewable commit или небольшой серией commits. Не смешивать исправления ядра с удалением unrelated файлов.

### Этап 0. Зафиксировать baseline worktree

1. Сохранить список tracked/untracked изменений.
2. Отделить model refactor files от unrelated deletions и mode changes.
3. Явно решить, что делать с `examples/`, `prompt_optimizer/`, external submodules и executable bits.
4. Зафиксировать текущий pure-logic baseline.
5. Создать отдельный detached worktree старой реализации на exact commit
   `504bd7fc2793c900e97010f124651460afcd4812` (текущий `origin/main`), не
   переключая и не очищая рабочий каталог с рефакторингом.
6. Выделить разные output roots для `legacy` и `refactor`; запрещено использовать
   result cache одной реализации в другой.

Результат: архитектурный diff можно ревьюить и бисектить.

### Этап 1. Добавить regression tests для известных P0 bugs

До исправлений добавить падающие tests для:

- vLLM/API phase-local stop token;
- backend kwargs precedence;
- hybrid thinking-disabled budget;
- API batch alignment;
- missing reasoning provenance;
- benchmark command propagation.

Результат: каждая найденная проблема воспроизводится без GPU, где это возможно.

### Этап 2. Реализовать dependency boundaries и Docker profiles

Scope: P0.7 и блокирующая часть P1.5.

1. Развязать API imports от torch/HF/vLLM.
2. Проверить draft-разделение requirements и убрать отсутствующие/лишние зависимости.
3. Собрать три добавленных Dockerfile `api`, `hf`, `vllm` из чистого checkout.
4. Проверить profile-specific import/build smoke tests.
5. Подтвердить, что `api` работает без NVIDIA runtime.

Результат: последующие исправления можно проверять в трёх воспроизводимых средах, а не в случайном host setup.

### Этап 3. Исправить phase-local generation config

Scope: P0.2 и guard для `num_return_sequences` из P1.3.

Результат: двухфазный flow получает одинаковые stop semantics на трёх backends.

### Этап 4. Нормализовать reasoning execution mode и benchmark config

Scope: P0.1 + P0.4.

1. Общий execution-mode resolver.
2. Thinking-aware MaxLenContext.
3. Положительный CLI flag с compatibility alias.
4. Общий benchmark config loader.
5. Миграция YAML и удаление silently ignored keys.

Результат: direct CLI и benchmark запускают один и тот же режим.

### Этап 5. Исправить backend kwargs

Scope: P0.3.

Результат: documented precedence, strict validation, tests для каждого backend.

### Этап 6. Исправить error integrity и exit statuses

Scope: P0.5 + P1.7.

Результат: partial failures не создают метрики и приводят к non-zero benchmark status.

### Этап 7. Добавить provenance и cache validation

Scope: P0.6.

Результат: каждый run воспроизводим, несовместимый cache не используется молча.

### Этап 8. Укрепить backend contract

Scope: P1.1, P1.2, P1.4, P1.5, P1.6.

Этот этап допустимо разбить на несколько commits:

1. typed result/info schemas;
2. capabilities и optional debug API;
3. окончательная cleanup dependency isolation и shared helpers;
4. configurable reasoning format;
5. API capability documentation/probing.

### Этап 9. Обновить документацию

1. README.
2. Benchmark YAML reference.
3. Task creation guide.
4. Migration notes для v3 -> v4.
5. Удалить выполненные пункты из `todo` или заменить ссылкой на отдельные планы.
6. Актуализировать build/run commands и mount/secret contract для Docker profiles.

### Этап 10. Реальная validation matrix

Выполняется только после прохождения всех dependency-free tests. До запуска
создать [`TEST_PLAN_v4.md`](TEST_PLAN_v4.md) с полностью подставляемыми командами
для нового кода и отдельным разделом команд, которые пользователь запускает в
legacy worktree. План должен фиксировать model names, выбранные datasets, Docker
image ids, generation settings, seeds, output roots и правила сравнения artifacts.

## 8. Обязательная validation matrix

Все реальные запуски делать на моделях из §2.4, одной conversation
template и одинаковых generation settings, где это возможно. На RTX 4090 не
пытаться использовать заявленный моделью максимальный context автоматически:
выбрать один явно записанный практичный `model_context_len` после smoke probe
(начальная кандидатура — 8192) и использовать его во всех сопоставимых runs.
Использовать отдельные output dirs или config fingerprint.

### 8.0. Docker/profile gate

До сравнения model outputs собрать profiles из чистого build context:

```bash
docker build -f docker/Dockerfile.api -t llmtf:api .
docker build -f docker/Dockerfile.hf -t llmtf:hf-cu129 .
docker build -f docker/Dockerfile.vllm --build-arg HF_BASE_IMAGE=llmtf:hf-cu129 -t llmtf:vllm-cu129 .
```

Проверить:

- `api` запускается без `--gpus` и без NVIDIA runtime;
- `api` выполняет unit/API-client tests без torch/vLLM;
- `hf` запускается с `--gpus` и проходит CUDA/HF smoke;
- `vllm` проходит все HF checks плюс vLLM imports/runtime;
- source checkout, model cache и outputs подключаются mounts, а не запекаются в image;
- API secrets отсутствуют в image layers и передаются только runtime.

### 8.1. Hybrid instruct: `Qwen/Qwen3.5-2B`

Матрица выполняется отдельно при `model_kind=hybrid` с thinking disabled и
enabled. Enabled-run использует explicit end token id, одинаковые reasoning
budgets и `num_return_sequences=1`.

| Backend | generate, off | proba, off | generate, on | proba, on | PPL |
|---|---:|---:|---:|---:|---:|
| HF | required | required | required | required | required, без reasoning-фазы |
| vLLM | required | required | required | required | expected unsupported |
| API against same local vLLM | required | required | required | required | expected unsupported |

PPL в текущем контракте — HF-only mean answer-token log probability и не имеет
thinking/reasoning phase. Для vLLM/API обязательный тест функции состоит в
проверке явного стабильного capability error. Реализация PPL для этих backends
является отдельным расширением scope и не должна имитироваться через другой
метод.

Дополнительно выполнить один strict `model_kind=reasoning` smoke для generate и
proba на каждом backend и один negative test для hybrid без end token id. Это
проверяет contract, но не подменяет основную off/on матрицу hybrid-модели.

### 8.2. Foundational: `Qwen/Qwen3.5-2B-Base`

Запускать с `--is_foundational`/эквивалентной typed config и
`conversation_configs/default_foundational.json`. Для API setup этот режим
должен быть явно передан в server/client path, а не предполагаться по имени.

| Backend | generate | token proba | PPL |
|---|---:|---:|---:|
| HF | required | required | required |
| vLLM | required | required | expected unsupported |
| API against same local vLLM | required | required | expected unsupported |

Проверить token counts, отсутствие chat assistant-template artifacts и то, что
foundational run не включает reasoning orchestration. Если API entry point не
умеет передать foundational semantics, это implementation blocker, а не
основание пропустить cell.

### 8.3. Размер и задачи быстрого прогона

- `--max_sample_per_dataset 8`, `few_shot_count=0`, deterministic generation;
- минимум одна generate-задача;
- минимум одна `calculate_tokens_proba`-задача;
- PPL запуск на HF с тем же ограничением samples;
- после smoke выполнить более длинный comparison-run только если ожидаемая
  длительность заранее оценена и согласована с пользователем.

Имена datasets должны быть записаны в testing plan после
проверки, что каждая задача действительно вызывает нужный метод. Нельзя считать
generate-задачу проверкой probability path или наоборот.

### 8.4. Failure tests

- Один HTTP request падает внутри batch.
- `/tokenize` недоступен: user-final generation продолжает работать, а
  whitespace-prefill probe даёт явную compatibility error.
- Контекст меньше answer budget.
- Cache fingerprint не совпадает.
- Unknown backend kwarg.
- Противоречивые thinking flags.
- Two-pass + `num_return_sequences > 1`.

### 8.5. Что сохранять после проверки

- точные команды;
- версии torch/transformers/vLLM/CUDA;
- stdout/stderr или evaluation log;
- `_params.jsonl`;
- `_total.jsonl`;
- краткое сравнение HF/vLLM/API;
- известные допустимые расхождения.

### 8.6. Сравнение со старой реализацией

Testing plan обязан содержать готовую инструкцию пользователю, а не только
описание намерения. Базовый безопасный способ получить старый код:

```bash
git worktree add --detach /path/to/llmtf_legacy \
  504bd7fc2793c900e97010f124651460afcd4812
git -C /path/to/llmtf_legacy rev-parse HEAD
```

Перед реальным запуском агент должен заменить `/path/to/...` на согласованные
абсолютные paths и выдать отдельные legacy-команды для каждой сравниваемой cell.
Старый и новый source tree по возможности монтируются в один и тот же validated
HF/vLLM image и используют общий model cache/path. Это изолирует изменение
framework code. Если legacy code несовместим с новым runtime и требует отдельный
image, точные package versions фиксируются, а различие runtime указывается как
confounder.

Соответствие режимов старого CLI:

- local legacy CLI всегда создаёт reasoning-capable facade; `--disable_thinking`
  соответствует hybrid/off, отсутствие флага — hybrid/on;
- legacy API CLI при `--disable_thinking` создаёт plain facade, без флага —
  reasoning facade;
- legacy `--max_prompt_len` не равен новому deployment context contract; выбрать
  значения так, чтобы effective prompt/answer budgets совпадали, и записать
  mapping в testing plan;
- foundational legacy run использовать только там, где entry point действительно
  поддерживает его; отсутствие поддержки фиксировать отдельно.

Условия честного сравнения:

1. Одинаковые model files/cache, tokenizer/chat template, dataset snapshot и sample ids.
2. Одинаковые seed, deterministic generation, few-shot count, budgets и penalties.
3. Раздельные output roots: например `results/legacy/<commit>/...` и
   `results/refactor/<commit-or-fingerprint>/...`.
4. Сначала сравнить количество и ids samples, затем per-sample inputs/outputs,
   затем aggregate metrics; несовпавший alignment запрещает сравнивать totals.
5. Для deterministic HF output требовать exact match там, где runtime один.
   Для floating log probabilities хранить absolute/relative deltas и установить
   tolerance после первого контрольного run, не скрывая backend-dependent drift.
6. vLLM/API результаты сравнивать с HF reference и legacy counterpart; различать
   framework regression, численную backend-разницу и template/server difference.
7. Не удалять legacy worktree и raw artifacts до принятия comparison report.

## 9. Dependency-free quality gate

Минимум для каждого commit:

```bash
python3 tests/test_refactor_logic.py
python3 -m compileall -q llmtf evaluate_model.py evaluate_model_api.py benchmark
git diff --check
```

Если установлен pytest:

```bash
python3 -m pytest tests/test_refactor_logic.py -q
```

Нужно добавить отдельные быстрые suites, не требующие model weights:

- `tests/test_cli_config.py`;
- `tests/test_benchmark_config.py`;
- `tests/test_backend_contract.py`;
- `tests/test_api_backend.py`;
- `tests/test_run_provenance.py`.

Не следует продолжать бесконечно расширять один `test_refactor_logic.py`.

## 10. Backward compatibility и миграция

### Сохраняем

- `LLM(backend=...)` construction;
- `model_kind` values `plain/reasoning/hybrid`;
- output filenames на первом этапе;
- tuple return adapter до завершения typed-result migration;
- `--disable_thinking` как временный deprecated alias;
- legacy YAML `extra_args.think` на один переходный цикл.

### Меняем осознанно

- thinking становится explicit opt-in по умолчанию;
- unknown backend/YAML keys становятся errors;
- incompatible cache становится error, а не silent skip;
- partial API batch становится failed task;
- CLI получает non-zero exit status при failed tasks;
- two-pass + multiple return sequences явно запрещается до отдельной реализации.

### Не делаем в v4

- auto-detection model kind;
- PPL на vLLM/API;
- native provider-specific reasoning content API;
- branch tree для нескольких reasoning candidates;
- bootstrap rewrite;
- полную замену benchmark scheduler;
- автоматическое удаление historical plans или unrelated directories.

## 11. Протокол следующего GPU/API-сеанса

Ожидаемый сценарий продолжения работы: агент запускается пользователем из
контейнера на машине с RTX 4090, получает ссылку на этот план и доступ к API
credentials. Модели уже выбраны в §2.4.

### 11.1. Что пользователь передаёт

- путь к [`REFACTOR_PLAN_v4.md`](REFACTOR_PLAN_v4.md) или указание следовать ему;
- активный Docker target/container (`hf` или `vllm`; для API-only проверки — `api`);
- API base URL и model name;
- API key через environment/secret mount, а не сообщением, которое затем копируется в команды или документы;
- доступные GPU ids и допустимый объём ресурсов, если это нельзя определить безопасно.

### 11.2. Что агент проверяет в начале

1. `git status` и текущий commit, не изменяя пользовательский worktree.
2. Какой Docker profile фактически запущен.
3. Версии Python, CUDA, torch, transformers и vLLM только там, где они должны присутствовать.
4. Доступность GPU через read-only checks.
5. Наличие required API variables без вывода их значений.
6. Dependency-free tests до model loading.
7. Свободные output paths; предыдущие результаты не переиспользуются без fingerprint check.
8. Доступность обеих моделей через общий cache/path для всех сравниваемых runs.

Отсутствие torch/vLLM в `api` target является успешным свойством profile, а не blocker. Отсутствие этих пакетов в `hf`/`vllm` target — ошибка сборки.

### 11.3. Порядок реальной проверки

1. Проверить model/runtime compatibility и при необходимости закрепить рабочие
   Transformers/vLLM builds до полной оценки.
2. HF hybrid/off: generate, token probability, PPL.
3. HF hybrid/on: generate и token probability с explicit end token id.
4. Local vLLM hybrid off/on: generate и token probability; PPL capability error.
5. Поднять локальный vLLM API server и повторить APIBackend off/on на том же
   checkpoint; проверить PPL capability error.
6. Повторить foundational matrix на Base checkpoint через HF, vLLM и API.
7. Затем проверить предоставленный внешний API endpoint как отдельный compatibility test.
8. Выполнить legacy comparison protocol из §8.6 и подготовить пользователю
   точные команды для тех baseline runs, которые он должен запустить сам.
9. Выполнить failure tests, не раскрывая credentials.
10. Сохранить sanitized commands, versions, params и результаты в отдельном validation report.

### 11.4. Правила работы с credentials

- Никогда не выполнять команды, печатающие всё окружение.
- Не включать secret value в shell history, process title, generated benchmark command или traceback.
- Не добавлять credentials в `--backend_kwargs`, YAML, `_params.jsonl` или validation report.
- При необходимости передавать key в процесс через environment variable с redaction в logging.
- Перед публикацией artifacts выполнить automated secret-key scan по изменённым текстовым файлам.

### 11.5. Результат сеанса

Создать короткий versioned validation report, содержащий:

- commit/config fingerprint;
- Docker target/image id без registry credentials;
- package/CUDA versions;
- выполненные matrix cells;
- ссылки на sanitized output directories;
- pass/fail и обнаруженные backend divergences;
- оставшиеся blockers.

Наличие GPU и credentials разрешает провести тесты, но не разрешает менять внешние сервисы, публиковать images или отправлять результаты третьим сторонам без отдельного запроса пользователя.

## 12. Definition of done

Model/reasoning/backend refactor считается завершённым только если одновременно выполнены все условия:

1. Все P0 проблемы исправлены и покрыты regression tests.
2. Benchmark способен явно запустить plain и two-pass reasoning через все три backends.
3. Phase-local end token подтверждён payload-level tests и реальными runs.
4. `backend_kwargs` имеет проверенную семантику приоритетов и не игнорирует опечатки.
5. Context budget совпадает с фактическим execution mode.
6. API partial failure не может породить смещённые метрики.
7. `_params.jsonl` содержит reasoning/evaluation provenance и fingerprint.
8. Cache validation не смешивает несовместимые runs.
9. CLI/benchmark возвращают non-zero при failed tasks.
10. Dependency-free suites и compile checks проходят.
11. Docker profiles `api`, `hf`, `vllm` собираются и проходят свои profile tests.
12. API profile не требует torch, vLLM, CUDA или GPU runtime; vLLM image расширяет HF profile.
13. Реальная validation matrix выполнена из воспроизводимых containers и задокументирована.
14. Credentials не присутствуют в images, logs, params или repository files.
15. README, AGENTS и benchmark YAML соответствуют коду.
16. Refactor разделён на reviewable commits; unrelated deletions и mode changes решены отдельно.
17. Полная §8 matrix выполнена на выбранных instruct/Base checkpoints: три backends,
    hybrid thinking off/on, foundational mode, generate, token probability и HF
    PPL; unsupported PPL cells дают явную capability error.
18. Выпущен `TEST_PLAN_v4.md` с точными legacy-командами, а результаты старой и
    новой реализаций сопоставлены на одинаковых sample ids.

До выполнения этих условий формулировка «архитектура реализована» допустима, а «рефакторинг завершён и безопасен» — нет.
