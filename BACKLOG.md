# Backlog

Этот файл содержит задачи, обнаруженные во время стабилизации refactor v4, но
не входящие в текущую стабильную версию. Snapshot-specific планы и отчёты
хранятся в `dev/`; реализованные пункты следует переносить из этого файла в
соответствующую документацию и тесты.

## Высокий приоритет: task layer и внешние задачи

### 1. Архитектурная доработка task layer

Корректностная стабилизация завершена и зафиксирована в
`dev/TASK_BUGFIX_REPORT.md`: базовый task/evaluator contract валидируется,
task identity участвует в cache fingerprint, а найденные дефекты встроенных
задач исправлены с migration notes и regression evidence. Следующий этап не
блокирует bugfix-коммит и сознательно оставлен как архитектурная работа:

- сократить дублирование dataset/few-shot/prompt-budget логики между
  историческими задачами без изменения их метрик;
- формализовать стабильный минимальный публичный API расширения задач поверх
  уже проверяемого внутреннего контракта;
- завершить ленивую загрузку optional tasks и внешних зависимостей, не полагаясь
  на import side effects;
- определить декларативные dataset/sample schemas и единый слой сообщений об
  ошибках для task-specific полей и split conventions;
- покрыть будущий публичный API отдельными contract tests до реализации plugin
  loader;
- определить migration policy для prompts и метрик, чтобы архитектурные
  изменения не меняли benchmark baselines молча.

### 2. Регистрация задач из примонтированных каталогов

После стабилизации task-контракта добавить возможность подключать один или
несколько каталогов с пользовательскими задачами вне репозитория. Основной
сценарий — приватная директория, примонтированная в Docker container, без
копирования её исходников в LLMTF и без обязательной публикации в Git.

- определить явный интерфейс подключения для single-model CLI и всех benchmark
  runners (например, repeatable CLI option и соответствующее поле YAML либо
  environment path); значение должно передаваться во все subprocesses;
- загружать задачи через документированный registration entry point/manifest,
  а не произвольное сканирование и импорт каждого Python-файла;
- поддержать несколько директорий, детерминированный порядок загрузки и явную
  политику конфликтов имён; встроенная или ранее зарегистрированная задача не
  должна незаметно переопределяться;
- валидировать внешние task classes тем же контрактом, что и встроенные, и
  выдавать локализованную ошибку с именем plugin/task без падения unrelated
  registrations;
- не устанавливать Python-зависимости автоматически. Пользователь должен
  добавлять их в собственный image/environment, а loader — сообщать понятную
  import error;
- учитывать версию/хеш внешнего task manifest или исходников в provenance и
  cache fingerprint, но не записывать приватный абсолютный путь или содержимое
  исходников в публичные results;
- документировать безопасный Docker mount, программную регистрацию и пример
  приватного task package; проверить HF, local vLLM и API runners;
- добавить dependency-free tests для discovery/validation/conflicts и
  integration smoke с временным внешним каталогом.

### 3. Архивное зеркало benchmark datasets в Hugging Face

Собрать фактически используемые LLMTF данные в публичном dataset repository
`RefalMachine/llmtf_benchmark`, чтобы воспроизводимость benchmark не зависела
от удаления, переименования или несовместимого обновления upstream datasets.
Пошаговая спецификация: `dev/DATASET_MIRROR_PLAN.md`.

- хранить данные физически в repository (предпочтительно sharded Parquet), а не
  оставлять ссылки на upstream loading scripts или внешние download URLs;
- использовать HF dataset configurations для отдельных логических datasets с
  независимыми schemas; сохранять исходные `train`/`validation`/`dev`/`test`
  splits внутри соответствующей configuration;
- не дублировать строки ради `fast`, `full` и `foundational`: хранить эти
  benchmark-варианты как versioned manifests со списком configurations,
  filters, sample limits и task parameters;
- сохранять внутренние срезы вроде MMLU subject/category в явных колонках;
  если upstream использовал отдельные configurations, записывать исходное имя
  в `source_config`, чтобы объединение оставалось обратимым;
- определить стабильную namespacing-схему configuration names и таблицу
  соответствия `LLMTF task -> mirror config -> upstream repo/config/revision`;
- закрепить точные upstream revisions, row counts, schemas и content hashes;
  сохранять авторство, исходную ссылку и citation metadata для каждого
  зеркалируемого набора;
- написать идемпотентный export/upload tool с dry-run, resume и валидацией
  результата через анонимный `load_dataset` для каждой configuration/split;
- после публикации переключить task loaders на зеркало либо добавить единый
  configurable dataset source с mirror по умолчанию и явным upstream fallback;
- добавить CI/maintenance audit, который сверяет manifests, registry,
  доступность mirror configs и отсутствие случайно закоммиченных HF tokens.

### 4. Очистка и обновление RuParam

Текущий публичный snapshot `RefalMachine/RuParam` содержит 9 505 строк и
известные data-quality проблемы, перечисленные в `dev/RUPARAM_DATA_AUDIT.md`.
После получения актуального файла провести отдельную data migration, не смешивая
её с изменениями task scoring.

- сохранить текущий raw snapshot и его revision как воспроизводимый legacy
  baseline; новую редакцию публиковать отдельной revision/version с changelog;
- повторить полный schema/content audit: row count, физические дубликаты,
  неуникальные исходные `id`, пустые поля, одинаковые `gram`/`ungram`, неверный
  `order`, whitespace/control characters и Unicode confusables;
- вручную разобрать либо удалить 100 известных строк с одинаковыми
  `gram`/`ungram`, строку без label и служебные labels `Разметка 1/2/3`; не
  применять эвристическую очистку без отчёта о каждой затронутой строке;
- ввести стабильный уникальный pair id, сохранив исходный id отдельным полем;
- нормализовать source/level/category значения и проверить соответствие новой
  авторской таксономии; если будут доступны 11 336 пар и 150 категорий,
  валидировать их по фактическому файлу, а не зашивать числа заранее;
- сформировать machine-readable migration report: removed/fixed/unchanged rows,
  old-to-new ids, распределения source/level/category и content hashes;
- повторить double-order scoring smoke и category/source/part/level aggregation
  tests на очищенном snapshot;
- изменить dataset revision в LLMTF provenance и явно потребовать новый
  baseline; результаты старой и новой редакций не объединять.

### 5. Leaderboard-категория «Русский язык»

Выделить отдельную категорию языковой компетенции и перенести в неё RuCoLA,
RuParam, RuBLiMP и SLAVA.

- определить точные result ids: `russiannlp/rucola_custom`, `ruparam`,
  канонический режим RuBLiMP (`russiannlp/rublimp-(classify)` и/или
  `russiannlp/rublimp-(choice)`) и будущий стабильный id SLAVA;
- добавить/зарегистрировать SLAVA в benchmark до включения в category config;
  сейчас такой task отсутствует в `TASK_REGISTRY`;
- решить, входят ли оба RuBLiMP-протокола как отдельные равновесные метрики или
  только один канонический протокол, чтобы один dataset не получил двойной вес;
- добавить категорию в `benchmark/categories.json`, удалить переносимые task ids
  из прежних категорий и проверить уникальность членства;
- определить состав категории отдельно для fast/full/foundational, если задачи
  доступны не во всех suites; отсутствующая задача должна отображаться как
  отсутствующая, а не как нулевой результат;
- зафиксировать правило category mean и влияние новой категории на общий mean;
  изменение структуры leaderboard требует новой версии baseline;
- добавить tests на точное членство, отсутствие дублей между категориями и
  корректное формирование таблицы при частично рассчитанном наборе задач.

## API и vLLM deployment

- Оптимизировать layout публикуемых HF/vLLM images. Текущий multi-stage build
  копирует `/opt/venv` одним слоем размером около 10.1 GB без сжатия; образ
  корректен, но первая публикация и восстановление upload после обрыва сети
  неудобны. Разбить dependency groups на переиспользуемые layers либо выбрать
  другой reproducible layout без повторной сборки compiled kernels.
- Добавить model-level `vllm_server_args` в benchmark YAML и безопасное
  преобразование mapping в аргументы запуска vLLM. Не определять reasoning и
  tool parsers автоматически: их значения зависят от модели. Запрещать
  переопределение runner-owned параметров (`model`, `port`, tensor parallel,
  context length, GPU memory, chat template), секреты и неоднозначные значения.
  Сохранять sanitized server manifest рядом с результатами.
- Подготовить отдельный request/response-контракт для function calling:
  task-level `tools`/`tool_choice`/дополнительное тело запроса и структурированный
  `tool_calls` в результате. Одних server parser flags для function-calling
  benchmark недостаточно.
- Пересмотреть исторические vLLM overrides. Отдельно проверить необходимость
  `enable_prefix_caching`, `disable_sliding_window` и чрезмерного
  `max_logprobs=1000000`; defaults и offline/API режимы должны различаться
  только по доказанной причине. Переименовать устаревшее внутреннее поле
  `max_seq_len_to_capture`, которое сейчас фактически задаёт `max_model_len`.

## Local benchmark runner

- Переработать `benchmark/calculate_benchmark.py`, чтобы local HF/vLLM worker
  загружал модель один раз на выделенную GPU-группу и переиспользовал её между
  всеми полученными task groups. Сейчас каждая группа запускает отдельный
  `evaluate_model.py`, поэтому веса повторно загружаются после завершения каждой
  группы.
- Сохранить параллельное выполнение при нескольких GPU: для local vLLM —
  несколько постоянных replicas с явным `tensor_parallel_size`; для HF —
  независимые replicas на отдельных GPU либо явно документированное
  `device_map`-sharding без ошибочного обозначения его как tensor parallel.
- Изолировать состояние между группами: восстанавливать generation/stop/
  reasoning config, не переносить task-specific параметры и гарантировать
  корректную очистку backend resources при ошибке или завершении worker-а.
- Не допускать одновременной записи разных workers в один artifact; финальный
  report должен по-прежнему собираться один раз после завершения всех workers.
- Добавить regression tests на единственную загрузку модели на worker,
  распределение групп между GPU-наборами, fail-fast одного worker-а и clean
  shutdown. Сравнить результаты с текущим subprocess runner на небольшом HF и
  local-vLLM smoke-наборе до удаления старого пути.

## API response contract

- Поддержать ответы, где reasoning возвращается отдельно от видимого текста.
  Единый extractor должен сохранять `content`, `reasoning_content`,
  `finish_reason`, provider-specific `stop_reason` и будущие `tool_calls`.
- Передавать backend-примитивам явную фазу (`one_pass`, `reasoning`,
  `continuation`). В reasoning-фазе разрешать отдельное reasoning-поле, но
  подтверждать удалённую parser-ом границу через `stop_reason` либо достижение
  лимита; неоднозначный ответ должен завершаться fail-closed.
- Для обычной/финальной фазы возвращать только видимый `content`, а reasoning
  сохранять в диагностическом `info`, не склеивая его с prediction.

## Stop configuration

- Заменить mutable `add_stop_strings`/`reset_stop_strings` на построение
  immutable effective sampling config: model/template stops + task stops +
  phase-local overlay.
- Загружать foundational conversation config единым helper-ом. Один и тот же
  config должен задавать Jinja-шаблон, базовую stop string и provenance hash для
  HF, offline vLLM и управляемого API server.
- Повторить 8-sample Base local/API parity после добавления
  foundational stop string в APIBackend. Managed-API one-sample generation
  уже прошла с `verified_exact`, но историческое расхождение 3/8
  требует повторного полного замера.
- Оставить backend-ам только адаптацию единого effective config: HF stop
  strings/token ids, vLLM `stop`/`stop_token_ids`, стандартный API `stop` и
  расширения профиля vLLM.

## Runtime parity и диагностика

- Нормализовать effective HF generation config: Transformers 5.9
  предупреждает, что одновременно заданы `max_new_tokens` и
  `max_length`, причём первое имеет приоритет. Оставить в payload
  только один эффективный length-механизм на фазу и добавить
  payload-level regression test, не меняя context budgeting.
- Исследовать Base generation divergence между HF и vLLM: prompt совпадает,
  но HF дошёл до лимита с повторениями, а vLLM завершил короткий перевод.
  Отдельно проверить stop-string/token-id семантику на нескольких samples.
- Перенести unsupported PPL check в vLLM/API CLI до дорогой загрузки модели или
  server discovery. Сейчас offline vLLM может потратить минуты на warmup перед
  ожидаемым HF-only отказом.
- Спроектировать portable prompts для задач с whitespace-terminated assistant
  prefill. vLLM 0.21 API может удалять такой suffix; `best_effort` остаётся
  только непаритетной диагностикой.

## Низкий приоритет: численный детерминизм

Эти расхождения не блокируют текущий merge/release candidate. Возвращаться к
ним следует только если benchmark потребует строгой численной
воспроизводимости между разными vLLM engine settings.

- Зафиксировать границы детерминизма vLLM. Повторный smoke с тем же model
  snapshot и greedy sampling, но с `gpu_memory_utilization=0.92` вместо `0.95`,
  дал малые сдвиги probability и иной one-pass перевод; в two-pass отклонение
  накапливается. Возможный эксперимент: A/A при одинаковых engine args, затем
  A/B по GPU memory.
- Исследовать Base probability divergence между offline vLLM и vLLM API при
  одинаковых prompt token IDs. В полном 8-sample прогоне пять max-absolute
  deltas не превысили 0.00031, три составили 0.029–0.031. Проверить engine
  flags, prefix caching, sliding-window semantics и logprobs mode только перед
  введением строгих численных tolerances.
