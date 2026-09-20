# Backlog

Этот файл содержит задачи, обнаруженные во время стабилизации refactor v4, но
не входящие в текущую стабильную версию. Snapshot-specific планы и отчёты
хранятся в `dev/`; реализованные пункты следует переносить из этого файла в
соответствующую документацию и тесты.

## Высокий приоритет: task layer и внешние задачи

### 1. Аудит и переработка task-логики

Эта задача является первоочередной и обязательной предпосылкой для внешней
регистрации задач. До расширения plugin surface провести системный аудит
`Task`, `SimpleFewShotHFTask`, `TASK_REGISTRY`, dataset loading и взаимодействия
task-а с evaluator/model:

- проверить и формализовать публичный task-контракт: обязательные методы и
  поля, scoring methods, message roles, answer budgets, stop configuration,
  few-shot semantics, aggregation и leaderboard aggregation;
- найти дублирование и неявные соглашения в существующих задачах, унифицировать
  повторяющуюся логику и удалить зависимости от случайных import side effects;
- проверить naming/run identity, task params, cache fingerprint и provenance,
  чтобы код или конфигурация задачи однозначно участвовали в совместимости
  результатов;
- унифицировать validation и сообщения об ошибках для некорректных datasets,
  splits, samples, messages, metrics и неподдерживаемых backend methods;
- проверить ленивую загрузку optional tasks и внешних зависимостей, чтобы одна
  недоступная задача не ломала импорт всего registry;
- определить стабильный минимальный API для пользовательских задач и покрыть
  его contract tests до реализации plugin loader;
- актуализировать встроенные задачи и документацию по результатам аудита, не
  меняя метрики молча: любые намеренные semantic changes должны иметь migration
  note и regression evidence.

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
- Обновить встроенный API server runner под vLLM 0.21: убрать устаревшие CLI
  flags, использовать `--language-model-only` для текстовых прогонов и не
  дублировать `json_to_jinja` из `llmtf.utils`.
- Пересмотреть исторические vLLM overrides. Отдельно проверить необходимость
  `enable_prefix_caching`, `disable_sliding_window` и чрезмерного
  `max_logprobs=1000000`; defaults и offline/API режимы должны различаться
  только по доказанной причине. Переименовать устаревшее внутреннее поле
  `max_seq_len_to_capture`, которое сейчас фактически задаёт `max_model_len`.

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
- Явно передавать foundational stop strings API-клиенту. Сейчас локальные
  backends извлекают их из conversation config, а APIBackend начинает с пустой
  базовой stop-конфигурации. На 8-sample Base exact API прогоне совпали
  только 3/8 generation predictions; несовпавшие API-ответы имели
  1400–2041 символ против 54–170 offline.
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
