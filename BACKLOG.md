# Backlog

Этот файл содержит задачи, обнаруженные во время стабилизации refactor v4, но
не входящие в текущую стабильную версию. Snapshot-specific планы и отчёты
хранятся в `dev/`; реализованные пункты следует переносить из этого файла в
соответствующую документацию и тесты.

## API и vLLM deployment

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
