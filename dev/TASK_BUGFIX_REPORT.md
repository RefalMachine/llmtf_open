# Task layer bugfix report

Дата: 2026-09-21.

Статус: исправлены обнаруженные в рамках аудита воспроизводимые дефекты.
Документ не является доказательством отсутствия любых будущих ошибок и не
закрывает архитектурный backlog task/plugin API.

## Исправления

- RuOpinionNE больше не исполняет model output через `eval`; parser принимает
  JSON и безопасный Python literal, отбрасывая scalar и элементы неверного
  типа.
- RuParam формирует canonical roles, работает строго zero-shot, корректно
  нормализует source order и считает строку правильной только при успехе в двух
  порядках предъявления. Неуникальный исходный `id` больше не смешивает пары;
  category/source/part/level slices сохраняются в aggregation details.
- CopyText снова получает local `leading_space` через `LLM`; API завершается
  ранней capability error вместо `AttributeError` или удалённого запроса.
- Normal evaluator dispatch поддерживает `calculate_logsoftmax`, а `LLM`
  проверяет alignment его batch result.
- LLM-as-a-Judge использует evaluator-compatible `max_prompt_len`; неверный
  few-shot режим, длины reference/model inputs и instruction alignment дают
  явные ошибки. Malformed generative judge output отмечается `invalid`, а не
  засчитывается как tie.
- PPL отделяет reference answer по character boundary exact rendered prompt и
  не смешивает assistant prefill с answer-token log probability.
- Shlepa сохраняет правильную метку `A`; distractors при малом sample limit
  берутся из полного split, а не из уже обрезанного evaluation subset.
- Oversized irreducible prompts завершаются `PromptTooLongError`; ложные
  сообщения о несуществующей truncation удалены, цикл сокращения ruTiE конечен.
- MMLU и RuBLiMP распределяют общий raw-record limit между subjects/slices без
  превышения. Пустой RuBLiMP output становится неправильным ответом, а не
  исключением.
- NER JSON parsing fail-closed; in-place NER требует полного совпадения
  tokenized source, а не только совпадающего префикса.
- RAG PPL берёт первый элемент `answers`; optional API judge инициализируется
  лениво и не выполняет сетевые запросы при импорте registry.
- IFEval больше не скачивает NLTK data в constructor; исправлен answer budget
  EnIFEval и обработка неверного prediction type. RuIFEval использует локальный
  `langdetect` с фиксированным seed вместо `fasttext-langdetect`, который
  пытался скачать модель при оценке и повторял минутные timeout для каждого
  языкового ограничения. Недетектируемый текст теперь не засчитывается.
- Удалены регистрации и benchmark-группы Libra `ru_gsm100`, `ru_trec` и
  `ru_qasper`: эти BuilderConfig отсутствуют в текущем публичном
  `ai-forever/LIBRA`, поэтому поставляемые конфигурации неизбежно падали.
  Основной fast benchmark теперь использует официальный LIBRA Mini из шести
  срезов, а обычный instruct benchmark — все 18 срезов редакции May 2026.
- Evaluator валидирует numeric limits, task contract, capability, dataset и
  metric schema; seed сбрасывается перед каждой задачей. Registry conflict
  требует явного override.
- Run config schema v2 включает task registry/init/dataset args и digest модуля.
  Summary пересобирается по всем завершённым totals того же output-каталога.
  Лениво обнаруживаемые API capabilities `logprobs`/`detokenize` остаются в
  provenance, но исключены из semantic fingerprint: cache resume больше не
  зависит от того, исполнилась предыдущая задача или была пропущена по cache
  hit. Старые schema-v2 totals проходят совместимую проверку по вложенному
  `run_config`; остальные различия по-прежнему fail closed.

## Изменения baseline

Новый baseline обязателен для Shlepa, PPL, NER in-place, MMLU/RuBLiMP sample
limits и cache artifacts до schema v2: task source digest намеренно
инвалидирует прежнюю совместимость. Schema-v2 artifacts с историческим
order-dependent API capability fingerprint пересчитывать не требуется.

## Проверки

Pure logic и contracts:

- `tests/test_refactor_logic.py`: 40 tests, 0 failures;
- полный unittest в актуальном контейнере: 72 passed, 1 skip;
- replay checker-части на 305 уже сгенерированных RuIFEval responses занял
  3.346 s; отдельный smoke с отключённой сетью различил русский и английский;
- host `unittest discover` не является полным gate: в активном host Python нет
  `PyYAML`, поэтому три YAML-dependent test-модуля не импортируются;
- API, HF и vLLM images импортировали и выполнили dependency-free suite;
- direct `py_compile`, repository `compileall` и `git diff --check` прошли;
- host `pytest` не запускался: модуль `pytest` не установлен.

Runtime environment:

- images: API `d10c2abd2015`, HF `2146b9f44aa2`, vLLM `d32605e9216b`;
- NVIDIA GeForce RTX 4090;
- torch 2.11.0+cu129, CUDA runtime 12.9, Transformers 5.9.0, vLLM 0.21.0.

Real-model smoke, `Qwen/Qwen3.5-2B`, one raw sample:

- HF `daru/treewayextractive`: success, normal logsoftmax dispatch;
- HF Shlepa movie task: success after full-split distractor fix;
- HF CopyText sentence RU: success (`len=1`, `lcs=1`);
- HF RuCoLA PPL: success, answer-token mean log probability `-0.25390625`;
- HF RuParam public snapshot through the default registry: одна исходная пара,
  два orientation, success; `_total` и category/part/level aggregation details
  созданы;
- local vLLM RuParam public snapshot: одна исходная пара, два orientation,
  success; `_total` и category aggregation details созданы;
- managed vLLM 0.21 API RuParam: runner сам поднял Qwen3.5-2B в
  text-only режиме; одна пара/два orientation прошли;
- managed vLLM 0.21 API foundational: Qwen3.5-2B-Base с Jinja из
  `default_foundational.json`, stop string `\n\n` и `probe_api_prefill=true`;
  one-sample Flores generation прошла, prefill отмечен `verified_exact`;
- managed vLLM 0.21 API Kinopoisk: четыре samples прошли с
  синхронным server/client top-100; candidate coverage сохранено в
  per-sample `info`;
- vLLM RuCoLA probability and CopyText: both success;
- API CopyText: expected early `NotImplementedError`, no total created.

Managed benchmark evidence после полного task-fix цикла:

- `Qwen/Qwen3.5-2B`, API Instruct Fast: итоговый output содержит 41 task total;
  cache-resume принял совместимые schema-v2 artifacts, а ранее прерванный
  `libra/matreshka_names` завершился на 180 samples за 41.65 s с четырьмя
  length slices;
- полный публичный RuParam snapshot оценён через managed API: 9 505 пар / 19 010
  orientations за 122.05 s, leaderboard pair accuracy `0.7034192530`;
- `Qwen/Qwen3.5-2B-Base`, managed API foundational: завершены все 13
  API-совместимых totals — translation, summarization, Shlepa, two MMLU groups,
  NER, Kinopoisk и RuCoLA. Последняя CopyText-группа вернула ожидаемый
  capability error и не создала totals; это намеренная граница APIBackend, а
  не незавершённая совместимая группа.

Runtime artifacts находятся в игнорируемом `results/` и не входят в source
change.

## Непроверенные границы

- Публичный `RefalMachine/RuParam` анонимно загружается как единственный split
  `test`: 9 505 строк, ожидаемые шесть колонок. Полная managed-API оценка
  выполнена; targeted HF и local-vLLM smokes также прошли. Полные HF/local-vLLM
  прогоны всего snapshot не выполнялись. Snapshot
  отличается от описанной в новой статье редакции на 11 336 пар; повторный data
  audit потребуется после замены файла.
- Live LLM-as-a-Judge endpoint smoke не выполнялся: endpoint/credentials и
  соответствующий input pair не были предоставлены.
- Полная multi-model hybrid/reasoning/base матрица v4 не повторялась. Завершён
  Instruct Fast API suite; в Base API suite завершены все поддерживаемые задачи,
  но весь конфиг ожидаемо возвращает non-zero из-за локального-only CopyText.
  Поэтому этот отчёт не расширяет parity claims из `TEST_REPORT_v4.md`.
- Dockerfiles не менялись, поэтому образы не пересобирались; текущий checkout
  монтировался в ранее валидированные profile images.
