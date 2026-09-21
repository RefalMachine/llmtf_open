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
  EnIFEval и обработка неверного prediction type.
- Evaluator валидирует numeric limits, task contract, capability, dataset и
  metric schema; seed сбрасывается перед каждой задачей. Registry conflict
  требует явного override.
- Run config schema v2 включает task registry/init/dataset args и digest модуля.
  Summary текущего запуска не смешивается со stale totals того же каталога.

## Изменения baseline

Новый baseline обязателен для Shlepa, PPL, NER in-place, MMLU/RuBLiMP sample
limits и любых старых cache artifacts: schema v2 и task source digest намеренно
инвалидируют прежнюю совместимость.

## Проверки

Pure logic и contracts:

- `tests/test_refactor_logic.py`: 38 tests, 0 failures;
- полный unittest внутри CPU-only `llmtf:api`: 62 passed, 1 skip;
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
- vLLM RuCoLA probability and CopyText: both success;
- API CopyText: expected early `NotImplementedError`, no total created.

Temporary runtime artifacts were written outside the repository under `/tmp`
and are not part of the source change.

## Непроверенные границы

- Публичный `RefalMachine/RuParam` анонимно загружается как единственный split
  `test`: 9 505 строк, ожидаемые шесть колонок. Полная оценка всех строк и
  API-ячейка не выполнялись; targeted HF и local-vLLM smokes прошли. Snapshot
  отличается от описанной в новой статье редакции на 11 336 пар; повторный data
  audit потребуется после замены файла.
- Live LLM-as-a-Judge endpoint smoke не выполнялся: endpoint/credentials и
  соответствующий input pair не были предоставлены.
- Полная multi-model hybrid/reasoning/base матрица v4 не повторялась; изменения
  проверены targeted smokes. Поэтому этот отчёт не расширяет parity claims из
  `TEST_REPORT_v4.md`.
- Dockerfiles не менялись, поэтому образы не пересобирались; текущий checkout
  монтировался в ранее валидированные profile images.
