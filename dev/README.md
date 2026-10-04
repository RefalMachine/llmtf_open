# Development artifacts

Эта директория хранит временные и исторические материалы разработки. Они
объясняют, как проект пришёл к текущей реализации, но не определяют её
публичный контракт.

- `REFACTOR_PLAN.md`, `REFACTOR_PLAN_v2.md`, `REFACTOR_PLAN_v3.md` —
  исторические design records;
- `REFACTOR_PLAN_v4.md` — план стабилизации текущей архитектуры;
- `TEST_PLAN_v4.md` — воспроизводимые команды validation matrix;
- `TEST_REPORT_v4.md` — результаты snapshot от 2026-09-20;
- `VALIDATION_STATUS_v4.md` — статус того же цикла разработки;
- [EXTERNAL_BENCHMARK_AUDIT.md](EXTERNAL_BENCHMARK_AUDIT.md) — аудит текущего
  task/eval, двух внешних snapshots и противоречий исходных планов;
- [TASK_EVAL_GRAPH_REFACTOR_PLAN_v2.md](TASK_EVAL_GRAPH_REFACTOR_PLAN_v2.md) —
  актуальный проект общего исполнения с разделением интеграционного выпуска
  R1 и полного task/multi-turn refactor R2;
- [EXTERNAL_BENCHMARK_INTEGRATION_PLAN_v2.md](EXTERNAL_BENCHMARK_INTEGRATION_PLAN_v2.md)
  — актуальный план RuBookSum/RuWikiBench: протоколы, этапы и validation gates;
- [LEGALBENCH_RU_INTEGRATION_PLAN.md](LEGALBENCH_RU_INTEGRATION_PLAN.md) —
  исследование данных и кода LegalBench-RU, план интеграции и проверки протокола;
- [LEGALBENCH_RU_VALIDATION_REPORT.md](LEGALBENCH_RU_VALIDATION_REPORT.md) —
  реализация, offline contracts и фактическое покрытие runtime-проверок;
- [SHLEPA_FEW_SHOT_FIX_REPORT.md](SHLEPA_FEW_SHOT_FIX_REPORT.md) — исправление
  игнорируемого few-shot count для всех четырёх задач Shlepa;
- [RUTAR_VALIDATION_REPORT.md](RUTAR_VALIDATION_REPORT.md) — протокол и
  фактические проверки бинарной задачи RuTaR;
- [RULEGALNER_MANUAL_VALIDATION_RESULTS.json](RULEGALNER_MANUAL_VALIDATION_RESULTS.json)
  — проверка ручного RuLegalNER: 12 наборов artifacts / 96 ответов на
  HF/local-vLLM/API, Base/Instruct, 0/5-shot, thinking off; версии, метрики,
  fingerprints и ограничения этого smoke;
- `TASK_EVAL_GRAPH_REFACTOR_PLAN.md`, `EXTERNAL_BENCHMARK_INTEGRATION_PLAN.md`
  — предыдущие версии этих проектов, сохранённые для истории;
- `TASK_BUGFIX_PLAN.md`, `TASK_BUGFIX_REPORT.md` — завершённая корректностная
  стабилизация; не смешивать её с ещё не реализованным task API v2.

Пользовательская документация находится в `../README.md` и `../docs/`.
Постоянные будущие задачи ведутся в `../BACKLOG.md`; правила для агентов — в
`../AGENTS.md`.

`tools/remap_qwen35_checkpoint.py` — параметризованная maintainer-утилита для
исправления дублированных module prefixes в исторических Qwen3.5 checkpoints.

`tools/legalbench_ru_report.py` — необязательная offline replay-проверка
LegalBench-RU artifacts и парных сравнений. Общий юридический бенчмарк использует
два `benchmark/llmtf_legal_*.yaml` и стандартный `show_results.py`, см.
[руководство](../docs/legal_benchmark.md).

`tools/validate_shlepa_few_shot.py` — воспроизводимая проверка реальных
0/1/5-shot промптов и artifacts на HF, local vLLM или API. Покрытие выполненных
серий и ограничения находятся в `SHLEPA_FEW_SHOT_FIX_REPORT.md`.

`tools/validate_rulegalner_manual.py` — task smoke на HF, local vLLM или API;
по умолчанию выполняет 0/5-shot, по 8 тестовых фрагментов. Это проверка generate
с thinking off, не полная v4 matrix. Запуск и протокол описаны в
[руководстве](../docs/legal_benchmark.md#проверка-интеграции-rulegalner-от-2026-10-04).
