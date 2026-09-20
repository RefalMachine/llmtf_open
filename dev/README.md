# Development artifacts

Эта директория хранит временные и исторические материалы разработки. Они
объясняют, как проект пришёл к текущей реализации, но не определяют её
публичный контракт.

- `REFACTOR_PLAN.md`, `REFACTOR_PLAN_v2.md`, `REFACTOR_PLAN_v3.md` —
  исторические design records;
- `REFACTOR_PLAN_v4.md` — план стабилизации текущей архитектуры;
- `TEST_PLAN_v4.md` — воспроизводимые команды validation matrix;
- `TEST_REPORT_v4.md` — результаты snapshot от 2026-09-20;
- `VALIDATION_STATUS_v4.md` — статус того же цикла разработки.

Пользовательская документация находится в `../README.md` и `../docs/`.
Постоянные будущие задачи ведутся в `../BACKLOG.md`; правила для агентов — в
`../AGENTS.md`.

`tools/remap_qwen35_checkpoint.py` — параметризованная maintainer-утилита для
исправления дублированных module prefixes в исторических Qwen3.5 checkpoints.
