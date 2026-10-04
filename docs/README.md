# Документация LLMTF

- [Release notes v0.3.0](releases/v0.3.0.md) — основные изменения, breaking
  changes, миграция, проверенный runtime и ограничения выпуска.
- [Архитектура](architecture.md) — слои `BaseLLM -> LLM -> Backend`, reasoning,
  budgeting и расширение framework.
- [Конфигурация и запуск](configuration.md) — CLI, YAML schema, backend kwargs,
  vLLM defaults и model-specific server options.
- [Assistant continuation и prefill](assistant_continuation.md) — точное
  продолжение assistant-сообщения, хвостовые пробелы, API-проверки и варианты
  следующего токена.
- [API backend](api_backend.md) — профили `auto`/`openai`/`vllm`, работа без
  tokenizer endpoint, capability errors и top-k scoring.
- [Результаты, кеш и ошибки](results.md) — artifacts, fingerprints и failure
  semantics.
- [LLM-as-a-Judge](llmaaj.md) — генерация candidates, judge pipeline и reports.
- [Юридический бенчмарк](legal_benchmark.md) — общие Base/Instruct конфиги,
  LawMC, режимы LegalBench-RU, RuTaR, ручной RuLegalNER и RuLaw-ProofBench;
  9 результатов Foundational и 11 Instruct, включая closed/open-book RuLaw.
- [RuLaw-ProofBench](../dev/rulaw_proofbench/README.md) — 300 вопросов и
  5 отдельных демонстраций на HF, открытые ответы и MCQ,
  [методика и ограничения](../dev/rulaw_proofbench/paper_methods.md).
- [LegalBench-RU](legalbench_ru.md) — экспериментальный набор, fixed few-shot,
  двойной scoring и проверка протокола.
- [RuTaR](rutar.md) — бинарные налоговые вопросы, закреплённый snapshot,
  фиксированные демонстрации и accuracy по вероятностям ответов.
- [Shlepa](shlepa.md) — настоящие few-shot demonstrations для всех четырёх
  задач, состав evaluation и миграция прежних результатов.
- [Docker-профили](../docker/README.md) — образы `api`, `hf` и `vllm`.
Код и тесты имеют приоритет над документацией при обнаружении расхождения.
Исторические планы и snapshot-specific validation artifacts находятся в
`../dev/` и не являются частью пользовательского контракта.
