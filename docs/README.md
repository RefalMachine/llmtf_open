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
- [Docker-профили](../docker/README.md) — образы `api`, `hf` и `vllm`.
Код и тесты имеют приоритет над документацией при обнаружении расхождения.
Исторические планы и snapshot-specific validation artifacts находятся в
`../dev/` и не являются частью пользовательского контракта.
