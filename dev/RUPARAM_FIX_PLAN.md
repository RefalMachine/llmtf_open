# RuParam functional fix plan

Статус: выполнено. Файл `RuParam.csv` является локальным входом и не входит
в состав репозитория.

## Найденные дефекты task-логики

- task использует первые 10 строк как few-shot, хотя протокол RuParam является
  zero-shot;
- исходный `order=i` меняется через chained assignment pandas, что ненадёжно и
  скрывает исходную ориентацию строки;
- две предъявленные ориентации группируются по полю `id`, но `id` в реальном
  CSV не уникален, поэтому aggregation падает или смешивает разные пары;
- агрегат не сохраняет диагностические срезы `label`, источник, часть корпуса и
  уровень TORFL;
- loader требует split `test`, хотя загруженный на Hugging Face CSV может быть
  автоматически представлен как `train`;
- загрузка без необходимости зависит от pandas и создаёт искусственный prompt
  split.

## Исправление

1. Нормализовать каждую исходную строку без изменения файла: для `order=i`
   поменять местами `gram` и `ungram`.
2. Создать стабильный уникальный `row_id` из индекса строки и digest её полей;
   исходный `id` оставить только как provenance.
3. Для каждой строки создать ровно два предъявления: grammatical-first и
   ungrammatical-first.
4. Считать пару правильной только когда обе ориентации выбраны верно.
5. Оставить основным leaderboard score micro accuracy по парам; сохранить
   вторичные срезы по `label`, `part`, `source`, TORFL level и позициям в
   aggregation details, включая category macro как диагностическую метрику.
6. Явно требовать `few_shot_count=0`.
7. Поддержать HF split `test` и fallback на `train`, а также опциональный
   локальный CSV для тестирования.
8. Добавить regression tests и прогнать обязательные dependency-free checks.

Все пункты выполнены. Локальный CSV дополнительно загружен через `datasets` в
API Docker profile: 9 505 rows дали 19 010 presentations и 9 505 уникальных
aggregation identities. Однопарный HF smoke с `Qwen/Qwen3.5-2B` и thinking
disabled успешно прошёл обе orientation, записал `_total` и
`_aggregation_details`; этот smoke проверяет исполнение task-контракта, а не
качество модели на корпусе. После публикации анонимная загрузка
`RefalMachine/RuParam` с чистым Hub cache подтвердила единственный split `test`
на 9 505 строк; однопарный smoke повторён через default registry без локального
dataset override на HF и local vLLM.

## Публикация

- добавить точное имя `/RuParam.csv` в `.gitignore`;
- публиковать файл в `data/test.csv`, чтобы Hub создавал ожидаемый split;
- токен вводится только через `hf auth login` и не записывается в команды,
  документацию или логи проекта.

Команды из корня checkout:

```bash
hf auth login
hf repos create RefalMachine/RuParam --repo-type dataset --exist-ok
hf upload RefalMachine/RuParam ./RuParam.csv data/test.csv \
  --repo-type dataset \
  --commit-message "Upload RuParam dataset snapshot"
```

Путь `data/test.csv` выбран намеренно: он даёт dataset split `test`, который
task использует непосредственно. Для совместимости loader также умеет читать
`train`, если CSV уже был загружен в корень HF repository.
