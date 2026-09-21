# Результаты, кеш и ошибки

## Файлы задачи

- `<task>.jsonl` — один pretty-printed JSON array с per-sample records;
- `<task>_params.jsonl` — model/task/run config и fingerprint;
- `<task>_total.jsonl` — aggregate metrics и fingerprint;
- `<task>_aggregation_details.jsonl` — optional details;
- `evaluation_results.txt` — сводная таблица;
- `evaluation_log.txt` — runtime log.

Несмотря на суффикс `.jsonl`, sample-файл является одним JSON array. Если
процесс был жёстко завершён до записи закрывающей скобки, перед разбором
необходимо вручную проверить последний объект и дописать закрывающую скобку.

Aggregation-функция задачи может вернуть пару `(primary_value, details)`.
Тогда primary value попадает в `_total.jsonl`, а JSON-совместимые details — в
`_aggregation_details.jsonl`. Например, RuParam оставляет leaderboard micro
accuracy по минимальным парам, а category/source/part/level slices и category
macro сохраняет только как диагностику.

## `predict` и `info`

`predict` — только raw newly generated continuation. Assistant-prefill и
reasoning в него не добавляются. В two-pass run phase diagnostics находятся в
`info.reasoning` и `info.response`.

Для token probability сохраняются semantic candidate scores и metadata о
surface-form aggregation. API top-k является censored ranking, а не полным
распределением.

## Fingerprint cache

Fingerprint строится из sanitized canonical run config. Он учитывает model,
backend, task, sampling, effective execution mode, reasoning и continuation
provenance. В schema v2 task identity также включает registry name, init params,
dataset arguments, полное имя класса и SHA-256 файла модуля задачи. API
credentials не сериализуются. Remote dataset revision учитывается только если
она явно присутствует в `dataset_args`; автоматическое разрешение Hub revision
пока не выполняется.

При существующем результате:

- совпадающий fingerprint разрешает cache hit;
- несовпадение — ошибка;
- для нового эксперимента используйте другой output/name suffix;
- для намеренного пересчёта — `--force_recalc`.

Это запрещает случайно смешивать plain/reasoning или разные backend settings в
одном output identity.

## Failure semantics

Backend batch errors сохраняют исходные indexes и fail closed. Задача с
ошибкой не получает новый `_total.jsonl`. `Evaluator` продолжает независимые
задачи, формирует `EvaluationSummary` и CLI возвращает ненулевой exit code,
если хотя бы одна запрошенная задача завершилась ошибкой.

Перед исполнением evaluator проверяет task method и answer budget, alignment
messages/samples, canonical message roles и единую metric schema. Пустой
dataset или несовместимый backend method являются task failure, а не пустым
успешным результатом.

Всегда проверяйте exit code и наличие totals для всех requested datasets.
Успех одного task не означает успех всего процесса.

`evaluation_results.txt` строится только из успешно завершённых либо
совместимо закэшированных задач текущего вызова. Другие `_total.jsonl` в том же
каталоге не попадают в среднее; если текущий вызов не дал ни одного total,
старый summary удаляется, чтобы не выдавать его за новый результат.

## Сводная таблица

`show_results.py` читает текущие JSON-array sample artifacts и исторические
последовательности JSON objects. Он не восстанавливает оборванный JSON array:
такой файл сначала нужно проверить и закрыть вручную.

```bash
python show_results.py \
  --log_dir /results/models \
  --output_dir /results/report \
  --benchmark_config benchmark/config_balanced.yaml \
  --category_path benchmark/categories.json
```

Добавьте `--n_bags N` для bootstrap и `--show_time` для колонок времени.
PPL totals без поля `time` отображаются с пустым значением времени.
