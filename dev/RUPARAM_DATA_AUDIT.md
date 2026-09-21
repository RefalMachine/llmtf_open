# RuParam snapshot audit

Дата проверки: 2026-09-21. Проверен локальный `/RuParam.csv`; сам CSV намеренно
исключён из Git.

Тот же snapshot опубликован в
`https://huggingface.co/datasets/RefalMachine/RuParam`. Анонимная проверка с
чистым HF cache видит только `data/test.csv`, единственный split `test` и 9 505
records с ожидаемой схемой.

## Фактический формат и объём

- 9 505 CSV records (физических строк больше из-за переводов строк внутри
  полей);
- поля: `id`, `gram`, `ungram`, `label`, `source`, `order`;
- 8 248 записей TORFL и 1 257 parametric;
- после удаления краевого whitespace у `label`: 67 уникальных labels — 38 в
  TORFL и 30 в parametric, общий label между частями: `npi`;
- TORFL levels после исправления кириллических конфузаблов в `source`:
  A1=1781, A2=937, B1=2458, B2=1609, C1=1175, C2=285;
- 4 805 строк имеют `order=s`, 4 700 — `order=i`.

Этот snapshot не совпадает с описанной в статье текущей редакцией на 11 336
пар и 150 категорий. Task не зашивает ожидаемое число строк/категорий и сможет
читать последующую версию с той же схемой, но результаты разных редакций нельзя
смешивать в одном baseline.

## Data-quality observations

- исходный `id` не является уникальным: поэтому он больше не используется как
  aggregation key;
- 1 строка не имеет `label` и попадает в диагностическую категорию
  `__unlabeled__`;
- 100 строк имеют одинаковые `gram` и `ungram` после `strip`; они не удаляются
  молча, их число записывается в aggregation details;
- `source` содержит кириллические конфузаблы в `torfl_А1` и `torfl_С1`; task
  нормализует их только для диагностических source/level slices;
- labels имеют краевой whitespace и несколько очевидно служебных значений
  (`Разметка 1`, `Разметка 2`, `Разметка 3`). Task удаляет только краевой
  whitespace, не объединяет и не переименовывает смысловые категории.

## Scoring decision

Одна аннотированная строка предъявляется дважды с обратным порядком вариантов.
Pair score равен 1 только при правильном выборе в обоих предъявлениях.

Основной `acc` и `leaderboard_result` — micro mean pair score по всем
аннотированным строкам. Это сохраняет прямой общий показатель протокола и вес
каждой строки. Macro mean по категориям не подменяет основной результат, потому
что размеры категорий резко различаются; он сохраняется как вторичная
`category_macro_accuracy`.

В `<task>_aggregation_details.jsonl` сохраняются:

- micro и category-macro accuracy;
- accuracy/count по `label`, `part`, `source` и TORFL level;
- отдельная accuracy для grammatical-first и grammatical-second;
- число одинаковых sentence pairs.

В per-sample JSON сохраняются `row_id`, исходный `original_id`, category,
source, part, level, исходный order и текущая presentation orientation.

## Наблюдаемый словарь labels

TORFL (38): `__unlabeled__`, `npi`, `вид`, `время`, `деепр`, `залог`, `коп`,
`лекс`, `лекс_глаг`, `лекс_диск`, `лекс_нар`, `лекс_предл`, `лекс_прил`,
`лекс_сущ`, `мест`, `мод`, `одуш`, `отриц`, `согл_атр`, `согл_пред`,
`согл_пред_л`, `согл_пред_р`, `согл_пред_ч`, `союз_подч`, `союз_соч`,
`упр_глаг`, `упр_нар`, `упр_предл`, `упр_прил`, `упр_сущ`, `утр_сущ`, `фин`,
`форма`, `числ`, `чр`, `Разметка 1`, `Разметка 2`, `Разметка 3`.

Parametric (30): `Non-proj`, `NonF`, `CoordCon`, `adj_n`, `adj_wh_isl`,
`agree`, `anaphor`, `aux_v`, `clitics`, `comp_s`, `compl_isl`, `conv_isl`,
`dep_adj`, `ind_q_isl`, `n_gen`, `n_rel`, `np_dep`, `np_isl`, `npi`, `p_np`,
`pred_case`, `pro-drop`, `rel_isl`, `subj_isl`, `voice_parta`, `voice_partp`,
`voice_refl`, `wh`, `wh_isl`, `wh_rel`.

Иерархии, связывающей эти значения с 29/121 категориями новой редакции,
текущий CSV не содержит. Поэтому task не пытается вывести её эвристически.
