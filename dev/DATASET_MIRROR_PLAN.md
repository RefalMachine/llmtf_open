# План архивного зеркала LLMTF datasets

Статус: **запланировано, реализация не начата**.

Дата: 2026-09-21.

Целевой Hugging Face repository: `RefalMachine/llmtf_benchmark`.

## 1. Цель

Сохранить собственную воспроизводимую копию всех данных, необходимых для
поставляемых LLMTF benchmarks, чтобы удаление, переименование или несовместимое
обновление upstream repository не делало benchmark неработоспособным и не
меняло выборку молча.

Зеркало должно:

- содержать сами данные и media assets, а не ссылки на upstream loaders;
- загружаться стандартным `datasets.load_dataset` без custom loading script;
- сохранять splits, категории и происхождение каждой строки;
- различать данные и benchmark-протокол: `fast`, `full` и `foundational` не
  должны создавать физические копии одних и тех же строк;
- иметь закреплённую revision, проверяемые hashes и анонимный read smoke;
- позволять LLMTF перейти на зеркало без изменения prompts, sample order и
  метрик.

## 2. Границы

### Этап A — обязательный состав

Все datasets, которые упоминаются в:

- `benchmark/llmtf_benchmark_instruct_fast.yaml`;
- `benchmark/llmtf_benchmark_instruct.yaml`;
- `benchmark/llmtf_benchmark_foundational.yaml`.

Включаются все необходимые few-shot splits, даже если leaderboard считается
только на test split.

### Этап B — расширенный состав

- остальные datasets зарегистрированных встроенных задач;
- optional RAG/LLM-as-a-Judge inputs, если сами данные публичны;
- данные примеров и smoke tasks, если они используются как поддерживаемый
  пользовательский сценарий;
- внешние benchmark submodules — только после отдельного решения о формате и
  праве на физическое зеркалирование.

### Не входит в первый этап

- изменение prompts, метрик или правил агрегации;
- объединение train/test либо пересэмплирование исходных строк;
- публикация model outputs и существующих `results/`;
- автоматическая установка зависимостей upstream loading scripts;
- хранение токена Hugging Face в config, shell history, логах или repository.

## 3. Модель данных в одном HF repository

Один repository используется как контейнер нескольких HF configurations.
Configuration соответствует логическому dataset с однородной schema, split —
исходному разбиению данных.

Примеры публичного API:

```python
load_dataset("RefalMachine/llmtf_benchmark", "ruparam", split="test")
load_dataset("RefalMachine/llmtf_benchmark", "mmlu_ru", split="dev")
load_dataset(
    "RefalMachine/llmtf_benchmark",
    "libra__ru_babilong_qa3",
    split="test",
)
```

### Правила configuration names

- lowercase ASCII, цифры и underscore;
- стабильное публичное имя не зависит от текущего username автора;
- семейство и upstream subset разделяются `__`;
- одно имя не используется для разных schemas;
- rename выполняется только через новый config и migration alias в LLMTF.

Предварительные примеры:

- `ruparam`;
- `flores__ru_en`, `flores__en_ru`;
- `mmlu_ru`, `mmlu_en`;
- `libra__ru_tpo`, `libra__ru_babilong_qa3`;
- `rublimp`;
- `ner__collection3`, `ner__nerel`, `ner__patient_queries`.

### Splits и внутренние срезы

- исходные `train`, `validation`, `dev`, `test` сохраняются раздельно;
- MMLU subjects хранятся в `subject`, крупная категория — в `category`;
- RuBLiMP phenomena хранятся в `phenomenon`/`source_config`;
- LIBRA subsets остаются отдельными configurations, поскольку они являются
  самостоятельно запускаемыми задачами;
- task-представления одних данных, например NER `json`/`in-place` или RAG
  `task-first`/`data-first`, не создают копии: представление формирует task code;
- при консолидации нескольких upstream configurations добавляются
  `source_config` и `source_split`, чтобы операцию можно было обратить.

### Физическое хранение

- основной формат — sharded Parquet;
- размер shard задаётся экспортёром и остаётся достаточно малым для resume;
- текст и annotations хранятся непосредственно в Parquet;
- изображения, аудио и другие media должны быть встроены либо сохранены рядом
  как versioned repository assets; внешняя URL-ссылка не считается архивом;
- порядок строк фиксируется и проверяется; технические поля не должны менять
  порядок benchmark sampling.

## 4. Benchmark manifests

В repository публикуются versioned manifests, но не дубликаты данных:

```text
manifests/
  instruct_fast.v1.yaml
  instruct_full.v1.yaml
  foundational.v1.yaml
```

Минимальная запись manifest:

```yaml
schema_version: 1
suite: instruct_fast
datasets_revision: <target-hf-commit>
tasks:
  - task: nlpcoreteam/ruMMLU
    config: mmlu_ru
    eval_split: test
    prompt_split: dev
    filters: {}
    max_sample_per_dataset: null
```

Manifest фиксирует:

- LLMTF task name и mirror configuration;
- evaluation/prompt splits;
- filters и category selection;
- sample limit и few-shot policy;
- ожидаемые row counts до и после filter;
- upstream и mirror revisions;
- schema/content fingerprint;
- совместимую версию manifest schema.

Generation settings и prompts остаются в LLMTF benchmark/task config. Manifest
данных не должен становиться вторым источником model sampling parameters.

## 5. Инвентаризация

До загрузки создать машинно-читаемый `dataset_inventory.yaml`.

Для каждого источника зафиксировать:

- LLMTF task names и benchmark suites;
- upstream repo id, configuration и точную commit revision;
- все доступные и реально используемые splits;
- schema/features, row count, порядок строк и dataset fingerprint;
- наличие streaming, loading script, gated access или remote media;
- целевой mirror config и преобразования при экспорте;
- attribution/citation metadata и статус готовности к публикации;
- expected anonymous accessibility после upload.

Инвентаризатор должен искать не только строковые константы, но и:

- `dataset_args()` и прямые вызовы `load_dataset`;
- registry init params;
- локальные JSON/JSONL ресурсы IFEval и других задач;
- datasets, уже собранные внутри `RefalMachine/darumeru`;
- конфигурации внутри multi-config datasets (MMLU, RuBLiMP, LIBRA);
- условно регистрируемые задачи и external submodules.

Результат этапа A считается полным только если каждый dataset из трёх основных
YAML сопоставлен ровно одному mirror config либо явно помечен как derived view
другого config.

## 6. Export/upload tool

Создать maintainer tool, предварительно
`dev/tools/mirror_benchmark_datasets.py`, со следующими режимами:

- `inventory` — получить metadata без записи в Hub;
- `export` — скачать закреплённую revision и собрать локальные Parquet shards;
- `validate-local` — проверить schema, splits, counts и hashes;
- `upload` — загрузить только уже проверенные artifacts;
- `validate-hub` — анонимно загрузить опубликованную revision и повторить
  проверки;
- `resume` — продолжить прерванный export/upload без перезаписи валидных
  shards;
- `dry-run` — показать план mutations без загрузки.

Требования к tool:

- токен принимается только через `HF_TOKEN` или approved secret mechanism;
- команды и логи никогда не печатают токен;
- upstream revision обязательна, floating `main` запрещён для export;
- upload выполняется staging batches, затем публикуется manifest с итоговой
  target revision;
- повторный запуск с теми же входами идемпотентен;
- несовпадение существующего hash завершается ошибкой, а не overwrite;
- временные файлы размещаются вне git worktree либо в явно ignored каталоге;
- source failures не оставляют configuration в состоянии, которое выглядит
  завершённым.

## 7. Валидация сохранности

Для каждой configuration/split проверить:

1. совпадение row count;
2. совместимость schema и nullable/list/class-label полей;
3. canonical per-row hashes и общий content hash;
4. сохранение row order;
5. распределение category/subject/source_config;
6. отсутствие URL-only media и недоступных внешних blobs;
7. `load_dataset(..., revision=<commit>)` без custom code;
8. анонимную загрузку после публикации;
9. повторную загрузку из чистого cache;
10. отсутствие секретов в dataset card, commits и artifacts.

Для каждого LLMTF task выполнить parity smoke old source vs mirror:

- одинаковые выбранные raw sample ids при фиксированном seed/limit;
- одинаковые prompt/reference payloads после task preprocessing;
- одинаковая task metric на фиксированном synthetic prediction;
- по одному реальному sample на используемый split/config;
- отдельный smoke для multi-config/category loaders.

Любое намеренное преобразование schema документируется и получает отдельный
mapping test. Совпадение row count без content comparison недостаточно.

## 8. Миграция LLMTF

После успешной публикации этапа A:

1. добавить центральный dataset catalog вместо разрозненных hard-coded repo
   paths;
2. сделать зеркало источником по умолчанию для поставляемых benchmarks;
3. оставить upstream fallback только как явную пользовательскую настройку;
4. передавать mirror repo/config/revision во все runners;
5. включить их в task provenance и cache fingerprint;
6. заменить локальные/старые алиасы единообразными catalog entries;
7. обновить benchmark manifests и документацию запуска;
8. инвалидировать старые cache artifacts только для задач, у которых реально
   изменились data identity либо row order;
9. выполнить небольшую HF, local-vLLM и API matrix на mirror source;
10. после parity удалить временный dual-read path, если он больше не нужен.

Переключение должно выполняться отдельным commit после публикации и проверки
Hub revision. Нельзя одновременно менять data source, prompts и metric.

## 9. Порядок выполнения

### P0 — спецификация и inventory

- [ ] Утвердить repo id и naming rules.
- [ ] Сгенерировать полный inventory этапа A.
- [ ] Зафиксировать upstream revisions и локальные source snapshots.
- [ ] Утвердить schema version для manifests и provenance fields.
- [ ] Посчитать ожидаемый объём хранения и число configurations/shards.

### P1 — exporter и первая вертикаль

- [ ] Реализовать export/validate/upload tool без upload по умолчанию.
- [ ] Провести вертикальный smoke на RuParam: export, local validation, upload в
  целевой config, anonymous reload, content parity.
- [ ] Проверить resume после искусственного прерывания.
- [ ] Проверить redaction токена в success и failure logs.

### P1 — основной benchmark

- [ ] Экспортировать простые single-config text datasets.
- [ ] Экспортировать multi-split datasets.
- [ ] Экспортировать MMLU/RuBLiMP с сохранением внутренних категорий.
- [ ] Экспортировать все 18 LIBRA configs и длинные shards.
- [ ] Экспортировать NER datasets и проверить nested annotations.
- [ ] Обработать datasets с media/remote assets.
- [ ] Опубликовать три manifests и закрепить target revision.

### P1 — миграция и parity

- [ ] Добавить dataset catalog и revision propagation.
- [ ] Переключить основные benchmark YAML/task loaders на зеркало.
- [ ] Выполнить per-task old-vs-mirror parity smokes.
- [ ] Выполнить fast benchmark dataset-loading smoke из чистого cache.
- [ ] Проверить полный состав и foundational config без model generation.

### P2 — расширение и maintenance

- [ ] Повторить процесс для optional/registry-only datasets.
- [ ] Добавить scheduled availability/integrity audit.
- [ ] Документировать процедуру обновления upstream snapshot как новую mirror
  revision без перезаписи старой воспроизводимой версии.
- [ ] Определить retention policy для superseded shards и manifests.

## 10. Критерии завершения этапа A

- все данные трёх основных benchmark YAML доступны из одного repository;
- ни один обязательный config не зависит от доступности upstream repo;
- fast/full/foundational представлены manifests без дублирования строк;
- configurations и splits анонимно загружаются на закреплённой revision;
- counts, schemas, row hashes и category distributions проверены;
- LLMTF использует зеркало по умолчанию и пишет его revision в provenance;
- old-vs-mirror sample/prompt parity подтверждена для каждой задачи;
- token отсутствует в git history, logs и result artifacts;
- точные команды, revisions и результаты сохранены в отдельном
  `dev/DATASET_MIRROR_REPORT.md`.

## 11. Основные риски

- очень крупные long-context datasets требуют shard/resume стратегии;
- upstream data могут содержать внешние media URLs, которые не являются
  полноценным архивом;
- custom loading scripts способны неявно преобразовывать исходные файлы;
- class labels и nested features могут измениться при наивной сериализации;
- объединение upstream configs может изменить row order или sample quotas;
- одновременная смена data source и task logic сделает parity недоказуемой;
- один repository упрощает discovery, но требует строгого namespacing и
  машинно проверяемого catalog, иначе configurations станут неоднозначными.
