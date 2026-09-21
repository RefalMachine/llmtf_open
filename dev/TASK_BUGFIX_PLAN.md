# Task layer bugfix plan

Статус документа: **реализация завершена; runtime evidence зафиксирован**.

Дата ревизии: 2026-09-21.

Этот этап предшествует архитектурной переработке task API. Его цель — получить
рабочую и проверяемую текущую реализацию, не вводя plugin loader, новый manifest
format или новый публичный декларативный API.

## 1. Граница этапа

В scope входят обнаруженные дефекты безопасности, исполнения, метрик,
ограничения выборки, cache identity и fail-closed validation:

1. убрать выполнение model output через `eval`;
2. исправить некорректную роль в `RuParam`;
3. восстановить local `CopyText` и выдавать раннюю capability error на API;
4. добавить normal-evaluation dispatch для `calculate_logsoftmax`;
5. исправить сигнатуру LLM-as-a-Judge task;
6. отделять answer tokens в PPL по символьной границе rendered prompt, а не
   сравнением token count с character offsets;
7. перестать отбрасывать правильный вариант `A` в Shlepa;
8. заменить ложные сообщения о truncation на явную ошибку при oversized
   zero-shot prompt и устранить бесконечный цикл в `ruTiE`;
9. сделать `max_sample_per_dataset` настоящей верхней границей исходных dataset
   records для multi-subject/multi-slice задач;
10. включить registry identity, task init params, dataset args и digest
    реализации задачи в run provenance/cache fingerprint;
11. локализовать очевидные parser failures: пустой RuBLiMP output, неверный тип
    JSON, неполную in-place NER копию и неправильное поле RAG answer;
12. добавить базовую validation входов evaluator, dataset batch и metric schema;
13. убрать сетевой side effect IFEval из конструктора задачи;
14. запретить неявное перезаписывание task registry entries;
15. покрыть перечисленные случаи dependency-free regression tests;
16. исключить stale totals из отчёта текущего запуска и сделать task sampling
    независимым от порядка запуска;
17. исправить выбор distractors Shlepa при малом sample limit.

Не входят в этот этап:

- новый декларативный `TaskSpec`;
- plugin/external task loader;
- полная lazy-модель встроенного registry;
- автоматическое разрешение и pinning remote Hugging Face dataset revisions;
- унификация всех исторических prompts и намеренное изменение определения
  существующих метрик.

Эти пункты остаются в `BACKLOG.md` и должны обсуждаться после стабилизации.

## 2. Инварианты

- Не возвращать удалённые legacy model classes.
- Reasoning orchestration остаётся в `LLM`.
- API по-прежнему не объявляет поддержку PPL или `return_tokens=True`.
- Невалидный model output получает нулевой/invalid результат там, где формат
  является частью task metric; программные и dataset-contract ошибки не
  маскируются и завершают task с non-zero summary.
- Изменения, влияющие на выборку или метрику, должны быть явно перечислены в
  итоговом отчёте и потребуют нового baseline.

## 3. Порядок реализации

### P0 — безопасность и недостижимые задачи

- [x] Безопасный parser RuOpinionNE.
- [x] Role fix RuParam.
- [x] CopyText local capability fix и явная API incompatibility.
- [x] `calculate_logsoftmax` dispatch и alignment validation.
- [x] LLM-as-a-Judge signature fix.
- [x] PPL answer-boundary fix.

### P1 — корректность samples и метрик

- [x] Shlepa label-zero fix и distractors из полного split.
- [x] Oversized prompt fail-closed во всех собственных loaders.
- [x] Конечный `ruTiE` context reduction.
- [x] Общая верхняя граница raw samples для MMLU и RuBLiMP.
- [x] Robust parsing для RuBLiMP, IFEval, RuOpinionNE и NER.
- [x] RAG PPL answer field fix.

### P1 — validation и provenance

- [x] Проверка evaluator numeric limits, task method, dataset alignment и
  metric keys.
- [x] Явный registry conflict error.
- [x] Task implementation/config identity в fingerprint.
- [x] Удаление NLTK download из IFEval constructor.
- [x] Отчёт только по успешным/совместимо закэшированным задачам текущего
  запуска.
- [x] Независимый от порядка задач seed перед каждым dataset load.

### P2 — документация и regression evidence

- [x] Dependency-free unit tests на исправленные инварианты.
- [x] Обновить task documentation и limitation notes.
- [x] Выполнить доступные проверки из `TASK_BUGFIX_TEST_PLAN.md`.
- [x] Создать `TASK_BUGFIX_REPORT.md` с границами runtime validation.

## 4. Изменения, требующие нового baseline

Следующие исправления намеренно меняют результаты и не должны смешиваться со
старыми totals:

- Shlepa больше не исключает samples с ответом `A`;
- PPL считает только токены reference answer;
- NER in-place требует полного воспроизведения исходного текста;
- multi-subject/multi-slice sample limits перестают превышать заданную границу;
- изменённый source digest задачи инвалидирует старый cache fingerprint.

## 5. Завершённый блок RuParam

Snapshot получен, опубликован и проверен. Аудит и исправления вынесены в
`dev/RUPARAM_FIX_PLAN.md` и `dev/RUPARAM_DATA_AUDIT.md`: устранена группировка
по неуникальному `id`, реализованы zero-shot double-order scoring и
диагностические срезы. Анонимная загрузка публичного split и однопарный HF
smoke прошли. Изменение правил требует нового baseline.
