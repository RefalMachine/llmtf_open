# Юридический бенчмарк LLMTF

Два общих конфига задают запуски для Base и Instruct моделей:

- [`llmtf_legal_foundational.yaml`](../benchmark/llmtf_legal_foundational.yaml)
  повторяет model/defaults из `llmtf_benchmark_foundational.yaml`;
- [`llmtf_legal_instruct.yaml`](../benchmark/llmtf_legal_instruct.yaml)
  повторяет model/defaults из `llmtf_benchmark_instruct_fast.yaml`.

Используется существующая схема `model/defaults/tasks`. Новые юридические
датасеты добавляются обычными записями в `tasks`; режимы одного датасета могут
перечисляться в его `datasets`. Конкретная модель и backend выбираются при запуске.

## Состав и параметры

| Компонент | Foundational | Instruct |
|---|---|---|
| `shlepa/lawmc` | 5-shot, temperature 0, без лимита | 0-shot, temperature 0, без лимита |
| `legalbench_ru/closed` | 5-shot, temperature 0, 816 записей | 0-shot, temperature 0.3, до 200 записей |
| `legalbench_ru/grounded` | 5-shot, temperature 0, 205 записей | 0-shot, temperature 0.3, до 200 записей |
| `legalbench_ru/distractor` | 5-shot, temperature 0, 189 записей | 0-shot, temperature 0.3, 189 записей |
| `legalbench_ru/temporal` | 5-shot, temperature 0, 8 записей | 0-shot, temperature 0.3, 8 записей |
| `rutar/closed` | 5-shot, все 202 вопроса | 0-shot, все 202 вопроса |
| `rulegalner_manual/legal` | 5-shot, все 201 фрагмент | 0-shot, все 201 фрагмент |
| RuLaw-ProofBench, закрытая книга | `rulaw_proofbench/mcq_closed`, 5-shot | `rulaw_proofbench/closed` и `rulaw_proofbench/mcq_closed`, 0-shot |
| RuLaw-ProofBench, открытая книга | `rulaw_proofbench/mcq_grounded`, 5-shot | `rulaw_proofbench/grounded` и `rulaw_proofbench/mcq_grounded`, 0-shot |

Для LawMC сохранён generation override группы Shlepa исходного Instruct Fast.
Foundational использует настоящие пять demonstrations; Instruct — zero-shot.
Демонстрации выбираются из первых пригодных уникальных вопросов train и
исключаются из evaluation вместе с дубликатами тех же вопросов. Запрошенное
число примеров сохраняется; недостаток данных или контекста даёт явную ошибку.
Пять соседних записей для формирования distractor options остаются отдельным
механизмом. Подробности — в [протоколе Shlepa](shlepa.md).

Для LegalBench-RU в Instruct перенесён лимит 200, используемый Instruct Fast
для ограничиваемых задач. Чтобы оценить весь корпус, удалите его
`max_sample_per_dataset`. Состав и скоринг LegalBench-RU описаны
[отдельно](legalbench_ru.md).

RuTaR — бинарные налоговые вопросы с оценкой вероятностей `0`/`1` и accuracy.
Пять постоянных демонстраций исключены из оценки даже при k=0; пустой вопрос
и один дубль также исключены. Исходные письма с ответами не входят в промпт.
Версия данных и split закреплены; подробности — в [протоколе RuTaR](rutar.md).
Лимит 200 группы LegalBench-RU к RuTaR не применяется.

`rulegalner_manual/legal` использует ручную переразметку части RuLegalNER
из [Bishop-Y/Bachelor_Thesis](https://github.com/Bishop-Y/Bachelor_Thesis).
Это экспериментальная оценка извлечения `LAW`, `PROVISION`, `PENALTY` из
судебных текстов. Основной результат — существующий NER macro-F1 по точным
строкам и типам с учётом повторов; порядок и координаты упоминаний не оцениваются.
Данные исходного автоматического RuLegalNER не используются как gold.
Лимит 200 группы LegalBench-RU к этой задаче не применяется.

`rulaw_proofbench/closed` — отдельный синтетический набор знания и локального
применения 30 статей четырёх актов на 01.01.2025. Данные и README опубликованы в
[RefalMachine/RuLaw-ProofBench](https://huggingface.co/datasets/RefalMachine/RuLaw-ProofBench);
загрузчик проверяет закреплённый коммит и SHA-256. Основной показатель — macro
accuracy по статьям. Instruct генерирует краткие открытые ответы: основной
scorer — [normalized exact match](../dev/rulaw_proofbench/SCORING.md) без предложений-алиасов;
дополнительная `llm_judge_accuracy` включается через
те же `LLMAAJ_API_BASE`, `LLMAAJ_API_KEY`, `LLMAAJ_MODEL_NAME`, что и RAG.
Судья проверяет эквивалентность неизменному gold и не заменяет юридическую валидацию.
Foundational использует дополняющую MCQ-версию тех же случаев: 3 варианта в
бинарных задачах (да/нет/недостаточно данных), 4 в остальных, выбор по вероятности.
Foundational использует ровно 5 демонстраций из отдельного `train` по статьям,
которых нет в тесте или его зависимостях; Instruct использует zero-shot.
Все 300 тестовых ID сохраняются при любом k от 0 до 5. Переполнение контекста — ошибка,
демонстрации не сокращаются. Оба YAML включают closed-book (`closed`, без текста
нормы) и open-book (`grounded`, с текстом нормы). Instruct дополнительно включает
zero-shot MCQ для обоих режимов; вероятность вариантов оценивается так же, как
в Foundational. Все постановки RuLaw используют 300 вопросов и temperature 0.
Открытый формат ответа и открытая книга — разные признаки: например,
`mcq_grounded` означает выбор варианта с предоставленным текстом нормы.
Пакет содержит первоисточники, правила, независимую
автоматическую реконструкцию и воспроизводимые проверки; юридической экспертной
разметки нет, различение моделей требует дальнейшего эксперимента.
См. [пакет и команды](../dev/rulaw_proofbench/README.md).

Итого Foundational создаёт 9 отдельных результатов, Instruct — 11.
Для открытых ответов RuLaw прямой скоринг и опциональный LLM-судья записываются
как метрики одного результата; MCQ использует только точность выбора варианта.

Thinking выключен в обоих конфигах. Foundational использует контекст 16 000 и
`probe_api_prefill: true`. Instruct наследует
`assistant_prefill_policy: best_effort` и не задаёт размер контекста:
действует настройка backend.
`best_effort` фиксируется в provenance как непроверенное продолжение, не как
подтверждение точной local/API parity.

`batch_size` не задан, как и в исходных конфигах. Общие runners выбирают 8 для HF
и 10 000 000 для local vLLM/API. Это размер evaluator batch; backend может
дополнительно ограничивать фактическую конкурентность запросов.

## Запуск

Из корня репозитория, внутри существующего совместимого Docker-образа:

```bash
python -m benchmark.calculate_benchmark \
  --model_dir Qwen/Qwen3.5-2B \
  --benchmark_config benchmark/llmtf_legal_instruct.yaml \
  --backend vllm --num_gpus 1 \
  --output_dir results/legal/instruct/Qwen3.5-2B

python -m benchmark.calculate_benchmark \
  --model_dir Qwen/Qwen3.5-2B-Base \
  --benchmark_config benchmark/llmtf_legal_foundational.yaml \
  --conv_path conversation_configs/default_foundational.json \
  --backend vllm --num_gpus 1 \
  --output_dir results/legal/foundational/Qwen3.5-2B-Base
```

Для HF передайте `--backend hf`. Те же YAML принимают
`benchmark.calculate_benchmark_api` и
`benchmark.calculate_benchmark_existing_api`. Для Base API необходим
соответствующий серверный foundational template; credentials передаются
через environment.

Каждый конфиг запускается в один каталог модели. LawMC, все четыре режима
LegalBench-RU, RuTaR и ручной RuLegalNER имеют отдельные имена artifacts. Base и Instruct результаты
хранятся раздельно. Дополнительный Instruct 5-shot можно добавить отдельной
записью `tasks` с `few_shot_count: 5` и `name_suffix: 5shot`.

## Результаты

Evaluator сохраняет отдельный `<task>_total.jsonl` для каждого компонента и
`evaluation_results.txt` с их значениями. Общая таблица моделей строится
существующим `show_results.py`, без файла категорий:

```bash
python show_results.py \
  --benchmark_config benchmark/llmtf_legal_instruct.yaml \
  --log_dir results/legal/instruct \
  --output_dir reports/legal/instruct

python show_results.py \
  --benchmark_config benchmark/llmtf_legal_foundational.yaml \
  --log_dir results/legal/foundational \
  --output_dir reports/legal/foundational
```

Отчёт содержит LawMC и отдельные значения closed, grounded, distractor,
temporal, accuracy RuTaR, macro-F1 ручного RuLegalNER и отдельные closed/open-book
результаты RuLaw для включённых форматов ответа. Парные сравнения не нужны для его формирования.
Стандартный `Mean`
остаётся обычным средним показанных задач; методология итогового балла
юридического бенчмарка пока не определена. Не считайте такой `Mean`
согласованным общим legal score. Проверяйте наличие всех 9 Foundational или 11 Instruct totals:
стандартный отчёт может отображать результаты неполного запуска.

`dev/tools/legalbench_ru_report.py` — дополнительная maintainer-утилита для
проверки скоринга и целостности сохранённых LegalBench-RU artifacts. Она не
требуется для запуска или обычной таблицы юридического бенчмарка.

## Ручной RuLegalNER

Snapshot от 2026-10-04 содержит train/val/test из 1000/134/201 фрагментов.
Все три исходных JSONL закреплены по SHA-256 в
[manifest.json](../llmtf/tasks/rulegalner_manual/manifest.json). Проверяются координаты, типы,
отсутствие пересечений spans и пересечения текстов/исходных документов между
split. Тест включает 27 документов, 145 LAW, 139 PROVISION, 50 PENALTY;
95 фрагментов не содержат ни одной сущности выбранных классов и сохраняются.
Кейсы одного документа зависимы: обычный sample bootstrap выключен.

Загрузчик скачивает оригинальные файлы с upstream Google Drive и кэширует их
в `$HF_HOME/llmtf/rulegalner_manual` (по умолчанию `~/.cache/huggingface/llmtf/rulegalner_manual`).
Загрузка требует сеть при отсутствии кэша; `HF_HUB_OFFLINE` сам по себе не
управляет запросами к Google Drive. Для offline-режима укажите
`LLMTF_RULEGALNER_MANUAL_DATA_DIR` с файлами `train_annotated.jsonl`,
`val_annotated.jsonl`, `test_annotated.jsonl`; hashes остаются обязательными.
Данные в репозитории LLMTF не распространяются. Явная лицензия на сам корпус
не подтверждена; CC BY-NC 4.0 в карточке upstream модели не считается
лицензией на автоматически перепубликуемые данные.

Пять demonstrations — фиксированные train IDs 1105, 291, 516, 546, 985 из
разных документов. Они показывают три класса, повторные упоминания и пустой
ответ. Это технически просмотренные примеры, не экспертная юридическая
валидация. `k=0..5` выбирает их префикс, полный evaluation содержит одни и те же
201 строки при любом k. Оба legal YAML используют полный test;
`max_sample_per_dataset` может явно ограничить его первыми N строками.
При доступном счётчике токенов переполнение контекста выявляется до генерации;
без счётчика превышение лимита обрабатывает backend. В обоих случаях k не
уменьшается. Answer budget — 1024 токена. Нет task assistant-prefill.
Задача исключена из общего `all` и включена явно в оба legal YAML.

Gold получается как `[class, text[start:end]]` без исправления upstream
аннотаций и нормализации пробелов. Ответ — JSON-массив таких пар; разрешена
одна внешняя Markdown-обёртка. Повторы сохраняются, порядок игнорируется.
Некорректный JSON, неверная структура или посторонний класс дают пустой набор
предсказаний и `format_valid=0`; частичные ответы из malformed JSON не извлекаются.
Корректный `[]` имеет `format_valid=1`. На пустом gold неверный формат получает
`exact_match=0`; F1 не добавляет искусственных сущностей для штрафования формата.
Лишние сущности на отрицательных примерах учитываются как FP основного F1.

TP/FP/FN суммируются по корпусу для каждого класса, затем усредняются три F1.
Класс без TP имеет F1=0, как в общем NER-скорере; маленький smoke без всех
классов нельзя интерпретировать как оценку полного теста. `format_valid` и
`exact_match` сохраняются как отдельные диагностические доли и не примешиваются
к leaderboard result. В aggregation details сохраняются TP/FP/FN/F1 по классам.
Сравнение с авторским strict span F1 не является целью протокола.

Разметка выполнена одним человеком и сохранена без экспертной коррекции.
Техническая проверка не доказывает её полноту: при просмотре встречались
сомнительные границы и пропуски. Результат измеряет соответствие этому
экспериментальному gold, а не юридическую правильность решения.

## Исправления общего NER скоринга от 2026-10-04

JSON-парсер проверяет форму каждой пары до скоринга и снимает только внешнюю
Markdown-обёртку, не изменяя строки сущностей. В словарном формате повторные
строки одного класса объединяются вместо перезаписи, пустые списки и неверный
тип ответа обрабатываются без падения. In-place извлечение поддерживает
многострочные сущности. In-place проверка сохраняет всю пунктуацию, включая
дефисы, тире и символы вне прежнего regex; пробелы между токенами не учитываются.
Collection3 проверяет полный исходный текст, а не только совпадающий префикс.
NEREL demonstrations сохраняют неразмеченный хвост текста;
BIO demonstrations с одиночными метками больше не добавляют лишнюю скобку.
Порядок сущностей, учёт повторов и формула F1 не изменены. В provenance общего
NER добавлены версия и hash общего скорера: старый кэш после исправления не
используется как результат нового протокола.

## Проверка интеграции RuLegalNER от 2026-10-04

[Машиночитаемый отчёт](../dev/RULEGALNER_MANUAL_VALIDATION_RESULTS.json)
содержит версии образов и моделей, параметры запуска, fingerprints и метрики.
Пути `/tmp/` в отчёте относятся к локальным artifacts проверочного запуска;
сами ответы моделей не включены в репозиторий.
Проверены HF, локальный vLLM и API-клиент без torch/vLLM: Base и Instruct,
0/5-shot, по 8 фрагментов, thinking off, generate. Всего 12 наборов artifacts
и 96 ответов; это smoke, не оценка качества на полном тесте и не полная v4 matrix.
Использованы существующие образы, без пересборки. В smoke temperature=0;
общий Instruct YAML сохраняет temperature=0.3.

Все сохранённые ответы пересчитаны финальным скорером, метрики и aggregation
details сверены с totals. HF/local-vLLM запускались до последнего исправления
проверки пунктуации in-place; JSON-парсер и F1 между этими версиями идентичны.
Исходные fingerprints сохранены, оба hash отражены в отчёте. Полная идентичность
генераций между бэкендами не утверждается.

Прошли 12 общих NER regressions, 8 тестов новой задачи (включая roundtrip gold
всех 201 фрагмента), 7 integration tests legal YAML и 40 dependency-free tests.
Также проверены загрузка оригиналов и повторное чтение кэша, compileall и
`git diff --check`. Pytest в используемом окружении отсутствует.

Для повторения task smoke внутри соответствующего Docker-образа:

```bash
python -m dev.tools.validate_rulegalner_manual \
  --backend vllm --model /path/to/pinned/model/snapshot \
  --data-dir /audit/rulegalner-manual-data \
  --output /audit/rulegalner-validation/vllm/instruct
```

Для Base добавьте `--base`; для HF — `--backend hf --batch-size 8`.
Для API — `--backend api --model <served-name> --api-base http://127.0.0.1:18765`.
Сервер Base API должен использовать foundational template; snapshot IDs и
серверные параметры приведены в отчёте. По умолчанию утилита выполняет оба
режима 0/5-shot с лимитом 8. Для повторного запуска после изменения скорера
используйте новый каталог результатов; обычные CLI также поддерживают
`--force_recalc`.

## Проверка расширенного RuLaw-ProofBench от 2026-10-04

Ревизия HF `cae70d59bdd44bd28e3fbd411b0c8acf7427ef98` содержит по 300 test
и 5 train в конфигурациях `open` и `mcq`. Анонимное чтение всех четырёх Parquet
и проверка SHA-256 прошли в API-образе без torch и vLLM. Прошли 13 тестов
RuLaw, 7 тестов общей legal-интеграции и dependency-free regressions.

Реальный запуск Qwen3.5-2B-Base в `llmtf:vllm-cu129` проверил 8 вопросов
MCQ closed: во всех запросах сохранены те же 5 демонстраций, thinking выключен,
вероятности корректны. Проверены `_params.jsonl`, `_total.jsonl` и записи
каждого примера. Micro-accuracy — 4/8, macro по 7 представленным статьям —
3/7; это проверка исполнения, не оценка качества на полном наборе.

Команда внутри контейнера с GPU и смонтированными репозиторием, модельным
кэшем и `/tmp` в `/audit`:

```bash
python -m dev.tools.validate_rulaw_proofbench \
  --backend vllm \
  --model /root/.cache/huggingface/hub/models--Qwen--Qwen3.5-2B-Base/snapshots/b1485b2fa6dfa1287294f269f5fb618e03d52d7c \
  --base --mcq --few-shot-count 5 --modes closed --limit 8 \
  --output /audit/rulaw-release-runtime/base-mcq-5shot
```

Артефакты проверки находятся в `/tmp/rulaw-release-runtime/base-mcq-5shot`;
нового сравнительного прогона моделей на всех 300 вопросах ещё нет.
