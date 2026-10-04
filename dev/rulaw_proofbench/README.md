# RuLaw-ProofBench

[Датасет и карточка на Hugging Face](https://huggingface.co/datasets/RefalMachine/RuLaw-ProofBench).
300 тестовых вопросов по 30 статьям четырёх актов РФ на 01.01.2025;
5 демонстраций по отдельным статьям, не входящим в тест или его зависимости.
Юридической экспертной разметки нет. Измеряется знание и локальное применение
названной нормы при явно заданных правовых квалификациях.

LLMTF берёт данные с HF по закреплённому коммиту и проверяет SHA-256 Parquet.
Пин и загрузчик находятся в `llmtf/tasks/rulaw_proofbench/`.
Локальные материалы построения не используются как запасной источник при оценке.

| Вариант | Задача общего legal-бенчмарка | Демонстрации | Метрика |
|---|---|---:|---|
| Instruct, открытый ответ | `rulaw_proofbench/closed`, `rulaw_proofbench/grounded` | 0 | normalized exact match; дополнительно LLM-судья |
| Instruct, MCQ | `rulaw_proofbench/mcq_closed`, `rulaw_proofbench/mcq_grounded` | 0 | Вероятности букв, точность выбранного варианта |
| Foundational, MCQ | `rulaw_proofbench/mcq_closed`, `rulaw_proofbench/mcq_grounded` | 5 | Вероятности букв, точность выбранного варианта |

`closed` означает закрытую книгу, `grounded` — открытую книгу с текстом нормы.
Оба режима входят в общие legal-конфиги; формат ответа задаётся отдельно.
В `context` используются обычные названия актов; внутренние ID остаются в
метаданных. MCQ содержит 3 варианта для бинарных вопросов и 4 для остальных.
Случайный выбор даёт 30,83% micro-accuracy. Открытая и MCQ-версии связаны по ID
и не являются независимыми выборками.

```bash
python -m dev.tools.validate_rulaw_proofbench \
  --backend vllm --model /models/Qwen3.5-2B-Base --base --mcq \
  --few-shot-count 5 --modes closed grounded --limit 300 --output /results/rulaw-base
```

Контекст должен вместить все запрошенные демонстрации; тихого сокращения k нет.
При k=0 и k=5 тест содержит те же 300 ID. Общие legal YAML включают задачу
отдельно от LegalBench-RU; `all` её не включает. PPL не реализован.

Открытые ответы оцениваются единственным [scorer](SCORING.md), без настроек
под конкретную модель. Семантическая метрика подключается через `LLMAAJ_API_BASE`,
`LLMAAJ_API_KEY`, `LLMAAJ_MODEL_NAME`. Для MCQ судья не нужен.

Полная [методика и ограничения](paper_methods.md).
[Результаты пилота на 150 вопросах](https://huggingface.co/datasets/RefalMachine/RuLaw-ProofBench/blob/main/PILOT_RESULTS.md)
сохранены отдельно: они не являются результатами расширенного набора.
Сырые ответы и журналы вынесены в HF `audit/pilot_results.tar.gz`;
первичные источники, доказательства и проверки — в `audit/construction.tar.gz`.
Архивы извлекаются в отдельный каталог; их пути начинаются от корня проекта.

В Git остаются код, методика, манифесты и аудиты. Тяжёлые первоисточники и
промежуточные JSONL хранятся в закреплённом HF-архиве и игнорируются Git.
Для оценки моделей они не нужны. Для воспроизведения построения восстановите
их из корня репозитория (команды внутри API-образа с установленным `hf`):

```bash
hf download RefalMachine/RuLaw-ProofBench audit/construction.tar.gz \
  --repo-type dataset --revision cae70d59bdd44bd28e3fbd411b0c8acf7427ef98 \
  --local-dir /tmp/rulaw-audit
sha256sum /tmp/rulaw-audit/audit/construction.tar.gz
tar -xzf /tmp/rulaw-audit/audit/construction.tar.gz --wildcards \
  'dev/rulaw_proofbench/sources/*' 'dev/rulaw_proofbench/*.jsonl'
```

SHA-256 архива должен совпасть с `audit/construction.tar.gz` в
`llmtf/tasks/rulaw_proofbench/manifest.json`. Локальный `dataset.jsonl` —
промежуточный пилот на 150 вопросов, из которого сборщик получает расширенный
набор; актуальные 300 тестовых вопросов загружаются с HF.

Дополнительные поправки для расширения извлекаются с исходными именами,
которые ожидает сборщик:

```bash
mkdir -p /tmp/rulaw-amendments
tar -xzf /tmp/rulaw-audit/audit/construction.tar.gz \
  -C /tmp/rulaw-amendments --wildcards \
  --transform='s|^expanded_release/additional_sources/amendment-|rulaw-amend-|' \
  'expanded_release/additional_sources/*.mht' \
  'expanded_release/additional_sources/*-metadata.html'
```

Для воспроизведения расширения из восстановленных официальных снимков:

```bash
python -m dev.rulaw_proofbench.build_expansion --output /tmp/rulaw-candidate
python -m dev.rulaw_proofbench.release \
  --output /tmp/rulaw-release --amendment-cache /tmp/rulaw-amendments
python -m dev.tools.publish_rulaw_proofbench \
  --expanded /tmp/rulaw-release --output /tmp/rulaw-publication/dataset
```

`build_expansion` не публикует кандидатов. `release` требует актуальных аудитов
источников и языка и повторяет проверки. `--upload` у publisher — явная публикация
в HF; обычная оценка никогда не изменяет данные или их пин. Дополнительные
поправки и их исходные имена перечислены в архиве и `expansion_source_review.json`.
