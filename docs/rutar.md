# RuTaR в LLMTF

`rutar/closed` — бинарная задача по налогообложению из
[RuTaR](https://github.com/rutar-anonymous/RuTaR). Используется
`question_for_llm`, gold берётся из `true_answer`: `1` означает «Да», `0` — «Нет».
LLMTF сравнивает вероятности двух цифр через `calculate_tokens_proba`;
основная метрика — accuracy. Равные вероятности считаются ошибкой; отсутствующие
или некорректные вероятности приводят к ошибке выполнения задачи.
Это протокол LLMTF, а не воспроизведение RAG-экспериментов авторов.

## Данные и split

Штатный источник — [RefalMachine/RuTaR](https://huggingface.co/datasets/RefalMachine/RuTaR),
HF revision `fdcc020429120cd81945b1b8060f5aa6a9840a18`, файл `data/test.parquet`.
Загрузчик использует `hf_hub_download`: повторные запуски читают локальный
Hub cache. SHA-256 Parquet проверяется перед разбором. Новые зависимости и
пересборка образов не нужны: pyarrow уже входит в профиль через datasets.

Зеркало сохраняет все 209 исходных записей, их порядок и аннотации, оригинальный
XLSX и корпус из 480 источников. Карточка содержит ссылки на исходный
[GitHub](https://github.com/rutar-anonymous/RuTaR), закреплённый upstream commit
`76c6ef0cafe89fe1b400b673478066108b0493ac`, документы и описание преобразования.
Upstream не задаёт лицензию; зеркало не добавляет новую лицензию.
Отбор для бенчмарка выполняется локально после загрузки полного зеркала.

Для первого запуска нужен доступ к HF, после кэширования возможна работа
с `HF_HUB_OFFLINE=1`. Maintainer-проверка принимает локальный Parquet или
оригинальный XLSX через `--data`; XLSX разбирается стандартной библиотекой.

В файле 209 непустых записей. ID соответствует исходной числовой колонке
индекса, а не номеру строки Excel. Исключены:

- ID `119`: отсутствует `question_for_llm`;
- ID `117`: нормализованный дубль вопроса ID `114` с тем же gold.

Первые пять пригодных уникальных записей, ID `0`–`4`, образуют постоянный
пул демонстраций. Они, дубликаты их вопросов и вопросы из тех же писем
(по `title`) исключаются из evaluation при любом k, включая zero-shot.
Остаются **202 вопроса**: 118 с ответом `0` и 84 с ответом `1`.
Это локальный split LLMTF; исходный репозиторий не задаёт train/test split.

`few_shot_count` принимает 0–5 и выбирает соответствующий префикс пула.
Все запрошенные демонстрации сохраняются. Переполнение контекста даёт явную
ошибку; при недоступном счётчике токенов k сохраняется.
Порядок evaluation совпадает с порядком исходного файла после исключений;
`max_sample_per_dataset` выбирает его префикс.

В промпт входят только инструкция, вопрос и ответы демонстраций. Поля
`full_text`, `question_letter`, `answer_letter`, `found_sources` и корпус
`sources_dataset_for_rutar.json` не используются. Конечное сообщение — `user`,
assistant prefill отсутствует. Это позволяет выполнять задачу и с
`assistant_prefill_policy=portable`.

## Запуск и результаты

Задача включена в оба [legal-конфига](legal_benchmark.md): Foundational —
5-shot, Instruct — 0-shot, в обоих случаях все 202 вопроса.
Отдельный запуск в совместимом Docker-образе:

```bash
python evaluate_model.py \
  --model_name_or_path Qwen/Qwen3.5-2B \
  --dataset_names rutar/closed --few_shot_count 0 \
  --model_kind hybrid --disable_thinking \
  --output_dir results/rutar/instruct
```

Для Base используйте `--is_foundational`, `--model_kind plain`,
`--conv_path conversation_configs/default_foundational.json` и k=5.
API запускается через обычный `evaluate_model_api.py` или общий benchmark runner.
Задача исключена из `all` и требует явного выбора.
PPL пропускается, поскольку `get_answer` отсутствует.

Артефакты имеют имена `rutar_closed_0shot` или `rutar_closed_5shot` в общих
конфигах. В sample сохраняется `_rutar` с ID демонстраций, запрошенным и
эффективным k, числом токенов и размером evaluation. Provenance содержит
upstream и HF commits, hashes файлов, исключения, split/scorer versions и hashes кода до
загрузки данных; эти поля участвуют в проверке cache fingerprint.

Точечная проверка штатного HF-источника:

```bash
python -m dev.tools.validate_rutar \
  --backend hf --model Qwen/Qwen3.5-2B \
  --output /audit/rutar-validation/hf/instruct
```

Утилита проверяет k=0/5 на восьми вопросах, с thinking off. `--base`,
`--backend vllm` и `--backend api --api-base URL` выбирают другие клетки.

В исходном репозитории на закреплённой версии нет файла лицензии; файлы данных
опубликованы в HF, а в Git-репозитории LLMTF хранятся только код и manifest. Ответы оцениваются относительно исходных
аннотаций и исторических документов, без обновления к действующему
налоговому законодательству.

Экспорт и публикация воспроизводятся через `dev.tools.publish_rutar`:

```bash
python -m dev.tools.publish_rutar --workbook /audit/rutar.xlsx \
  --sources /audit/rutar-sources.json --output /audit/rutar-hf
# После проверки локальных файлов, при авторизации через hf auth login:
python -m dev.tools.publish_rutar --workbook /audit/rutar.xlsx \
  --sources /audit/rutar-sources.json --output /audit/rutar-hf --upload
```

Публикация проверяет Parquet roundtrip, не перезаписывает отличающееся зеркало
и анонимно читает опубликованный commit из нового кэша. Токен не передаётся
в команде и не входит в карточку, manifest или результаты.
