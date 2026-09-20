# Примеры

Эта директория содержит небольшие исходные примеры для текущего публичного
API. Результаты запусков сюда намеренно не коммитятся: их формат описан в
[`docs/results.md`](../docs/results.md), а реальные runtime-проверки — в
[`dev/TEST_REPORT_v4.md`](../dev/TEST_REPORT_v4.md).

## Готовый smoke benchmark

[`benchmark_smoke.yaml`](benchmark_smoke.yaml) ограничивает каждую задачу
восемью примерами и проверяет генерацию вместе с token probability.

Для уже запущенного vLLM endpoint:

```bash
OPENAI_API_KEY=EMPTY python benchmark/calculate_benchmark_existing_api.py \
  --model_name /models/instruct \
  --base_url http://127.0.0.1:8000 \
  --api_profile vllm \
  --benchmark_config examples/benchmark_smoke.yaml \
  --output_dir /results/smoke
```

Для локального vLLM backend:

```bash
python benchmark/calculate_benchmark.py \
  --model_dir /models/instruct \
  --benchmark_config examples/benchmark_smoke.yaml \
  --output_dir /results/smoke-local \
  --num_gpus 1 \
  --backend vllm
```

`model_kind`, thinking mode и reasoning token id всегда задавайте под
конкретную модель. Пример специально использует безопасный one-pass режим и не
пытается автоматически выбирать model-specific reasoning/tool parsers.

## Своя задача и программный API

[`custom_task.py`](custom_task.py) показывает минимальную реализацию
`SimpleFewShotHFTask` на локальном JSONL-датасете. В ней есть обязательный
`_max_task_new_tokens`, few-shot messages, список вариантов для token
probability, sample metric и aggregation.

[`run_custom_task.py`](run_custom_task.py) регистрирует эту задачу в
`Evaluator`, собирает `LLM(backend=...)` и возвращает ненулевой код при ошибке
оценки. Запуск через API:

```bash
OPENAI_API_KEY=EMPTY python -m examples.run_custom_task \
  --backend api \
  --base-url http://127.0.0.1:8000 \
  --api-profile vllm \
  --model-name /models/instruct \
  --output-dir /results/custom-task
```

Тот же пример с локальным vLLM:

```bash
CUDA_VISIBLE_DEVICES=0 python -m examples.run_custom_task \
  --backend vllm \
  --model-name /models/instruct \
  --model-context-len 8192 \
  --gpu-memory-utilization 0.92 \
  --output-dir /results/custom-task-local
```

Для Transformers используй `--backend hf`. `--conversation-template auto`
оставляет chat template tokenizer; для base-модели добавь `--is-foundational`.

Датасет из [`data/`](data/) учебный и не предназначен для сравнения моделей.
Чтобы добавить постоянную задачу, перенеси класс в `llmtf/tasks/` и
зарегистрируй его в `llmtf/tasks/__init__.py`.

## Conversation templates

При `--conv_path auto` используется chat template tokenizer. Поддерживаемые
ручные шаблоны лежат в [`conversation_configs/`](../conversation_configs/),
включая `default_foundational.json` для base-моделей. Отдельные копии шаблонов
в `examples/` не поддерживаются.
