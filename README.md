# LLMTF Open

LLMTF Open — фреймворк для оценки языковых моделей на русскоязычных и
мультиязычных задачах. Он поддерживает локальные Hugging Face и vLLM модели,
а также OpenAI-compatible API.

Текущая версия — `v0.3.0`. Изменения и инструкция по миграции находятся в
[`docs/releases/v0.3.0.md`](docs/releases/v0.3.0.md).

Текущая архитектура проверена на `Qwen/Qwen3.5-2B` и
`Qwen/Qwen3.5-2B-Base` через HF, local vLLM и vLLM-compatible API. Известные
ограничения находятся в [`BACKLOG.md`](BACKLOG.md).

## Возможности

- генерация и next-token probability через HF, локальный vLLM и API;
- HF-only PPL как средний log probability токенов ответа;
- plain, hybrid и обязательный two-pass reasoning режимы;
- точное локальное продолжение assistant-prefill без скрытого trimming;
- YAML benchmark runner с распределением локальных задач или API servers по GPU;
- fingerprinted results cache и fail-closed обработка частичных ошибок;
- foundational/base модели с отдельным conversation config;
- LLM-as-a-Judge pipeline.

PPL не поддерживается vLLM/API. Native provider reasoning fields и function
calling пока не входят в transport-контракт.

## Установка

### Docker — рекомендуемый путь

Профили разделены по назначению:

- `api` — CPU-only клиент без torch, CUDA и vLLM;
- `hf` — CUDA 12.9, torch 2.11, Transformers и kernels для Qwen3.5;
- `vllm` — расширяет HF-образ vLLM 0.21.

Проверенные `v0.3.0` runtime-образы для `linux/amd64` опубликованы в
[`refalmachine/llmtf`](https://hub.docker.com/r/refalmachine/llmtf):

```bash
docker pull refalmachine/llmtf:v0.3.0-api
docker pull refalmachine/llmtf:v0.3.0-hf-cu129
docker pull refalmachine/llmtf:v0.3.0-vllm-cu129
```

Образы содержат зависимости и CUDA runtime, но не копию исходников LLMTF.
Запускайте их из checkout репозитория, смонтированного в `/workdir`.

```bash
docker build -f docker/Dockerfile.api -t llmtf:api .
docker build -f docker/Dockerfile.hf -t llmtf:hf-cu129 .
docker build \
  -f docker/Dockerfile.vllm \
  --build-arg HF_BASE_IMAGE=llmtf:hf-cu129 \
  -t llmtf:vllm-cu129 .
```

```bash
docker run --rm -it --gpus all --ipc=host \
  -v "$PWD:/workdir" -w /workdir \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  refalmachine/llmtf:v0.3.0-vllm-cu129
```

Сборка, proxy и GPU architecture options описаны в
[`docker/README.md`](docker/README.md).

### Локальная установка

CPU/API профиль можно установить отдельно:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements/profiles/api.txt
```

Для HF/vLLM рекомендуется Docker: там согласованы torch, CUDA и compiled
kernels. Единственным источником зависимостей являются файлы в
`requirements/profiles/`; монолитный legacy-профиль удалён.

## Быстрый запуск

### Локальный Hugging Face

```bash
CUDA_VISIBLE_DEVICES=0 python evaluate_model.py \
  --model_name_or_path /models/instruct \
  --output_dir /results/hf \
  --dataset_names russiannlp/rucola_custom \
  --model_context_len 8192 \
  --model_kind plain \
  --disable_thinking \
  --few_shot_count 5 \
  --batch_size 1
```

Без `--vllm` используется `HFBackend`. Для PPL добавьте `--ppl_scoring`.

### Локальный vLLM

```bash
CUDA_VISIBLE_DEVICES=0 python evaluate_model.py \
  --model_name_or_path /models/instruct \
  --output_dir /results/vllm \
  --dataset_names russiannlp/rucola_custom \
  --model_context_len 8192 \
  --model_kind hybrid \
  --disable_thinking \
  --gpu_memory_utilization 0.92 \
  --vllm
```

`gpu_memory_utilization` по умолчанию также равен `0.92`. Флаг в команде
оставлен явно для воспроизводимости.

### Two-pass reasoning

Thinking всегда opt-in. Для hybrid/reasoning запуска требуется model-specific
id закрывающего reasoning-токена:

```bash
CUDA_VISIBLE_DEVICES=0 python evaluate_model.py \
  --model_name_or_path /models/hybrid \
  --output_dir /results/reasoning \
  --dataset_names darumeru/flores_ru_en \
  --model_context_len 8192 \
  --model_kind hybrid \
  --enable_thinking \
  --end_thinking_token_id ID \
  --max_new_tokens_reasoning 2048 \
  --min_new_tokens_reasoning 512 \
  --gpu_memory_utilization 0.92 \
  --vllm
```

Получайте `ID` из tokenizer конкретной модели для `THINK_CLOSE_MARKER`; не
переносите его между моделями. Reasoning/tool parsers vLLM также являются
model-specific и не включаются framework'ом автоматически.

### OpenAI-compatible API

Пример текстового vLLM server:

```bash
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
  --model /models/instruct \
  --port 8000 \
  --tensor-parallel-size 1 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.92 \
  --language-model-only
```

```bash
OPENAI_API_KEY=EMPTY python evaluate_model_api.py \
  --base_url http://127.0.0.1:8000 \
  --api_profile vllm \
  --model_name_or_path /models/instruct \
  --output_dir /results/api \
  --dataset_names russiannlp/rucola_custom \
  --model_context_len 8192 \
  --model_kind plain \
  --disable_thinking
```

Используйте `api_profile=openai` для generic API и `vllm` только для сервера с
соответствующими extensions. Ключ передавайте через окружение; не записывайте
его в YAML или image. Подробности: [`docs/api_backend.md`](docs/api_backend.md).

### Foundational/base модель

```bash
CUDA_VISIBLE_DEVICES=0 python evaluate_model.py \
  --model_name_or_path /models/base \
  --conv_path conversation_configs/default_foundational.json \
  --output_dir /results/base \
  --dataset_names darumeru/flores_ru_en \
  --model_context_len 8192 \
  --model_kind plain \
  --disable_thinking \
  --is_foundational \
  --gpu_memory_utilization 0.92 \
  --vllm
```

## Режимы модели

| `model_kind` | Thinking выключен | Thinking включён |
|---|---|---|
| `plain` | one-pass | warning и one-pass |
| `hybrid` | one-pass | two-pass reasoning |
| `reasoning` | ошибка конфигурации | обязательный two-pass |

Thinking включается только `--enable_thinking`; deprecated
`--disable_thinking` оставлен как явный compatibility alias. Two-pass требует
`num_return_sequences=1`. PPL всегда one-pass и только HF.

Контекстный бюджет вычисляется из `model_context_len`, максимума ответа задачи
и reasoning budget. Удалённый `max_prompt_len` больше не используется.

## Benchmark YAML

```yaml
model:
  model_kind: hybrid
  enable_thinking: false
  model_context_len: 8192
  max_new_tokens_reasoning: 2048
  min_new_tokens_reasoning: 512
  end_thinking_token_id: null

defaults:
  evaluation:
    few_shot_count: 0
    batch_size: 8
  generation:
    temperature: 0.0

tasks:
  - name: classification
    datasets: [russiannlp/rucola_custom]
    enable_thinking: false
```

Неизвестные поля и противоречивые reasoning-настройки завершаются ошибкой до
загрузки модели. `backend_kwargs` не должен содержать secrets. Точный schema и
приоритеты: [`docs/configuration.md`](docs/configuration.md).

Локальный runner:

```bash
python benchmark/calculate_benchmark.py \
  --model_dir /models/instruct \
  --benchmark_config benchmark/config_balanced.yaml \
  --output_dir /results/benchmark \
  --num_gpus 4 \
  --tensor_parallel_size 1 \
  --backend vllm
```

Управляемые vLLM API servers:

```bash
python benchmark/calculate_benchmark_api.py \
  --model_dir /models/instruct \
  --benchmark_config benchmark/config_balanced.yaml \
  --output_dir /results/api-benchmark \
  --num_gpus 4 \
  --tensor_parallel_size 1 \
  --gpu_memory_utilization 0.92 \
  --base_port 8000
```

Для уже запущенного endpoint используйте
`benchmark/calculate_benchmark_existing_api.py`.

## Результаты и ошибки

На каждую задачу создаются:

- `<task>.jsonl` — JSON array с per-sample результатами;
- `<task>_params.jsonl` — sanitized run config и fingerprint;
- `<task>_total.jsonl` — агрегаты и тот же fingerprint;
- `evaluation_results.txt` и `evaluation_log.txt`.

Кеш используется только при совпадении fingerprint. Частичная backend-ошибка
даёт ненулевой exit code и не создаёт новый total. Подробнее:
[`docs/results.md`](docs/results.md).

Сводную Markdown-таблицу можно построить так:

```bash
python show_results.py \
  --log_dir /results/models \
  --output_dir /results/report \
  --benchmark_config benchmark/config_balanced.yaml \
  --category_path benchmark/categories.json
```

## Архитектура и расширение

Основной путь: `BaseLLM -> LLM -> Backend`. `LLM` владеет reasoning dispatch,
backend выполняет только primitives. Старые model facades удалены и не должны
возвращаться. Описание компонентов: [`docs/architecture.md`](docs/architecture.md).

Новая задача наследует `SimpleFewShotHFTask`, задаёт
`_max_task_new_tokens`, dataset/split methods, `create_messages`, `evaluate` и
`aggregation`, затем регистрируется в `llmtf/tasks/__init__.py`.
Минимальная задача на локальном датасете, программный запуск и небольшой
benchmark YAML находятся в [`examples/`](examples/README.md).

LLM-as-a-Judge запускается отдельным pipeline, описанным в
[`docs/llmaaj.md`](docs/llmaaj.md).

## Проверка изменений

```bash
python3 tests/test_refactor_logic.py
python3 -m unittest discover -s tests -p 'test_*.py'
python3 -m compileall -q llmtf evaluate_model.py evaluate_model_api.py benchmark show_results.py dev/tools examples
git diff --check
```

Успешные unit-тесты не заменяют GPU/API smoke после изменения runtime,
контейнеров, reasoning или backend payload.

## Документация

Навигация по документации находится в [`docs/README.md`](docs/README.md).
