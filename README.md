# LLMTF Open

**LLMTF Open** — фреймворк для оценки больших языковых моделей, с фокусом на русскоязычные задачи и запуск как локальных Hugging Face/vLLM моделей, так и моделей через OpenAI-compatible API.

## Возможности

- Оценка инструктивных, базовых и reasoning-моделей.
- Запуск через `transformers`, локальный `vllm` или vLLM OpenAI API server.
- Автоматизированный benchmark с распределением задач по нескольким GPU.
- Message-based формат задач: `system`/`user`/`assistant`.
- Методы оценки: генерация, вероятности токенов, PPL для локальных моделей.
- Набор русскоязычных задач: классификация, MMLU, перевод, суммаризация, NER, RAG, IFEval, Libra, DaruMeru.
- LLM-as-a-Judge benchmark с сохранением результатов и сравнением моделей.

## Установка

### Вариант 1: Docker

Рекомендуемый вариант для GPU-запуска и vLLM. 

```bash
docker build -t llmtf-open:ngc-vllm .
```
Запуск контейнера из корня репозитория:
```bash
docker run --gpus all --rm -it \
  -v "$PWD":/workdir \
  -w /workdir \
  llmtf-open:ngc-vllm
```
### Вариант 2: локальное окружение
Для локального запуска без Docker:
```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools packaging ninja
pip install -r requirements.txt
```

Для GPU/vLLM лучше использовать CUDA-совместимое окружение. 

## Быстрый запуск

### Запуск полного benchmark

`benchmark/calculate_benchmark.py` запускает локальную оценку и сам распределяет группы задач по GPU.

```bash
python benchmark/calculate_benchmark.py \
  --model_dir /path/to/model \
  --benchmark_config benchmark/config_balanced.yaml \
  --conv_path conversation_configs/qwen3.json \
  --output_dir /path/to/output/benchmark \
  --num_gpus 8 \
  --tensor_parallel_size 2 \
  --backend vllm
```

### Запуск benchmark через vLLM API

`benchmark/calculate_benchmark_api.py` поднимает несколько vLLM OpenAI API server'ов и распределяет задачи между ними.

```bash
python benchmark/calculate_benchmark_api.py \
  --model_dir /path/to/model \
  --benchmark_config benchmark/config_balanced.yaml \
  --conv_path conversation_configs/qwen3.json \
  --output_dir /path/to/output/api_benchmark \
  --num_gpus 4 \
  --tensor_parallel_size 1 \
  --base_port 8000 \
  --api_key EMPTY \
  --force_recalc
```

### Оценка foundational/base модели

```bash
CUDA_VISIBLE_DEVICES=0 python evaluate_model.py \
  --model_name_or_path /path/to/base_model \
  --conv_path conversation_configs/default_foundational.json \
  --output_dir /path/to/output/llmtf_eval_base \
  --few_shot_count 5 \
  --max_prompt_len 4000 \
  --batch_size 8 \
  --vllm \
  --is_foundational
```

## Поддерживаемые задачи

### Знания и рассуждения

- **MMLU** — тесты общих знаний на русском и английском языках.
- **Shlepa** — специализированные домены: фильмы, музыка, право, книги. 
- **DaruMeru** — комплексный русскоязычный бенчмарк: reasoning, QA, world knowledge и NLI-подобные задачи. 

### Навыки и способности

- **Перевод** — Flores ru/en и en/ru.
- **Суммаризация** — генерация кратких версий новостных и текстовых документов. 
- **Анализ тональности и классификация** — оценка классификационных способностей и извлечение мнений. 
- **NER** — распознавание именованных сущностей, включая вложенные и биомедицинские сущности. 
- **RAG** — вопросно-ответные задачи с контекстом из поисковой выдачи.

### Специализированные задачи

- **IFEval** — проверка следования инструкциям.
- **Libra** — задачи на работу с длинными контекстами до 32K токенов. 
- **Математика и физика** — задачи с проверкой финального ответа.
- **Copy tasks** — проверка устойчивости на копировании предложений, абзацев и документов. 

Полный список задач в `llmtf/tasks/__init__.py`.


## Архитектура

### Основные компоненты

- **`llmtf/`** — ядро фреймворка:
  - `base.py` — базовые классы `Task` и `LLM`;
  - `model.py` — реализации `HFModel`, `VLLMModel` и API-моделей;
  - `evaluator.py` — основной класс для оценки;
  - `tasks/` — коллекция задач для оценки.

- **`benchmark/`** — автоматизированный benchmark:
  - `calculate_benchmark.py` — параллельное выполнение задач локальными моделями;
  - `calculate_benchmark_api.py` — параллельное выполнение задач через vLLM API servers;
  - `llmaaj/` — LLM-as-a-Judge оценка.

- **`conversation_configs/`** — конфигурации chat templates для разных моделей.


## Конфигурации

### Conversation configs

Файлы в `conversation_configs/` описывают chat template для моделей.

Для instruct-моделей обычно указывается конкретный chat config. Для base/foundational моделей используйте `conversation_configs/default_foundational.json` и флаг `--is_foundational`.

### Benchmark configs

Файлы в `benchmark/*.yaml` группируют датасеты и параметры:
- `benchmark/config_balanced.yaml` — полный benchmark.
- `benchmark/llmtf_benchmark_instruct.yaml` и `benchmark/llmtf_benchmark_instruct_fast.yaml` — instruct-наборы.

В YAML можно задавать task groups, `datasets`, `few_shot_count`, `max_prompt_len`, `max_sample_per_dataset`, `batch_size`, `name_suffix`, `max_new_tokens_reasoning`, параметры `generation` и `extra_args`.

## Результаты

Для каждой задачи в `output_dir` создаются:

- `<task>_params.jsonl` — параметры запуска.
- `<task>.jsonl` — результаты по отдельным примерам.
- `<task>_total.jsonl` — агрегированные метрики.
- `evaluation_results.txt` — сводная таблица.
- `evaluation_log.txt` — лог запуска.

Если файл `<task>_total.jsonl` уже существует, задача пропускается. Чтобы пересчитать результаты, используйте `--force_recalc`.


## Дополнительные сценарии запуска

### Локальная оценка одной модели через vLLM

```bash
CUDA_VISIBLE_DEVICES=0 python evaluate_model.py \
  --model_name_or_path /path/to/model \
  --conv_path conversation_configs/qwen3.json \
  --output_dir /path/to/output/llmtf_eval \
  --dataset_names russiannlp/rucola_custom \
  --few_shot_count 5 \
  --max_prompt_len 4000 \
  --batch_size 8 \
  --vllm \
  --force_recalc
```

### Оценка одной модели через `transformers`

```bash
CUDA_VISIBLE_DEVICES=0 python evaluate_model.py \
  --model_name_or_path /path/to/model \
  --conv_path conversation_configs/qwen3.json \
  --output_dir /path/to/output/llmtf_eval_hf \
  --dataset_names nlpcoreteam/rummlu \
  --few_shot_count 5 \
  --max_prompt_len 4000 \
  --batch_size 1
```

### Оценка моделей через vLLM API

Сначала поднимите vLLM server:

```bash
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
  --model /path/to/model \
  --port 8000 \
  --tensor-parallel-size 1 \
  --no-enable-log-requests \
  --uvicorn-log-level error \
  --disable-uvicorn-access-log \
  --disable-log-stats \
  --model-impl transformers
```

Затем запустите оценку:

```bash
python evaluate_model_api.py \
  --base_url http://localhost:8000 \
  --model_name_or_path /path/to/model \
  --api_key EMPTY \
  --output_dir /path/to/output/api_eval \
  --dataset_names russiannlp/rucola_custom \
  --few_shot_count 5 \
  --max_prompt_len 4000 \
  --batch_size 16 \
  --disable_thinking
```

## Параметры генерации

У модели есть `generation_config`, который используется по умолчанию. Локальные модели читают его из Hugging Face config, API vLLM модели создают стандартный config. Можно задать произвольный конфиг.

Для reasoning-классов `HFModelReasoning`, `VLLMModelReasoning`, `ApiVLLMModelReasoning` доступен `max_new_tokens_reasoning`: отдельный бюджет на рассуждения, который не входит в `max_new_tokens`.

Приоритеты такие:

1. Явный `generation_config`, переданный в `Evaluator.evaluate(...)`.
2. Параметры задачи: `task.max_task_new_tokens`, `task.additional_stop_strings`, `task.method_additional_args`.
3. `model.generation_config`.

`max_prompt_len` ограничивает бюджет промпта. Если контекст модели меньше, чем `max_prompt_len + max_new_tokens` плюс reasoning-бюджет, сначала уменьшается reasoning-бюджет, затем бюджет промпта.

## LLM-as-a-Judge
Фреймворк включает систему оценки моделей с помощью LLM-судей.

Генерация ответов модели:

```bash
python benchmark/llmaaj/generate_llmaaj.py \
  --base_url http://localhost:8000 \
  --model_name_or_path /path/to/model \
  --api_key EMPTY \
  --model_name my_model \
  --benchmark_name ru_arena-hard-v0.1
```

Оценка судьей:

```bash
python benchmark/llmaaj/judge_llmaaj.py \
  --judge_base_url http://localhost:8001 \
  --judge_model_name_or_path /path/to/judge_model \
  --judge_api_key EMPTY \
  --judge_model_name deepseek \
  --benchmark_name ru_arena-hard-v0.1 \
  --model_name my_model
```

Показ результатов:

```bash
python benchmark/llmaaj/show_benchmark.py \
  --benchmark_name ru_arena-hard-v0.1 \
  --judge_model_name deepseek
```

## Добавление новой задачи
Для добавления новой задачи необходимо создать класс, наследующий от `SimpleFewShotHFTask` и зарегистрировать задачу в `TASK_REGISTRY`.

```python
from llmtf.base import SimpleFewShotHFTask
from llmtf.metrics import mean


class MyTask(SimpleFewShotHFTask):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._max_task_new_tokens = 512

    def dataset_args(self):
        return {"path": "my_dataset", "name": "default"}

    def create_messages(self, sample, with_answer=False):
        messages = [{"role": "user", "content": sample["question"]}]
        if with_answer:
            messages.append({"role": "assistant", "content": sample["answer"]})
        return messages

    def evaluate(self, sample, prediction):
        return {"accuracy": int(sample["answer"] == prediction)}

    def aggregation(self):
        return {"accuracy": mean}
```

Примеры notebooks находятся в `examples/`.

## Замечания

- Квантизация поддерживается экспериментально и может требовать отдельной проверки.
- PPL в текущей реализации считается для локальных моделей как средний logprob без экспоненцирования.

## Лицензия

Проект распространяется под открытой лицензией. См. файл `LICENSE`, если он присутствует в поставке.
