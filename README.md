# LLMTF Open

**LLMTF Open** — это открытый фреймворк для комплексной оценки больших языковых моделей (LLM) с поддержкой русского языка и реализованным на его основе бенчмарком.

## Основные возможности

- **Быстрая оценка моделей** с поддержкой VLLM для ускорения инференса
- **Гибкая архитектура задач** на основе message-формата
- **Поддержка различных методов оценки**: генерация текста, расчет вероятностей токенов, перплексия
- **Богатый набор задач**: от классификации до RAG и длинных контекстов
- **Автоматизированный бенчмарк** с параллельным выполнением на нескольких GPU
- **LLM-as-a-Judge** оценка с ELO рейтингами

## Быстрый старт

### Установка

**TODO**

### Базовое использование

```bash
# Оценка модели с VLLM
CUDA_VISIBLE_DEVICES=0 python evaluate_model.py \
  --model_name_or_path openchat/openchat-3.5-0106 \
  --conv_path conversation_configs/openchat_3.5_1210.json \
  --output_dir results_openchat \
  --few_shot_count 1 \
  --max_len 4000 \
  --batch_size 8 \
  --vllm

# Запуск полного бенчмарка
python benchmark/calculate_benchmark_api.py \
  --model_dir /path/to/model \
  --gen_config_settings benchmark/config_balanced.json \
  --api_key EMPTY \
  --num_gpus 4
```

## Архитектура

### Основные компоненты

- **`llmtf/`** — ядро фреймворка
  - `base.py` — базовые классы Task и LLM
  - `model.py` — реализации HFModel и VLLMModel
  - `evaluator.py` — основной класс для оценки
  - `tasks/` — коллекция задач для оценки

- **`benchmark/`** — автоматизированный бенчмарк
  - `calculate_benchmark_api.py` — параллельное выполнение задач через API
  - `llmaaj/` — LLM-as-a-Judge оценка

- **`conversation_configs/`** — конфигурации чат-шаблонов для различных моделей

## Поддерживаемые задачи

### Знания и рассуждения
- **MMLU** (русский/английский) — тест общих знаний
- **Shlepa** — специализированные домены (фильмы, музыка, право, книги)
- **DaruMeru** — комплексный русскоязычный бенчмарк

### Навыки и способности
- **Перевод** — Flores ru↔en
- **Суммаризация** — новостные тексты
- **Анализ тональности** — извлечение мнений
- **NER** — распознавание именованных сущностей
- **RAG** — вопросно-ответные системы с контекстом

### Специализированные задачи
- **IFEval** — следование инструкциям
- **Libra** — работа с длинными контекстами (до 32K токенов)
- **Математика** — решение задач по математике и физике


## Добавление новых задач

Создайте класс, наследующий от `SimpleFewShotHFTask`:

```python
from llmtf.base import SimpleFewShotHFTask

class MyTask(SimpleFewShotHFTask):
    def dataset_args(self):
        return {"path": "my_dataset", "name": "default"}
    
    def create_messages(self, sample, with_answer=False):
        messages = [{"role": "user", "content": sample["question"]}]
        if with_answer:
            messages.append({"role": "assistant", "content": sample["answer"]})
        return messages
    
    def evaluate(self, sample, prediction):
        return {"accuracy": int(sample["answer"] == prediction)}
```

## LLM-as-a-Judge

Фреймворк включает систему оценки моделей с помощью LLM-судей:

```bash
# Генерация ответов моделей
python benchmark/llmaaj/generate_llmaaj.py \
  --base_url http://localhost:8000 \
  --model_name_or_path /path/to/model \
  --api_key EMPTY \
  --model_name my_model \
  --benchmark_name ru_arena-hard-v0.1

# Оценка судьей
python benchmark/llmaaj/judge_llmaaj.py \
  --judge_base_url http://localhost:8001 \
  --judge_model_name_or_path /path/to/judge_model \
  --judge_api_key EMPTY \
  --judge_model_name deepseek \
  --benchmark_name ru_arena-hard-v0.1 \
  --model_name my_model

# Показ результатов
python benchmark/llmaaj/show_benchmark.py \
  --benchmark_name ru_arena-hard-v0.1 \
  --judge_model_name deepseek
```

## Особенности использования

- **VLLM**: Рекомендуется для ускорения, но требует `CUDA_VISIBLE_DEVICES`
- **Длинные контексты**: Параметр `max_len` может автоматически корректироваться
- **Квантизация**: Экспериментальная поддержка, возможны проблемы, лучше не использовать

## Примеры

В директории `examples/` находятся Jupyter notebook'и с примерами:
- Создание новой задачи
- Настройка конфигурации модели
- Результаты оценки различных моделей

## Лицензия

Проект распространяется под открытой лицензией. См. файл LICENSE для деталей.
