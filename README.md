# LLMTF Open

**LLMTF Open** — это открытый фреймворк для комплексной оценки больших языковых моделей (LLM) с поддержкой русского языка и реализованным на его основе бенчмарком.

## Основные возможности

- **Быстрая оценка моделей** с поддержкой VLLM для ускорения инференса
- **Гибкая архитектура задач** на основе message-формата
- **Поддержка различных методов оценки**: генерация текста, расчет вероятностей токенов, перплексия
- **Оценка разных классов моделей**: базовых, инструктивных, рассуждающих
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

## Добавление новых задач

Создайте класс, наследующий от `SimpleFewShotHFTask`:

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

## Параметры генерации

Класс модели имеет поле `generation_config`, которое используется по умолчанию при генерации.
Локальные модели инициализируют это поле своим `generation_config` из `HF`. `api vllm` модели используют стандартный конфиг. Вы также можете задать произвольный конфиг.

В случае использования рассуждающих моделей - а именно классов `HFModelReasoning`, `VLLMModelReasoning`, `ApiVLLMModelReasoning` - помимо параметра `max_new_tokens` вы можете задать `max_new_tokens_reasoning`, ограничивающий бюджет на рассуждения (не идет в счет `max_new_tokens`).

Каждая задача предоставляет бюджет токенов - `task.max_task_new_tokens` - который имеет более высокий приоритет, чем
`max_new_tokens` в `model.generation_config`. Попмимо этого могут быть указаны дополнительные стоп-строки - `task.additional_stop_strings` - и параметры метода задачи - `task.method_additional_args`.

Помимо этого, метод `evaluate` класса `Evaluator`  имеет параметр `max_prompt_len` - бюджет на токены промпт. Если общий бюджет токенов  - `model.model_max_len` - окажется меньше чем `max_new_tokens` + `max_prompt_len` (+ `max_new_tokens_reasoning` в случае рассуждающей модели), то сначала будет укорачиваться бюджет на рассуждения, а потом на промпт.

Также вы можете передать произвольный конфиг в качестве параметра методу `evaluate` класса `Evaluator`, однако это может вызвать проблемы с некоторыми задачами, так как этот параметр имеет наивысший приоритет.

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
