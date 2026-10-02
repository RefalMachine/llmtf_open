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

Каждый конфиг запускается в один каталог модели. LawMC и все четыре режима
LegalBench-RU имеют отдельные имена artifacts. Base и Instruct результаты
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
temporal. Парные сравнения не нужны для его формирования. Стандартный `Mean`
остаётся обычным средним показанных задач; методология итогового балла
юридического бенчмарка пока не определена. Не считайте такой `Mean`
согласованным общим legal score. Проверяйте наличие всех пяти totals:
стандартный отчёт может отображать результаты неполного запуска.

`dev/tools/legalbench_ru_report.py` — дополнительная maintainer-утилита для
проверки скоринга и целостности сохранённых LegalBench-RU artifacts. Она не
требуется для запуска или обычной таблицы юридического бенчмарка.
