# LLM-as-a-Judge

Команды запускаются из корня репозитория. Для real credentials используйте
`OPENAI_API_KEY`; CLI key options оставлены для локального placeholder.

## Генерация ответов оцениваемой модели

```bash
OPENAI_API_KEY=EMPTY python -m benchmark.llmaaj.generate_llmaaj \
  --base_url http://127.0.0.1:8000 \
  --model_name_or_path /models/candidate \
  --model_name candidate \
  --benchmark_name ru_arena-hard-v0.1
```

Результат записывается в
`benchmark/llmaaj/<benchmark>/model_results/<model_name>.json`.

## Судейство

```bash
OPENAI_API_KEY=EMPTY python -m benchmark.llmaaj.judge_llmaaj \
  --judge_base_url http://127.0.0.1:8001 \
  --judge_model_name_or_path /models/judge \
  --judge_model_name judge \
  --benchmark_name ru_arena-hard-v0.1 \
  --model_name candidate \
  --force_recalc
```

Judge использует `APIBackend` и общий `Evaluator`, включая новые JSON-array
artifacts, fingerprint и fail-closed summary. `--max_len` задаёт effective API
model context override.

## Таблица результатов

```bash
python -m benchmark.llmaaj.show_benchmark \
  --benchmark_name ru_arena-hard-v0.1 \
  --judge_model_name judge
```

Параметризованный full launcher находится в
[`benchmark/llmaaj/run_full.sh`](../benchmark/llmaaj/run_full.sh). Он принимает
manifest со строками `MODEL_PATH RESULT_NAME TENSOR_PARALLEL_SIZE`, запускает
candidate server в `llmtf:vllm-cu129` и передаёт judge credentials только через
окружение. Список обязательных и дополнительных переменных доступен через:

```bash
benchmark/llmaaj/run_full.sh --help
```

Model-specific параметры vLLM передаются по одному аргументу на строку через
`VLLM_EXTRA_ARGS_FILE`; launcher не выбирает parser автоматически. Перед
большим запуском сделайте небольшой endpoint smoke выбранной candidate/judge
пары.

Скрипты не пытаются автоматически выбирать модельный reasoning-протокол.
Настройте его на стороне API-сервера; универсальные дополнительные поля
запроса будут добавлены отдельно согласно [`BACKLOG.md`](../BACKLOG.md).
