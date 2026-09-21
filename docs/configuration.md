# Конфигурация и запуск

## Single-model CLI

`evaluate_model.py` выбирает HF по умолчанию и local vLLM с `--vllm`.
`evaluate_model_api.py` всегда создаёт `APIBackend`.

Общие ключевые параметры:

- `--model_kind {plain,hybrid,reasoning}`;
- `--enable_thinking` или deprecated explicit alias `--disable_thinking`;
- `--model_context_len` — deployment context, не prompt length;
- `--max_new_tokens_reasoning` и `--min_new_tokens_reasoning`;
- `--end_thinking_token_id` — обязательный model-specific id для two-pass;
- `--assistant_prefill_policy {auto,exact,portable,best_effort}`;
- `--backend_kwargs JSON_OBJECT`.

CLI options, которые пользователь передал явно, имеют приоритет над
одноимёнными полями `backend_kwargs`. Не переданные CLI defaults не затирают
JSON. Неизвестный constructor key отклоняется до model loading.

## vLLM defaults

Offline `VLLMBackend` и управляемый API runner используют
`gpu_memory_utilization=0.92`. Значение можно переопределить:

```bash
python evaluate_model.py ... --vllm --gpu_memory_utilization 0.90
```

или для benchmark server:

```bash
python benchmark/calculate_benchmark_api.py ... --gpu_memory_utilization 0.90
```

Managed API runner передаёт один `--max_logprobs` и server-у, и
API-клиенту. Default равен 100; это снижает риск потери candidate
в top-k для probability tasks, но не превращает censored top-k в полное
распределение.

Текущие offline defaults также включают prefix caching и отключают sliding
window. Это исторические overrides, отмеченные для отдельной ревизии в
backlog; фиксируйте effective params из `_params.jsonl` при сравнении runs.

Reasoning parser, tool parser и другие model-specific server args framework не
угадывает. Безопасный `vllm_server_args` для управляемого runner ещё находится
в backlog. До его реализации поднимайте сервер отдельно и используйте
`calculate_benchmark_existing_api.py`.

## API profiles

- `auto` — conservative OpenAI payload плюс необязательное discovery;
- `openai` — только стандартные chat-поля;
- `vllm` — явные vLLM extensions.

Профиль не определяет модель автоматически и не включает reasoning/tool
parsers. Детали находятся в [`api_backend.md`](api_backend.md).

## Benchmark YAML schema

Верхний уровень содержит только `model`, `defaults`, `tasks`.

`model`:

- `model_kind`, `enable_thinking`, `model_context_len`;
- `max_new_tokens_reasoning`, `min_new_tokens_reasoning`;
- `end_thinking_token_id`, `backend_kwargs`, `is_foundational`;
- `assistant_prefill_policy`, `probe_api_prefill`, `api_profile`.

`defaults.evaluation` и task-level evaluation:

- `few_shot_count`, `batch_size`, `max_sample_per_dataset`, `name_suffix`.

`defaults.generation` и task-level generation:

- `temperature`, `repetition_penalty`, `presence_penalty`,
  `num_return_sequences`.

Task содержит `name`, непустой `datasets` и может переопределить
`enable_thinking`, evaluation и generation. Старый `extra_args.think`
поддерживается один переходный цикл с warning.

Secrets в `backend_kwargs` запрещены. Передавайте credentials через runtime
environment или secret manager.

## Sampling и stops

Задача временно накладывает `_max_task_new_tokens` и
`additional_stop_strings` поверх model config. Reasoning phase получает
отдельную копию с reasoning budget и end id; answer phase использует исходные
answer stops. После задачи базовая конфигурация восстанавливается.

Foundational conversation config определяет server chat template и
базовую stop string API-клиента. Managed runner передаёт один и тот же
conversation JSON обеим сторонам. Для задач с assistant-prefill включайте
`probe_api_prefill: true`; это уже задано в `llmtf_benchmark_foundational.yaml`.
Последняя группа этого конфига — CopyText: она поддерживается локальными HF и
vLLM backends, но ожидаемо завершается capability error через APIBackend,
который не возвращает необходимые token ids и `leading_space` metadata.

## Context budget

Prompt budget равен deployment context минус effective answer/reasoning
reserves. Старые task-level ключи `max_prompt_len`/`max_len` удалены, а strict
loader отклоняет их как неизвестные; deployment capacity задаётся через
model-level `model_context_len`. PPL и hybrid-disabled не резервируют reasoning
budget.
