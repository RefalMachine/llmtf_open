# API backend

`APIBackend` работает с OpenAI-compatible chat API. Серверу не обязательно
реализовывать `/v1/models`, `/tokenize` или `/detokenize`: обычная генерация
без assistant-prefill от них не зависит.

## Профили

### Преждевременный EOS в двухэтапном reasoning

При подтверждённом EOS до `THINK_CLOSE_MARKER` LLMTF закрывает reasoning
и выполняет отдельную стадию ответа. Фраза о лимите добавляется только при
исчерпании reasoning budget. Терминальный EOS удаляется из assistant-prefill;
текст рассуждения не используется как готовый итоговый ответ.

Для явного vLLM-профиля `finish_reason=stop` вместе с возвращённым
`stop_reason=null` означает EOS; явно известный штатный EOS ID также
распознаётся. Сам по себе стандартный `finish_reason=stop`
такой гарантии не даёт. При сохранении специальных токенов клиент запрашивает
`return_token_ids` и декодирует последний ID через `/detokenize`, чтобы удалить
точный конечный маркер. Неизвестный короткий stop или ошибка генерации
по-прежнему не считаются успешным завершением reasoning.

В sample `info.reasoning` записываются `termination_reason` (`think_close`,
`length`, `eos`), `synthetic_close`, `eos_removed` и исходные сведения backend
в `generation_termination`. В params указан
`termination_policy=close_on_confirmed_eos_v1`. Для анализа абляций следует
отдельно считать долю EOS: такие примеры не получают завершающую фразу.
6 октября 2026 года прошли проверки HF, local vLLM и CPU-only API-клиента
на Qwen3.5-2B с одной RTX 4090: обычные off/on вызовы, принудительный
пустой и непустой EOS, отдельный strict reasoning, generate и probability.
Артефакты и ограничения проверки сохранены в
`papers/LLMTF_legal/experiments/validation/eos-2026-10-06/`.
Это проверка исполнения, без утверждения общей local/API parity.

Общий YAML принимает `model.reasoning_truncing_prompt`. `null` (или отсутствие
поля) сохраняет штатную фразу, пустая строка убирает только текст фразы,
другая строка заменяет его. Фраза используется лишь при обрыве reasoning по
бюджету; закрытие reasoning и разделитель ответа сохраняются. Параметр также
доступен как `--reasoning_truncing_prompt` в обоих single-model CLI и входит
в существующий reasoning provenance/fingerprint. Минимальный резерв
`min_new_tokens_reasoning` не заставляет модель генерировать заданное число
токенов. API reasoning требует проверки assistant continuation, описанной ниже.

```text
--api_profile {auto,openai,vllm}
```

- `auto` (по умолчанию) отправляет консервативный OpenAI payload и пробует
  только необязательные возможности `/v1/models` и `/tokenize`. Обнаруженный
  `/tokenize` используется для budget, но не включает остальные
  vLLM-расширения автоматически.
- `openai` отправляет только стандартные chat-поля и не обращается к
  `/tokenize`.
- `vllm` включает `continue_final_message`, `add_generation_prompt`,
  `chat_template_kwargs`, `stop_token_ids`, `top_k`, `repetition_penalty` и
  другие используемые framework'ом расширения vLLM.

Профиль выбирается явно по контракту сервера. Успешный `/tokenize` сам по себе
не доказывает поддержку continuation.

`api_profile` отвечает только за transport capability. Отдельная
`assistant_prefill_policy` задаёт, достаточно ли объявленной возможности,
нужен ли probe или prefill должен быть запрещён. Поэтому консервативный
`api_profile=auto` не превращается во `vllm` по результату эвристического
запроса.

Reasoning parser, tool parser и остальные model-specific server options не
выбираются автоматически. Если они нужны, сервер следует поднять с явно
проверенными аргументами, а evaluation запустить через
`benchmark/calculate_benchmark_existing_api.py` или `evaluate_model_api.py`.

`/v1/models` является необязательным. Если endpoint недоступен, используется
точно переданное `--model_name_or_path`. Если сервер также не сообщает
`max_model_len`, необходимо передать `--model_context_len`.

## Подсчёт prompt

`count_tokens_for_messages` возвращает:

- целое число, если доступен серверный `/tokenize`;
- `None`, если предварительный счётчик отсутствует.

`None` не означает нулевую длину и не заменяется эвристикой. Framework
сохраняет запрошенный few-shot prompt без скрытого урезания. Если он не
помещается, server context error завершает batch ошибкой. Поле
`usage.prompt_tokens`, если сервер его вернул, сохраняется уже после запроса.

Пустые платные генерации только ради измерения prompt автоматически не
выполняются.

## Assistant continuation

Перед вторым проходом LLMTF нормализует конец reasoning: удаляет только
завершающие пробельные символы перед `THINK_CLOSE_MARKER` и ставит ровно
один перевод строки перед ним. Это применяется при естественном закрытии,
лимите и EOS; внутренние абзацы, разделитель после маркера и assistant-prefill
задачи сохраняются. Граница соответствует шаблону Qwen и устраняет ошибку
`continue_final_message` из-за лишних переводов строк перед закрытием.
В params записывается `normalization_policy=single_newline_before_close_v1`;
для нового запуска нужен отдельный output или явный пересчёт старого кеша.
Обычный transport по умолчанию остаётся `chat`.

Профили `auto` и `openai` не обещают, что финальное сообщение `assistant`
будет продолжено. При политиках `auto` и `exact` такой prompt отклоняется до
генерации. Возможные действия:

1. переписать task prompt так, чтобы он заканчивался сообщением `user`;
2. выбрать `--api_profile vllm` для соответствующего сервера;
3. использовать `best_effort` только для явно непроверенного запуска.

`portable` запрещает любой assistant-prefill на всех backend'ах. Это позволяет
запускать один и тот же reviewed prompt локально и через закрытое API.

Two-pass emulated reasoning также требует continuation. На профилях `auto` и
`openai` он завершается capability error до первого reasoning-запроса. Режим
`best_effort` разрешает непроверенный запуск, но он не доказывает parity.

Текущий transport ожидает видимый ответ в обычном `content`. Ответы provider'а,
где reasoning приходит только в отдельном `reasoning_content`, ещё не входят в
поддерживаемый контракт. То же относится к структурированным `tool_calls`.

## Token probability

`calculate_tokens_proba` не требует `/tokenize`, если chat API возвращает
`logprobs` с текстами токенов. Параметр backend'а
`calculate_tokens_proba_logprobs_count` задаёт запрашиваемый top-k (по
умолчанию 20).

Top-k не содержит полное распределение. Поэтому API-результат помечается:

```json
{
  "candidate_score_semantics": "top_k_censored_ranking",
  "candidate_surface_coverage": {"A": [" A"], "B": []},
  "top_logprobs_count": 20
}
```

Отсутствующему варианту для совместимости task-интерфейса соответствует
нижняя граница `0.0`, а не заявленная точная вероятность. Если не найден ни
один вариант, все значения равны `0.0`, а `info` содержит
`candidate_ranking_resolved: false` и semantics
`top_k_censored_all_candidates_below_cutoff`. Это сохраняет sample,
но не доказывает относительный ranking. API без logprobs следует
настроить с `supports_logprobs=false` в `--backend_kwargs`; probability-задачи
тогда отклоняются как unsupported до загрузки датасета.

## Воспроизводимость

`_params.jsonl` содержит выбранный профиль и `server_capabilities` с
трёхзначными значениями `true` / `false` / `null` (`null` означает, что
возможность не была проверена). Endpoint по умолчанию редактируется, API key
никогда не входит в params.

На проверенном vLLM 0.21 server trailing whitespace assistant-prefill
удаляется. Поэтому affected instruct runs выполнялись только как явный
`best_effort`. Foundational API generation также пока не получает local stop
strings единообразно с HF/vLLM; этот gap описан в `BACKLOG.md`.

Пример generic API без tokenizer:

```bash
python evaluate_model_api.py \
  --base_url https://provider.example/v1 \
  --api_profile openai \
  --model_name_or_path closed-model \
  --model_context_len 8192 \
  --output_dir /tmp/llmtf-api \
  --dataset_names task/with_user_final_prompt \
  --model_kind plain \
  --disable_thinking
```

Пример vLLM API:

```bash
python evaluate_model_api.py \
  --base_url http://127.0.0.1:8000 \
  --api_profile vllm \
  --model_name_or_path model \
  --model_context_len 8192 \
  --output_dir /tmp/llmtf-vllm-api \
  --dataset_names russiannlp/rucola_custom \
  --assistant_prefill_policy auto \
  --probe_api_prefill
```
