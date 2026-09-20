# Рефакторинг v3 — план

> Исторический development design record. Актуальный контракт находится в
> `../docs/`.

Контекст: v1/v2 выполнены (см. `REFACTOR_PLAN.md`, `REFACTOR_PLAN_v2.md`). Архитектура: `BaseLLM` (abc) + `LLM` (concrete, держит `backend`) + `Backend` abc + 3 concrete (`HFBackend`/`VLLMBackend`/`APIBackend`) в `llmtf/backends/`. `ReasoningConfig` отделён от `generation_config`, единый диспетчер ризонинга на уровне `LLM`. Фасады и `*Reasoning` подклассы удалены. Этот план фиксирует решений для четырёх обсуждённых вопросов + мелочи и формализует бэклог следующих шагов.

## Зафиксированные решения с пользователем

1. **`apply_model_prompt` / `count_tokens_for_prompt` остаются** как debug-API для локальных бекендов (HF/vLLM): первый рендерит chat-template → str, второй токенизирует голую строку → int. На **API** — `NotImplementedError` с понятным сообщением (chat template живёт на сервере, локальный рендер невозможен). Из abc убираются как обязательные абстрактные методы (optional на backend-е).
2. **Pass-through backend-параметров через `--backend_kwargs <json>`** (escape-hatch для спецов). Дублировать в configs-on-disk не нужно («городить конфиги не хочется»). Явно в argparse экспонируется ограниченный набор (~10) ключевых параметров; остальное — JSON-пасграундом в конструктор бекенда.
3. **`presence_penalty`** на HF не поддерживается нативно (нет такого поля в HF `GenerationConfig`/`generate`). Решение: если в `generation_config` выставлен `presence_penalty` (отличен от default 0.0) и бекенд — HF, писать `logger.warning('presence_penalty is not supported by the HF backend; use repetition_penalty')`. На vLLM/API — добавить реальный проброс в `SamplingParams` и в JSON-пейлоад `/v1/chat/completions`. `num_return_sequences` — **не трогаем** (поведение на всех бекендах работает, расхождение API (N запросов vs `n:N`) приемлемо).
4. **`sniff_think_close_token_id` удаляется**; ответственность за тип модели — на пользователе через `--model_kind`. Дефолт `--model_kind` меняется с `auto` на `plain`.
   - По умолчанию `plain` <=> one-pass — корректно для всех instruct/base моделей и для hybrid-моделей в plain-режиме (enable_thinking=False → one-pass, think-токен не нужен).
   - Hybrid-модели в reasoning-режиме: пользователь явно ставит `--model_kind hybrid` (+ `--enable_thinking` по умолчанию, т.к. `enable_thinking = not args.disable_thinking`). Если think-токен не предоставлен (`end_thinking_token_id`) и `--model_kind hybrid`, ризонинг-фаза будет останавливаться только по текстовому stop-стрингу `THINK_CLOSE_MARKER` — допустимо, но работает менее надёжно (особенно на HF при генерации без `skip_special_tokens`). Предупреждение остаётся в `_setup_reasoning`.
   - `--model_kind` option `auto` убирается из argparse; в коде `auto` больше не валидный `model_kind` (если придёт — `ModelKind(kind)` кинет ValueError).Sidestep: explicit `--end_thinking_token_id` получает добавление в **local** CLI (`evaluate_model.py`) — раньше был только в API CLI (`evaluate_model_api.py`). Симметрируем.

---

## Слой A. Debug-API: `apply_model_prompt` / `count_tokens_for_prompt`

**A1.** `llmtf/base.py:BaseLLM`:
- Убрать `@abstractmethod` с `apply_model_prompt` и `count_tokens_for_prompt`. Оставить **только** `count_tokens_for_messages` как обязательный message-oriented контракт. Опциональные два метода оставить в `BaseLLM` как не-abstract методы с дефолтом `raise NotImplementedError('[debug] apply_model_prompt is supported only on local backends (HF/vLLM); API backend exposes no chat-template rendering')`.

**A2.** `llmtf/llm.py:LLM`: оставить прокси-методы `apply_model_prompt`/`count_tokens_for_prompt`, проксирующие в backend. Если бекенд их не реализует — получит `NotImplementedError` от backend-а (для API). Ничего не трогать сверх этого.

**A3.** `llmtf/backends/api.py`: удалить текущую «legacy-compat» `apply_model_prompt` (возвращает int через `/tokenize`) и `count_tokens_for_prompt(prompt_tokens)` (int→int). Заменить на:
```python
def apply_model_prompt(self, *args, **kwargs):
    raise NotImplementedError(
        "[debug] APIBackend does not render chat templates locally; "
        "the chat template lives on the server. Use count_tokens_for_messages "
        "directly (it hits /tokenize on the server)."
    )

def count_tokens_for_prompt(self, prompt = None, *args, **kwargs):
    raise NotImplementedError(
        "[debug] APIBackend has no local tokenizer; "
        "use count_tokens_for_messages on messages."
    )
```
Документировать в `Backends/backend base.py` comment-block-е: эти два метода — optional debug-helpers, обязательный контракт — `count_tokens_for_messages`.

**Проверка:** py_compile; smoke-test `LLM(HFBackend(...)).apply_model_prompt(messages)` возвращает str; `LLM(APIBackend(...)).apply_model_prompt(messages)` поднимает `NotImplementedError` с читаемым сообщением.

## Слой B. Pass-through backend-параметров (`--backend_kwargs`)

**B1.** Добавить в `evaluate_model.py` и `evaluate_model_api.py` argparse:
```python
parser.add_argument('--backend_kwargs', type=str, default=None,
    help='JSON string of extra kwargs forwarded to the backend constructor, '
         'e.g. \'{"gpu_memory_utilization":0.9,"limit_mm_per_prompt":{"image":2}}\'. '
         'Explicit CLI flags below override these.')
```

**B2.** Явно экспонировать (в argparse, с типами и help) «ключевые» параметры:

| Флаг | Тип | Default | Бекенд | Comment |
|---|---|---|---|---|
| `--max_prompt_len` | int | 4000 | (all) | Уже есть. На vLLM маппится в `max_seq_len_to_capture`. |
| `--tensor_parallel_size` | int | 1 | HF+vLLM | Уже есть. |
| `--device_map` | str | 'auto' | HF | Уже есть. |
| `--torch_dtype` | str | 'auto' | HF | **новый** |
| `--attn_implementation` | str | 'flash_attention_2' | HF | **новый** |
| `--trust_remote_code` | flag | False | HF+vLLM | **новый** |
| `--load_in_8bit` | flag | False | HF | **новый** |
| `--use_fast_tokenizer` | flag | True | HF+vLLM | **новый** |
| `--gpu_memory_utilization` | float | 0.95 | vLLM | **новый** |
| `--calculate_tokens_proba_logprobs_count` | int | 50 | vLLM | **новый** |
| `--limit_mm_per_prompt` (json) | json | `{"image":0,"video":0}` | vLLM | **новый** |

Параметры reasoning (`--model_kind`, `--max_new_tokens_reasoning`, `--end_thinking_token_id`, `--reasoning_truncing_prompt` если ещё нет) — на `LLM.from_pretrained`, **не** в backend.-going to LLM._setup_reasoning через `from_pretrained`.

**B3.** Механика merge в `evaluate_model.py`:
```python
explicit = {
    'device_map': args.device_map,
    'tensor_parallel_size': args.tensor_parallel_size,
}
if args.backend_kwargs:
    kwargs_from_json = json.loads(args.backend_kwargs)
else:
    kwargs_from_json = {}

# Per-backend explicit fields
if args.vllm:
    explicit.update(
        conversation_template_path=args.conv_path,
        max_seq_len_to_capture=args.max_prompt_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        calculate_tokens_proba_logprobs_count=args.calculate_tokens_proba_logprobs_count,
        limit_mm_per_prompt=args.limit_mm_per_prompt,
        trust_remote_code=args.trust_remote_code,
        use_fast_tokenizer=args.use_fast_tokenizer,
        enable_prefix_caching=not args.disable_prefix_caching,
        disable_sliding_window=args.disable_sliding_window,
    )
else:
    explicit.update(
        conversation_template_path=args.conv_path,
        alpha_scale=args.alpha_scale,
        not_scale_lm_head=args.not_scale_lm_head,
        torch_dtype=args.torch_dtype,
        attn_implementation=args.attn_implementation,
        load_in_8bit=args.load_in_8bit,
        trust_remote_code=args.trust_remote_code,
        use_fast_tokenizer=args.use_fast_tokenizer,
    )

# explicit overrides --backend_kwargs: kwargs_from_json first, then explicit (latter wins)
backend_kwargs = {**kwargs_from_json, **explicit}
backend = (VLLMBackend if args.vllm else HFBackend)(**backend_kwargs)
```

**B4.** **Починить уже существующие мёртвые флаги** `--disable_sliding_window` / `--disable_prefix_caching`: сейчас парсятся, никуда не передаются (см. `evaluate_model.py:23-24`). После B3 они попадают в explicit kwargs для VLLMBackend. Если принято решение совсем их убрать в пользу `--backend_kwargs` — убрать и из argparse; но раз они уже есть — оставить и завести (минимальные правки).

**B5.** В `evaluate_model_api.py`: добавить те же параметры-escape, но из explicit — только `--backend_kwargs` и, опционально, `--num_procs`/`--openai_max_concurrency`-env, который **тоже** починим (см. E1 ниже). Большинство explicit-полей HF/vLLM на API не применимы.

**B6.** Документация в docstrings/AGENTS.md: «ключевые параметры открываем явно; всё остальное — через `--backend_kwargs`». Не дублировать argparse на каждый backend-параметр.

**Проверка:** tiny eval: `--backend_kwargs '{"calculate_tokens_proba_logprobs_count":10}' --dataset_names ... --max_sample_per_dataset 8`, потом inspect `_params.jsonl` (через `get_params()`), там должно быть `calculate_tokens_proba_logprobs_count: 10`. Explicit `--calculate_tokens_proba_logprobs_count 20` должен перебить `--backend_kwargs = '{"..._count":10}'` (`explicit` wins after merge в B3). Проверить на обычно-warn-ящем параметре (например, `gpu_memory_utilization=0.5`).

## Слой C. `presence_penalty`

**C1.** `llmtf/backends/hf.py`: в начале `generate_batch` (и `calculate_tokens_proba_batch`, и `calculate_logsoftmax_batch` если gc туда попадает) — check:
```python
if getattr(generation_config, 'presence_penalty', 0.0) not in (None, 0.0):
    # логировать по разу на таску/процесс чтобы не спамить
    if not getattr(self, '_presence_penalty_warned', False):
        self.logger.warning(
            "presence_penalty=%s is set but HF generate does not support it; "
            "it will be ignored. Use repetition_penalty instead.",
            generation_config.presence_penalty,
        )
        self._presence_penalty_warned = True
```
(Флаг _presence_penalty_warned на backend-е, чтобы не пищать на каждый батч.)

**C2.** `llmtf/backends/vllm.py`: добавить `presence_penalty=generation_config.presence_penalty` в `SamplingParams` (если поле None — vLLM берёт свой default 0.0; safe):
```python
sampling_params = SamplingParams(
    temperature=generation_config.temperature,
    top_p=generation_config.top_p,
    top_k=generation_config.top_k,
    max_tokens=generation_config.max_new_tokens,
    repetition_penalty=generation_config.repetition_penalty,
    presence_penalty=getattr(generation_config, 'presence_penalty', 0.0) or 0.0,
    stop=..., stop_token_ids=..., n=...,
    include_stop_str_in_output=...,
)
```

**C3.** `llmtf/backends/api.py`: добавить `'presence_penalty': getattr(generation_config, 'presence_penalty', 0.0) or 0.0` в JSON-payload `/v1/chat/completions` в `generate`.

**Проверка:** на vLLM/API — tiny eval с `--presence_penalty 1.5` не падает и параметр виден в `_params.jsonl` (через `get_params()`-логику, хотя `get_params` сейчас логгит только конфиг через `to_json_string` — там presence_penalty тоже должен попасть, проверить). На HF — expected warning вoncéр лога.

## Слой D. Удаление `sniff_think_close_token_id` + смена дефолта `model_kind` на plain

**D1.** `llmtf/llm.py:LLM._setup_reasoning`: убрать ветку `if kind == 'auto'` с sniff-ом. Прямой парсинг: `kind = ModelKind(model_kind)` (`'auto'` больше не валидно — `ModelKind('auto')` упадёт). Если `max_new_tokens_reasoning is None` и kind != plain — fallback на `self.backend.generation_config.max_new_tokens` (как сейчас). Если `end_thinking_token_id is None` и kind не plain — оставляем `ReasoningFormat.end_thinking_token_id=None`, **warning** остаётся: «hybrid without explicit end_thinking_token_id; reasoning phase will stop on text-marker `THINK_CLOSE_MARKER` only — less robust on HF».

**D2.** `llmtf/backends/base.py:Backend`: удалить `sniff_think_close_token_id` (default-реализация и docstring).

**D3.** `llmtf/backends/hf.py` / `vllm.py` / `api.py`: удалить override-ы `sniff_think_close_token_id`.

**D4.** `evaluate_model.py` / `evaluate_model_api.py` / `benchmark/llmaaj/generate_llmaaj.py` / `judge_llmaaj.py`:
- `--model_kind` choices изменить с `['auto','plain','reasoning','hybrid']` на `['plain','reasoning','hybrid']`, default `'plain'`.
- В `evaluate_model_api.py` убрать ad-hoc `if explicit_kind == 'auto': model_kind = 'plain' if args.disable_thinking else 'hybrid'` — больше не нужно: дефолт `'plain'`, явный override через `--model_kind`.
- В `evaluate_model.py`: добавить `parser.add_argument('--end_thinking_token_id', type=int, default=None)` (сейчас есть только в API). Проброс в `from_pretrained`.
- В `evaluate_model_api.py`: `--max_new_tokens_reasoning` default менять с `3000` на `None` для симметрии с local CLI; с `_setup_reasoning` дефолт берётся из `gc.max_new_tokens` если None.

**D5.** `llmtf/reasoning.py:EmulatedReasoningStrategy._build_reasoning_config`: проверить, что при `fmt.end_thinking_token_id is None` eos_token_id не переопределяется (сейчас — `if self.fmt.end_thinking_token_id is not None: cfg.eos_token_id = [...]`). ✓ Документировать в ReasoningFormat: «end_thinking_token_id is optional; if None, reasoning phase relies on text stop-string `think_close` only».

**D6.** Эргономика: добавить в `_setup_reasoning` для `reasoning`-kind без явного `end_thinking_token_id` — `ValueError` с подсказкой «reasoning model_kind requires explicit --end_thinking_token_id». Для `hybrid` — только warning (гибрид может работать в plain-режиме, поэтому一刀切 error не годится).Решение: если `model_kind=reasoning` и `end_thinking_token_id is None` — `raise ValueError`. Для `hybrid` и отсутствия `end_thinking_token_id` — warning (как в D1) и continuation через текст-стоп.

**D7.** Тесты `tests/test_refactor_logic.py`: убрать тесты, явно использующие `sniff_think_close_token_id` (или адаптировать — FakeBackend без sniff, пользователь передаёт `hybrid` явно). Добавить тест: default `model_kind='plain'` → `_reasoning is None`. Тест: `model_kind='auto'` → `ValueError` в `_setup_reasoning`. Тест: `model_kind='reasoning'` без `end_thinking_token_id` → `ValueError`. Тест: `model_kind='hybrid'` без `end_thinking_token_id` → warning, но стратегия создаётся и `_build_reasoning_config` не выставляет eos.

**Проверка:** на reasoning-модели (Qwen3-think) на HF: `--model_kind hybrid --enable_thinking --end_thinking_token_id <id>` — метрики как в v2-базе; `--model_kind plain` — one-pass (как `*Reasoning`+`disable_thinking` в v1-базе); `--model_kind reasoning --end_thinking_token_id <id>` — two-pass; `--model_kind reasoning` без id — ValueError; `--model_kind hybrid` без id — warning + two-pass с text-stop-only.

## Слой E. Мелкие баги/чистки (фикс независимо,赛前 commit)

**E1.** `llmtf/backends/api.py:18`: `self.num_procs = int(os.getenv('OPENAI_MAX_CONCURRENCY', '20'))`. Сейчас str→`ThreadPoolExecutor(max_workers=str)` упадёт TypeError при первом batch. (Один из bug-fixed пофиксить раньше, можно в первом же commit.)

**E2.** `llmtf/evaluator.py:1-22`: убрать дубликаты `import json`, `import os`, `import numpy` (по одному в каждом). Убрать мёртвые `import inspect`, `import re` (grep no use), `#from llmtf.task import *` — trailing comment.

**E3.** `llmtf/evaluator.py:66`: перенести `model.logger.warning('Custom generation_config receives full priority...')` из цикла по датасетам в начало `evaluate` (после `set_out_handler_to_main_logger`, перед `for dataset_name in datasets_names`). Чтобы один warning на run, а не N.

**E4.** `llmtf/evaluator.py:227-235` (`evaluate_dataset_ppl`): убрать локальную `backend = getattr(model, 'name', lambda: 'unknown')()` (не используется); убрать двойной `assert`. Оставить единый `assert model.support_method('calculate_logsoftmax')`, перед ним — `logger.error(...)` с указанием `type(model).__name__` (это уже есть на строках 229-233), дальше один assert с тем же сообщением.

**E5.** `llmtf/backends/api.py`: `print(r.text)` (строки 41, 207, 307), `print(data['data'])` (45) — заменить на `self.logger.debug(...)`. `{print("Задача {idx} завершилась с ошибкой: {e}")}` (266, 345) — `self.logger.error(...)`. `print('Can\'t tokenize, fallback to 0 len assumtion')` (132) — `self.logger.warning(...)`. Two user-facing warnings (198, 299): `print("You requested more tokens than maximum model context length...")` — оставить print (user-facing, в stdout ожидаемо) либо `logger.warning` (видно в лог-файл); предпочтительно `logger.warning`.

## Слой F. Бэклог следующего раунда (не в текущем заходе)

**F1.** Дублирование backend-методов HF/vLLM: `_check_if_lora`, `_check_if_leading_space`, `_update_chat_template`, `_init_default_gen_params`, `_override_eos_token_conv_template`, `_check_word_is_token`, `add_stop_token`, `_augment_tokens_of_interest` — вынести в общий helper-миксин `TokenizerBackendMixin` (для бекендов с локальным токенайзером — HF+vLLM, не API). Учитывая, что `_add_stop_string` различается (HF делает vocab-scan, vLLM — `_check_word_is_token`) — выделить общую логику и оставить точку overridingления для HF. Унификация ~150 строк копипасты → одно место. См. `REFACTOR_PLAN_v2.md` «Что в текущем рефакторинге коряво», пункт 5 — корневая причина частично осталась.

**F2.** Двойной assert в `evaluate_dataset_ppl` и общая проверка support_method для гибридных runs. Сейчас `evaluator.evaluate_ppl` идёт HF-only; если когда-то захотим PPL на reasoning-фазе — нужно расширять `_run_reasoning_then_continuation` на mode='ppl' (continuation_fn = `calculate_logsoftmax_batch`). vllm-api-runtime — вне scope.

**F3.** Стейл-трэш на верхнем уровне репо (правит пользователь, не в этом заходе):
- `.gitmodules` описывает сабмодули `external_benchmarks/ruwikibench` и `external_benchmarks/rubooksum`, которые в рабочем дереве удалены (git status: deletions). Решить: либо `git rm --cached` + убрать запись из `.gitmodules`, либо восстановить.
- `REFACTOR_PLAN.md` (v1, 49 КБ) и `REFACTOR_PLAN_v2.md` (v2, 29 КБ) — решённая история. Опционально: оставить только v3当前位置, v1/v2 в `docs/history/`.
- `remap_qwen35_checkpoint.py` — одноразовый скрипт Qwen3.5 checkpoint remap. Если повторного использования нет — удалить.
- `run_evaluate_singlenode_multigpu.py` — старый runner с hardcoded `task_groups`; AGENTS.md сам не рекомендует его использовать. Опционально: удалить или пометить как deprecated в начале файла.

**F4.** AGENTS.md терминология: «`tests/` сейчас держит scratch eval artifacts, не test suite» — устарело. `tests/test_refactor_logic.py` — рабочий pure-logic тест-сьют (14/14). Обновить формулировку, чтобы никто не «почистил» `tests/` по умолчанию. Заодно проверить, что `.gitignore` не покрывает регрессию `tests/test_refactor_logic.py` (зависит от status-кода `tests/*` glob-а в .gitignore — см. `tests/qwen3-0.6B/` scratch-директорию, она оставлена ignore-ом).

**F5.** `APIBackend.num_return_sequences` делает N независимых HTTP-запросов вместо `'n': N` в одном запросе (как на HF/vLLM, vLLM-сервер поддерживает `'n'` нативно). Эффективнее — `'n': num_return_sequences`. Семантика одинаковая (random sampling), выигрыш — пропускная способность. Решить, стоит ли. Не критично.

**F6.** `_common.py` с `from llmtf.backends._common import *` — перес房企ца wider-импортов. Можно сделать явные импорты в каждом бекенде. Дело стиля, не функциональность.

**F7.** Прочий `todo` (из репо-файла `todo`): бутстрап rewrite (пункт 7 todo), отладка api/hf/vllm совпадений (пункт 8 todo), конфиг бенча для мультигпу (пункт 2 todo). Не в этом заходе.

---

## Порядок исполнения

Каждый слой — отдельный commit с small smoke-test (import-check или tiny eval HF).

1. **Commit 1 (Слой E)** — мелкие баги-чистки (`num_procs` int-cast, дубликаты импортов, cosmetique evaluator, api print→logger). Чистый и независимый.
2. **Commit 2 (Слой C)** — `presence_penalty` на vLLM/API, warning на HF. Независимый, safe.
3. **Commit 3 (Слой A)** — debug-API: drop abstracts, API → `NotImplementedError`. Независимый, safe.
4. **Commit 4 (Слой D)** — removal `sniff_think_close_token_id`, default `model_kind='plain'`, симметрия `--end_thinking_token_id` между CLIs, тесты. Breaking по semantике (старые бенч YAML, которые не выставляли `--model_kind`, теперь получают plain — для instruct моделей это **то же поведение** по выводу, для reasoning-моделей, которые ехали auto→hybrid, придется дописать `model_kind: hybrid` в YAML-конфиге. Обновить один-два YAML-конфига в `benchmark/` для reasoning-моделей.
5. **Commit 5 (Слой B)** — `--backend_kwargs` + explicit-фильтры + починка уже существующих dead flags (`--disable_sliding_window`, `--disable_prefix_caching`). Финальный слой, требующий tiny eval с vLLM.
6. **Commit 6 (AGENTS.md sync)** — синхронизация AGENTS.md с v3: обновить `sniff`-описание (убрать), добавить `--backend_kwargs` описание, обновить «Must-test matrix» (default plain теперь), добавив explicit `model_kind` requirement для reasoning runs.

Каждый commit — после py_compile + `python3 -m pytest tests/test_refactor_logic.py -q` (если тесты адаптированы) + tiny HF eval при возможности. GPU-okружения для vLLM/API — на пользователе, но механика проверяется импортом и mock-конструкцией.

## Backward compatibility

- **Слой A:** external consumers, зовущие `apply_model_prompt` на API — должны были и так не работать; теперь получат читаемое `NotImplementedError` вместо странной int-семантики. Net win. На HF/vLLM — без изменений.
- **Слой B:** новые флаги, override-semantics explicit > --backend_kwargs. Старые CLI-вызовы без `--backend_kwargs` — не меняются (explicit defaults).
- **Слой C:** добавочный warning на HF при presence_penalty != 0; метрики HF не меняются (presence_penalty и так игнорился). На vLLM/API — теперь реально влияет (изменение выхода, но **новое** поведение — старое было «флаг парсится и игнорится», теперь — применяется).
- **Слой D breaking:** `--model_kind auto` больше не валиден — любой внешний скрипт/конфиг с этим значением падает с ValueError (осмысленно). Бенч-configи в репо, использующие `auto` (если есть — grep по `benchmark/*.yaml`), заменить на `hybrid` или `plain`. Дефолт `auto`→`plain` меняет поведение reasoning-моделей «втихую» (раньше они ехали hybrid по auto-detect, теперь plain one-pass) — это исправлено в бенч-YAML-ах в Commit 4.

## Risk-блок (regression watch)

- **Слой C:** vLLM и API теперь реально применяют `presence_penalty`. Если в существующих бенчconfig-ах стоит `presence_penalty: 1.5` с расчётом что он игнорится — метрики поменяются. Проверить `grep -r presence_penalty benchmark/` перед deploy; в single-node и быстром tiny eval.
- **Слой D:** reasoning-модели без explicit `--end_thinking_token_id` работают только по text stop-string `THINK_CLOSE_MARKER`. На HF при генерации без skip_special=True stop-token-id ловится надёжно, как только reasoning-фаза генерит токен .Без явного id _вынуждаем_ text-stop — на HF клиническое может не остановиться (если модель генерит token-first без marker-string). Если `enable_thinking=True` ставит `skip_special_tokens=False` в reasoning_phase_kwargs (llm.py:198 — уже так теперь), то `THINK_CLOSE_MARKER` появляётся как строка в output → Стратегия ловит. Но risk real. mitigate — D6 (ValueError на reasoning-kind без id, warning на hybrid без id).
- **Слой B:** `--backend_kwargs` JSON inline — пользователь может нарушить Контрукктор-инвариант (передать `gpu_memory_utilization=2.0`). Нужен в Backend `__init__` осмысленный assert или fallback — но это уже backend-specific. Не фиксим в этом плане; mocking — твоё.
- **Слой D таблица YAML:** если в `benchmark/*.yaml` есть записи для reasoning-моделей без `model_kind`, после смены default на plain они поедут plain one-pass, **метрики reasoning-моделей упадут** (one-pass даёт без reasoning-block нужно pre-validated в Commit 4 и дописать `model_kind: hybrid` в этих YAMLs.

## Зафиксированные NOT делаем

- `num_return_sequences` не трогаем.
- vLLM `_add_stop_string` (vocab-scan vs `_check_word_is_token`) различен с HF — унификация в F1 (бэклог).
- API `'n': N` оптимизация — F5 (бэклог).
- `tests/` в `.gitignore` — не трогаем (F4 только AGENTS.md формулировка).
- `.gitmodules`/staged deletions верхнего уровня — пользователь разберётся (F3 комментарий).
- Бутстрап rewrite, мультигпу bench-config — из `todo` (F7), не в этом заходе.

## Проверка (acceptance)

- `python3 -m pytest tests/test_refactor_logic.py -q` — все тесты проходят (включая адаптированные/новые из D7).
- `python evaluate_model.py --model ... --dataset_names <one> --max_sample_per_dataset 8 --backend_kwargs '{"temperature":0.3}'` — параметр применён.
- `python evaluate_model_api.py --base_url ... --model_kind plain` и `--model_kind hybrid --end_thinking_token_id <id>` — работает reasoning-two-pass.
- `python evaluate_model.py --model_kind auto` — `ValueError` с читаемым сообщением.
- `python evaluate_model.py --model_kind reasoning` без `--end_thinking_token_id` — `ValueError` с читаемым сообщением.
- `python evaluate_model.py --presence_penalty 1.5 --vllm` — параметр виден в `_params.jsonl` и применён.
- `python evaluate_model.py --presence_penalty 1.5` (без --vllm, HF) — warning в логе оффline.
- `python -c "from llmtf.llm import LLM; from llmtf.backends import APIBackend; LLM(backend=APIBackend(api_base='')).apply_model_prompt([])"` — `NotImplementedError` с readable message.
