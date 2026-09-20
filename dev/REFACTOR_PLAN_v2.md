# Рефакторинг бекендов и ризонинга — план v2

> Исторический development design record. Актуальный контракт находится в
> `../docs/`.

## Зафиксированные решения

- `BaseLLM` (abc, контракт для evaluator/tasks) + `LLM` (concrete, держит `backend`).
- `Backend` abstract + 3 concrete (`HFBackend`/`VLLMBackend`/`APIBackend`) в новом пакете `llmtf/backends/`.
- Бекенд принимает **messages**; ризонинг-флоу (двойной вызов, манипуляция messages) — на уровне `LLM`, не бекенда.
- `Backend.count_tokens_for_messages(messages, **kw) -> int` — единый контракт; `apply_model_prompt`/`count_tokens_for_prompt` двойной семантики (str/int) исчезают.
- `ReasoningConfig` выносится из `generation_config` (читают `MaxLenContext` и `EmulatedReasoningStrategy`).
- `generate` = `generate_batch(bs=1)`; бекенд expose-ит батч (для API — `ThreadPoolExecutor` поверх single, как сейчас).
- PPL (`calculate_logsoftmax_batch`) — optional метод backend-а, default `NotImplementedError`; HF-only.
- `return_tokens` — optional, API кидает `NotImplementedError` (как сейчас).
- Пакет: `llmtf/backends/` + `llmtf/llm.py`; `llmtf/models/__init__.py` — shim re-export для `from llmtf.model import HFModel, ...`.
- Migration: поэтапно, eval-чек после каждого слоя — гоняется интерактивно (golden не снимается, сравнение с версией до всех рефакторингов).

---

## Анализ текущего состояния

### Чем реально отличаются `LocalHostedLLM` и `ApiVLLMModel`

Поверхность реальных отличий крошечная, остальное — дублирование.

| Аспект | Local (HF/vLLM) | API |
|---|---|---|
| Токенизатор | реальный `AutoTokenizer` локально | эмуляция через HTTP `/tokenize` |
| Загрузка модели | `_load_plain_model`/`_load_lora` (веса в процесс) | `GET /v1/models` (только метаданные) |
| Примитив генерации | `model.generate(...)` in-process | `POST /v1/chat/completions` |
| Batch | настоящий тензорный батч | `ThreadPoolExecutor` поверх single-sample |
| `calculate_logsoftmax` (PPL) | да (HF) / нет (vLLM) | нет |
| `return_tokens=True` | да | `NotImplementedError` |
| Stop-token logic | `_check_word_is_token`/`add_stop_token`/`_augment_tokens_of_interest` (нужен vocab) | просто `stop_strings.append(...)` |
| Промпт-рендер | `apply_model_prompt` -> str | `apply_model_prompt` -> int (на самом деле это `count_tokens` под чужим именем) |

Всё остальное — `generate_batch`/`calculate_tokens_proba_batch` диспетчеры ризонинга, `_setup_reasoning`, `_reasoning`/`_reasoning_kind`/`_reasoning_fmt` тройка, `add_stop_strings`/`reset_stop_strings`, `get_max_model_len`, `get_params` — идентично или тривиально параметризуемо.

### Что в текущем рефакторинге коряво

1. **Три копии одного диспетчера ризонинга.** `generate_batch` и `calculate_tokens_proba_batch` в `hf.py`/`vllm.py`/`api.py` — это ~80 строк `if kind == plain / reasoning / hybrid ... self._reasoning._run_reasoning_then_continuation(...)` буквально копипаст между тремя файлами. x3 = ~240 строк мёртвого дубликата. При этом `EmulatedReasoningStrategy` уже бекенд-агностик (принимает `reasoning_fn`/`continuation_fn` колбэки) — нужно было просто поднять диспетчер на уровень LLM, а не дублировать.
2. **Две копии `_setup_reasoning`.** `LocalHostedLLM._setup_reasoning` и `ApiVLLMModel._setup_api_reasoning` совпадают на 90% — отличается только `_sniff_think_close_token_id` (tokenizer) vs `_sniff_think_close_token_id_api` (HTTP /tokenize). Это один метод с инжектируемым сниффером.
3. **`generation_config` — год-объект.** На нём висят и сэмплинг-параметры, и `max_new_tokens_reasoning`/`reasoning_truncing_prompt`/`end_thinking_token_id` (последние три — только ради `MaxLenContext` и `EmulatedReasoningStrategy`). Это корневая причина «ризонинг размазан по бекендам»: бекенд вынужден таскать через себя поля, которые его не касаются.
4. **`enable_thinking` протекает в примитив бекенда.** Это kwarg ризонинг-слоя, а пробрасывается в `_generate_batch_impl` -> `apply_model_prompt(add_think_token=...)` -> на API в `chat_template_kwargs`. Примитив бекенда не должен знать про thinking.
5. **`apply_model_prompt` — разные контракты.** Local возвращает `str`, API возвращает `int` (токен-каунт). `evaluator.evaluate_dataset_ppl` делает `len(model.apply_model_prompt(...))` — на API бы упало, спасает только то, что PPL HF-only. Это не «разные классы», это «одинаковый интерфейс с разной семантикой» — хуже дублирования.
6. **`add_assistant_prompt_to_output`** — флаг ризонинг-флоу (нужен только двухпроходному режиму), но торчит в сигнатуре бекенда как легитимный generate-параметр.

### Ключевые потребители контракта `LLM` (что нельзя сломать)

- `evaluator.py`: `generate_batch`/`calculate_tokens_proba_batch`/`calculate_logsoftmax_batch`, `apply_model_prompt` (в PPL-пути для shift), `count_tokens_for_prompt`, `support_method`, `add_stop_strings`/`reset_stop_strings`, `get_max_model_len`, `get_params`, `model.generation_config` (через `MaxLenContext`).
- `base.py:SimpleFewShotHFTask._prepare_messages`: `count_tokens_for_prompt`, `apply_model_prompt`, `support_method`.
- `utils.py:MaxLenContext`: `generation_config.max_new_tokens` + `max_new_tokens_reasoning` (мутирует).
- CLI: `evaluate_model*.py`, `benchmark/calculate_benchmark*.py` прокидывают `max_new_tokens_reasoning`/`end_thinking_token_id`/`reasoning_truncing_prompt` в `from_pretrained`.

---

## Слои рефакторинга

### Слой 1. `Backend` abstract + 3 concrete

Новый пакет `llmtf/backends/`, `base.py`:

```python
class Backend(Base):                       # abstract
    def support_method(self, method) -> bool: ...
    def from_pretrained(self, model_dir, *, conversation_template_path, is_foundational, **kw): ...
    # примитивы — messages in, (prompts, outputs, infos) out
    def generate_batch(self, messages_batch, *, sampling_config, enable_thinking=False,
                       return_tokens=False, include_stop_str_in_output=False,
                       skip_special_tokens=True, **kw): ...
    def calculate_tokens_proba_batch(self, messages_batch, tokens_of_interest, **kw): ...
    def calculate_logsoftmax_batch(self, messages_batch, *, log_only_last=True, **kw):
        raise NotImplementedError
    # токен-счёт — один вызов, opaque внутри
    def count_tokens_for_messages(self, messages, *, incomplete_last_bot_message=True,
                                  add_think_token=False) -> int: ...
    # конфиг бекенда
    def get_max_model_len(self) -> int: ...
    def get_params(self) -> dict: ...
    def add_stop_strings(self, ss): ...
    def reset_stop_strings(self): ...
    @property
    def generation_config(self) -> GenerationConfig: ...   # чисто сэмплинг
```

`llmtf/backends/hf.py` — `HFBackend(Backend)`:
- из `hf.py`: `_load_plain_model`, `_load_lora`, `_resolve_model_class`, `calculate_logsoftmax_batch`, vocab-based `_augment_tokens_of_interest`, `_check_word_is_token`, `add_stop_token`, `_add_stop_string` (переопределённая с vocab-сканом).
- `generate_batch`/`calculate_tokens_proba_batch` — то, что сейчас `_generate_batch_impl`/`_calculate_tokens_proba_batch_impl`.
- `count_tokens_for_messages`: `apply_chat_template` + `len(tokenizer(...))`.

`llmtf/backends/vllm.py` — `VLLMBackend(Backend)`:
- из `vllm.py`: `_load_plain_model`, `_load_lora`, `_get_lora_request`, `_get_max_lora_rank`, flash-attn warning, `SamplingParams`-маппинг.
- `calculate_logsoftmax_batch` — не override (default `NotImplementedError`).
- `support_method` возвращает `['generate','calculate_tokens_proba']`.

`llmtf/backends/api.py` — `APIBackend(Backend)`:
- из `api.py`: HTTP `/v1/models`, `/v1/chat/completions`, `/tokenize`, `ThreadPoolExecutor`.
- `count_tokens_for_messages` — `/tokenize` -> `data['count']` (один вызов вместо нынешнего `apply_model_prompt`-возвращающего-int).
- `add_stop_strings` — простое `stop_strings.append` (без vocab-логики).
- `_augment_tokens_of_interest` — client-side эмуляция через `[' '+token, token]` (как сейчас в `calculate_tokens_proba`).
- `calculate_logsoftmax_batch` — не override.
- `return_tokens=True` -> `NotImplementedError`.

**Что уходит из бекендов навсегда:** `_reasoning`/`_reasoning_kind`/`_reasoning_fmt`, `_setup_reasoning`/`_setup_api_reasoning`, три копии диспетчера `generate_batch`/`calculate_tokens_proba_batch` с `if kind == plain/reasoning/hybrid ...`.

### Слой 2. `ReasoningConfig`

`llmtf/models/reasoning.py`:

```python
@dataclass
class ReasoningConfig:
    model_kind: ModelKind = ModelKind.plain
    max_new_tokens_reasoning: Optional[int] = None
    truncing_prompt: str = ReasoningFormat.truncation_prompt
    end_thinking_token_id: Optional[int] = None
```

- `EmulatedReasoningStrategy._build_reasoning_config` / `_process_reasoning_outputs` читают из `ReasoningConfig` вместо `getattr(generation_config, ...)`.
- `_setup_reasoning` (бывший, в LocalHostedLLM) переезжает в `LLM._setup_reasoning` — одна реализация; сниффер think-close токена инжектируется: `backend.sniff_think_close_token_id()` (HF/vLLM -> tokenizer, API -> `/tokenize`).
- `MaxLenContext` (`utils.py:62`) переписывается: `model.reasoning_config.max_new_tokens_reasoning` вместо `model.generation_config.max_new_tokens_reasoning`; restore на выходе — тоже через `reasoning_config`.

### Слой 3. `BaseLLM` (abc) + `LLM` (concrete)

`llmtf/base.py`:

```python
class BaseLLM(abc.ABC):
    # публичный контракт — то, от чего зависит evaluator/tasks
    def generate(self, messages, **kw)               # = generate_batch([messages])[0]
    def generate_batch(self, messages_batch, **kw)
    def calculate_tokens_proba(self, messages, toi, **kw)
    def calculate_tokens_proba_batch(self, ...)
    def calculate_logsoftmax(self, messages, **kw)
    def calculate_logsoftmax_batch(self, ...)         # delegating; HF-only
    def count_tokens_for_prompt(self, messages, **kw) -> int   # делегирует backend
    def support_method(self, m) -> bool
    def get_max_model_len(self) -> int
    def get_params(self) -> dict
    def add_stop_strings(self, ss); def reset_stop_strings(self)
    @property
    def reasoning_config(self) -> ReasoningConfig
    @property
    def generation_config(self) -> GenerationConfig
```

`llmtf/llm.py`:

```python
class LLM(BaseLLM):
    def __init__(self, backend: Backend, *, reasoning_config=None, generation_config=None):
        self.backend = backend
        self._reasoning_config = reasoning_config or ReasoningConfig()
        self._reasoning: Optional[EmulatedReasoningStrategy] = None
        self._setup_reasoning(...)   # вызывается из from_pretrained

    def from_pretrained(self, model_dir, *, model_kind="auto",
                        max_new_tokens_reasoning=None, ...):
        self.backend.from_pretrained(model_dir, ...)
        self._setup_reasoning(model_kind=model_kind,
                              max_new_tokens_reasoning=max_new_tokens_reasoning, ...)

    # ЕДИНЫЙ диспетчер ризонинга (вместо 3 копий)
    def generate_batch(self, messages, *, enable_thinking=False, generation_config=None,
                       add_reasoning_truncing_prompt=False, add_reasoning_info=True,
                       add_assistant_prompt_to_output=True, **kw):
        kind = self._reasoning_config.model_kind
        if kind == ModelKind.plain or not enable_thinking:
            if kind == ModelKind.reasoning and not enable_thinking: raise ValueError(...)
            return self.backend.generate_batch(messages, sampling_config=gc_or_default,
                                                enable_thinking=False, **kw)
        gc = generation_config or self.generation_config
        return self._reasoning._run_reasoning_then_continuation(
            messages,
            reasoning_fn=lambda mb, generation_config, **kw:
                self.backend.generate_batch(mb, sampling_config=generation_config, **kw),
            continuation_fn=lambda mb, generation_config=None, **kw:
                self.backend.generate_batch(mb, sampling_config=gc, **kw),
            reasoning_config=self._reasoning_config,   # вместо generation_config
            mode='generate', ...)
```

Аналогично — единый `calculate_tokens_proba_batch` с `mode='ctp'`.

`enable_thinking` больше не торчит как kwarg ризонинга в примитив бекенда — это флаг ризонинг-слоя LLM; бекенд получает его только как флаг рендера внутри `count_tokens_for_messages`/`apply_chat_template`. `add_assistant_prompt_to_output`/`add_reasoning_truncing_prompt`/`add_reasoning_info` — тоже остаются на уровне LLM (они управляют пост-обработкой `EmulatedReasoningStrategy`).

### Слой 4. Consumers

- **`evaluator.py`**: PPL-путь — `len(model.apply_model_prompt(m['messages']))` -> `model.count_tokens_for_prompt(m['messages'])`. Остальное без изменений (ходит через `BaseLLM`). `MaxLenContext` см. Слой 2.
- **`base.py:SimpleFewShotHFTask._prepare_messages`** (строки 181–192): `model.count_tokens_for_prompt(model.apply_model_prompt(messages))` -> `model.count_tokens_for_prompt(messages)` — message-oriented.
- **`evaluate_model.py`/`evaluate_model_api.py`**: `model = LLM(backend=HFBackend(...))` (или `VLLMBackend`/`APIBackend`); `from_pretrained` пробрасывает `model_kind`/`max_new_tokens_reasoning`/`end_thinking_token_id`/`reasoning_truncing_prompt` в `LLM._setup_reasoning`. CLI-args без изменений.
- **`benchmark/calculate_benchmark*.py`**: без изменений (прокидывают те же CLI args).
- **`llmtf/models/__init__.py`**: shim —

```python
def HFModel(**kw): return LLM(backend=HFBackend(**kw))
def VLLMModel(**kw): return LLM(backend=VLLMBackend(**kw))   # с vllm-import guard
def ApiVLLMModel(**kw): return LLM(backend=APIBackend(**kw))
```

  чтобы `from llmtf.model import HFModel, ApiVLLMModel` продолжало работать.

### Слой 5. Cleanup

Удалить:
- `llmtf/models/local.py` (`LocalHostedLLM` — содержимое разделится).
- `llmtf/models/hf.py`/`vllm.py`/`api.py` (перемещены в `backends/`).
- Три копии `_setup_*reasoning`.
- Три копии диспетчера ризонинга (~240 строк дубликата).
- Двойной контракт `apply_model_prompt` (str) vs API (int).
- Все упоминания `VLLM_USE_V1` (stale по AGENTS.md).

`llmtf/model.py` (shim, если есть) — обновить re-export.

---

## Порядок исполнения

Каждый слой — отдельный коммит, после которого гоняется eval (интерактивно, сравнение с версией до всех рефакторингов).

1. **Слой 1a**: `llmtf/backends/base.py` (abstract `Backend`) — пустые concrete-классы-заглушки, re-export из `models/__init__.py`. Чек: `python -c "from llmtf.backends import HFBackend"` + tiny eval.
2. **Слой 1b**: перенести реализацию из `hf.py`/`vllm.py`/`api.py` в `HFBackend`/`VLLMBackend`/`APIBackend`. Старые `HFModel`/`VLLMModel`/`ApiVLLMModel` временно делегируют в backend. Чек: tiny eval на HF.
3. **Слой 2**: `ReasoningConfig`, `EmulatedReasoningStrategy` читает из него, `MaxLenContext` переписан. Чек: tiny eval hybrid.
4. **Слой 3**: `BaseLLM` abc + `LLM` concrete с единым диспетчером. Чек: tiny eval на HF/vLLM/API × plain/hybrid × generate/ctp.
5. **Слой 4**: consumers (`evaluator` PPL-путь, `_prepare_messages`, CLI). Чек: tiny eval PPL.
6. **Слой 5**: cleanup старых файлов, `VLLM_USE_V1`. Чек: импорт + tiny eval.

---

## Отчёт о выполненной работе

Рефакторинг выполнен полностью (все 5 слоёв + последующий cleanup фасадов). Ниже — что фактически сделано, с подтверждениями.

### Итоговая архитектура

```
llmtf/
├── base.py            # BaseLLM (abc) — публичный контракт; Task, SimpleFewShotHFTask
├── llm.py             # LLM(BaseLLM) — единственный concrete-класс, держит backend
├── reasoning.py       # ModelKind, ReasoningFormat, ReasoningConfig, EmulatedReasoningStrategy
├── backends/
│   ├── base.py        # Backend (abc) — интерфейс примитива
│   ├── _common.py      # shared third-party imports для concrete бекендов
│   ├── hf.py          # HFBackend (generate / ctp / logsoftmax / PPL)
│   ├── vllm.py        # VLLMBackend (generate / ctp; vllm lazy/guarded)
│   ├── api.py         # APIBackend (generate / ctp; HTTP)
│   └── __init__.py
├── evaluator.py       # Evaluator (ходит через BaseLLM)
├── sample_logger.py   # JsonArrayLogger / PrettyJsonLogger
├── utils.py           # MaxLenContext (читает reasoning_config)
└── tasks/             # TASK_REGISTRY; type-hints на BaseLLM
```

**Фасадов нет.** `llmtf/models/` и `llmtf/model.py` удалены. Конструкция модели — только прямой `LLM(backend=HFBackend(...))`.

### Слой 1a — `Backend` abstract + stubs [ВЫПОЛНЕНО]

- Создан пакет `llmtf/backends/` с `base.py` (абстрактный `Backend`): `generate_batch`, `calculate_tokens_proba_batch`, `calculate_logsoftmax_batch` (default `NotImplementedError`), `count_tokens_for_messages` (единый message→int контракт), `sniff_think_close_token_id` (default `(None, False)`), `support_method`, `from_pretrained`, `get_max_model_len`/`get_params`/`add_stop_strings`/`reset_stop_strings`, `generation_config` (атрибут, не abstract property — реалистичнее для backend-а, который выставляет его в `_load_model`).
- `hf.py`/`vllm.py`/`api.py` — заглушки `HFBackend`/`VLLMBackend`/`APIBackend`.

### Слой 1b — перенос реализации + единый диспетчер [ВЫПОЛНЕНО]

- `HFBackend`/`VLLMBackend`/`APIBackend` получили всю примитивную логику (loading, gen, ctp, PPL для HF, stop-strings, render/token-count, sniff). Из бекендов убраны ризонинг-диспетчер и `_setup_*reasoning`.
- `LocalHostedLLM` (в `llmtf/models/local.py`, промежуточный слой) переписан как composition-база: держит `self.backend`, содержит **единую** копию диспетчера ризонинга + единый `_setup_reasoning` (через `backend.sniff_think_close_token_id()`); прокси к бекенду для `generation_config`/`apply_model_prompt`/`count_tokens_*`/`stop_strings`/`support_method`/`get_params`/`get_max_model_len`.
- `HFModel`/`VLLMModel`/`ApiVLLMModel` (в `llmtf/models/`) — тонкие фасады, создающие соответствующий бекенд и наследующие диспетчер. `ApiVLLMModel` теперь наследует тот же base, что и HF/vLLM (раньше был отдельным классом) — копия диспетчера сократилась с 3 до 1.
- **Побочная правка:** `llmtf/models/reasoning.py` → `llmtf/reasoning.py` (вынесен из пакета `llmtf.models`, чтобы разорвать цикл импортов: бекенды зависят от `reasoning`, а `llmtf.models.__init__` — от бекендов через фасады). `reasoning.py` больше не тянет `llmtf.models._common` (только stdlib).
- **Дублирование диспетчера:** ~240 строк (3 копии) → ~80 (одна).

### Слой 2 — `ReasoningConfig` [ВЫПОЛНЕНО]

- `llmtf/reasoning.py`: добавлен `ReasoningConfig` (dataclass: `model_kind`, `max_new_tokens_reasoning`, `fmt: ReasoningFormat`) с property `is_reasoning`.
- `EmulatedReasoningStrategy` конструируется от `ReasoningConfig` и читает ризонинг-поля **live из конфига** (не из `generation_config`); тримы `MaxLenContext` применяются на лету. `_process_reasoning_outputs` больше не принимает `generation_config` — только флаги постобработки.
- `_setup_reasoning` строит `ReasoningConfig` и `EmulatedReasoningStrategy(rc)`. Ризонинг-поля **больше не пишутся в `generation_config`** — он остаётся чисто сэмплинговым.
- `MaxLenContext` (`utils.py`) переписан: читает/мутирует `model.reasoning_config.max_new_tokens_reasoning` вместо `model.generation_config.max_new_tokens_reasoning`; флаг reasoning-active — `rc.is_reasoning`, а не `hasattr(gc, ...)`.
- **Корневая причина «ризонинг размазан по бекендам» устранена:** `generation_config`-год-объект больше не существует.

### Слой 3 — `BaseLLM` (abc) + `LLM` (concrete) [ВЫПОЛНЕНО]

- `llmtf/base.py`: абстрактный `LLM` переименован в `BaseLLM` (публичный контракт). Добавлены `count_tokens_for_messages` и `reasoning_config` в абстрактный интерфейс. Type hints в `SimpleFewShotHFTask` обновлены на `BaseLLM`.
- `llmtf/llm.py` (новый): concrete `LLM(BaseLLM)` — единая реализация с `backend` и диспетчером ризонинга (перенесена из `models/local.py`).
- `llmtf/models/local.py` — shim (`from llmtf.llm import LLM as LocalHostedLLM`).
- Массовый rename `LLM`→`BaseLLM` в ~18 файлах задач и `_common.py` (word-boundary sed, безопасно для `LLMAAJ`/`VLLMModel`).

### Слой 4 — consumers [ВЫПОЛНЕНО]

Миграция потребителей на message-oriented `count_tokens_for_messages`:
- `llmtf/evaluator.py` (PPL-путь): `len(model.apply_model_prompt(m['messages']))` → `model.count_tokens_for_messages(m['messages'])`.
- `llmtf/base.py:SimpleFewShotHFTask._prepare_messages`: двойной вызов `count_tokens_for_prompt(apply_model_prompt(...))` → `count_tokens_for_messages(...)`.
- 6 task-файлов с собственной `_prepare_messages`: `llm_as_a_judge`, `ruopinionne` (×2), `shlepa`, `darumeru` (×2), `nlpcoreteam` (×2), `daru_treeway_summ`.
- CLI без изменений: `from_pretrained` пробрасывает `model_kind`/`max_new_tokens_reasoning`/`end_thinking_token_id`/`reasoning_truncing_prompt` в `LLM._setup_reasoning` → `ReasoningConfig`.

### Слой 5 — cleanup [ВЫПОЛНЕНО]

- **`VLLM_USE_V1` удалён** из `evaluate_model.py`, `benchmark/calculate_benchmark.py`, `benchmark/calculate_benchmark_api.py`, `run_llmaaj_full.sh`, `llmtf/base.py` (закомментированные строки).
- AGENTS.md синхронизирован с новой архитектурой.

### Cleanup фасадов (пост-Слой 5, по запросу пользователя) [ВЫПОЛНЕНО]

Легаси-фасады были чистой индирекцией без внешних потребителей (все 5 call sites — внутри репо). Удалены:
- Пакет `llmtf/models/` целиком (`local.py`, `hf.py`, `vllm.py`, `api.py`, `_common.py`, `__init__.py`).
- `llmtf/model.py` shim.
- `llmtf/models/sample_logger.py` → `llmtf/sample_logger.py` (import в `evaluator.py` обновлён).
- `LLM.from_pretrained` поднят из фасадов в `LLM` (единственный lifecycle-метод: грузит backend + настраивает reasoning).
- `VLLMBackend.__init__` теперь зовёт `_ensure_vllm_available()` (раньше это делал facade `VLLMModel.__init__`).
- 5 call sites мигрированы на прямой `LLM(backend=...)`:
  - `evaluate_model.py` → `LLM(backend=HFBackend(...))` / `LLM(backend=VLLMBackend(...))`
  - `evaluate_model_api.py` → `LLM(backend=APIBackend(...))`
  - `llmtf/tasks/__init__.py` (RAG-LLM-judge init) → `LLM(backend=APIBackend(...))`
  - `benchmark/llmaaj/generate_llmaaj.py`, `benchmark/llmaaj/judge_llmaaj.py` → `LLM(backend=APIBackend(...))`
- AGENTS.md обновлён: убраны упоминания фасадов/`llmtf.models`/`model.py`; добавлен явный паттерн `LLM(backend=...)`.

### Тесты [ВЫПОЛНЕНО]

`tests/test_refactor_logic.py` переписан под новую архитектуру — 13/13 pure-logic тестов проходят (без torch/vllm/transformers, через stub-модули):
- импорты + `ModelKind` enum + `ReasoningConfig` defaults
- удаление фасадов из публичного API (`llmtf.model`/`llmtf.models` не импортируются)
- `EmulatedReasoningStrategy` two-pass: stop-by-think, stop-by-length (appends close marker + truncation prompt), ctp-mode payload shape
- `JsonArrayLogger` valid array / empty / salvage-after-kill; `PrettyJsonLogger` historical format
- `LLM._setup_reasoning` auto-detect: hybrid когда backend знает think-токен, plain иначе; `generation_config` остаётся чистым от ризонинг-полей
- LLM dispatcher: plain+enable_thinking warns + one-pass; reasoning-kind + enable_thinking=False raises `ValueError`
- `MaxLenContext` тримит `reasoning_config.max_new_tokens_reasoning` (не `generation_config`), восстанавливает на exit

### Проверено без GPU-окружения

- `py_compile` всех затронутых файлов (бэкенды, модели, задачи, CLI, utils, evaluator, sample_logger).
- Импорты: `BaseLLM`/`LLM`/`HFBackend`/`VLLMBackend`/`APIBackend`; `issubclass(LLM, BaseLLM)`; `VLLMModel=None` при отсутствии vllm (ожидаемо).
- Реестр задач (87) собирается; `Evaluator`/`MaxLenContext` импортируются.
- Smoke-тесты через `FakeBackend`: plain/hybrid generate, two-pass ризонинг (stop-by-length), live-read `max_new_tokens_reasoning`, `MaxLenContext` trim+restore, прокси-методы.
- `grep` подтверждает: в коде (`.py`, кроме `REFACTOR_PLAN*.md` и теста) нет ссылок на `HFModel`/`VLLMModel`/`ApiVLLMModel`/`LocalHostedLLM`/`from llmtf.model`/`from llmtf.models`.

### Что осталось на пользователя (GPU-окружение)

Tiny eval из AGENTS.md (матрица HF/vLLM/API × plain/hybrid × generate/ctp, одна таска, `--max_sample_per_dataset 8`) и сравнение с версией до всех рефакторингов. Цикл тестирования — интерактивно с пользователем.
