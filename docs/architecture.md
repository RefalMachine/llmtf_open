# Архитектура

## Слои

```text
Task / Evaluator
       |
       v
   BaseLLM
       |
       v
      LLM  ---- reasoning / continuation orchestration
       |
       v
 Backend primitive: HFBackend | VLLMBackend | APIBackend
```

`BaseLLM` задаёт task-facing контракт. Единственная concrete модель — `LLM`.
Она решает, нужен ли one-pass или two-pass запуск, но не содержит runtime-код
конкретного движка. Backend отвечает за загрузку, chat rendering, generate,
token probability и доступные ему дополнительные primitives.

Старые `llmtf/model.py`, `llmtf/models/`, model facades и reasoning subclasses
удалены. Совместимого shim для них нет.

## Основные файлы

- `llmtf/base.py` — `Task`, `BaseLLM`, `SimpleFewShotHFTask`;
- `llmtf/llm.py` — единый `LLM` и dispatch;
- `llmtf/reasoning.py` — model kinds, execution mode и two-pass strategy;
- `llmtf/continuation.py` — assistant-prefill и token surface forms;
- `llmtf/backends/base.py` — backend ABC и capability errors;
- `llmtf/backends/{hf,vllm,api}.py` — runtime adapters;
- `llmtf/evaluator.py` — dataset loop, budgeting, cache и reports;
- `llmtf/provenance.py` — canonical config, redaction и fingerprint;
- `llmtf/sample_logger.py` — валидный JSON array и pretty JSON sidecars.

## One-pass и two-pass

`resolve_reasoning_execution` один раз нормализует режим для budgeting и
dispatch. Plain и hybrid-disabled выполняют один backend-вызов. Hybrid-enabled
и strict reasoning выполняют:

1. reasoning phase с отдельной копией sampling config и phase-local stop id;
2. проверку завершения по `THINK_CLOSE_MARKER` или лимиту;
3. точную сборку assistant continuation;
4. answer phase через `generate` либо `calculate_tokens_proba`.

Базовый sampling config после reasoning не мутируется. `predict` всегда
содержит только вновь сгенерированный видимый ответ; reasoning сохраняется в
`info.reasoning`.

## Continuation

Внутренние роли канонические: `system`, `user`, `assistant`. Исторический
`bot` нормализуется только на task boundary. HF и local vLLM используют общий
точный renderer. API применяет capability/policy gate. Полный контракт описан
в [`assistant_continuation.md`](assistant_continuation.md).

## Context budgeting

`MaxLenContext` использует deployment context, `_max_task_new_tokens` и
фактический execution mode. Hybrid reasoning может быть отключён для одной
задачи, если не помещается minimum budget; strict reasoning завершается
ошибкой. Состояние budget восстанавливается даже после исключения.

## Backend capabilities

| Primitive | HF | local vLLM | API |
|---|---:|---:|---:|
| generate | да | да | да |
| token probability | да | да | при наличии logprobs |
| answer-token log probability (PPL path) | да | нет | нет |
| exact local assistant continuation | да | да | зависит от profile/probe |

API exports и torch imports ленивые, поэтому CPU/API профиль не требует
локального model runtime.

## Добавление backend

Новый backend реализует primitives из `llmtf/backends/base.py`, объявляет
capabilities и возвращает выровненные batch results. Он не должен копировать
reasoning, continuation, candidate-space или cache logic.

## Добавление задачи

Concrete task должен задать `_max_task_new_tokens`, dataset arguments, test и
prompt splits, `create_messages`, `evaluate` и `aggregation`. Метод задачи —
один из `generate`, `calculate_tokens_proba`, `calculate_logsoftmax`.
Регистрация находится в `llmtf/tasks/__init__.py`.

Обычный evaluator dispatch поддерживает все три метода. До backend-вызова он
проверяет capability, task budget, выравнивание dataset payloads и canonical
roles; после вызова — alignment batch result и согласованность metric keys с
`aggregation()`. Повторная программная регистрация имени требует явного
`allow_override=True`.
