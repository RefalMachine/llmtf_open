# AGENTS.md

Compact guide for agents working in `llmtf_open` (LLM evaluation framework for Russian NLP tasks). Trust the code and tests when documentation conflicts. Snapshot-specific development records live in `dev/`; remaining work is in `BACKLOG.md`.

## Format rules

- Запрещено использовать в рассуждениях и ответах специальные model tokens. Если такой токен необходимо обсудить, используй плейсхолдеры вроде `|THINK_TOKEN_START|`, `|THINK_TOKEN_END|` или имя константы `THINK_CLOSE_MARKER`.

## Current refactor status

- The layered architecture is implemented in the working tree: `BaseLLM` -> concrete `LLM` -> `Backend`, with reasoning orchestration in `llmtf/reasoning.py` and three concrete backends in `llmtf/backends/`.
- The old `llmtf/model.py`, `llmtf/models/`, legacy model facades, and `*Reasoning` subclasses are removed. Do not reintroduce them.
- The layered refactor is the validated `v0.3.0` release. This version is delivered as an ordinary commit on `main`, not as a Python package or a separate Git tag. Release notes are in `docs/releases/v0.3.0.md`. Obsolete notebooks and committed example outputs were replaced by maintained source examples; external benchmark submodules remain unchanged. The obsolete root Docker/requirements path and hard-coded launch wrappers were removed after an explicit repository-root audit.
- The fresh 2026-09-20 HF/vLLM/API matrix completed: 26 artifact sets / 208 samples, plus four aligned legacy vLLM non-reasoning controls. See `dev/TEST_REPORT_v4.md`; do not generalize beyond its models, tasks and runtime.
- Benchmark propagation, phase-local stop ids, backend-kwargs precedence, execution-aware budgeting, provenance/cache validation, API batch alignment and centralized continuation are implemented and runtime-tested.
- Task-layer correctness stabilization is recorded in `dev/TASK_BUGFIX_REPORT.md`: evaluator contracts, normal `calculate_logsoftmax` dispatch, PPL answer boundaries, stable task provenance/cache resume and the listed built-in task fixes are implemented. Managed API Instruct Fast completed with 41 totals; the public `RefalMachine/RuParam` snapshot contains 9,505 rows and completed all 19,010 orientations through managed API. Targeted one-pair HF/local-vLLM smokes also passed. Do not generalize API evidence to full local runs or to the 11,336-pair edition described in the newer paper.
- Exact local/API parity remains unavailable for unprobed whitespace-ended assistant prefills on vLLM 0.21. Foundational stop propagation is implemented and a one-sample managed-API run passed with `verified_exact`; the historical 8-sample Base parity cell has not yet been repeated.

## Environment

- Work may be performed from different hosts and containers. Inspect the active environment before testing; a missing local GPU or package is not evidence that the framework does not support it.
- In this workspace the sandbox shell's `nvidia-smi` cannot see the driver, but Docker GPU passthrough works. On 2026-09-20, `docker run --rm --gpus all llmtf:vllm-cu129 ...` reported one NVIDIA GeForce RTX 4090 and `torch.cuda.is_available() == True`. Always test CUDA inside the target container with `--gpus all`; never treat host/sandbox `nvidia-smi` failure as a blocker here.
- Docker is the intended execution path. The target installation profiles are:
  - `api`: CPU-only/API-client image, with no torch, vLLM, CUDA, or GPU requirement;
  - `hf`: extends the logical API/common dependency profile with torch/CUDA and local Hugging Face support;
  - `vllm`: extends the `hf` dependency/runtime profile with vLLM and vLLM-specific verification.
- Profile definitions live in [`docker/Dockerfile.api`](docker/Dockerfile.api), [`docker/Dockerfile.hf`](docker/Dockerfile.hf), and [`docker/Dockerfile.vllm`](docker/Dockerfile.vllm), with split dependencies under [`requirements/profiles/`](requirements/profiles/) and build instructions in [`docker/README.md`](docker/README.md). They use public Python/NVIDIA CUDA images, not NGC. The validated HF line is CUDA 12.9 + torch 2.11 and includes `flash-attn`, `flash-linear-attention`, and `causal-conv1d`; vLLM 0.21 extends the locally tagged HF image.
- The validation models are `Qwen/Qwen3.5-2B` (hybrid thinking off/on) and `Qwen/Qwen3.5-2B-Base` (foundational). The tested snapshots loaded and generated with Transformers 5.9.0 and vLLM 0.21.0. Re-run the matrix when changing model revision or runtime pins; do not replace validated pins with floating main/nightly dependencies without recording a new runtime snapshot.
- API import boundaries are split: backend exports are lazy, API uses `SamplingConfig`, evaluator imports torch optionally, and the old mixed `_common.py` is removed. A clean no-cache API image build plus live vLLM endpoint generate/probability checks passed on 2026-09-20.
- The profile requirements intentionally use `numpy>=2,<2.3` because vLLM 0.21 requires `opencv-python-headless>=4.13`, whose Python 3.9+ wheels require NumPy 2; the pure-logic suite passes on NumPy 2.2.6. Do not reintroduce a conflicting monolithic requirements file.
- `VLLM_USE_V1` references were removed because current vLLM only ships the V1 engine. Do not add them back.
- `benchmark/calculate_benchmark.py` downloads `punkt_tab` at startup, so the first run needs network access. Some IFEval helpers also download it lazily when absent.
- `.gitmodules` declares `external_benchmarks/ruwikibench` and `external_benchmarks/rubooksum`. They are outside the model refactor; do not change their gitlinks without an explicit repository-cleanup decision.
- API credentials must be supplied at runtime through environment variables or an approved secret mechanism. Never bake credentials into an image, commit them, print them in commands/logs, or serialize them into result params.

## Entry points

- `evaluate_model.py` — single-model local evaluation. Without `--vllm` it uses `HFBackend`; with `--vllm` it uses `VLLMBackend`; both are wrapped in `LLM(backend=...)`. `--ppl_scoring` selects HF-only PPL. `--model_kind {plain,reasoning,hybrid}` defaults to `plain`; `auto` is not supported.
- `evaluate_model_api.py` — evaluation against an OpenAI-compatible API via `APIBackend`; `--api_profile {auto,openai,vllm}` selects conservative standard payloads or explicit vLLM extensions. It returns `EvaluationSummary.exit_code` and performs normal cleanup.
- Both single-model CLIs use explicit opt-in `--enable_thinking`; `--disable_thinking` remains a deprecated compatibility alias and the default is false.
- `--backend_kwargs 'JSON'` accepts an object only. Explicitly supplied CLI backend options override JSON, omitted CLI options do not, and unknown constructor keys fail before model loading.
- `benchmark/calculate_benchmark.py` — YAML-driven local parallel runner with GPU allocation.
- `benchmark/calculate_benchmark_api.py` — starts vLLM API servers and evaluates through `evaluate_model_api.py`.
- `benchmark/calculate_benchmark_existing_api.py` — evaluates against an already running server.
- All three benchmark runners use `benchmark/config.py`, validate `model/defaults/tasks` sections and propagate model kind, explicit thinking mode, context/reasoning budgets, end token id, API profile, backend kwargs and sampling settings. Legacy `extra_args.think` is supported for one transition cycle with a warning.
- `benchmark/llmaaj/{generate,judge,show_benchmark}_llmaaj.py` must be invoked from the repository root. Prefer `python -m benchmark.llmaaj.generate_llmaaj ...`; the parameterized full launcher is `benchmark/llmaaj/run_full.sh`.
- `show_results.py` formats model result directories. Category definitions live in `benchmark/categories.json`; current JSON arrays and historical concatenated sample objects are both accepted.
- `dev/tools/remap_qwen35_checkpoint.py` is a maintainer-only conversion utility for historical Qwen3.5 checkpoints, not a framework entry point.

## Core (`llmtf/`)

- `base.py` — `Task`, `BaseLLM`, and `SimpleFewShotHFTask`. Every concrete task must set `_max_task_new_tokens`. Tasks use `method` in `{generate, calculate_tokens_proba, calculate_logsoftmax}`.
- `llm.py` — the single concrete `LLM(BaseLLM)`. It owns reasoning dispatch and proxies primitive work to its backend.
- `reasoning.py` — `ModelKind`, `ReasoningFormat`, `ReasoningConfig`, `ReasoningResult`, and the shared two-pass `EmulatedReasoningStrategy`.
- `continuation.py` — the single assistant-prefill contract: continuation capability, exact local rendering, trailing-whitespace verification, and next-token surface variants such as `A`/` A`. Do not duplicate trimming or candidate-space logic in backends or tasks. See `docs/assistant_continuation.md`.
- `backends/base.py` — primitive backend ABC. Concrete backends are `HFBackend`, `VLLMBackend`, and `APIBackend` in `hf.py`, `vllm.py`, and `api.py`.
- `evaluator.py` — dataset loop, aggregation, result caching, PPL path, and report creation. The random seed is fixed at 555.
- `sample_logger.py` — `JsonArrayLogger` for per-sample results and `PrettyJsonLogger` for aggregate/params files.
- There is no compatibility shim for legacy model classes. Construct models directly with `LLM(backend=HFBackend(...))`, `LLM(backend=VLLMBackend(...))`, or `LLM(backend=APIBackend(...))`, then call `from_pretrained`.

## Reasoning contract and current limitations

- `model_kind=plain`: always one-pass. `enable_thinking=True` currently warns and falls back to one-pass.
- `model_kind=hybrid`: one-pass when thinking is disabled; two-pass when enabled.
- `model_kind=reasoning`: two-pass is mandatory; disabling thinking is an error. An explicit `end_thinking_token_id` is required.
- Reasoning fields live in `LLM.reasoning_config`, not in the backend sampling `generation_config`. Defaults are `max_new_tokens_reasoning=4096` and `min_new_tokens_reasoning=1024`.
- The two-pass dispatcher supports `generate` and `calculate_tokens_proba`. PPL is HF-only and does not implement a reasoning phase.
- The current strategy assumes one reasoning candidate. `num_return_sequences > 1` is not defined for two-pass reasoning and should not be treated as supported.
- All backends consume stop token ids from the effective phase-local generation config; payload-level tests verify reasoning and continuation ids and base-config restoration.
- `ReasoningFormat` exists, but `LLM._setup_reasoning` still fixes the text close marker to the framework constant. Supporting other reasoning protocols requires additional work.

## Assistant continuation contract

- Internal and logged message roles are canonical `system`/`user`/`assistant`; historical `bot` is accepted only at the task-data boundary and normalized immediately.
- `predict` is always the raw newly generated continuation and never includes the assistant prefill.
- `--assistant_prefill_policy {auto,exact,portable,best_effort}` controls prefills. Local HF/vLLM render them exactly through `llmtf.continuation`; API `auto` first requires a profile with declared continuation support and then verifies hazardous whitespace. `portable` forbids every assistant-prefill. `best_effort` is explicit and logged as unverified.
- The validated vLLM 0.21 API probe reports that `continue_final_message` strips trailing whitespace. Do not claim local/API parity for an affected task; use a reviewed portable prompt or record a `best_effort` run as a non-parity diagnostic.
- Token-probability candidate/leading-space variants are centralized and aggregated with the documented historical `max` rule. Multi-token variants are not silently reduced to their first token.

## Context budgeting

- Primary values are `model_context_len`, `task._max_task_new_tokens`, and `reasoning_config.max_new_tokens_reasoning`/`min_new_tokens_reasoning`.
- `MaxLenContext(task, model, custom_generation_config)` returns the effective prompt budget and temporarily adjusts answer/reasoning budgets. `--max_prompt_len` was removed from the CLI.
- Repository benchmark YAMLs no longer contain legacy `max_prompt_len` or task-level `max_len`. The strict loader rejects those unknown keys; configure deployment capacity with model-level `model_context_len`.
- `MaxLenContext` receives the normalized execution mode. Hybrid-disabled and PPL reserve no reasoning budget; hybrid can skip below its floor for one task, while strict reasoning fails explicitly.

## Results, caching, and failure behaviour

- Output files are `<task>_params.jsonl`, `<task>.jsonl`, `<task>_total.jsonl`, optional `<task>_aggregation_details.jsonl`, plus `evaluation_results.txt` and `evaluation_log.txt`.
- `<task>.jsonl` is a valid pretty-printed JSON array. If a process is killed before the array closes, append a closing bracket before parsing.
- `_params.jsonl` and `_total.jsonl` contain a sanitized canonical run config and fingerprint. Cache hits require the same fingerprint; mismatch is an error unless force-recalculation or a different output identity is selected.
- `LLM.get_params()` includes additive `_llmtf_reasoning` and `_llmtf_continuation` provenance; task artifacts distinguish configured/effective reasoning budgets and requested/effective thinking.
- API batches fail closed with `BackendBatchError` carrying original indexes. Bounded retries and request timeouts preserve ordering, context errors remain errors, and an unavailable token counter returns `None` rather than zero. Requested few-shot prompts are not silently trimmed when no counter exists.
- API `/v1/models`, `/tokenize`, and `/detokenize` are optional. `auto` and `openai` use a conservative payload without assistant continuation; `vllm` explicitly enables the vLLM extensions. See `docs/api_backend.md`.
- `Evaluator` records task failures in `EvaluationSummary`; both single-model CLIs return its non-zero exit code when any requested task fails. Still inspect requested task totals because a run can contain a mixture of succeeded, skipped and failed tasks.

## Task and repository conventions

- New task: subclass `SimpleFewShotHFTask`, implement `dataset_args`, `test_split_name`, `prompt_split_name`, `create_messages`, `evaluate`, and `aggregation`, set `_max_task_new_tokens`, then register it in `llmtf/tasks/__init__.py`.
- `conversation_configs/*.json` define local chat templates. Use `default_foundational.json` with `--is_foundational`; `--conv_path auto` uses the tokenizer template.
- RAG LLM-judge tasks are registered only when `LLMAAJ_API_BASE`, `LLMAAJ_API_KEY`, and `LLMAAJ_MODEL_NAME` are all set.
- `tests/test_refactor_logic.py` is a real pure-logic regression suite and must not be removed. `.gitignore` no longer ignores all `test*`; scratch artifacts under `tests/` must be handled explicitly.
- PPL is the mean answer-token log probability, not exponentiated perplexity, and is currently HF-only.

## Verification requirements

Always run the dependency-free checks after model/reasoning/backend changes:

```bash
python3 tests/test_refactor_logic.py
python3 -m compileall -q llmtf evaluate_model.py evaluate_model_api.py benchmark show_results.py dev/tools examples
git diff --check
```

If `pytest` is installed, also run:

```bash
python3 -m pytest tests/test_refactor_logic.py -q
```

Before claiming runtime safety, build and test all three Docker profiles:

- `api`: prove that torch, vLLM and CUDA are not required, then run API generate/probability checks against the supplied endpoint.
- `hf`: run local HF generate, probability and PPL checks on the GPU host.
- `vllm`: run local vLLM checks and start a local vLLM API server; this target must extend the HF profile rather than duplicate it.

Then run a small real-model matrix (`--max_sample_per_dataset 8`, one generate task and one probability task) on the models above:

- Instruct/hybrid model on HF, vLLM and API with thinking explicitly off and on: generate and calculate-token-probability; enabled runs use an explicit end token.
- PPL on HF for the instruct model. PPL has no reasoning phase; vLLM/API must return an explicit unsupported-capability result.
- Base model in foundational mode on HF, vLLM and API: generate and calculate-token-probability; additionally PPL on HF and unsupported-capability checks on vLLM/API.
- One strict reasoning contract smoke per backend, separate from the main hybrid off/on matrix.
- Legacy comparison from a detached worktree at `504bd7fc2793c900e97010f124651460afcd4812`, with separate outputs and the same model cache/runtime wherever compatible. The reference commands are in `dev/TEST_PLAN_v4.md`.

Use `--enable_thinking` for enabled cells and `--disable_thinking` for explicit disabled cells. Record exact commands and inspect both `_params.jsonl` and `_total.jsonl`. The completed reference run is documented in `dev/TEST_REPORT_v4.md`; rerun the relevant cells after runtime-sensitive changes.

When a future task is started from a GPU-capable container with a reference to `dev/REFACTOR_PLAN_v4.md` and API access supplied by the user:

1. verify Docker GPU access with `docker run --rm --gpus all ...`; do not use sandbox-shell `nvidia-smi` as the deciding check;
2. record the target image id plus torch/CUDA/Transformers/vLLM and GPU versions;
3. run dependency-free tests first;
4. run HF and local vLLM one-sample smoke tests on the supplied GPU;
5. run API tests using credentials from the environment without echoing them;
6. execute the v4 matrix and save only sanitized commands/configuration and non-secret results.

## Quick smoke command

Use the containerized smoke commands in `dev/TEST_PLAN_v4.md`; include
`docker run --gpus all` and use `bash -c`, not a login shell.

No lint/typecheck/test targets are configured. Do not substitute a successful import or pure-logic test for a real backend run.
