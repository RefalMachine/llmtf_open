# Validation report: refactor v4

> Development validation artifact for one recorded runtime snapshot.

Date: 2026-09-20

This report records the fresh validation run described by `TEST_PLAN_v4.md`.
Raw outputs are outside the repository at:

```text
/var/mtikhomi/work_folder/projects/devel/llmtf_validation_v4_fresh_20260920
```

## Runtime snapshot

- GPU: NVIDIA GeForce RTX 4090.
- vLLM image: `llmtf:vllm-cu129`, image id
  `sha256:d32605e9216b680faa60a510628cbc385d4ae58507125e9ea499697f03ec4722`.
- torch/CUDA/Transformers/vLLM: 2.11.0+cu129 / 12.9 / 5.9.0 / 0.21.0.
- Models: the same local Hugging Face snapshots of `Qwen/Qwen3.5-2B` and
  `Qwen/Qwen3.5-2B-Base` were mounted into every refactor and legacy run.
- Refactor offline vLLM used `gpu_memory_utilization=0.92`; the unmodified
  legacy constructor used its historical 0.95 default. Legacy also left prefix
  caching disabled, while the refactor run enabled it. These are recorded
  runtime confounders, not silently treated as parity.

## Dependency-free and unit gates

- `python3 tests/test_refactor_logic.py`: 26 passed, 0 failed.
- At the matrix snapshot, `unittest` discovery contained 41 passing tests. The
  final v0.3.0 release gate contains 57 tests: 56 pass on the minimal host and
  one dataset runtime test is skipped there; that test and all six example
  checks pass in the `llmtf:api` image. The backend suite also passed inside
  the vLLM image, including vLLM-specific payload/default checks.
- `compileall`: passed.
- `git diff --check`: passed at the validation snapshot.
- Host pytest was not installed; the same dependency-free tests were executed
  directly instead.

## Refactor real-model matrix

The refactor produced 26 successful task artifact sets containing 208 sample
records:

- HF: 8 instruct/Base generate-or-probability cells plus 2 HF-only PPL cells;
- offline vLLM: 8 instruct/Base generate-or-probability cells;
- vLLM-compatible API: 8 instruct/Base generate-or-probability cells.

The instruct model was exercised with thinking explicitly disabled and enabled;
the enabled cells used an explicit end-token id. A separate strict reasoning
contract was exercised on HF, offline vLLM, and API. The Base model was run in
foundational mode. vLLM/API PPL returned the expected unsupported-capability
result rather than a false success.

The vLLM 0.21 API probe confirmed that trailing whitespace in an assistant
continuation is stripped. Consequently, instruct API continuation cells were
recorded under explicit `best_effort`; they are successful diagnostics but not
local/API prompt parity. Base API `exact` passed the continuation gate.

## Reasoning evidence

Offline vLLM `hybrid_on` and strict runs each produced 8 generate and 8
probability records. Every record contains distinct reasoning and response
phase diagnostics. The response phase generated normal answer continuations
for generation and exactly one answer token for probability scoring. Hybrid
and strict results matched under the same enabled configuration:

| Task | Hybrid result | Strict result |
|---|---:|---:|
| `darumeru/flores_ru_en` | ROUGE-L 0.615925 | ROUGE-L 0.615925 |
| `russiannlp/rucola_custom` | accuracy 1.0, MCC 1.0 | accuracy 1.0, MCC 1.0 |

The dependency-free suite additionally verifies phase-local stop ids, insertion
of `THINK_CLOSE_MARKER` after a length-limited reasoning phase, restoration of
the base sampling config, execution-aware context budgeting, and rejection of
multiple reasoning sequences. Together with the real two-pass artifacts this
is the evidence for the new reasoning implementation; the legacy reasoning
facade is not used as an oracle.

## Legacy vLLM comparison

The nested checkout `llmtf_open/` was verified at commit
`504bd7fc2793c900e97010f124651460afcd4812`. Four non-reasoning cells completed
with 8 aligned samples each:

| Cell | Legacy | Refactor | Alignment summary |
|---|---:|---:|---|
| instruct, thinking off, generate | ROUGE-L 0.563501 | ROUGE-L 0.554646 | same inputs; refactor preserves one trailing prefill space |
| instruct, thinking off, probability | accuracy 0.125, MCC 0.0 | accuracy 0.875, MCC 0.654654 | same inputs; legacy candidate-id collision |
| Base generate | ROUGE-L 0.517617 | ROUGE-L 0.523431 | prompts identical; 7/8 predictions identical |
| Base probability | accuracy 0.625, MCC -0.218218 | identical | prompts, predictions, sample metrics, and total all identical |

The instruct prompt difference is systematic and intentional: legacy removes
one trailing ASCII space from every assistant prefill, while the refactor exact
continuation contract preserves it. Legacy probability scoring also takes the
first id of a multi-token alternate surface; for this tokenizer both semantic
classes then receive the same score. The refactor keeps only genuine
single-token surface variants and aggregates them by the documented maximum.

Both legacy thinking-enabled cells failed before inference with
`max_tokens must be at least 1, got 0`. The old CLI uses `--max_prompt_len` both
as the engine context length and as the evaluator prompt budget; after reserving
the answer and reasoning budgets, the reasoning phase is reduced to zero. Per
the project decision, the historically unused legacy reasoning path was not
patched or rerun. Its failure does not invalidate the four aligned
non-reasoning controls or serve as evidence against the independently tested
refactor reasoning implementation.

Two runtime-compatibility edits were required only inside the disposable nested
legacy checkout:

1. accept an already rendered prompt string in the legacy image-sanitizing
   logger;
2. explicitly shut down the vLLM 0.21 engine so the CLI process exits.

The first affects result logging only; the second affects process teardown
only. Neither changes prompts, inference, probabilities, or metrics. The
disposable nested checkout was removed manually before the v0.3.0 release
commit.

## Remaining limitations

- Base API generation lacks the foundational stop strings used by local
  backends: only 3/8 exact predictions matched and several API responses ran
  close to the token ceiling. This is tracked in `BACKLOG.md`.
- Instruct API continuation with a whitespace-ended prefill cannot claim exact
  parity on vLLM 0.21; portable task prompts or a server implementation that
  preserves the suffix are still needed.
- A/A then A/B study of different vLLM engine arguments is explicitly
  low-priority and does not block the refactor merge. It is needed only before
  introducing strict cross-runtime numeric probability tolerances.
