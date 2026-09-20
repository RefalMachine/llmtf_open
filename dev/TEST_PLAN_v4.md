# Validation plan v4

> Reproduction plan for the v4 development snapshot, not user documentation.

This plan separates framework validation from the legacy comparison. Do not
reuse output directories between cells. The fixed seed is 555 in `Evaluator`;
all commands below use deterministic generation, zero few-shot examples and at
most eight evaluation samples.

Current execution status is recorded in
[`VALIDATION_STATUS_v4.md`](VALIDATION_STATUS_v4.md). Keep that file factual:
imports and unit tests are not model-runtime validation.

The sandbox shell cannot access the NVIDIA driver in this workspace. This is
not a blocker: Docker GPU passthrough was verified on 2026-09-20 with one RTX
4090. All GPU decisions and tests must use `docker run --gpus all`.

## 1. Paths and immutable inputs

Run from a shell on the GPU host:

```bash
export REFACTOR_SRC=/var/mtikhomi/work_folder/projects/devel/llmtf_open
export LEGACY_SRC=/var/mtikhomi/work_folder/projects/devel/llmtf_legacy_v4
export RESULT_ROOT=/var/mtikhomi/work_folder/projects/devel/llmtf_validation_v4
export HF_CACHE_DIR="${HF_HOME:-$HOME/.cache/huggingface}"
export INSTRUCT_MODEL=Qwen/Qwen3.5-2B
export BASE_MODEL=Qwen/Qwen3.5-2B-Base
export MODEL_CONTEXT_LEN=8192
export MAX_REASONING=2048
export MIN_REASONING=512
export HF_IMAGE=llmtf:hf-cu129
export VLLM_IMAGE=llmtf:vllm-cu129

mkdir -p "$RESULT_ROOT" "$HF_CACHE_DIR"
git -C "$REFACTOR_SRC" rev-parse HEAD
git -C "$REFACTOR_SRC" status --short
git -C "$REFACTOR_SRC" worktree add --detach "$LEGACY_SRC" \
  504bd7fc2793c900e97010f124651460afcd4812
git -C "$LEGACY_SRC" rev-parse HEAD
```

If the legacy worktree already exists, omit only the `git worktree add` line and
verify that the printed legacy commit is exactly the requested commit. Both
trees must use the same mounted cache. Do not update either model between runs.

Resolve the explicit close-token id from the same tokenizer that will be used
for evaluation, without embedding the model token text in commands:

```bash
export END_THINKING_TOKEN_ID="$(docker run --rm \
  -v "$REFACTOR_SRC:/workdir:ro" \
  -v "$HF_CACHE_DIR:/root/.cache/huggingface" \
  "$HF_IMAGE" python -c \
  'from transformers import AutoTokenizer; from llmtf.reasoning import THINK_CLOSE_MARKER; t=AutoTokenizer.from_pretrained("Qwen/Qwen3.5-2B"); ids=t.encode(THINK_CLOSE_MARKER, add_special_tokens=False); assert len(ids)==1, ids; print(ids[0])')"
test -n "$END_THINKING_TOKEN_ID"
```

Record image ids and runtime versions before model tests:

```bash
docker image inspect "$HF_IMAGE" --format '{{.Id}}'
docker image inspect "$VLLM_IMAGE" --format '{{.Id}}'
docker run --rm --gpus all "$VLLM_IMAGE" python -c \
  'import torch, transformers, vllm; print(torch.__version__, torch.version.cuda, transformers.__version__, vllm.__version__); print(torch.cuda.get_device_name(0))'
```

This container command is authoritative for the current workspace. Do not
replace it with host/sandbox `nvidia-smi`.

## 2. Dependency-free and profile gate

```bash
cd "$REFACTOR_SRC"
python3 tests/test_refactor_logic.py
python3 -m unittest discover -s tests -p 'test_*.py' -v
python3 -m compileall -q llmtf evaluate_model.py evaluate_model_api.py benchmark
git diff --check
bash docker/validate_profiles.sh
```

The profile script does not prove model compatibility. Before the full matrix,
run one one-sample HF generation and one one-sample local-vLLM generation. If
either checkpoint fails to load, stop: determine a working official immutable
Transformers/vLLM build and record its wheel/commit and hash before continuing.

Exact one-sample generate and probability smoke commands:

```bash
run_refactor_smoke() {
  image=$1
  backend_args=$2
  backend_name=$3
  docker run --rm --gpus all --ipc=host \
    -e CUDA_VISIBLE_DEVICES=0 \
    -v "$REFACTOR_SRC:/workdir" \
    -v "$HF_CACHE_DIR:/root/.cache/huggingface" \
    -v "$RESULT_ROOT:/results" \
    "$image" bash -c "
      set -euo pipefail
      common='--few_shot_count 0 --batch_size 1 --max_sample_per_dataset 1 --model_context_len $MODEL_CONTEXT_LEN --temperature 0 --force_recalc'
      python evaluate_model.py --model_name_or_path '$INSTRUCT_MODEL' --conv_path auto --output_dir /results/refactor/smoke/$backend_name/generate --dataset_names darumeru/flores_ru_en --model_kind hybrid --disable_thinking \$common $backend_args
      python evaluate_model.py --model_name_or_path '$INSTRUCT_MODEL' --conv_path auto --output_dir /results/refactor/smoke/$backend_name/probability --dataset_names russiannlp/rucola_custom --model_kind hybrid --disable_thinking \$common $backend_args
    "
}

run_refactor_smoke "$HF_IMAGE" '' hf
run_refactor_smoke "$VLLM_IMAGE" '--vllm' vllm
```

Inspect both smoke output directories before starting §3. Each requested task
must have `_params.jsonl` and `_total.jsonl`; a zero subprocess exit alone is
not sufficient.

### Assistant-prefill gate

Local matrix commands use the default `--assistant_prefill_policy auto`, which
preserves the final assistant message exactly. API parity additionally requires
a profile that declares continuation. For a vLLM API task with a
whitespace-terminated prefill, add:

```bash
--api_profile vllm --assistant_prefill_policy auto --probe_api_prefill
```

If the probe reports `verified_stripped`, the server does not satisfy exact
prefill for that task. Record the failed command and do not claim API/local
parity. The cell may be rerun only as an explicitly non-parity diagnostic with
`--assistant_prefill_policy best_effort`, or with a reviewed portable task
prompt that has no assistant-prefill at all. A portable rewrite is a different
experiment identity. See `docs/assistant_continuation.md`.

## 3. Refactor local matrix

The generate task is `darumeru/flores_ru_en`; the token-probability task is
`russiannlp/rucola_custom`. Run the following block once with `BACKEND_ARGS`
empty in the HF image and once with `BACKEND_ARGS=--vllm` in the vLLM image.

```bash
run_refactor_local_matrix() {
  image=$1
  backend_args=$2
  backend_name=$3
  docker run --rm --gpus all --ipc=host \
    -e CUDA_VISIBLE_DEVICES=0 \
    -v "$REFACTOR_SRC:/workdir" \
    -v "$HF_CACHE_DIR:/root/.cache/huggingface" \
    -v "$RESULT_ROOT:/results" \
    "$image" bash -c "
      set -euo pipefail
      common='--few_shot_count 0 --batch_size 1 --max_sample_per_dataset 8 --model_context_len $MODEL_CONTEXT_LEN --temperature 0 --force_recalc'
      for task in darumeru/flores_ru_en russiannlp/rucola_custom; do
        python evaluate_model.py --model_name_or_path '$INSTRUCT_MODEL' --conv_path auto --output_dir /results/refactor/$backend_name/hybrid_off/\${task//\//_} --dataset_names \"\$task\" --model_kind hybrid --disable_thinking \$common $backend_args
        python evaluate_model.py --model_name_or_path '$INSTRUCT_MODEL' --conv_path auto --output_dir /results/refactor/$backend_name/hybrid_on/\${task//\//_} --dataset_names \"\$task\" --model_kind hybrid --enable_thinking --end_thinking_token_id '$END_THINKING_TOKEN_ID' --max_new_tokens_reasoning '$MAX_REASONING' --min_new_tokens_reasoning '$MIN_REASONING' \$common $backend_args
        python evaluate_model.py --model_name_or_path '$INSTRUCT_MODEL' --conv_path auto --output_dir /results/refactor/$backend_name/strict/\${task//\//_} --dataset_names \"\$task\" --model_kind reasoning --enable_thinking --end_thinking_token_id '$END_THINKING_TOKEN_ID' --max_new_tokens_reasoning '$MAX_REASONING' --min_new_tokens_reasoning '$MIN_REASONING' \$common $backend_args
      done
      for task in darumeru/flores_ru_en russiannlp/rucola_custom; do
        python evaluate_model.py --model_name_or_path '$BASE_MODEL' --conv_path conversation_configs/default_foundational.json --output_dir /results/refactor/$backend_name/base/\${task//\//_} --dataset_names \"\$task\" --model_kind plain --disable_thinking --is_foundational \$common $backend_args
      done
    "
}

run_refactor_local_matrix "$HF_IMAGE" '' hf
run_refactor_local_matrix "$VLLM_IMAGE" '--vllm' vllm
```

HF-only PPL cells:

```bash
docker run --rm --gpus all --ipc=host -e CUDA_VISIBLE_DEVICES=0 \
  -v "$REFACTOR_SRC:/workdir" -v "$HF_CACHE_DIR:/root/.cache/huggingface" \
  -v "$RESULT_ROOT:/results" "$HF_IMAGE" bash -c "
  set -euo pipefail
  python evaluate_model.py --model_name_or_path '$INSTRUCT_MODEL' --conv_path auto --output_dir /results/refactor/hf/hybrid_ppl --dataset_names russiannlp/rucola_custom --model_kind hybrid --disable_thinking --ppl_scoring --few_shot_count 0 --batch_size 1 --max_sample_per_dataset 8 --model_context_len '$MODEL_CONTEXT_LEN' --temperature 0 --force_recalc
  python evaluate_model.py --model_name_or_path '$BASE_MODEL' --conv_path conversation_configs/default_foundational.json --output_dir /results/refactor/hf/base_ppl --dataset_names russiannlp/rucola_custom --model_kind plain --disable_thinking --is_foundational --ppl_scoring --few_shot_count 0 --batch_size 1 --max_sample_per_dataset 8 --model_context_len '$MODEL_CONTEXT_LEN' --temperature 0 --force_recalc
"
```

For vLLM and API, run the same PPL-shaped command with `--ppl_scoring` and
require a non-zero exit plus an explicit HF-only capability message. No
`_total.jsonl` may be created.

## 4. API matrix

Start two separate server sessions: instruct with the tokenizer's native chat
template, then Base with
`conversation_configs/default_foundational.json` converted/passed as the server
chat template. Use text-only limits and `--max-model-len 8192`. Record the exact
server command because vLLM server flags may differ between the validated pinned
version and the draft image.

In the client shell export endpoint metadata without printing the key:

```bash
export OPENAI_BASE_URL=http://127.0.0.1:8000
export OPENAI_MODEL_NAME="$INSTRUCT_MODEL"
test -n "${OPENAI_API_KEY:-}"
```

Then run generate/probability for hybrid off, hybrid on and strict reasoning by
reusing the two task names from §3 with `evaluate_model_api.py`. The exact
per-cell substitution is:

```bash
for task in darumeru/flores_ru_en russiannlp/rucola_custom; do
  python evaluate_model_api.py --base_url "$OPENAI_BASE_URL" --api_profile vllm --model_name_or_path "$OPENAI_MODEL_NAME" --output_dir "$RESULT_ROOT/refactor/api/hybrid_off/${task//\//_}" --dataset_names "$task" --model_kind hybrid --disable_thinking --few_shot_count 0 --batch_size 8 --max_sample_per_dataset 8 --model_context_len "$MODEL_CONTEXT_LEN" --temperature 0 --assistant_prefill_policy auto --probe_api_prefill --force_recalc
  python evaluate_model_api.py --base_url "$OPENAI_BASE_URL" --api_profile vllm --model_name_or_path "$OPENAI_MODEL_NAME" --output_dir "$RESULT_ROOT/refactor/api/hybrid_on/${task//\//_}" --dataset_names "$task" --model_kind hybrid --enable_thinking --end_thinking_token_id "$END_THINKING_TOKEN_ID" --max_new_tokens_reasoning "$MAX_REASONING" --min_new_tokens_reasoning "$MIN_REASONING" --few_shot_count 0 --batch_size 8 --max_sample_per_dataset 8 --model_context_len "$MODEL_CONTEXT_LEN" --temperature 0 --assistant_prefill_policy auto --probe_api_prefill --force_recalc
  python evaluate_model_api.py --base_url "$OPENAI_BASE_URL" --api_profile vllm --model_name_or_path "$OPENAI_MODEL_NAME" --output_dir "$RESULT_ROOT/refactor/api/strict/${task//\//_}" --dataset_names "$task" --model_kind reasoning --enable_thinking --end_thinking_token_id "$END_THINKING_TOKEN_ID" --max_new_tokens_reasoning "$MAX_REASONING" --min_new_tokens_reasoning "$MIN_REASONING" --few_shot_count 0 --batch_size 8 --max_sample_per_dataset 8 --model_context_len "$MODEL_CONTEXT_LEN" --temperature 0 --assistant_prefill_policy auto --probe_api_prefill --force_recalc
done
```

Thinking-enabled and strict reasoning API cells also create a
whitespace-ended assistant continuation between phases. They require the same
probe gate even when the original task has no assistant prefill.

Restart the server for the Base checkpoint/template, set
`OPENAI_MODEL_NAME="$BASE_MODEL"`, and run the same two tasks with
`--model_kind plain --disable_thinking --is_foundational` into
`$RESULT_ROOT/refactor/api/base/...`.

## 5. Legacy commands for the user

These are the baseline runs that may be delegated. Run them in the same HF and
vLLM images and with the same cache as §3. The old CLI has no deployment-context
argument; its `--max_prompt_len` is a prompt budget and, for legacy vLLM, also
affects engine capture length. The closest task-specific mapping is:

| Mode | generate (`flores`) | probability (`rucola`) |
|---|---:|---:|
| thinking off | 7680 | 8191 |
| thinking on, 2048 reasoning | 5632 | 6143 |

Most eight-sample prompts are shorter than all four ceilings, so this mapping
normally preserves the evaluated inputs. Record actual prompt lengths and flag
the legacy-vLLM dual-use of this argument as a confounder.

Observed on 2026-09-20: the two legacy vLLM thinking-enabled commands below
reduce the reasoning phase to zero and fail with a vLLM validation error. The
legacy CLI uses `--max_prompt_len` as both engine context length and evaluator
prompt budget, so the proposed mapping cannot represent an 8192-token engine
context plus a smaller task prompt budget. This historically unused reasoning
path is retained below only as an exact reproduction command; it is not a
required parity oracle and should not be patched merely to validate the
refactor. The four thinking-off/Base controls remain valid. See
`TEST_REPORT_v4.md`.

Run this block for legacy HF, then legacy vLLM:

```bash
run_legacy_local_matrix() {
  image=$1
  backend_args=$2
  backend_name=$3
  docker run --rm --gpus all --ipc=host \
    -e CUDA_VISIBLE_DEVICES=0 \
    -v "$LEGACY_SRC:/workdir" \
    -v "$HF_CACHE_DIR:/root/.cache/huggingface" \
    -v "$RESULT_ROOT:/results" \
    "$image" bash -c "
      set -euo pipefail
      common='--few_shot_count 0 --batch_size 1 --max_sample_per_dataset 8 --temperature 0 --force_recalc'
      python evaluate_model.py --model_name_or_path '$INSTRUCT_MODEL' --conv_path auto --output_dir /results/legacy/$backend_name/hybrid_off_generate --dataset_names darumeru/flores_ru_en --disable_thinking --max_prompt_len 7680 \$common $backend_args
      python evaluate_model.py --model_name_or_path '$INSTRUCT_MODEL' --conv_path auto --output_dir /results/legacy/$backend_name/hybrid_off_proba --dataset_names russiannlp/rucola_custom --disable_thinking --max_prompt_len 8191 \$common $backend_args
      python evaluate_model.py --model_name_or_path '$INSTRUCT_MODEL' --conv_path auto --output_dir /results/legacy/$backend_name/hybrid_on_generate --dataset_names darumeru/flores_ru_en --max_prompt_len 5632 --max_new_tokens_reasoning '$MAX_REASONING' \$common $backend_args
      python evaluate_model.py --model_name_or_path '$INSTRUCT_MODEL' --conv_path auto --output_dir /results/legacy/$backend_name/hybrid_on_proba --dataset_names russiannlp/rucola_custom --max_prompt_len 6143 --max_new_tokens_reasoning '$MAX_REASONING' \$common $backend_args
      python evaluate_model.py --model_name_or_path '$BASE_MODEL' --conv_path conversation_configs/default_foundational.json --output_dir /results/legacy/$backend_name/base_generate --dataset_names darumeru/flores_ru_en --disable_thinking --is_foundational --max_prompt_len 7680 \$common $backend_args
      python evaluate_model.py --model_name_or_path '$BASE_MODEL' --conv_path conversation_configs/default_foundational.json --output_dir /results/legacy/$backend_name/base_proba --dataset_names russiannlp/rucola_custom --disable_thinking --is_foundational --max_prompt_len 8191 \$common $backend_args
    "
}

run_legacy_local_matrix "$HF_IMAGE" '' hf
run_legacy_local_matrix "$VLLM_IMAGE" '--vllm' vllm
```

Legacy HF PPL:

```bash
docker run --rm --gpus all --ipc=host -e CUDA_VISIBLE_DEVICES=0 \
  -v "$LEGACY_SRC:/workdir" -v "$HF_CACHE_DIR:/root/.cache/huggingface" \
  -v "$RESULT_ROOT:/results" "$HF_IMAGE" bash -c "
  set -euo pipefail
  common='--few_shot_count 0 --batch_size 1 --max_sample_per_dataset 8 --temperature 0 --force_recalc --ppl_scoring --disable_thinking'
  python evaluate_model.py --model_name_or_path '$INSTRUCT_MODEL' --conv_path auto --output_dir /results/legacy/hf/hybrid_ppl --dataset_names russiannlp/rucola_custom --max_prompt_len 8191 \$common
  python evaluate_model.py --model_name_or_path '$BASE_MODEL' --conv_path conversation_configs/default_foundational.json --output_dir /results/legacy/hf/base_ppl --dataset_names russiannlp/rucola_custom --is_foundational --max_prompt_len 8191 \$common
"
```

For legacy API comparison against the same instruct server, run four cells:

```bash
for task in darumeru/flores_ru_en russiannlp/rucola_custom; do
  prompt_len=8191; test "$task" = darumeru/flores_ru_en && prompt_len=7680
  python "$LEGACY_SRC/evaluate_model_api.py" --base_url "$OPENAI_BASE_URL" --model_name_or_path "$OPENAI_MODEL_NAME" --api_key EMPTY --output_dir "$RESULT_ROOT/legacy/api/hybrid_off/${task//\//_}" --dataset_names "$task" --disable_thinking --few_shot_count 0 --batch_size 8 --max_sample_per_dataset 8 --max_prompt_len "$prompt_len" --temperature 0 --force_recalc
  prompt_len=6143; test "$task" = darumeru/flores_ru_en && prompt_len=5632
  python "$LEGACY_SRC/evaluate_model_api.py" --base_url "$OPENAI_BASE_URL" --model_name_or_path "$OPENAI_MODEL_NAME" --api_key EMPTY --output_dir "$RESULT_ROOT/legacy/api/hybrid_on/${task//\//_}" --dataset_names "$task" --end_thinking_token_id "$END_THINKING_TOKEN_ID" --max_new_tokens_reasoning "$MAX_REASONING" --few_shot_count 0 --batch_size 8 --max_sample_per_dataset 8 --max_prompt_len "$prompt_len" --temperature 0 --force_recalc
done
```

The legacy API CLI exposes neither foundational mode nor PPL. Mark those legacy
cells `unsupported by legacy entry point`; do not emulate them with a different
method. Also note that the legacy API CLI puts the key in the process arguments;
use only the non-secret local-server placeholder for this baseline. Do not use
that command form for a real credential.

## 6. Failure tests

The dependency-free suites cover phase-local stop ids, malformed/unknown
backend kwargs, contradictory YAML, context-budget restoration, partial API
batch failure, tokenize failure, multiple reasoning sequences and cache
fingerprint mismatch. On the real API server additionally:

1. stop one request in an eight-request batch and require non-zero exit;
2. disable `/tokenize`, run one `ruifeval` sample ending in `user`, and require
   successful generation with `server_capabilities.tokenize=false`; separately
   require a clear compatibility failure for a whitespace-prefill task using
   `--probe_api_prefill`;
3. pass a context length smaller than a task answer budget and require non-zero exit;
4. rerun into an existing output directory with a changed thinking mode and
   require cache mismatch unless `--force_recalc` is supplied.

No failed task may create a new `_total.jsonl`.

## 7. Artifact comparison

For every successful cell retain stdout/stderr, `evaluation_log.txt`, the sample
JSON array, `_params.jsonl` and `_total.jsonl`. Compare in this order:

1. sample count and dataset sample ids;
2. rendered prompts/token counts;
3. per-sample predictions/probabilities;
4. aggregate metrics.

Do not compare totals if alignment differs. Require exact deterministic HF text
when runtime and inputs match. Report absolute/relative probability deltas for
floating results and set any tolerance only after inspecting the first control
run. Keep the legacy worktree and raw artifacts until the comparison report is
accepted.
