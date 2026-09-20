#!/usr/bin/env bash
set -euo pipefail

usage() {
    sed -n '8,41p' "$0" | sed 's/^# \{0,1\}//'
}

# Run candidate generation and judging for every row in a model manifest.
#
# Usage:
#   JUDGE_BASE_URL=https://judge.example/v1 \
#   JUDGE_MODEL=/models/judge \
#   JUDGE_API_KEY=... \
#   benchmark/llmaaj/run_full.sh models.tsv
#
# models.tsv contains whitespace-separated fields without spaces in values:
#   MODEL_PATH  RESULT_NAME  TENSOR_PARALLEL_SIZE
#
# Required environment:
#   JUDGE_BASE_URL   OpenAI-compatible judge endpoint
#   JUDGE_MODEL      model id exposed by the judge endpoint
#   JUDGE_API_KEY    judge credential; never forwarded on the command line
#
# Optional environment:
#   JUDGE_NAME=judge
#   BENCHMARK_NAME=ru_arena-hard-v0.1
#   VLLM_IMAGE=llmtf:vllm-cu129
#   HOST_PORT=7060
#   CONTAINER_PORT=8000
#   MAX_MODEL_LEN=32000
#   GPU_MEMORY_UTILIZATION=0.92
#   CANDIDATE_MAX_NEW_TOKENS=4096
#   JUDGE_MAX_LEN=16384
#   MODEL_VOLUME=/host/models:/models:ro
#   HF_CACHE_VOLUME=/host/hf-cache:/root/.cache/huggingface
#   VLLM_EXTRA_ARGS_FILE=/path/to/args.txt
#   JUDGE_FORCE_RECALC=0
#
# VLLM_EXTRA_ARGS_FILE contains exactly one argument per line. This is the
# explicit place for model-specific options such as parser names; the script
# never guesses them.

if [[ ${1:-} == "-h" || ${1:-} == "--help" ]]; then
    usage
    exit 0
fi
if [[ $# -ne 1 ]]; then
    usage >&2
    exit 2
fi

models_file=$1
if [[ ! -f "$models_file" ]]; then
    echo "Model manifest not found: $models_file" >&2
    exit 2
fi

: "${JUDGE_BASE_URL:?JUDGE_BASE_URL is required}"
: "${JUDGE_MODEL:?JUDGE_MODEL is required}"
: "${JUDGE_API_KEY:?JUDGE_API_KEY is required}"

judge_name=${JUDGE_NAME:-judge}
benchmark_name=${BENCHMARK_NAME:-ru_arena-hard-v0.1}
vllm_image=${VLLM_IMAGE:-llmtf:vllm-cu129}
host_port=${HOST_PORT:-7060}
container_port=${CONTAINER_PORT:-8000}
max_model_len=${MAX_MODEL_LEN:-32000}
gpu_memory_utilization=${GPU_MEMORY_UTILIZATION:-0.92}
judge_force_recalc=${JUDGE_FORCE_RECALC:-0}
candidate_max_new_tokens=${CANDIDATE_MAX_NEW_TOKENS:-4096}
judge_max_len=${JUDGE_MAX_LEN:-16384}

active_container=""
cleanup() {
    if [[ -n "$active_container" ]]; then
        docker stop "$active_container" >/dev/null 2>&1 || true
        active_container=""
    fi
}
trap cleanup EXIT INT TERM

wait_for_server() {
    local health_url=$1
    local attempts=${HEALTH_CHECK_ATTEMPTS:-120}
    local interval=${HEALTH_CHECK_INTERVAL_SECONDS:-5}
    local attempt
    for ((attempt = 1; attempt <= attempts; attempt++)); do
        if curl -fsS "$health_url" >/dev/null 2>&1; then
            return 0
        fi
        sleep "$interval"
    done
    echo "vLLM server did not become ready: $health_url" >&2
    return 1
}

read_vllm_extra_args() {
    vllm_extra_args=()
    if [[ -z ${VLLM_EXTRA_ARGS_FILE:-} ]]; then
        return
    fi
    if [[ ! -f "$VLLM_EXTRA_ARGS_FILE" ]]; then
        echo "VLLM extra args file not found: $VLLM_EXTRA_ARGS_FILE" >&2
        exit 2
    fi
    while IFS= read -r arg || [[ -n "$arg" ]]; do
        [[ -z "$arg" || "$arg" == \#* ]] && continue
        vllm_extra_args+=("$arg")
    done < "$VLLM_EXTRA_ARGS_FILE"
}

start_candidate_server() {
    local model_path=$1
    local tp_size=$2
    local container_name="llmtf-candidate-$$"
    local docker_args=(
        run -d --rm
        --gpus all
        --ipc=host
        --name "$container_name"
        -p "127.0.0.1:${host_port}:${container_port}"
    )
    if [[ -n ${MODEL_VOLUME:-} ]]; then
        docker_args+=(-v "$MODEL_VOLUME")
    fi
    if [[ -n ${HF_CACHE_VOLUME:-} ]]; then
        docker_args+=(-v "$HF_CACHE_VOLUME")
    fi
    docker_args+=(
        "$vllm_image"
        python -m vllm.entrypoints.openai.api_server
        --model "$model_path"
        --port "$container_port"
        --tensor-parallel-size "$tp_size"
        --max-model-len "$max_model_len"
        --gpu-memory-utilization "$gpu_memory_utilization"
        --language-model-only
    )
    docker_args+=("${vllm_extra_args[@]}")
    active_container=$(docker "${docker_args[@]}")
    wait_for_server "http://127.0.0.1:${host_port}/health"
}

run_candidate() {
    local model_path=$1
    local result_name=$2
    OPENAI_API_KEY=EMPTY python3 -m benchmark.llmaaj.generate_llmaaj \
        --base_url "http://127.0.0.1:${host_port}" \
        --model_name_or_path "$model_path" \
        --model_name "$result_name" \
        --max_new_tokens "$candidate_max_new_tokens" \
        --benchmark_name "$benchmark_name"
}

run_judge() {
    local result_name=$1
    local force_args=()
    if [[ "$judge_force_recalc" == "1" ]]; then
        force_args+=(--force_recalc)
    fi
    OPENAI_API_KEY="$JUDGE_API_KEY" python3 -m benchmark.llmaaj.judge_llmaaj \
        --judge_base_url "$JUDGE_BASE_URL" \
        --judge_model_name_or_path "$JUDGE_MODEL" \
        --judge_model_name "$judge_name" \
        --model_name "$result_name" \
        --max_len "$judge_max_len" \
        --benchmark_name "$benchmark_name" \
        "${force_args[@]}"
}

mkdir -p "benchmark/llmaaj/${benchmark_name}/model_results"
read_vllm_extra_args

while read -r model_path result_name tp_size extra; do
    [[ -z "$model_path" || "$model_path" == \#* ]] && continue
    if [[ -z "$result_name" || -z "$tp_size" || -n ${extra:-} ]]; then
        echo "Invalid model manifest row: expected MODEL_PATH RESULT_NAME TP_SIZE" >&2
        exit 2
    fi
    echo "Evaluating ${result_name} (${model_path}, TP=${tp_size})"
    start_candidate_server "$model_path" "$tp_size"
    run_candidate "$model_path" "$result_name"
    cleanup
    run_judge "$result_name"
done < "$models_file"
