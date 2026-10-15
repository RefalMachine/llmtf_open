#!/usr/bin/env bash
# Full preparation and dataset construction; launch from any directory.
set +x
set -euo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$project_dir"

# This is the local proxy used by the successful API pilot.
# Override it with RULAW_PROXY=http://host:port when needed.
export HTTPS_PROXY="${RULAW_PROXY:-http://127.0.0.1:8118}"
export HTTP_PROXY="$HTTPS_PROXY"
export https_proxy="$HTTPS_PROXY"
export http_proxy="$HTTP_PROXY"

# Credentials are inherited or entered privately, never stored in this file.
if [[ -n "${RULAW_API_BASE:-}" ]]; then
  # The known adapter on 8080 implements a restricted subscription protocol.
  case "${RULAW_API_BASE%/}" in
    http://127.0.0.1:8080/v1|http://localhost:8080/v1)
      export RULAW_API_PROFILE="${RULAW_API_PROFILE:-subscription}"
      ;;
  esac
  if [[ -z "${RULAW_API_KEY:-}" ]]; then
    read -rs -p 'Токен вашей API-прослойки: ' RULAW_API_KEY
    printf '\n'
  fi
  if [[ -z "$RULAW_API_KEY" ]]; then
    printf 'Не задан RULAW_API_KEY.\n' >&2
    exit 1
  fi
  export RULAW_API_KEY
  printf 'API: %s (успешные ответы сохраняются в кеше).\n' "$RULAW_API_BASE"
  python3 -m dev.rulaw_proofbench.builder check-api \
    --config dev/rulaw_proofbench/configs/release.json
else
  if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
    read -rs -p 'OpenRouter API key: ' OPENROUTER_API_KEY
    printf '\n'
  fi
  if [[ -z "$OPENROUTER_API_KEY" ]]; then
    printf 'Не задан OPENROUTER_API_KEY.\n' >&2
    exit 1
  fi
  export OPENROUTER_API_KEY
fi

# v1 contains a recorded HTTP 403; retain it and use fresh output directories.
prepare_dir="${RULAW_PREPARE_DIR:-dev/rulaw_artifacts/prepared-release-v2}"
build_dir="${RULAW_BUILD_DIR:-dev/rulaw_artifacts/runs/full-300-v2}"
workers="${RULAW_WORKERS:-4}"
retry_args=()
if [[ "${1:-}" == "--retry-failed" ]]; then
  retry_args+=(--retry-failed)
  shift
fi
if (( $# )); then
  printf 'Использование: bash run_rulaw_proofbench.sh [--retry-failed]\n' >&2
  exit 1
fi

if [[ -f "$prepare_dir/input-package.json" ]]; then
  printf 'Этап 1/2 уже завершён: используем подготовленный пакет.\n'
  python3 -m dev.rulaw_proofbench.builder check-package \
    --inputs "$prepare_dir/input-package.json"
else
  printf 'Этап 1/2: подготовка входов → %s\n' "$prepare_dir"
  python3 -m dev.rulaw_proofbench.builder prepare \
    --config dev/rulaw_proofbench/configs/prepare-release.json \
    --output "$prepare_dir" \
    --workers "$workers" "${retry_args[@]}"
fi

printf 'Этап 2/2: построение датасета → %s\n' "$build_dir"
exec bash dev/rulaw_proofbench/build.sh \
  --config dev/rulaw_proofbench/configs/release.json \
  --inputs "$prepare_dir/input-package.json" \
  --output "$build_dir" \
  --workers "$workers" "${retry_args[@]}"
