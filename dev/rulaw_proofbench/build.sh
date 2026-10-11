#!/usr/bin/env bash
# Run from any directory. The API key is inherited, never written to disk.
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$project_dir"
exec python3 -m dev.rulaw_proofbench.builder build "$@"
