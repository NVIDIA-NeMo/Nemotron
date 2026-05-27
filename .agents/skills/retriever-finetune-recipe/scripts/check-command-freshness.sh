#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
cd "$ROOT"

log="${TMPDIR:-/tmp}/retriever-skill-command-check.out"
commands=(
  "uv run --no-sync nemotron embed --help"
  "uv run --no-sync nemotron embed run -c default -d --from sdg --to prep"
  "uv run --no-sync nemotron embed run -c default -d --from prep --to eval"
  "uv run --no-sync nemotron embed finetune -c default -d"
  "uv run --no-sync nemotron rerank --help"
  "uv run --no-sync nemotron rerank run -c default -d --from prep --to eval"
  "uv run --no-sync nemotron rerank finetune -c default -d"
  "uv run --no-sync nemotron rerank eval -c default -d eval_nim=true eval_base=false"
)

for cmd in "${commands[@]}"; do
  printf 'checking: %s\n' "$cmd"
  if ! bash -lc "$cmd" >"$log" 2>&1; then
    cat "$log"
    exit 1
  fi
done

echo "all documented command freshness checks passed"
