#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
export UNIFIED_REFUSAL_OUTPUT_DIR="${UNIFIED_REFUSAL_OUTPUT_DIR:-$ROOT/outputs}"

if [[ -z "${UNIFIED_REFUSAL_DATASET:-}" ]]; then
  echo "ERROR: set UNIFIED_REFUSAL_DATASET=/path/to/local_quadruples.jsonl" >&2
  exit 1
fi

TARGET="${1:-Meta-Llama-3-8B-Instruct}"
MAX_PAIRS="${2:-}"

run_for_model() {
  local model="$1"
  local max_arg=()
  if [[ -n "$MAX_PAIRS" ]]; then
    max_arg=(--max-pairs "$MAX_PAIRS")
  fi

  python "$ROOT/src/extract_hidden_states.py" --model "$model" "${max_arg[@]}"
  python "$ROOT/src/analyze.py" --model "$model"
  python "$ROOT/src/cross_probe.py" --model "$model"
  python "$ROOT/src/decompose.py" --model "$model"
  python "$ROOT/src/logit_lens.py" --model "$model"
}

if [[ "$TARGET" == "--all" ]]; then
  mapfile -t MODELS < <(python - <<'PY'
from config import MODELS
for name in MODELS:
    print(name)
PY
)
  for model in "${MODELS[@]}"; do
    run_for_model "$model"
  done
else
  run_for_model "$TARGET"
fi

