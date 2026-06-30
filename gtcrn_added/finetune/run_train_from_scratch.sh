#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

STAMP="$(date +%Y%m%d-%H%M%S)"
OUTPUT_ROOT="${OUTPUT_ROOT:-experiments/gtcrn_dns3_paired_scratch}"
RUN_DIR="${RUN_DIR:-${OUTPUT_ROOT}/${STAMP}}"
LOG_FILE="${LOG_FILE:-${RUN_DIR}/train.log}"
CONFIG_PATH="${CONFIG_PATH:-finetune/config.yaml}"
EPOCHS="${EPOCHS:-100}"

mkdir -p "$RUN_DIR"

echo "run_dir=${RUN_DIR}"
echo "log_file=${LOG_FILE}"
echo "epochs=${EPOCHS}"
echo "config=${CONFIG_PATH}"
echo "mode=from_scratch"

exec conda run --no-capture-output -n vinp310 \
  python -m finetune.train \
  --config "$CONFIG_PATH" \
  --from-scratch \
  --epochs "$EPOCHS" \
  --output-dir "$RUN_DIR" \
  --save-every-epoch \
  "$@" 2>&1 | tee "$LOG_FILE"
