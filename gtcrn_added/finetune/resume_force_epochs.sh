#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 <run_dir> [target_epochs]" >&2
  echo "example: $0 experiments/gtcrn_dns3_paired_scratch/20260625-144014 50" >&2
  exit 2
fi

RUN_DIR="$1"
TARGET_EPOCHS="${2:-50}"
shift
if [[ $# -gt 0 ]]; then
  shift
fi
CONFIG_PATH="${CONFIG_PATH:-finetune/config.yaml}"
LOG_FILE="${LOG_FILE:-${RUN_DIR}/train_resume_to_${TARGET_EPOCHS}.log}"
RESUME_CKPT="${RUN_DIR}/checkpoints/latest.tar"

if [[ ! -f "$RESUME_CKPT" ]]; then
  echo "missing checkpoint: $RESUME_CKPT" >&2
  exit 1
fi

mkdir -p "$RUN_DIR"

echo "run_dir=${RUN_DIR}"
echo "resume_ckpt=${RESUME_CKPT}"
echo "target_epochs=${TARGET_EPOCHS}"
echo "log_file=${LOG_FILE}"
echo "config=${CONFIG_PATH}"
echo "mode=resume_disable_early_stop"

exec conda run --no-capture-output -n vinp310 \
  python -m finetune.train \
  --config "$CONFIG_PATH" \
  --resume "$RESUME_CKPT" \
  --output-dir "$RUN_DIR" \
  --epochs "$TARGET_EPOCHS" \
  --save-every-epoch \
  --disable-early-stop \
  "$@" 2>&1 | tee "$LOG_FILE"
