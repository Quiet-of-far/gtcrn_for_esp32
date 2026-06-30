#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

exec conda run --no-capture-output -n vinp310 \
  python -m finetune.train \
  --config finetune/config.yaml \
  --pretrained checkpoints/model_trained_on_dns3.tar \
  "$@"
