#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/home/quiet/miniforge3/envs/vinp310/bin/python}"

cd "${ROOT_DIR}"
exec "${PYTHON}" apps/gtcrn_rknn_realtime_tk.py "$@"
