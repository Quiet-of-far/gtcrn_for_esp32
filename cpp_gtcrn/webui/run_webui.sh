#!/usr/bin/env bash
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-7860}"

exec python3 "${DIR}/server.py" --host "${HOST}" --port "${PORT}" "$@"
