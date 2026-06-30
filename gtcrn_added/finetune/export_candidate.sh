#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHECKPOINT="$1"
OUTPUT_DIR="$2"
AUDIO="${3:-test_wavs/mix.wav}"
PYTHON="${GTCRN_PYTHON:-/home/quiet/miniforge3/envs/vinp310/bin/python}"

mkdir -p "$OUTPUT_DIR"
cd "$ROOT"

ONNX="$OUTPUT_DIR/gtcrn_dns3_finetuned_stream_4d_no_deconv.onnx"
RAW_ONNX="$OUTPUT_DIR/gtcrn_dns3_finetuned_stream_4d_no_deconv_raw.onnx"
WEIGHTS="$OUTPUT_DIR/gtcrn_weights_dns3_finetuned.h"
REFERENCE="$OUTPUT_DIR/gtcrn_dns3_finetuned_ref_1000.bin"
EQUIVALENCE="$OUTPUT_DIR/stream_equivalence_1000.json"
BUILD_DIR="$OUTPUT_DIR/cpp_build"

"$PYTHON" tools/rknn/export_gtcrn_stream_onnx_4d_nodeconv.py \
  --checkpoint "$CHECKPOINT" \
  --output "$ONNX" \
  --raw-output "$RAW_ONNX" \
  --compare-repeats 16 \
  --no-simplify

"$PYTHON" -m finetune.verify_stream_equivalence \
  --checkpoint "$CHECKPOINT" \
  --frames 1000 \
  --chunk-size 100 \
  --output "$EQUIVALENCE"

"$PYTHON" tools/cpp_gtcrn/export_cpp_gtcrn_assets.py \
  --checkpoint "$CHECKPOINT" \
  --weights-out "$WEIGHTS" \
  --ref-out "$REFERENCE" \
  --audio "$AUDIO" \
  --frames 1000

"$PYTHON" - "$ONNX" <<'PY'
import sys
import onnx

path = sys.argv[1]
model = onnx.load(path)
onnx.checker.check_model(model)
count = sum(node.op_type == "ConvTranspose" for node in model.graph.node)
if count:
    raise SystemExit(f"ONNX contains {count} ConvTranspose nodes")
print("onnx_check=ok ConvTranspose=0")
PY

cmake -S cpp_gtcrn -B "$BUILD_DIR" \
  -DGTCRN_WEIGHTS_HEADER="$WEIGHTS" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build "$BUILD_DIR" --target benchmark_gtcrn -j"$(nproc)"
"$BUILD_DIR/benchmark_gtcrn" --ref "$REFERENCE" --frames 1000 --warmup 20

"$PYTHON" - "$CHECKPOINT" "$ONNX" "$WEIGHTS" "$REFERENCE" "$EQUIVALENCE" "$BUILD_DIR/benchmark_gtcrn" "$OUTPUT_DIR/export_report.json" <<'PY'
import json
import sys
from pathlib import Path

keys = ["checkpoint", "onnx", "weights", "reference", "stream_equivalence", "cpp_benchmark"]
report = dict(zip(keys, sys.argv[1:7]))
report["conv_transpose_nodes"] = 0
Path(sys.argv[7]).write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
PY
