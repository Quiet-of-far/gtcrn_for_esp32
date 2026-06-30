import argparse
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from rknn.api import RKNN


INPUT_NAMES = ["mix", "conv_cache", "tra_cache", "inter_cache"]
OUTPUT_NAMES = ["enh", "conv_cache_out", "tra_cache_out", "inter_cache_out"]
INPUT_SHAPES = {
    "mix": (1, 257, 1, 2),
    "conv_cache": (2, 1, 16, 16, 33),
    "tra_cache": (2, 3, 1, 1, 16),
    "inter_cache": (2, 1, 33, 16),
}


def make_inputs(seed):
    rng = np.random.default_rng(seed)
    return {
        "mix": rng.normal(0.0, 1.0, INPUT_SHAPES["mix"]).astype(np.float32),
        "conv_cache": np.zeros(INPUT_SHAPES["conv_cache"], dtype=np.float32),
        "tra_cache": np.zeros(INPUT_SHAPES["tra_cache"], dtype=np.float32),
        "inter_cache": np.zeros(INPUT_SHAPES["inter_cache"], dtype=np.float32),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", default="stream/onnx_models/gtcrn_simple.onnx")
    parser.add_argument("--target", default="rk3568")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--save-dir", default=None)
    args = parser.parse_args()

    inputs = make_inputs(args.seed)
    ort_session = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    ort_outputs = ort_session.run(None, inputs)

    rknn = RKNN(verbose=False)
    try:
        for label, ret in [
            ("config", rknn.config(target_platform=args.target)),
            ("load_onnx", rknn.load_onnx(model=args.onnx)),
            ("build", rknn.build(do_quantization=False)),
            ("init_runtime", rknn.init_runtime()),
        ]:
            if ret != 0:
                raise RuntimeError(f"{label} failed: {ret}")

        input_list = [inputs[name] for name in INPUT_NAMES]
        data_format = ["nchw"] * len(input_list)
        for _ in range(args.warmup):
            rknn.inference(inputs=input_list, data_format=data_format)

        times = []
        rknn_outputs = None
        for _ in range(args.repeat):
            started = time.perf_counter()
            rknn_outputs = rknn.inference(inputs=input_list, data_format=data_format)
            times.append((time.perf_counter() - started) * 1000)
    finally:
        rknn.release()

    metrics = []
    for name, ort_output, rknn_output in zip(OUTPUT_NAMES, ort_outputs, rknn_outputs):
        diff = ort_output - rknn_output
        metrics.append(
            {
                "name": name,
                "shape": rknn_output.shape,
                "max_abs": float(np.max(np.abs(diff))),
                "mean_abs": float(np.mean(np.abs(diff))),
                "p99_abs": float(np.percentile(np.abs(diff), 99)),
            }
        )

    for item in metrics:
        print(
            "{name}: shape={shape} max_abs={max_abs:.8f} "
            "mean_abs={mean_abs:.8f} p99_abs={p99_abs:.8f}".format(**item)
        )
    print(
        "pc_sim_time_ms avg={:.3f} min={:.3f} max={:.3f}".format(
            float(np.mean(times)), float(np.min(times)), float(np.max(times))
        )
    )

    if args.save_dir:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        for name in INPUT_NAMES:
            np.save(save_dir / f"{name}.npy", inputs[name])
        for name, ort_output, rknn_output in zip(OUTPUT_NAMES, ort_outputs, rknn_outputs):
            np.save(save_dir / f"{name}_onnx.npy", ort_output)
            np.save(save_dir / f"{name}_rknn.npy", rknn_output)
        lines = [
            "{name}: shape={shape} max_abs={max_abs:.8f} "
            "mean_abs={mean_abs:.8f} p99_abs={p99_abs:.8f}".format(**item)
            for item in metrics
        ]
        lines.append(
            "pc_sim_time_ms avg={:.3f} min={:.3f} max={:.3f}".format(
                float(np.mean(times)), float(np.min(times)), float(np.max(times))
            )
        )
        (save_dir / "metrics.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
