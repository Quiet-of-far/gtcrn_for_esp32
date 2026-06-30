import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]


def initial_state():
    return {
        "conv_a": np.zeros((2, 1, 16, 16, 33), np.float32),
        "tra_a": np.zeros((2, 3, 1, 1, 16), np.float32),
        "inter_a": np.zeros((2, 1, 33, 16), np.float32),
        "conv_b": np.zeros((2, 1, 16, 16, 33), np.float32),
        "tra_b": np.zeros((2, 3, 1, 1, 16), np.float32),
        "inter_b": np.zeros((2, 1, 33, 16), np.float32),
    }


def worker(args):
    sys.path.insert(0, (ROOT / "tools" / "rknn").as_posix())
    from export_gtcrn_stream_onnx_4d_nodeconv import make_models

    torch.set_num_threads(1)
    original, no_deconv = make_models(args.checkpoint, torch.device("cpu"))
    mixes = np.load(args.input)
    state_np = dict(np.load(args.state))
    conv_a, tra_a, inter_a = (torch.from_numpy(state_np[key]) for key in ("conv_a", "tra_a", "inter_a"))
    conv_b, tra_b, inter_b = (torch.from_numpy(state_np[key]) for key in ("conv_b", "tra_b", "inter_b"))
    max_abs = 0.0
    sum_abs = 0.0
    count = 0
    with torch.no_grad():
        for mix_np in mixes:
            mix = torch.from_numpy(mix_np)
            out_a = original(mix, conv_a, tra_a, inter_a)
            out_b = no_deconv(mix, conv_b, tra_b, inter_b)
            for left, right in zip(out_a, out_b):
                diff = (left - right).abs()
                max_abs = max(max_abs, float(diff.max()))
                sum_abs += float(diff.sum())
                count += diff.numel()
            _, conv_a, tra_a, inter_a = out_a
            _, conv_b, tra_b, inter_b = out_b
    np.savez(
        args.next_state,
        conv_a=conv_a.numpy(), tra_a=tra_a.numpy(), inter_a=inter_a.numpy(),
        conv_b=conv_b.numpy(), tra_b=tra_b.numpy(), inter_b=inter_b.numpy(),
    )
    Path(args.result).write_text(
        json.dumps({"max_abs": max_abs, "sum_abs": sum_abs, "count": count}), encoding="utf-8"
    )


def parent(args):
    rng = np.random.default_rng(0)
    all_mixes = rng.standard_normal((args.frames, 1, 257, 1, 2), dtype=np.float32)
    max_abs = 0.0
    sum_abs = 0.0
    count = 0
    with tempfile.TemporaryDirectory(prefix="gtcrn_equiv_") as tmp:
        tmp = Path(tmp)
        state_path = tmp / "state_0.npz"
        np.savez(state_path, **initial_state())
        for start in range(0, args.frames, args.chunk_size):
            end = min(args.frames, start + args.chunk_size)
            input_path = tmp / f"input_{start}.npy"
            next_state = tmp / f"state_{end}.npz"
            result_path = tmp / f"result_{start}.json"
            np.save(input_path, all_mixes[start:end])
            subprocess.run(
                [
                    sys.executable, "-m", "finetune.verify_stream_equivalence", "--worker",
                    "--checkpoint", args.checkpoint, "--input", input_path,
                    "--state", state_path, "--next-state", next_state, "--result", result_path,
                ],
                cwd=ROOT,
                check=True,
            )
            result = json.loads(result_path.read_text(encoding="utf-8"))
            max_abs = max(max_abs, result["max_abs"])
            sum_abs += result["sum_abs"]
            count += result["count"]
            state_path = next_state
    report = {"frames": args.frames, "max_abs": max_abs, "mean_abs": sum_abs / count}
    if report["max_abs"] >= 1e-4 or report["mean_abs"] >= 1e-5:
        raise RuntimeError(f"stream equivalence failed: {report}")
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--frames", type=int, default=1000)
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument("--output")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--input")
    parser.add_argument("--state")
    parser.add_argument("--next-state")
    parser.add_argument("--result")
    args = parser.parse_args()
    if args.worker:
        worker(args)
    else:
        if not args.output:
            parser.error("--output is required")
        parent(args)


if __name__ == "__main__":
    main()
