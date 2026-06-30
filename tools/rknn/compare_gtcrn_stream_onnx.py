import argparse
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import soundfile as sf


SR = 16000
N_FFT = 512
HOP = 256
INPUT_NAMES = ["mix", "conv_cache", "tra_cache", "inter_cache"]
OUTPUT_NAMES = ["enh", "conv_cache_out", "tra_cache_out", "inter_cache_out"]


def load_audio(path, frames):
    wav, sr = sf.read(path, dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != SR:
        raise ValueError(f"{path} sample rate is {sr}, expected {SR}")
    needed = frames * HOP
    if wav.size < needed:
        repeats = int(np.ceil(needed / max(1, wav.size)))
        wav = np.tile(wav, repeats)
    return wav[:needed].astype(np.float32, copy=False)


def iter_mix_frames(audio):
    window = np.sqrt(np.hanning(N_FFT).astype(np.float32))
    buf = np.zeros(N_FFT, dtype=np.float32)
    for pos in range(0, len(audio), HOP):
        buf[:-HOP] = buf[HOP:]
        buf[-HOP:] = audio[pos : pos + HOP]
        spec = np.fft.rfft(buf * window, n=N_FFT)
        yield np.stack([spec.real, spec.imag], axis=-1).astype(np.float32).reshape(
            1, N_FFT // 2 + 1, 1, 2
        )


def count_ops(path):
    model = onnx.load(path)
    counts = {}
    for node in model.graph.node:
        counts[node.op_type] = counts.get(node.op_type, 0) + 1
    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", default="stream/onnx_models/gtcrn_simple_4d.onnx")
    parser.add_argument("--candidate", default="stream/onnx_models/gtcrn_simple_4d_no_deconv.onnx")
    parser.add_argument("--audio", default="test_wavs/mix.wav")
    parser.add_argument("--frames", type=int, default=1000)
    parser.add_argument("--max-abs-threshold", type=float, default=1e-4)
    parser.add_argument("--mean-abs-threshold", type=float, default=1e-5)
    args = parser.parse_args()

    base_counts = count_ops(args.baseline)
    cand_counts = count_ops(args.candidate)
    print(f"baseline ConvTranspose={base_counts.get('ConvTranspose', 0)}")
    print(f"candidate ConvTranspose={cand_counts.get('ConvTranspose', 0)}")

    base = ort.InferenceSession(args.baseline, providers=["CPUExecutionProvider"])
    cand = ort.InferenceSession(args.candidate, providers=["CPUExecutionProvider"])
    audio = load_audio(args.audio, args.frames)

    base_inputs = [
        np.zeros((2, 16, 16, 33), dtype=np.float32),
        np.zeros((2, 3, 1, 16), dtype=np.float32),
        np.zeros((2, 1, 33, 16), dtype=np.float32),
    ]
    cand_inputs = [x.copy() for x in base_inputs]
    stats = {name: {"max_abs": 0.0, "sum_abs": 0.0, "count": 0} for name in OUTPUT_NAMES}

    for mix in iter_mix_frames(audio):
        out_base = base.run(None, dict(zip(INPUT_NAMES, [mix] + base_inputs)))
        out_cand = cand.run(None, dict(zip(INPUT_NAMES, [mix] + cand_inputs)))
        for name, a, b in zip(OUTPUT_NAMES, out_base, out_cand):
            diff = np.abs(a - b)
            stats[name]["max_abs"] = max(stats[name]["max_abs"], float(diff.max()))
            stats[name]["sum_abs"] += float(diff.sum())
            stats[name]["count"] += int(diff.size)
        base_inputs = [np.ascontiguousarray(x, dtype=np.float32) for x in out_base[1:]]
        cand_inputs = [np.ascontiguousarray(x, dtype=np.float32) for x in out_cand[1:]]

    failed = cand_counts.get("ConvTranspose", 0) != 0
    for name in OUTPUT_NAMES:
        mean_abs = stats[name]["sum_abs"] / stats[name]["count"]
        max_abs = stats[name]["max_abs"]
        print(f"{name}: max_abs={max_abs:.8g} mean_abs={mean_abs:.8g}")
        if max_abs > args.max_abs_threshold or mean_abs > args.mean_abs_threshold:
            failed = True
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
