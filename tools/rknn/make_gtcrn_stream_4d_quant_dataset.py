import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf


SR = 16000
N_FFT = 512
HOP = 256
INPUT_NAMES = ["mix", "conv_cache", "tra_cache", "inter_cache"]


def load_audio(path, seconds):
    wav, sr = sf.read(path, dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != SR:
        raise ValueError(f"{path} sample rate is {sr}, expected {SR}")
    if seconds > 0:
        wav = wav[: int(seconds * SR)]
    if wav.size < N_FFT:
        wav = np.pad(wav, (0, N_FFT - wav.size))
    peak = np.max(np.abs(wav))
    if peak > 1.0:
        wav = wav / peak
    return wav.astype(np.float32, copy=False)


def synthetic_audio(seconds, seed):
    rng = np.random.default_rng(seed)
    n = max(N_FFT, int(seconds * SR))
    t = np.arange(n, dtype=np.float32) / SR
    speech_like = (
        0.12 * np.sin(2 * np.pi * 180 * t)
        + 0.08 * np.sin(2 * np.pi * 440 * t)
        + 0.04 * np.sin(2 * np.pi * 1240 * t)
    )
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 2.3 * t)
    noise = rng.normal(0.0, 0.035, n).astype(np.float32)
    return (speech_like * envelope + noise).astype(np.float32)


def iter_mix_frames(audio):
    window = np.sqrt(np.hanning(N_FFT).astype(np.float32))
    buf = np.zeros(N_FFT, dtype=np.float32)
    padded = np.pad(audio, (0, HOP))
    for pos in range(0, len(padded) - HOP + 1, HOP):
        buf[:-HOP] = buf[HOP:]
        buf[-HOP:] = padded[pos : pos + HOP]
        spec = np.fft.rfft(buf * window, n=N_FFT)
        yield np.stack([spec.real, spec.imag], axis=-1).astype(np.float32).reshape(
            1, N_FFT // 2 + 1, 1, 2
        )


def save_sample(out_dir, index, mix, conv_cache, tra_cache, inter_cache):
    sample_dir = out_dir / f"{index:05d}"
    sample_dir.mkdir(parents=True, exist_ok=True)
    arrays = {
        "mix": mix,
        "conv_cache": conv_cache,
        "tra_cache": tra_cache,
        "inter_cache": inter_cache,
    }
    paths = []
    for name in INPUT_NAMES:
        path = sample_dir / f"{name}.npy"
        np.save(path, np.ascontiguousarray(arrays[name], dtype=np.float32))
        paths.append(path.resolve().as_posix())
    return " ".join(paths)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", default="stream/onnx_models/gtcrn_simple_4d.onnx")
    parser.add_argument("--audio", default="test_wavs/mix.wav")
    parser.add_argument("--output-dir", default="deploy_rknn/quant_calib_gtcrn_4d")
    parser.add_argument("--seconds", type=float, default=8.0)
    parser.add_argument("--max-samples", type=int, default=256)
    parser.add_argument("--synthetic-seconds", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    audio_parts = []
    audio_path = Path(args.audio)
    if audio_path.exists():
        audio_parts.append(load_audio(audio_path, args.seconds))
    if args.synthetic_seconds > 0:
        audio_parts.append(synthetic_audio(args.synthetic_seconds, args.seed))
    if not audio_parts:
        raise FileNotFoundError(f"no calibration audio found: {audio_path}")
    audio = np.concatenate(audio_parts)

    session = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    conv_cache = np.zeros((2, 16, 16, 33), dtype=np.float32)
    tra_cache = np.zeros((2, 3, 1, 16), dtype=np.float32)
    inter_cache = np.zeros((2, 1, 33, 16), dtype=np.float32)

    lines = []
    for index, mix in enumerate(iter_mix_frames(audio)):
        if index >= args.max_samples:
            break
        lines.append(save_sample(out_dir, index, mix, conv_cache, tra_cache, inter_cache))
        enh, conv_cache, tra_cache, inter_cache = session.run(
            None,
            {
                "mix": mix,
                "conv_cache": conv_cache,
                "tra_cache": tra_cache,
                "inter_cache": inter_cache,
            },
        )
        del enh
        conv_cache = np.ascontiguousarray(conv_cache, dtype=np.float32)
        tra_cache = np.ascontiguousarray(tra_cache, dtype=np.float32)
        inter_cache = np.ascontiguousarray(inter_cache, dtype=np.float32)

    dataset = out_dir / "dataset.txt"
    dataset.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"saved {len(lines)} samples: {dataset}")


if __name__ == "__main__":
    main()
