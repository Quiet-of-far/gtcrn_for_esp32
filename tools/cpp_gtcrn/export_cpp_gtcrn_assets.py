import argparse
import re
import struct
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, ROOT.as_posix())
sys.path.insert(0, (ROOT / "stream").as_posix())
sys.path.insert(0, (ROOT / "tools" / "rknn").as_posix())

from export_gtcrn_stream_onnx_4d_nodeconv import make_models


SR = 16000
N_FFT = 512
HOP = 256


def symbol(name):
    return "w_" + re.sub(r"[^0-9A-Za-z_]", "_", name)


def c_float(value):
    text = f"{float(value):.9g}"
    if "e" not in text and "." not in text:
        text += ".0"
    return text + "f"


def write_header(model, output):
    state = {
        key: value.detach().cpu().contiguous().numpy().astype(np.float32)
        for key, value in model.state_dict().items()
        if value.ndim > 0
    }
    with output.open("w", encoding="utf-8") as f:
        f.write("#pragma once\n")
        f.write("#include <cstddef>\n")
        f.write("#include <string>\n")
        f.write("#include <unordered_map>\n")
        f.write("#include <vector>\n\n")
        f.write("namespace gtcrn {\n")
        f.write("struct TensorView { const float* data; std::vector<int> shape; size_t size; };\n")
        for key, arr in state.items():
            flat = arr.reshape(-1)
            f.write(f"static const float {symbol(key)}[] = {{\n")
            for i in range(0, flat.size, 8):
                f.write("  " + ", ".join(c_float(x) for x in flat[i : i + 8]) + ",\n")
            f.write("};\n")
        f.write("inline std::unordered_map<std::string, TensorView> make_weight_map() {\n")
        f.write("  std::unordered_map<std::string, TensorView> m;\n")
        for key, arr in state.items():
            shape = ", ".join(str(x) for x in arr.shape)
            f.write(
                f'  m.emplace("{key}", TensorView{{{symbol(key)}, std::vector<int>{{{shape}}}, {arr.size}}});\n'
            )
        f.write("  return m;\n}\n")
        f.write("} // namespace gtcrn\n")


def load_audio(path, frames):
    wav, sr = sf.read(path, dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != SR:
        raise ValueError(f"{path} sample rate is {sr}, expected {SR}")
    needed = frames * HOP
    if wav.size < needed:
        reps = int(np.ceil(needed / max(1, wav.size)))
        wav = np.tile(wav, reps)
    return wav[:needed].astype(np.float32, copy=False)


def iter_mix_frames(audio):
    window = np.sqrt(np.hanning(N_FFT).astype(np.float32))
    buf = np.zeros(N_FFT, dtype=np.float32)
    for pos in range(0, len(audio), HOP):
        buf[:-HOP] = buf[HOP:]
        buf[-HOP:] = audio[pos : pos + HOP]
        spec = np.fft.rfft(buf * window, n=N_FFT)
        yield np.stack([spec.real, spec.imag], axis=-1).astype(np.float32).reshape(1, 257, 1, 2)


def write_reference(model, audio_path, frames, output):
    audio = load_audio(audio_path, frames)
    conv = torch.zeros(2, 1, 16, 16, 33)
    tra = torch.zeros(2, 3, 1, 1, 16)
    inter = torch.zeros(2, 1, 33, 16)
    with output.open("wb") as f:
        f.write(b"GTCRNREF1")
        f.write(struct.pack("<I", frames))
        with torch.no_grad():
            for mix_np in iter_mix_frames(audio):
                mix = torch.from_numpy(mix_np)
                enh, conv, tra, inter = model(mix, conv, tra, inter)
                parts = [
                    mix_np.reshape(-1),
                    enh.detach().cpu().numpy().astype(np.float32).reshape(-1),
                    conv.detach().cpu().numpy().astype(np.float32).reshape(-1),
                    tra.detach().cpu().numpy().astype(np.float32).reshape(-1),
                    inter.detach().cpu().numpy().astype(np.float32).reshape(-1),
                ]
                for part in parts:
                    f.write(np.ascontiguousarray(part, dtype=np.float32).tobytes())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=(ROOT / "checkpoints" / "model_trained_on_dns3.tar").as_posix())
    parser.add_argument("--weights-out", default=(ROOT / "cpp_gtcrn" / "gtcrn_weights_dns3.h").as_posix())
    parser.add_argument("--ref-out", default=(ROOT / "cpp_gtcrn" / "gtcrn_ref_1000.bin").as_posix())
    parser.add_argument("--audio", default=(ROOT / "test_wavs" / "mix.wav").as_posix())
    parser.add_argument("--frames", type=int, default=1000)
    args = parser.parse_args()

    _, model = make_models(args.checkpoint, torch.device("cpu"))
    model.eval()
    weights_out = Path(args.weights_out)
    ref_out = Path(args.ref_out)
    weights_out.parent.mkdir(parents=True, exist_ok=True)
    ref_out.parent.mkdir(parents=True, exist_ok=True)
    write_header(model, weights_out)
    write_reference(model, Path(args.audio), args.frames, ref_out)
    print(f"wrote {weights_out}")
    print(f"wrote {ref_out}")


if __name__ == "__main__":
    main()
