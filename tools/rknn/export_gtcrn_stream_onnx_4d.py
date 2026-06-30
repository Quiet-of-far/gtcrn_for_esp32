import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn


ROOT_DIR = Path(__file__).resolve().parents[2]
STREAM_DIR = ROOT_DIR / "stream"
sys.path.insert(0, ROOT_DIR.as_posix())
sys.path.insert(0, STREAM_DIR.as_posix())

from gtcrn import GTCRN
from gtcrn_stream import StreamGTCRN
from modules.convert import convert_to_stream


class StreamGTCRN4DWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, mix, conv_cache, tra_cache, inter_cache):
        conv_cache_5d = conv_cache.reshape(2, 1, 16, 16, 33)
        tra_cache_5d = tra_cache.reshape(2, 3, 1, 1, 16)
        enh, conv_out, tra_out, inter_out = self.model(
            mix, conv_cache_5d, tra_cache_5d, inter_cache
        )
        return (
            enh,
            conv_out.reshape(2, 16, 16, 33),
            tra_out.reshape(2, 3, 1, 16),
            inter_out,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default=(ROOT_DIR / "checkpoints" / "model_trained_on_dns3.tar").as_posix(),
    )
    parser.add_argument(
        "--output",
        default=(STREAM_DIR / "onnx_models" / "gtcrn_simple_4d.onnx").as_posix(),
    )
    parser.add_argument("--raw-output", default=None)
    parser.add_argument("--no-simplify", action="store_true")
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    raw_output = Path(args.raw_output) if args.raw_output else output.with_name("gtcrn_4d.onnx")

    device = torch.device("cpu")
    offline_model = GTCRN().to(device).eval()
    checkpoint = torch.load(args.checkpoint, map_location=device)
    offline_model.load_state_dict(
        {key.replace("module.", ""): value for key, value in checkpoint["model"].items()}
    )

    stream_model = StreamGTCRN().to(device).eval()
    convert_to_stream(stream_model, offline_model)
    wrapper = StreamGTCRN4DWrapper(stream_model).to(device).eval()

    mix = torch.randn(1, 257, 1, 2, device=device)
    conv_cache = torch.zeros(2, 16, 16, 33, device=device)
    tra_cache = torch.zeros(2, 3, 1, 16, device=device)
    inter_cache = torch.zeros(2, 1, 33, 16, device=device)

    torch.onnx.export(
        wrapper,
        (mix, conv_cache, tra_cache, inter_cache),
        raw_output.as_posix(),
        input_names=["mix", "conv_cache", "tra_cache", "inter_cache"],
        output_names=["enh", "conv_cache_out", "tra_cache_out", "inter_cache_out"],
        opset_version=11,
        verbose=False,
    )

    if args.no_simplify:
        if raw_output != output:
            output.write_bytes(raw_output.read_bytes())
        print(f"saved: {output}")
        return

    try:
        import onnx
        from onnxsim import simplify

        onnx_model = onnx.load(raw_output.as_posix())
        model_simp, check = simplify(onnx_model)
        if not check:
            raise RuntimeError("onnxsim validation failed")
        onnx.save(model_simp, output.as_posix())
    except Exception as exc:
        print(f"onnx simplify failed, keeping raw export: {exc}")
        if raw_output != output:
            output.write_bytes(raw_output.read_bytes())
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
