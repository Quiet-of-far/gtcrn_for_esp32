import argparse
from pathlib import Path

from rknn.api import RKNN


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--onnx",
        default="stream/onnx_models/gtcrn_simple.onnx",
        help="Streaming GTCRN ONNX model.",
    )
    parser.add_argument(
        "--output",
        default="deploy_rknn/gtcrn_dns3_stream_rk3568_fp16.rknn",
        help="Output RKNN model path.",
    )
    parser.add_argument("--target", default="rk3568", help="RKNN target platform.")
    parser.add_argument("--quant", action="store_true", help="Enable int8 quantization.")
    parser.add_argument("--dataset", default=None, help="Calibration dataset for quantization.")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rknn = RKNN(verbose=True)
    try:
        print("--> Config")
        ret = rknn.config(target_platform=args.target)
        if ret != 0:
            raise RuntimeError(f"rknn.config failed: {ret}")

        print("--> Load ONNX")
        ret = rknn.load_onnx(model=args.onnx)
        if ret != 0:
            raise RuntimeError(f"rknn.load_onnx failed: {ret}")

        print("--> Build")
        ret = rknn.build(do_quantization=args.quant, dataset=args.dataset)
        if ret != 0:
            raise RuntimeError(f"rknn.build failed: {ret}")

        print("--> Export RKNN")
        ret = rknn.export_rknn(output_path.as_posix())
        if ret != 0:
            raise RuntimeError(f"rknn.export_rknn failed: {ret}")
        print(f"saved: {output_path}")
    finally:
        rknn.release()


if __name__ == "__main__":
    main()
