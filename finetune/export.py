import argparse
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--audio", default="test_wavs/mix.wav")
    args = parser.parse_args()
    script = Path(__file__).with_name("export_candidate.sh").resolve()
    os.environ["GTCRN_PYTHON"] = sys.executable
    os.execv(
        "/usr/bin/bash",
        ["bash", str(script), str(Path(args.checkpoint).resolve()), str(Path(args.output_dir).resolve()), args.audio],
    )


if __name__ == "__main__":
    main()
