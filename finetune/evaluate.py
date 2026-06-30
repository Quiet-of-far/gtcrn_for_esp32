import argparse
import json
from pathlib import Path

import torch
import yaml

from finetune.data import PairedWavDataset
from finetune.losses import HybridLoss
from finetune.model import EndToEndGTCRN, load_model_weights
from finetune.train import evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="finetune/config.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EndToEndGTCRN(config["n_fft"], config["hop_length"], config["win_length"]).to(device)
    load_model_weights(model, args.checkpoint, map_location=device)
    loss_fn = HybridLoss(config["n_fft"], config["hop_length"], config["win_length"]).to(device)
    dataset = PairedWavDataset(config["valid_dir"], config["sample_rate"])
    metrics = evaluate(model, dataset, loss_fn, device, config["metric_workers"])
    Path(args.output).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
