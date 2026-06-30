import argparse
import csv
import json
import math
import os
import random
import shutil
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from finetune.data import PairedWavDataset, validate_dataset
from finetune.losses import HybridLoss
from finetune.metrics import MetricAccumulator
from finetune.model import EndToEndGTCRN, load_model_weights


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path):
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def make_scheduler(optimizer, warmup_steps, total_steps, min_lr, max_lr):
    min_ratio = min_lr / max_lr

    def scale(step):
        if step < warmup_steps:
            return max(min_ratio, (step + 1) / max(1, warmup_steps))
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_ratio + 0.5 * (1 - min_ratio) * (1 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


@torch.inference_mode()
def evaluate(model, dataset, loss_fn, device, metric_workers, sample_dir=None, metric_limit=None):
    model.eval()
    loss_total = 0.0
    metric_accumulator = MetricAccumulator(metric_workers)
    iterator = tqdm(dataset, desc="validate", leave=False)
    for index, (noisy, clean, name) in enumerate(iterator):
        noisy = noisy.unsqueeze(0).to(device)
        clean = clean.unsqueeze(0).to(device)
        enhanced = model(noisy)
        loss_total += float(loss_fn(enhanced, clean))
        if metric_limit is None or index < metric_limit:
            metric_accumulator.submit((clean[0].cpu().numpy(), enhanced[0].cpu().numpy(), 16000))
        if sample_dir is not None and index < 5:
            sample_dir.mkdir(parents=True, exist_ok=True)
            sf.write(sample_dir / f"{Path(name).stem}_enh.wav", enhanced[0].cpu().numpy(), 16000)
            sf.write(sample_dir / f"{Path(name).stem}_clean.wav", clean[0].cpu().numpy(), 16000)
            sf.write(sample_dir / f"{Path(name).stem}_noisy.wav", noisy[0].cpu().numpy(), 16000)
    metrics = metric_accumulator.finish()
    metrics["loss"] = loss_total / len(dataset)
    return metrics


def save_checkpoint(path, model, optimizer, scheduler, epoch, global_step, best_pesq, stale_epochs, config, metrics):
    torch.save(
        {
            "epoch": epoch,
            "global_step": global_step,
            "model": model.model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_pesq": best_pesq,
            "stale_epochs": stale_epochs,
            "config": config,
            "metrics": metrics,
        },
        path,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="finetune/config.yaml")
    parser.add_argument("--pretrained", default="checkpoints/model_trained_on_dns3.tar")
    parser.add_argument("--resume")
    parser.add_argument("--output-dir")
    parser.add_argument("--output-root")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--from-scratch", action="store_true")
    parser.add_argument("--save-every-epoch", action="store_true")
    parser.add_argument("--disable-early-stop", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--baseline-only", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.output_root:
        config["output_root"] = args.output_root
    if args.epochs is not None:
        config["epochs"] = args.epochs
    seed_everything(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and not args.smoke:
        raise RuntimeError("formal fine-tuning requires CUDA")

    train_report = validate_dataset(config["train_dir"], config["sample_rate"])
    valid_report = validate_dataset(config["valid_dir"], config["sample_rate"])
    print(json.dumps({"train": train_report, "valid": valid_report}, ensure_ascii=False))

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = Path(args.output_dir or Path(config["output_root"]) / stamp).resolve()
    checkpoints = output_dir / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    train_limit = 4 if args.smoke else None
    valid_limit = 2 if args.smoke else None
    train_dataset = PairedWavDataset(
        config["train_dir"], config["sample_rate"], config["segment_seconds"], training=True, limit=train_limit
    )
    valid_dataset = PairedWavDataset(config["valid_dir"], config["sample_rate"], limit=valid_limit)
    loader = DataLoader(
        train_dataset,
        batch_size=2 if args.smoke else config["batch_size"],
        shuffle=True,
        num_workers=0 if args.smoke else config["num_workers"],
        pin_memory=device.type == "cuda",
        drop_last=True,
        persistent_workers=not args.smoke and config["num_workers"] > 0,
    )

    model = EndToEndGTCRN(config["n_fft"], config["hop_length"], config["win_length"]).to(device)
    loss_fn = HybridLoss(config["n_fft"], config["hop_length"], config["win_length"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    epochs = 1 if args.smoke else config["epochs"]
    total_steps = epochs * len(loader)
    scheduler = make_scheduler(
        optimizer, config["warmup_steps"], total_steps, config["min_learning_rate"], config["learning_rate"]
    )
    start_epoch = 1
    global_step = 0
    best_pesq = -float("inf")
    stale_epochs = 0

    if args.resume:
        resume = torch.load(args.resume, map_location=device)
        model.model.load_state_dict(resume["model"], strict=True)
        optimizer.load_state_dict(resume["optimizer"])
        scheduler.load_state_dict(resume["scheduler"])
        start_epoch = resume["epoch"] + 1
        global_step = resume["global_step"]
        best_pesq = resume["best_pesq"]
        stale_epochs = resume.get("stale_epochs", 0)
    elif not args.from_scratch:
        load_model_weights(model, args.pretrained, map_location=device)

    baseline_path = output_dir / "baseline_metrics.json"
    if baseline_path.exists():
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    else:
        baseline = evaluate(
            model,
            valid_dataset,
            loss_fn,
            device,
            1 if args.smoke else config["metric_workers"],
            output_dir / "baseline_samples",
        )
        baseline_path.write_text(json.dumps(baseline, indent=2), encoding="utf-8")
    print("baseline", baseline)
    if args.baseline_only:
        return

    writer = SummaryWriter(output_dir / "tensorboard")
    history_path = output_dir / "metrics.csv"
    history_exists = history_path.exists()
    history_file = history_path.open("a", newline="", encoding="utf-8")
    history = csv.DictWriter(history_file, fieldnames=["epoch", "train_loss", "loss", "pesq", "stoi", "si_snr", "lr"])
    if not history_exists:
        history.writeheader()

    amp_enabled = config["amp"] and device.type == "cuda"
    amp_dtype = torch.bfloat16 if config.get("amp_dtype", "bfloat16") == "bfloat16" else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled and amp_dtype == torch.float16, init_scale=1024.0)
    try:
        for epoch in range(start_epoch, epochs + 1):
            model.train()
            running = 0.0
            bar = tqdm(loader, desc=f"train {epoch}/{epochs}")
            for step, (noisy, clean, _) in enumerate(bar, 1):
                noisy = noisy.to(device, non_blocking=True)
                clean = clean.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                    enhanced = model(noisy)
                with torch.autocast(device_type=device.type, enabled=False):
                    loss = loss_fn(enhanced, clean)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"non-finite loss at epoch={epoch} step={step}: {loss}")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
                old_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() >= old_scale:
                    scheduler.step()
                global_step += 1
                running += float(loss)
                bar.set_postfix(loss=running / step, lr=optimizer.param_groups[0]["lr"])

            train_loss = running / len(loader)
            metrics = evaluate(
                model,
                valid_dataset,
                loss_fn,
                device,
                1 if args.smoke else config["metric_workers"],
                output_dir / "validation_samples" / f"epoch_{epoch:03d}",
            )
            improved = metrics["pesq"] > best_pesq + config["early_stop_min_delta"]
            if improved:
                best_pesq = metrics["pesq"]
                stale_epochs = 0
            else:
                stale_epochs += 1
            save_checkpoint(
                checkpoints / "latest.tar", model, optimizer, scheduler, epoch, global_step,
                best_pesq, stale_epochs, config, metrics,
            )
            if args.save_every_epoch:
                save_checkpoint(
                    checkpoints / f"epoch_{epoch:03d}.tar", model, optimizer, scheduler, epoch, global_step,
                    best_pesq, stale_epochs, config, metrics,
                )
            if improved:
                shutil.copy2(checkpoints / "latest.tar", checkpoints / "best_pesq.tar")
            row = {"epoch": epoch, "train_loss": train_loss, "lr": optimizer.param_groups[0]["lr"], **metrics}
            history.writerow(row)
            history_file.flush()
            for key, value in row.items():
                if key != "epoch":
                    writer.add_scalar(key, value, epoch)
            print("epoch", row, "best_pesq", best_pesq, "stale", stale_epochs)
            if (not args.disable_early_stop) and stale_epochs >= config["early_stop_patience"]:
                print(f"early stopping at epoch {epoch}")
                break
    finally:
        history_file.close()
        writer.close()


if __name__ == "__main__":
    main()
