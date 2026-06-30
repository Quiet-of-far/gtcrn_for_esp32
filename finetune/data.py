from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset


class PairedWavDataset(Dataset):
    def __init__(self, root, sample_rate=16000, segment_seconds=None, training=False, limit=None):
        self.root = Path(root).expanduser().resolve()
        self.clean_dir = self.root / "clean"
        self.noisy_dir = self.root / "noisy"
        self.sample_rate = sample_rate
        self.segment_samples = None if segment_seconds is None else round(segment_seconds * sample_rate)
        self.training = training
        if not self.clean_dir.is_dir() or not self.noisy_dir.is_dir():
            raise FileNotFoundError(f"expected clean/ and noisy/ under {self.root}")
        noisy = {p.name: p for p in self.noisy_dir.glob("*.wav")}
        clean = {p.name: p for p in self.clean_dir.glob("*.wav")}
        missing_clean = sorted(noisy.keys() - clean.keys())
        missing_noisy = sorted(clean.keys() - noisy.keys())
        if missing_clean or missing_noisy:
            raise ValueError(
                f"unpaired data: missing_clean={len(missing_clean)} missing_noisy={len(missing_noisy)}"
            )
        names = sorted(noisy)
        if limit is not None:
            names = names[:limit]
        if not names:
            raise ValueError(f"no paired WAV files under {self.root}")
        self.items = [(noisy[name], clean[name], name) for name in names]

    def __len__(self):
        return len(self.items)

    def _load_pair(self, noisy_path, clean_path):
        noisy, noisy_sr = sf.read(noisy_path, dtype="float32", always_2d=False)
        clean, clean_sr = sf.read(clean_path, dtype="float32", always_2d=False)
        if noisy.ndim != 1 or clean.ndim != 1:
            raise ValueError(f"mono audio required: {noisy_path.name}")
        if noisy_sr != self.sample_rate or clean_sr != self.sample_rate:
            raise ValueError(f"sample rate mismatch: {noisy_path.name}")
        if noisy.shape != clean.shape:
            raise ValueError(f"frame count mismatch: {noisy_path.name}")
        return noisy, clean

    def __getitem__(self, index):
        noisy_path, clean_path, name = self.items[index]
        noisy, clean = self._load_pair(noisy_path, clean_path)
        if self.segment_samples is not None:
            length = noisy.shape[0]
            if length > self.segment_samples:
                start = np.random.randint(0, length - self.segment_samples + 1) if self.training else 0
                noisy = noisy[start : start + self.segment_samples]
                clean = clean[start : start + self.segment_samples]
            elif length < self.segment_samples:
                pad = self.segment_samples - length
                noisy = np.pad(noisy, (0, pad))
                clean = np.pad(clean, (0, pad))
        return torch.from_numpy(noisy.copy()), torch.from_numpy(clean.copy()), name


def validate_dataset(root, sample_rate=16000):
    dataset = PairedWavDataset(root, sample_rate=sample_rate)
    total_frames = 0
    for noisy_path, clean_path, name in dataset.items:
        noisy_info = sf.info(noisy_path)
        clean_info = sf.info(clean_path)
        expected = (sample_rate, 1, "PCM_16")
        noisy_format = (noisy_info.samplerate, noisy_info.channels, noisy_info.subtype)
        clean_format = (clean_info.samplerate, clean_info.channels, clean_info.subtype)
        if noisy_format != expected or clean_format != expected:
            raise ValueError(f"invalid WAV format for {name}: {noisy_format} / {clean_format}")
        if noisy_info.frames != clean_info.frames:
            raise ValueError(f"frame count mismatch for {name}")
        total_frames += noisy_info.frames
    return {"pairs": len(dataset), "hours": total_frames / sample_rate / 3600.0}
