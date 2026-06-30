import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from finetune.data import PairedWavDataset
from finetune.model import EndToEndGTCRN, load_model_weights


class DatasetTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        (root / "clean").mkdir()
        (root / "noisy").mkdir()
        self.root = root

    def tearDown(self):
        self.tmp.cleanup()

    def write_pair(self, name, samples):
        clean = np.arange(samples, dtype=np.float32) / max(1, samples)
        noisy = clean + 0.01
        sf.write(self.root / "clean" / name, clean, 16000, subtype="PCM_16")
        sf.write(self.root / "noisy" / name, noisy, 16000, subtype="PCM_16")

    def test_short_pair_is_padded(self):
        self.write_pair("short.wav", 100)
        dataset = PairedWavDataset(self.root, segment_seconds=0.01, training=True)
        noisy, clean, _ = dataset[0]
        self.assertEqual(noisy.shape[-1], 160)
        self.assertTrue(torch.all(noisy[100:] == 0))
        self.assertTrue(torch.all(clean[100:] == 0))

    def test_long_pair_uses_aligned_crop(self):
        self.write_pair("long.wav", 1000)
        dataset = PairedWavDataset(self.root, segment_seconds=0.01, training=True)
        noisy, clean, _ = dataset[0]
        difference = noisy - clean
        self.assertLess(float(difference.std()), 1e-4)


class ModelTest(unittest.TestCase):
    def test_checkpoint_and_output_length(self):
        model = EndToEndGTCRN()
        load_model_weights(model, "checkpoints/model_trained_on_dns3.tar")
        waveform = torch.randn(1, 4096)
        with torch.no_grad():
            output = model(waveform)
        self.assertEqual(output.shape, waveform.shape)


if __name__ == "__main__":
    unittest.main()
