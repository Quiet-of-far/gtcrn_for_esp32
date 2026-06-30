import torch
import torch.nn as nn


class HybridLoss(nn.Module):
    def __init__(self, n_fft=512, hop_length=256, win_length=512, compress_factor=0.3, eps=1e-12):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.compress_factor = compress_factor
        self.eps = eps
        self.register_buffer("window", torch.hann_window(win_length), persistent=False)

    def forward(self, prediction, target):
        prediction = prediction.float()
        target = target.float()
        pred = torch.stft(
            prediction, self.n_fft, self.hop_length, self.win_length, self.window, return_complex=True
        )
        true = torch.stft(target, self.n_fft, self.hop_length, self.win_length, self.window, return_complex=True)
        pred_mag = pred.abs().clamp_min(self.eps)
        true_mag = true.abs().clamp_min(self.eps)
        c = self.compress_factor
        pred_c = pred / pred_mag.pow(1 - c)
        true_c = true / true_mag.pow(1 - c)
        ri = (pred_c.real - true_c.real).square().mean() + (pred_c.imag - true_c.imag).square().mean()
        mag = (pred_mag.pow(c) - true_mag.pow(c)).square().mean()
        projection = (
            (target * prediction).sum(-1, keepdim=True)
            * target
            / (target.square().sum(-1, keepdim=True) + 1e-8)
        )
        sisnr = -2 * torch.log10(
            projection.norm(dim=-1, keepdim=True)
            / (prediction - projection).norm(dim=-1, keepdim=True).clamp_min(self.eps)
            + self.eps
        ).mean()
        return 30 * ri + 70 * mag + sisnr
