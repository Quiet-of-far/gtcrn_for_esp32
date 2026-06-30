import torch
import torch.nn as nn

from gtcrn import GTCRN


class EndToEndGTCRN(nn.Module):
    def __init__(self, n_fft=512, hop_length=256, win_length=512):
        super().__init__()
        self.model = GTCRN()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.register_buffer("window", torch.hann_window(win_length).sqrt(), persistent=False)

    def forward(self, waveform):
        length = waveform.shape[-1]
        spec = torch.stft(
            waveform,
            self.n_fft,
            self.hop_length,
            self.win_length,
            self.window,
            return_complex=True,
        )
        enhanced = self.model(torch.view_as_real(spec))
        enhanced = torch.view_as_complex(enhanced.contiguous())
        return torch.istft(
            enhanced,
            self.n_fft,
            self.hop_length,
            self.win_length,
            self.window,
            length=length,
        )


def normalize_state_dict(state):
    return {key.removeprefix("module.").removeprefix("model."): value for key, value in state.items()}


def load_model_weights(model, checkpoint_path, map_location="cpu"):
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    state = checkpoint["model"] if "model" in checkpoint else checkpoint
    model.model.load_state_dict(normalize_state_dict(state), strict=True)
    return checkpoint
