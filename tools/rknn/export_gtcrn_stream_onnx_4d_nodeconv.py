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
from gtcrn_stream import (
    DPGRNN,
    ERB,
    Mask,
    SFE,
    StreamEncoder,
    StreamGTConvBlock,
)
from modules.convert import convert_to_stream


def deconv_weight_to_conv2d(weight, groups):
    """Convert ConvTranspose2d weight to equivalent zero-insert + Conv2d weight."""
    in_channels, out_per_group, _, _ = weight.shape
    in_per_group = in_channels // groups
    chunks = []
    for group_idx in range(groups):
        part = weight[
            group_idx * in_per_group : (group_idx + 1) * in_per_group
        ]
        chunks.append(part.permute(1, 0, 2, 3))
    return torch.flip(torch.cat(chunks, dim=0), dims=[-2, -1])


class ConvTransposeAsConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation=1, groups=1, bias=True):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)
        if isinstance(dilation, int):
            dilation = (dilation, dilation)
        self.t_stride, self.f_stride = stride
        self.t_pad, self.f_pad = padding
        self.t_dilation, self.f_dilation = dilation
        self.t_size, self.f_size = kernel_size
        if self.t_stride != 1 or self.t_pad != 0 or self.t_size != 1:
            raise ValueError("Only the GTCRN frequency upsampling ConvTranspose2d is supported.")
        self.groups = groups
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=(1, 1),
            padding=(0, 0),
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, x):
        if self.f_stride > 1:
            b, c, t, f = x.shape
            zeros = x.new_zeros(b, c, t, f, self.f_stride - 1)
            x = torch.cat([x[:, :, :, :, None], zeros], dim=-1).reshape(b, c, t, -1)
            left_pad = self.f_stride - 1
            base_pad = (self.f_size - 1) * self.f_dilation - self.f_pad
            if left_pad > self.f_size - 1:
                raise RuntimeError("Unsupported frequency stride/kernel combination.")
            x = torch.nn.functional.pad(x, [base_pad, base_pad - left_pad, 0, 0])
        else:
            base_pad = (self.f_size - 1) * self.f_dilation - self.f_pad
            x = torch.nn.functional.pad(x, [base_pad, base_pad, 0, 0])
        return self.conv(x)


class ConvBlockNoDeconv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups=1, is_last=False):
        super().__init__()
        self.conv = ConvTransposeAsConv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            groups=groups,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.Tanh() if is_last else nn.PReLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class StreamGTConvBlockNoPointDeconv(StreamGTConvBlock):
    def __init__(self, in_channels, hidden_channels, kernel_size, stride, padding, dilation):
        super().__init__(
            in_channels,
            hidden_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            use_deconv=True,
        )
        self.point_conv1 = nn.Conv2d(in_channels // 2 * 3, hidden_channels, 1)
        self.point_conv2 = nn.Conv2d(hidden_channels, in_channels // 2, 1)


class StreamDecoderNoDeconv(nn.Module):
    def __init__(self):
        super().__init__()
        self.de_convs = nn.ModuleList(
            [
                StreamGTConvBlockNoPointDeconv(16, 16, (3, 3), stride=(1, 1), padding=(0, 1), dilation=(5, 1)),
                StreamGTConvBlockNoPointDeconv(16, 16, (3, 3), stride=(1, 1), padding=(0, 1), dilation=(2, 1)),
                StreamGTConvBlockNoPointDeconv(16, 16, (3, 3), stride=(1, 1), padding=(0, 1), dilation=(1, 1)),
                ConvBlockNoDeconv(16, 16, (1, 5), stride=(1, 2), padding=(0, 2), groups=2, is_last=False),
                ConvBlockNoDeconv(16, 2, (1, 5), stride=(1, 2), padding=(0, 2), groups=1, is_last=True),
            ]
        )

    def forward(self, x, en_outs, conv_cache, tra_cache):
        x, conv_cache[:, :, 6:16, :], tra_cache[0] = self.de_convs[0](
            x + en_outs[4], conv_cache[:, :, 6:16, :], tra_cache[0]
        )
        x, conv_cache[:, :, 2:6, :], tra_cache[1] = self.de_convs[1](
            x + en_outs[3], conv_cache[:, :, 2:6, :], tra_cache[1]
        )
        x, conv_cache[:, :, :2, :], tra_cache[2] = self.de_convs[2](
            x + en_outs[2], conv_cache[:, :, :2, :], tra_cache[2]
        )
        for i in range(3, 5):
            x = self.de_convs[i](x + en_outs[4 - i])
        return x, conv_cache, tra_cache


class StreamGTCRNNoDeconv(nn.Module):
    def __init__(self):
        super().__init__()
        self.erb = ERB(65, 64)
        self.sfe = SFE(3, 1)
        self.encoder = StreamEncoder()
        self.dpgrnn1 = DPGRNN(16, 33, 16)
        self.dpgrnn2 = DPGRNN(16, 33, 16)
        self.decoder = StreamDecoderNoDeconv()
        self.mask = Mask()

    def forward(self, spec, conv_cache, tra_cache, inter_cache):
        spec_ref = spec
        spec_real = spec[..., 0].permute(0, 2, 1)
        spec_imag = spec[..., 1].permute(0, 2, 1)
        spec_mag = torch.sqrt(spec_real**2 + spec_imag**2 + 1e-12)
        feat = torch.stack([spec_mag, spec_real, spec_imag], dim=1)
        feat = self.erb.bm(feat)
        feat = self.sfe(feat)
        feat, en_outs, conv_cache[0], tra_cache[0] = self.encoder(
            feat, conv_cache[0], tra_cache[0]
        )
        feat, inter_cache[0] = self.dpgrnn1(feat, inter_cache[0])
        feat, inter_cache[1] = self.dpgrnn2(feat, inter_cache[1])
        m_feat, conv_cache[1], tra_cache[1] = self.decoder(
            feat, en_outs, conv_cache[1], tra_cache[1]
        )
        m = self.erb.bs(m_feat)
        spec_enh = self.mask(m, spec_ref.permute(0, 3, 2, 1))
        spec_enh = spec_enh.permute(0, 3, 2, 1)
        return spec_enh, conv_cache, tra_cache, inter_cache


TRANSPOSE_EQUIV_KEYS = {
    "decoder.de_convs.0.point_conv1.weight": 1,
    "decoder.de_convs.0.point_conv2.weight": 1,
    "decoder.de_convs.1.point_conv1.weight": 1,
    "decoder.de_convs.1.point_conv2.weight": 1,
    "decoder.de_convs.2.point_conv1.weight": 1,
    "decoder.de_convs.2.point_conv2.weight": 1,
    "decoder.de_convs.3.conv.conv.weight": 2,
    "decoder.de_convs.4.conv.conv.weight": 1,
}


def load_no_deconv_state(no_deconv_model, original_stream_model):
    src = original_stream_model.state_dict()
    dst = no_deconv_model.state_dict()
    converted = {}
    for key, value in dst.items():
        src_key = key.replace(".conv.conv.", ".conv.")
        if key in TRANSPOSE_EQUIV_KEYS:
            converted[key] = deconv_weight_to_conv2d(src[src_key], TRANSPOSE_EQUIV_KEYS[key])
        elif src_key in src:
            converted[key] = src[src_key]
        else:
            raise KeyError(f"missing source parameter for {key}")
        if converted[key].shape != value.shape:
            raise ValueError(f"shape mismatch for {key}: {converted[key].shape} != {value.shape}")
    no_deconv_model.load_state_dict(converted)


class StreamGTCRN4DWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, mix, conv_cache, tra_cache, inter_cache):
        enh, conv_out, tra_out, inter_out = self.model(
            mix,
            conv_cache.reshape(2, 1, 16, 16, 33),
            tra_cache.reshape(2, 3, 1, 1, 16),
            inter_cache,
        )
        return (
            enh,
            conv_out.reshape(2, 16, 16, 33),
            tra_out.reshape(2, 3, 1, 16),
            inter_out,
        )


def make_models(checkpoint_path, device):
    offline_model = GTCRN().to(device).eval()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    offline_model.load_state_dict(
        {key.replace("module.", ""): value for key, value in checkpoint["model"].items()}
    )

    from gtcrn_stream import StreamGTCRN

    original_stream = StreamGTCRN().to(device).eval()
    convert_to_stream(original_stream, offline_model)

    no_deconv = StreamGTCRNNoDeconv().to(device).eval()
    load_no_deconv_state(no_deconv, original_stream)
    return original_stream, no_deconv


def compare_torch(original_stream, no_deconv, repeats):
    rng = torch.Generator().manual_seed(0)
    conv_a = torch.zeros(2, 1, 16, 16, 33)
    tra_a = torch.zeros(2, 3, 1, 1, 16)
    inter_a = torch.zeros(2, 1, 33, 16)
    conv_b = conv_a.clone()
    tra_b = tra_a.clone()
    inter_b = inter_a.clone()
    max_abs = 0.0
    mean_abs = 0.0
    with torch.no_grad():
        for _ in range(repeats):
            mix = torch.randn(1, 257, 1, 2, generator=rng)
            out_a = original_stream(mix, conv_a, tra_a, inter_a)
            out_b = no_deconv(mix, conv_b, tra_b, inter_b)
            for a, b in zip(out_a, out_b):
                diff = (a - b).abs()
                max_abs = max(max_abs, float(diff.max()))
                mean_abs = max(mean_abs, float(diff.mean()))
            _, conv_a, tra_a, inter_a = out_a
            _, conv_b, tra_b, inter_b = out_b
    print(f"torch_compare max_abs={max_abs:.8g} max_mean_abs={mean_abs:.8g}")
    return max_abs, mean_abs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default=(ROOT_DIR / "checkpoints" / "model_trained_on_dns3.tar").as_posix(),
    )
    parser.add_argument(
        "--output",
        default=(STREAM_DIR / "onnx_models" / "gtcrn_simple_4d_no_deconv.onnx").as_posix(),
    )
    parser.add_argument("--raw-output", default=None)
    parser.add_argument("--no-simplify", action="store_true")
    parser.add_argument("--compare-repeats", type=int, default=16)
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    raw_output = Path(args.raw_output) if args.raw_output else output.with_name("gtcrn_4d_no_deconv.onnx")

    device = torch.device("cpu")
    original_stream, no_deconv = make_models(args.checkpoint, device)
    max_abs, mean_abs = compare_torch(original_stream, no_deconv, args.compare_repeats)
    if max_abs > 1e-4 or mean_abs > 1e-5:
        raise RuntimeError("no-deconv model is not numerically equivalent enough to export")

    wrapper = StreamGTCRN4DWrapper(no_deconv).to(device).eval()
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
