import argparse
import queue
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

import matplotlib

matplotlib.use("TkAgg")

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
import sounddevice as sd
from rknn.api import RKNN


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_RKNN = ROOT_DIR / "deploy_rknn" / "gtcrn_dns3_stream_rk3568_fp16.rknn"
DEFAULT_ONNX = ROOT_DIR / "stream" / "onnx_models" / "gtcrn_simple.onnx"
INPUT_NAMES = ["mix", "conv_cache", "tra_cache", "inter_cache"]
INPUT_SHAPES = {
    "mix": (1, 257, 1, 2),
    "conv_cache": (2, 1, 16, 16, 33),
    "tra_cache": (2, 3, 1, 1, 16),
    "inter_cache": (2, 1, 33, 16),
}


class GTCRNRKNNStreamer:
    def __init__(self, rknn_path, onnx_path, target="rk3568", backend="auto"):
        self.rknn_path = Path(rknn_path)
        self.onnx_path = Path(onnx_path)
        self.target = target
        self.backend = backend
        self.rknn = RKNN(verbose=False)
        self.data_format = ["nchw"] * len(INPUT_NAMES)
        self.backend_name = ""
        self._load_runtime()
        self.reset()

    def _load_runtime(self):
        if self.backend in {"auto", "rknn"} and self.rknn_path.exists():
            ret = self.rknn.load_rknn(self.rknn_path.as_posix())
            if ret == 0:
                ret = self.rknn.init_runtime()
                if ret == 0:
                    self.backend_name = "rknn"
                    return
            if self.backend == "rknn":
                raise RuntimeError(f"load_rknn/init_runtime failed: {ret}")
            self.rknn.release()
            self.rknn = RKNN(verbose=False)

        if self.backend in {"auto", "sim"}:
            if not self.onnx_path.exists():
                raise FileNotFoundError(self.onnx_path)
            for label, ret in [
                ("config", self.rknn.config(target_platform=self.target)),
                ("load_onnx", self.rknn.load_onnx(model=self.onnx_path.as_posix())),
                ("build", self.rknn.build(do_quantization=False)),
                ("init_runtime", self.rknn.init_runtime()),
            ]:
                if ret != 0:
                    raise RuntimeError(f"{label} failed: {ret}")
            self.backend_name = "pc_sim"
            return

        raise RuntimeError("No usable RKNN runtime backend.")

    def reset(self):
        self.conv_cache = np.zeros(INPUT_SHAPES["conv_cache"], dtype=np.float32)
        self.tra_cache = np.zeros(INPUT_SHAPES["tra_cache"], dtype=np.float32)
        self.inter_cache = np.zeros(INPUT_SHAPES["inter_cache"], dtype=np.float32)

    def infer(self, spec_frame):
        inputs = [spec_frame, self.conv_cache, self.tra_cache, self.inter_cache]
        outputs = self.rknn.inference(inputs=inputs, data_format=self.data_format)
        self.conv_cache = outputs[1].astype(np.float32, copy=False)
        self.tra_cache = outputs[2].astype(np.float32, copy=False)
        self.inter_cache = outputs[3].astype(np.float32, copy=False)
        return outputs[0].astype(np.float32, copy=False)

    def release(self):
        self.rknn.release()


class RealtimeDenoiser:
    def __init__(self, streamer, sr=16000, n_fft=512, hop_length=256):
        self.streamer = streamer
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.window = np.sqrt(np.hanning(n_fft).astype(np.float32))
        self.reset()

    def reset(self):
        self.streamer.reset()
        self.sample_buffer = np.zeros(self.n_fft, dtype=np.float32)
        self.ola_signal = np.zeros(self.n_fft, dtype=np.float32)
        self.ola_weight = np.zeros(self.n_fft, dtype=np.float32)

    def process_hop(self, samples):
        self.sample_buffer[:-self.hop_length] = self.sample_buffer[self.hop_length :]
        self.sample_buffer[-self.hop_length :] = samples.astype(np.float32, copy=False)
        frame = self.sample_buffer * self.window
        spec = np.fft.rfft(frame, n=self.n_fft)
        spec_frame = np.stack([spec.real, spec.imag], axis=-1).astype(np.float32)
        spec_frame = spec_frame.reshape(1, self.n_fft // 2 + 1, 1, 2)

        out = self.streamer.infer(spec_frame)[0, :, 0, :]
        enhanced_spec = out[:, 0] + 1j * out[:, 1]
        enhanced_frame = np.fft.irfft(enhanced_spec, n=self.n_fft).astype(np.float32)
        enhanced_frame *= self.window

        self.ola_signal += enhanced_frame
        self.ola_weight += self.window * self.window
        hop = self.ola_signal[: self.hop_length] / np.maximum(
            self.ola_weight[: self.hop_length], 1e-6
        )
        self.ola_signal[:-self.hop_length] = self.ola_signal[self.hop_length :]
        self.ola_signal[-self.hop_length :] = 0
        self.ola_weight[:-self.hop_length] = self.ola_weight[self.hop_length :]
        self.ola_weight[-self.hop_length :] = 0
        return np.clip(hop, -1.0, 1.0).astype(np.float32)


class AudioEngine:
    def __init__(self, denoiser, status_callback=None):
        self.denoiser = denoiser
        self.status_callback = status_callback
        self.input_queue = queue.Queue(maxsize=8)
        self.output_queue = queue.Queue(maxsize=16)
        self.stream = None
        self.worker = None
        self.running = False
        self.last_latency_ms = 0.0
        self.input_meter = 0.0
        self.output_meter = 0.0
        self.last_error = ""

    def start(self):
        if self.running:
            return
        self.denoiser.reset()
        self.input_queue = queue.Queue(maxsize=8)
        self.output_queue = queue.Queue(maxsize=16)
        self.running = True
        self.worker = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker.start()
        self.stream = sd.Stream(
            samplerate=self.denoiser.sr,
            blocksize=self.denoiser.hop_length,
            channels=(1, 1),
            dtype="float32",
            latency="low",
            callback=self._audio_callback,
        )
        self.stream.start()

    def stop(self):
        self.running = False
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        self.worker = None

    def _audio_callback(self, indata, outdata, frames, _time_info, status):
        if status:
            self.last_error = str(status)
        samples = indata[:, 0].copy()
        self.input_meter = float(min(1.0, np.sqrt(np.mean(samples * samples) + 1e-12) * 10))
        try:
            self.input_queue.put_nowait(samples)
        except queue.Full:
            self.last_error = "input queue full"
        try:
            out = self.output_queue.get_nowait()
        except queue.Empty:
            out = np.zeros(frames, dtype=np.float32)
        if len(out) < frames:
            out = np.pad(out, (0, frames - len(out)))
        out = out[:frames].astype(np.float32, copy=False)
        self.output_meter = float(min(1.0, np.sqrt(np.mean(out * out) + 1e-12) * 10))
        outdata[:, 0] = out

    def _worker_loop(self):
        while self.running:
            try:
                samples = self.input_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            started = time.perf_counter()
            try:
                enhanced = self.denoiser.process_hop(samples)
                self.last_latency_ms = (time.perf_counter() - started) * 1000
                try:
                    self.output_queue.put_nowait(enhanced)
                except queue.Full:
                    self.last_error = "output queue full"
            except Exception as exc:
                self.last_error = repr(exc)
                if self.status_callback:
                    self.status_callback(self.last_error)


class App(tk.Tk):
    def __init__(self, args):
        super().__init__()
        self.title("GTCRN DNS3 RKNN Real-time Denoiser")
        self.geometry("760x460")
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.args = args
        self.engine = None
        self.streamer = None
        self.meter_history = np.zeros(120, dtype=np.float32)
        self._build_ui()
        self.after(100, self._load_model)

    def _build_ui(self):
        root = ttk.Frame(self, padding=12)
        root.pack(fill=tk.BOTH, expand=True)

        self.state_var = tk.StringVar(value="Loading RKNN runtime...")
        self.backend_var = tk.StringVar(value="backend: --")
        self.latency_var = tk.StringVar(value="latency: -- ms")
        self.error_var = tk.StringVar(value="")

        ttk.Label(root, text="GTCRN DNS3 RKNN Real-time Denoiser", font=("", 16, "bold")).pack(
            anchor=tk.W
        )
        ttk.Label(root, textvariable=self.state_var).pack(anchor=tk.W, pady=(6, 0))
        ttk.Label(root, textvariable=self.backend_var).pack(anchor=tk.W)
        ttk.Label(root, textvariable=self.latency_var).pack(anchor=tk.W)
        ttk.Label(root, textvariable=self.error_var, foreground="#b00020").pack(anchor=tk.W)

        actions = ttk.Frame(root)
        actions.pack(anchor=tk.W, pady=12)
        self.start_button = ttk.Button(actions, text="Start", command=self.start_audio, state=tk.DISABLED)
        self.stop_button = ttk.Button(actions, text="Stop", command=self.stop_audio, state=tk.DISABLED)
        self.start_button.pack(side=tk.LEFT, padx=(0, 8))
        self.stop_button.pack(side=tk.LEFT)

        self.fig = Figure(figsize=(7.2, 2.2), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_ylim(0, 1)
        self.ax.set_xlim(0, len(self.meter_history) - 1)
        self.ax.set_title("Output level")
        self.ax.set_yticks([0, 0.5, 1.0])
        self.line, = self.ax.plot(self.meter_history, color="#0b6efd", linewidth=2)
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, pady=(8, 0))

        hint = (
            "Uses system default microphone and speaker. Wear headphones to avoid feedback."
        )
        ttk.Label(root, text=hint).pack(anchor=tk.W, pady=(8, 0))

    def _load_model(self):
        try:
            self.streamer = GTCRNRKNNStreamer(
                self.args.rknn, self.args.onnx, target=self.args.target, backend=self.args.backend
            )
            denoiser = RealtimeDenoiser(self.streamer)
            self.engine = AudioEngine(denoiser, status_callback=self._set_error_from_thread)
            self.backend_var.set(f"backend: {self.streamer.backend_name}")
            self.state_var.set("Ready")
            self.start_button.configure(state=tk.NORMAL)
            self.after(100, self._refresh_ui)
        except Exception as exc:
            self.state_var.set("Failed to load RKNN runtime")
            self.error_var.set(repr(exc))
            messagebox.showerror("GTCRN RKNN", repr(exc))

    def start_audio(self):
        try:
            self.engine.start()
            self.state_var.set("Running")
            self.start_button.configure(state=tk.DISABLED)
            self.stop_button.configure(state=tk.NORMAL)
        except Exception as exc:
            self.error_var.set(repr(exc))
            messagebox.showerror("Audio start failed", repr(exc))

    def stop_audio(self):
        if self.engine:
            self.engine.stop()
        self.state_var.set("Stopped")
        self.start_button.configure(state=tk.NORMAL)
        self.stop_button.configure(state=tk.DISABLED)

    def _set_error_from_thread(self, text):
        self.after(0, lambda: self.error_var.set(text))

    def _refresh_ui(self):
        if self.engine:
            self.latency_var.set(f"latency: {self.engine.last_latency_ms:.2f} ms")
            if self.engine.last_error:
                self.error_var.set(self.engine.last_error)
            self.meter_history[:-1] = self.meter_history[1:]
            self.meter_history[-1] = self.engine.output_meter
            self.line.set_ydata(self.meter_history)
            self.canvas.draw_idle()
        self.after(100, self._refresh_ui)

    def on_close(self):
        if self.engine:
            self.engine.stop()
        if self.streamer:
            self.streamer.release()
        self.destroy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rknn", default=DEFAULT_RKNN.as_posix())
    parser.add_argument("--onnx", default=DEFAULT_ONNX.as_posix())
    parser.add_argument("--target", default="rk3568")
    parser.add_argument(
        "--backend",
        choices=["auto", "rknn", "sim"],
        default="auto",
        help="auto tries the exported RKNN first, then falls back to PC simulator.",
    )
    args = parser.parse_args()
    app = App(args)
    app.mainloop()


if __name__ == "__main__":
    main()
