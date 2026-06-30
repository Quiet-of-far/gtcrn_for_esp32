import argparse
import queue
import subprocess
import threading
import time

import numpy as np
from rknnlite.api import RKNNLite


SR = 16000
N_FFT = 512
HOP = 256
HOP_BYTES = HOP * 2


def percentile(values, q):
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


class GTCRNRKNN:
    def __init__(self, path):
        self.rknn = RKNNLite()
        ret = self.rknn.load_rknn(path)
        if ret != 0:
            raise RuntimeError(f"load_rknn failed: {ret}")
        ret = self.rknn.init_runtime()
        if ret != 0:
            raise RuntimeError(f"init_runtime failed: {ret}")
        self.reset()

    def reset(self):
        self.conv = np.zeros((2, 16, 16, 33), dtype=np.float32)
        self.tra = np.zeros((2, 3, 1, 16), dtype=np.float32)
        self.inter = np.zeros((2, 1, 33, 16), dtype=np.float32)

    def infer(self, mix):
        outs = self.rknn.inference(inputs=[mix, self.conv, self.tra, self.inter])
        if outs is None or len(outs) != 4:
            raise RuntimeError("RKNN inference returned no outputs")
        self.conv = np.ascontiguousarray(outs[1], dtype=np.float32)
        self.tra = np.ascontiguousarray(outs[2], dtype=np.float32)
        self.inter = np.ascontiguousarray(outs[3], dtype=np.float32)
        return np.ascontiguousarray(outs[0], dtype=np.float32)

    def release(self):
        self.rknn.release()


class Denoiser:
    def __init__(self, model):
        self.model = model
        self.window = np.sqrt(np.hanning(N_FFT).astype(np.float32))
        self.reset()

    def reset(self):
        self.model.reset()
        self.buf = np.zeros(N_FFT, dtype=np.float32)
        self.ola = np.zeros(N_FFT, dtype=np.float32)
        self.wsum = np.zeros(N_FFT, dtype=np.float32)

    def process(self, samples):
        self.buf[:-HOP] = self.buf[HOP:]
        self.buf[-HOP:] = samples.astype(np.float32, copy=False)
        spec = np.fft.rfft(self.buf * self.window, n=N_FFT)
        mix = np.stack([spec.real, spec.imag], axis=-1).astype(np.float32).reshape(
            1, N_FFT // 2 + 1, 1, 2
        )
        enh = self.model.infer(mix)[0, :, 0, :]
        frame = np.fft.irfft(enh[:, 0] + 1j * enh[:, 1], n=N_FFT).astype(np.float32)
        frame *= self.window
        self.ola += frame
        self.wsum += self.window * self.window
        hop = self.ola[:HOP] / np.maximum(self.wsum[:HOP], 1e-6)
        self.ola[:-HOP] = self.ola[HOP:]
        self.ola[-HOP:] = 0
        self.wsum[:-HOP] = self.wsum[HOP:]
        self.wsum[-HOP:] = 0
        return np.clip(hop, -1, 1).astype(np.float32)


def print_stats(label, times, dropped=None):
    print(
        "{} count={} avg={:.3f} p50={:.3f} p95={:.3f} p99={:.3f} max={:.3f}{}".format(
            label,
            len(times),
            float(np.mean(times)),
            percentile(times, 50),
            percentile(times, 95),
            percentile(times, 99),
            float(np.max(times)),
            "" if dropped is None else f" dropped={dropped}",
        ),
        flush=True,
    )


def inference_benchmark(path, warmup, frames):
    model = GTCRNRKNN(path)
    rng = np.random.default_rng(0)
    try:
        for _ in range(warmup):
            mix = rng.normal(0.0, 1.0, (1, 257, 1, 2)).astype(np.float32)
            model.infer(mix)
        times = []
        for _ in range(frames):
            mix = rng.normal(0.0, 1.0, (1, 257, 1, 2)).astype(np.float32)
            started = time.perf_counter()
            model.infer(mix)
            times.append((time.perf_counter() - started) * 1000)
        print_stats("inference_ms", times)
    finally:
        model.release()


def stft_benchmark(path, warmup, frames):
    model = GTCRNRKNN(path)
    denoiser = Denoiser(model)
    rng = np.random.default_rng(1)
    try:
        for _ in range(warmup):
            denoiser.process(rng.normal(0.0, 0.05, HOP).astype(np.float32))
        times = []
        for _ in range(frames):
            samples = rng.normal(0.0, 0.05, HOP).astype(np.float32)
            started = time.perf_counter()
            denoiser.process(samples)
            times.append((time.perf_counter() - started) * 1000)
        print_stats("stft_model_istft_ms", times)
    finally:
        model.release()


def read_exact(pipe, n):
    chunks = []
    total = 0
    while total < n:
        data = pipe.read(n - total)
        if not data:
            raise RuntimeError("arecord stopped")
        chunks.append(data)
        total += len(data)
    return b"".join(chunks)


def audio_benchmark(path, seconds):
    model = GTCRNRKNN(path)
    denoiser = Denoiser(model)
    inq = queue.Queue(maxsize=3)
    outq = queue.Queue(maxsize=8)
    running = True
    dropped = 0
    times = []
    errors = []
    arecord = subprocess.Popen(
        [
            "arecord",
            "-q",
            "-D",
            "default",
            "-t",
            "raw",
            "-f",
            "S16_LE",
            "-r",
            str(SR),
            "-c",
            "1",
            "--buffer-size",
            str(HOP * 16),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    aplay = subprocess.Popen(
        [
            "aplay",
            "-q",
            "-D",
            "default",
            "-t",
            "raw",
            "-f",
            "S16_LE",
            "-r",
            str(SR),
            "-c",
            "1",
            "--buffer-size",
            str(HOP * 16),
        ],
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    def capture():
        nonlocal dropped, running
        try:
            while running:
                raw = read_exact(arecord.stdout, HOP_BYTES)
                samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
                if inq.full():
                    inq.get_nowait()
                    dropped += 1
                inq.put_nowait(samples)
        except Exception as exc:
            if running:
                errors.append(repr(exc))
                running = False

    def process():
        nonlocal dropped, running
        try:
            while running:
                try:
                    samples = inq.get(timeout=0.2)
                except queue.Empty:
                    continue
                started = time.perf_counter()
                y = denoiser.process(samples)
                times.append((time.perf_counter() - started) * 1000)
                if outq.full():
                    outq.get_nowait()
                    dropped += 1
                outq.put_nowait(y)
        except Exception as exc:
            if running:
                errors.append(repr(exc))
                running = False

    def play():
        nonlocal running
        silence = np.zeros(HOP, dtype=np.float32)
        try:
            while running:
                try:
                    y = outq.get(timeout=HOP / SR)
                except queue.Empty:
                    y = silence
                pcm = (np.clip(y, -1, 1) * 32767).astype("<i2").tobytes()
                aplay.stdin.write(pcm)
                aplay.stdin.flush()
        except Exception as exc:
            if running:
                errors.append(repr(exc))
                running = False

    threads = [threading.Thread(target=fn, daemon=True) for fn in (capture, process, play)]
    for thread in threads:
        thread.start()
    deadline = time.time() + seconds
    try:
        while running and time.time() < deadline:
            time.sleep(0.1)
    finally:
        running = False
        for pipe in (arecord.stdout, arecord.stderr, aplay.stdin, aplay.stderr):
            try:
                pipe.close()
            except Exception:
                pass
        for proc in (arecord, aplay):
            try:
                proc.terminate()
            except Exception:
                pass
        for proc in (arecord, aplay):
            try:
                proc.wait(timeout=1.0)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        for thread in threads:
            thread.join(timeout=0.5)
        model.release()
    if errors:
        print("audio_errors", errors, flush=True)
        raise SystemExit(1)
    print_stats("audio_process_ms", times, dropped=dropped)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--mode", choices=["inference", "stft", "audio"], default="inference")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--frames", type=int, default=1000)
    parser.add_argument("--seconds", type=float, default=60.0)
    args = parser.parse_args()

    if args.mode == "inference":
        inference_benchmark(args.model, args.warmup, args.frames)
    elif args.mode == "stft":
        stft_benchmark(args.model, args.warmup, args.frames)
    else:
        audio_benchmark(args.model, args.seconds)


if __name__ == "__main__":
    main()
