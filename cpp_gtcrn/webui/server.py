import argparse
import audioop
import cgi
import json
import os
import re
import signal
import subprocess
import threading
import time
import uuid
import wave
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RUNS_DIR = ROOT / "runs"
OFFLINE_DIR = ROOT / "offline"
DEFAULT_BINARY = "/home/firefly/gtcrn_cpp_float32/cpp_gtcrn/build/realtime_gtcrn_alsa"
DEFAULT_OFFLINE_BINARY = "/home/firefly/gtcrn_cpp_float32/cpp_gtcrn/build/offline_gtcrn"
DEFAULT_FINETUNED_BINARY = "/home/firefly/gtcrn_cpp_float32/cpp_gtcrn/build_finetuned/realtime_gtcrn_alsa"
DEFAULT_FINETUNED_OFFLINE_BINARY = "/home/firefly/gtcrn_cpp_float32/cpp_gtcrn/build_finetuned/offline_gtcrn"


def list_alsa_devices(mode):
    if mode not in ("capture", "playback"):
        raise ValueError(f"unknown ALSA mode: {mode}")
    command = "arecord" if mode == "capture" else "aplay"
    proc = subprocess.run(
        [command, "-l"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=5,
    )
    output = "\n".join((proc.stdout, proc.stderr))
    devices = [
        {"id": "default", "label": "系统默认 (PulseAudio)"},
        {"id": "pulse", "label": "PulseAudio"},
    ]
    pattern = re.compile(
        r"^card\s+(\d+):\s+([^\s]+)\s+\[(.*?)\],\s+"
        r"device\s+(\d+):\s+([^\[]+?)\s+\[(.*?)\]\s*$"
    )
    for line in output.splitlines():
        match = pattern.match(line.strip())
        if not match:
            continue
        card_index, card_id, card_label, device_index, device_name, _ = match.groups()
        devices.append({
            "id": f"plughw:CARD={card_id},DEV={device_index}",
            "label": f"{card_label} - {device_name.strip()} (card {card_index}, device {device_index})",
        })
    return devices


def web_url(path):
    return "/" + path.relative_to(ROOT).as_posix()


def parse_kv_stats(text):
    result = {}
    for line in text.splitlines():
        for part in line.split():
            if "=" not in part:
                continue
            key, value = part.split("=", 1)
            try:
                result[key] = float(value)
            except ValueError:
                result[key] = value
    return result


def convert_wav_to_16k_mono(src, dst):
    with wave.open(src.as_posix(), "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        frames = wav.readframes(wav.getnframes())

    if sample_width == 1:
        frames = audioop.bias(frames, 1, -128)
        frames = audioop.lin2lin(frames, 1, 2)
        sample_width = 2
    elif sample_width in (3, 4):
        frames = audioop.lin2lin(frames, sample_width, 2)
        sample_width = 2
    elif sample_width != 2:
        raise RuntimeError(f"unsupported WAV sample width: {sample_width}")

    if channels == 2:
        frames = audioop.tomono(frames, sample_width, 0.5, 0.5)
        channels = 1
    elif channels != 1:
        raise RuntimeError(f"unsupported WAV channel count: {channels}")

    if sample_rate != 16000:
        frames, _ = audioop.ratecv(frames, sample_width, channels, sample_rate, 16000, None)
        sample_rate = 16000

    with wave.open(dst.as_posix(), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(frames)


class RealtimeController:
    def __init__(self, binary, capture, playback, periods, prefill_periods, model_id="dns3", model_label=""):
        self.binary = Path(binary)
        self.capture = capture
        self.playback = playback
        self.periods = periods
        self.prefill_periods = prefill_periods
        self.model_id = model_id
        self.model_label = model_label or model_id
        self.proc = None
        self.reader = None
        self.lock = threading.RLock()
        self.last_status = self._base_status()
        self.last_error = ""
        self.last_stdout = ""
        self.recording = False
        self.record_dir = None
        self.input_url = ""
        self.output_url = ""

    def _base_status(self):
        return {
            "running": False,
            "binary": self.binary.as_posix(),
            "capture": self.capture,
            "playback": self.playback,
            "periods": self.periods,
            "prefill_periods": self.prefill_periods,
            "model_id": self.model_id,
            "model": self.model_label,
            "recording": False,
            "input_url": "",
            "output_url": "",
        }

    def _cmd(self, extra=None, record_dir=None):
        cmd = [
            self.binary.as_posix(),
            "--capture",
            self.capture,
            "--playback",
            self.playback,
            "--periods",
            str(self.periods),
            "--prefill-periods",
            str(self.prefill_periods),
        ]
        if record_dir is not None:
            cmd.extend(["--record-dir", Path(record_dir).as_posix()])
        if extra:
            cmd.extend(extra)
        return cmd

    def _record_urls(self):
        if self.record_dir is None:
            return "", ""
        return web_url(self.record_dir / "realtime_input.wav"), web_url(self.record_dir / "realtime_output.wav")

    def _new_record_dir(self):
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        path = RUNS_DIR / f"{stamp}-{uuid.uuid4().hex[:8]}"
        path.mkdir(parents=True, exist_ok=False)
        return path

    def _read_loop(self):
        assert self.proc is not None
        for line in self.proc.stdout:
            text = line.strip()
            if not text:
                continue
            with self.lock:
                self.last_stdout = text[-2000:]
            if text.startswith("{"):
                try:
                    payload = json.loads(text)
                except json.JSONDecodeError:
                    continue
                with self.lock:
                    payload.update(self._base_status())
                    payload["running"] = self.proc.poll() is None
                    payload["error"] = self.last_error
                    payload["recording"] = self.recording
                    payload["input_url"] = self.input_url
                    payload["output_url"] = self.output_url
                    self.last_status = payload
        rc = self.proc.poll()
        stderr = self.proc.stderr.read() if self.proc.stderr else ""
        with self.lock:
            self.last_error = stderr[-2000:] if rc not in (0, None, -signal.SIGTERM) else ""
            self.last_status["running"] = False
            self.last_status["returncode"] = rc
            self.last_status["error"] = self.last_error
            self.last_status["recording"] = self.recording
            self.last_status["input_url"] = self.input_url
            self.last_status["output_url"] = self.output_url

    def start(self, record_dir=None):
        with self.lock:
            if self.proc is not None and self.proc.poll() is None:
                return self.status()
            if not self.binary.exists():
                raise FileNotFoundError(self.binary)
            self.last_error = ""
            self.last_stdout = ""
            self.last_status = self._base_status()
            self.last_status["running"] = True
            self.recording = record_dir is not None
            self.record_dir = Path(record_dir) if record_dir is not None else None
            self.input_url, self.output_url = self._record_urls()
            self.last_status["recording"] = self.recording
            self.last_status["input_url"] = self.input_url
            self.last_status["output_url"] = self.output_url
            self.proc = subprocess.Popen(
                self._cmd(["--status-json", "--quiet"], record_dir=record_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
            self.reader = threading.Thread(target=self._read_loop, daemon=True)
            self.reader.start()
            return self.status()

    def select_model(self, model_id, binary, model_label):
        with self.lock:
            if self.proc is not None and self.proc.poll() is None:
                raise RuntimeError("请先停止实时降噪再切换模型")
            binary = Path(binary)
            if not binary.exists():
                raise FileNotFoundError(binary)
            self.binary = binary
            self.model_id = model_id
            self.model_label = model_label
            self.last_status = self._base_status()
            return self.status()

    def select_audio_devices(self, capture, playback):
        with self.lock:
            if self.proc is not None and self.proc.poll() is None:
                raise RuntimeError("请先停止实时降噪再切换音频设备")
            if not capture or not playback:
                raise ValueError("麦克风和扬声器设备不能为空")
            self.capture = capture
            self.playback = playback
            self.last_status = self._base_status()
            return self.status()

    def stop(self):
        proc = None
        with self.lock:
            proc = self.proc
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=3)
        with self.lock:
            self.last_status["running"] = False
            self.recording = False
            self.last_status["recording"] = False
        return self.status()

    def start_recording(self):
        if self.status().get("running", False):
            self.stop()
            time.sleep(0.2)
        record_dir = self._new_record_dir()
        return self.start(record_dir=record_dir)

    def stop_recording(self):
        if not self.status().get("recording", False):
            raise RuntimeError("recording is not running")
        self.stop()
        with self.lock:
            self.recording = False
            self.last_status["recording"] = False
            self.last_status["input_url"] = self.input_url
            self.last_status["output_url"] = self.output_url
        return self.status()

    def status(self):
        with self.lock:
            status = dict(self.last_status)
            status["error"] = self.last_error
            status["last_stdout"] = self.last_stdout
            status["recording"] = self.recording
            status["input_url"] = self.input_url
            status["output_url"] = self.output_url
            if self.proc is not None:
                status["running"] = self.proc.poll() is None
                status["pid"] = self.proc.pid
            return status

    def latency_test(self):
        was_running = self.status().get("running", False)
        if was_running:
            self.stop()
            time.sleep(0.3)
        proc = subprocess.run(
            self._cmd(["--latency-test", "--seconds", "6", "--quiet"]),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=15,
        )
        result = {
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr[-2000:],
        }
        for line in proc.stdout.splitlines():
            if "latency_ms=" in line:
                for part in line.split():
                    if "=" not in part:
                        continue
                    key, value = part.split("=", 1)
                    try:
                        result[key] = float(value)
                    except ValueError:
                        result[key] = value
        if was_running:
            self.start()
        return result


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=ROOT.as_posix(), **kwargs)

    def end_headers(self):
        if self.path == "/" or self.path.split("?", 1)[0].endswith((".html", ".js", ".css")):
            self.send_header("Cache-Control", "no-store, max-age=0")
        super().end_headers()

    def _json(self, code, payload):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/api/status":
            self._json(200, self.server.ctrl.status())
            return
        if self.path == "/api/config":
            self._json(200, self.server.ctrl._base_status())
            return
        if self.path == "/api/models":
            self._json(200, {"models": self.server.public_models()})
            return
        if self.path == "/api/audio-devices":
            self._json(200, self.server.audio_devices())
            return
        if self.path == "/":
            self.path = "/index.html"
        return super().do_GET()

    def do_POST(self):
        try:
            if self.path == "/api/start":
                self._json(200, self.server.ctrl.start())
                return
            if self.path == "/api/stop":
                self._json(200, self.server.ctrl.stop())
                return
            if self.path == "/api/model/select":
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length) or b"{}")
                self._json(200, self.server.select_model(payload.get("model_id", "dns3_finetuned")))
                return
            if self.path == "/api/audio-devices/select":
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length) or b"{}")
                self._json(200, self.server.select_audio_devices(
                    payload.get("capture", ""), payload.get("playback", "")
                ))
                return
            if self.path == "/api/record/start":
                self._json(200, self.server.ctrl.start_recording())
                return
            if self.path == "/api/record/stop":
                self._json(200, self.server.ctrl.stop_recording())
                return
            if self.path == "/api/latency-test":
                self._json(200, self.server.ctrl.latency_test())
                return
            if self.path == "/api/offline/process":
                self._json(200, self.server.process_offline_upload(self))
                return
        except Exception as exc:
            self._json(500, {"error": repr(exc)})
            return
        self.send_error(404)


class Server(ThreadingHTTPServer):
    def __init__(self, address, handler, ctrl, models):
        super().__init__(address, handler)
        self.ctrl = ctrl
        self.models = models
        self.offline_binary = Path(models[ctrl.model_id]["offline_binary"])
        self.offline_lock = threading.Lock()

    def public_models(self):
        return [
            {"id": model_id, "label": spec["label"]}
            for model_id, spec in self.models.items()
        ]

    def select_model(self, model_id):
        if model_id not in self.models:
            raise ValueError(f"unknown model: {model_id}")
        spec = self.models[model_id]
        status = self.ctrl.select_model(model_id, spec["binary"], spec["label"])
        self.offline_binary = Path(spec["offline_binary"])
        return status

    def audio_devices(self):
        capture = list_alsa_devices("capture")
        playback = list_alsa_devices("playback")
        if self.ctrl.capture not in {item["id"] for item in capture}:
            capture.append({"id": self.ctrl.capture, "label": f"当前配置 ({self.ctrl.capture})"})
        if self.ctrl.playback not in {item["id"] for item in playback}:
            playback.append({"id": self.ctrl.playback, "label": f"当前配置 ({self.ctrl.playback})"})
        return {
            "capture": capture,
            "playback": playback,
            "selected_capture": self.ctrl.capture,
            "selected_playback": self.ctrl.playback,
        }

    def select_audio_devices(self, capture, playback):
        available = self.audio_devices()
        capture_ids = {item["id"] for item in available["capture"]}
        playback_ids = {item["id"] for item in available["playback"]}
        if capture not in capture_ids:
            raise ValueError(f"不可用的麦克风设备: {capture}")
        if playback not in playback_ids:
            raise ValueError(f"不可用的扬声器设备: {playback}")
        return self.ctrl.select_audio_devices(capture, playback)

    def process_offline_upload(self, handler):
        if self.ctrl.status().get("running", False):
            raise RuntimeError("请先停止实时降噪，再处理离线上传音频")
        if not self.offline_binary.exists():
            raise FileNotFoundError(self.offline_binary)
        content_type = handler.headers.get("Content-Type", "")
        if not content_type.startswith("multipart/form-data"):
            raise RuntimeError("expected multipart/form-data")

        form = cgi.FieldStorage(
            fp=handler.rfile,
            headers=handler.headers,
            environ={
                "REQUEST_METHOD": "POST",
                "CONTENT_TYPE": content_type,
                "CONTENT_LENGTH": handler.headers.get("Content-Length", "0"),
            },
        )
        file_item = form["audio"] if "audio" in form else None
        if file_item is None or not getattr(file_item, "file", None):
            raise RuntimeError("missing audio file")

        with self.offline_lock:
            OFFLINE_DIR.mkdir(parents=True, exist_ok=True)
            stamp = time.strftime("%Y%m%d-%H%M%S")
            work_dir = OFFLINE_DIR / f"{stamp}-{uuid.uuid4().hex[:8]}"
            work_dir.mkdir(parents=True, exist_ok=False)
            upload_path = work_dir / "upload.wav"
            input_path = work_dir / "input_16k_mono.wav"
            output_path = work_dir / "enhanced_stream.wav"

            with upload_path.open("wb") as out:
                while True:
                    chunk = file_item.file.read(1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)

            convert_wav_to_16k_mono(upload_path, input_path)
            proc = subprocess.run(
                [
                    self.offline_binary.as_posix(),
                    "--input",
                    input_path.as_posix(),
                    "--output",
                    output_path.as_posix(),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=300,
            )
            if proc.returncode != 0:
                raise RuntimeError(proc.stderr[-2000:] or proc.stdout[-2000:] or "offline processing failed")

            payload = {
                "returncode": proc.returncode,
                "stdout": proc.stdout.strip(),
                "stderr": proc.stderr.strip(),
                "input_url": web_url(input_path),
                "output_url": web_url(output_path),
                "upload_url": web_url(upload_path),
            }
            payload.update(parse_kv_stats(proc.stdout))
            return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=os.environ.get("HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "7860")))
    parser.add_argument("--binary", default=os.environ.get("GTCRN_BINARY", DEFAULT_BINARY))
    parser.add_argument("--offline-binary", default=os.environ.get("GTCRN_OFFLINE_BINARY", DEFAULT_OFFLINE_BINARY))
    parser.add_argument("--finetuned-binary", default=os.environ.get("GTCRN_FINETUNED_BINARY", DEFAULT_FINETUNED_BINARY))
    parser.add_argument("--finetuned-offline-binary", default=os.environ.get("GTCRN_FINETUNED_OFFLINE_BINARY", DEFAULT_FINETUNED_OFFLINE_BINARY))
    parser.add_argument("--capture", default=os.environ.get("GTCRN_CAPTURE", "plughw:0,0"))
    parser.add_argument("--playback", default=os.environ.get("GTCRN_PLAYBACK", "plughw:0,0"))
    parser.add_argument("--periods", type=int, default=int(os.environ.get("GTCRN_PERIODS", "3")))
    parser.add_argument(
        "--prefill-periods",
        type=int,
        default=int(os.environ.get("GTCRN_PREFILL_PERIODS", "2")),
    )
    args = parser.parse_args()
    models = {
        "dns3": {
            "label": "GTCRN DNS3 原版 C++ float32",
            "binary": args.binary,
            "offline_binary": args.offline_binary,
        },
        "dns3_finetuned": {
            "label": "GTCRN DNS3 微调版 C++ float32",
            "binary": args.finetuned_binary,
            "offline_binary": args.finetuned_offline_binary,
        },
    }
    default_id = "dns3"
    default_spec = models[default_id]
    ctrl = RealtimeController(
        default_spec["binary"], args.capture, args.playback, args.periods,
        args.prefill_periods, default_id, default_spec["label"]
    )
    server = Server((args.host, args.port), Handler, ctrl, models)
    print(f"GTCRN C++ WebUI: http://{args.host}:{args.port}", flush=True)
    print(f"binary={args.binary}", flush=True)
    print(f"offline_binary={args.offline_binary}", flush=True)
    print(f"finetuned_binary={args.finetuned_binary}", flush=True)
    print(f"finetuned_offline_binary={args.finetuned_offline_binary}", flush=True)
    try:
        server.serve_forever()
    finally:
        ctrl.stop()


if __name__ == "__main__":
    main()
