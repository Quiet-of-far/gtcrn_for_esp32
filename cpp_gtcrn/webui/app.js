const stateEl = document.querySelector("#state");
const modelSelect = document.querySelector("#modelSelect");
const captureSelect = document.querySelector("#captureSelect");
const playbackSelect = document.querySelector("#playbackSelect");
const applyDevicesBtn = document.querySelector("#applyDevicesBtn");
const refreshDevicesBtn = document.querySelector("#refreshDevicesBtn");
const model = document.querySelector("#model");
const capture = document.querySelector("#capture");
const playback = document.querySelector("#playback");
const buffer = document.querySelector("#buffer");
const frames = document.querySelector("#frames");
const log = document.querySelector("#log");
const startBtn = document.querySelector("#startBtn");
const stopBtn = document.querySelector("#stopBtn");
const recordStartBtn = document.querySelector("#recordStartBtn");
const recordStopBtn = document.querySelector("#recordStopBtn");
const recordState = document.querySelector("#recordState");
const inputAudio = document.querySelector("#inputAudio");
const outputAudio = document.querySelector("#outputAudio");
const inputDownload = document.querySelector("#inputDownload");
const outputDownload = document.querySelector("#outputDownload");
const offlineForm = document.querySelector("#offlineForm");
const offlineFile = document.querySelector("#offlineFile");
const offlineBtn = document.querySelector("#offlineBtn");
const offlineState = document.querySelector("#offlineState");
const offlineInputAudio = document.querySelector("#offlineInputAudio");
const offlineOutputAudio = document.querySelector("#offlineOutputAudio");
const offlineInputDownload = document.querySelector("#offlineInputDownload");
const offlineOutputDownload = document.querySelector("#offlineOutputDownload");

function fmt(value, suffix = "") {
  if (value === undefined || value === null || Number.isNaN(Number(value))) return "--";
  return `${Number(value).toFixed(2)}${suffix}`;
}

function latencyText(status) {
  if (status.error) return status.error;
  if (status.frames === undefined || status.frames === null) {
    return status.running ? "等待实时推理统计输出..." : "未启动实时推理。";
  }
  return [
    `frames=${status.frames}`,
    `model_ms avg=${fmt(status.model_avg_ms)} p50=${fmt(status.model_p50_ms)} p95=${fmt(status.model_p95_ms)} p99=${fmt(status.model_p99_ms)} max=${fmt(status.model_max_ms)}`,
    `frame_ms avg=${fmt(status.frame_avg_ms)} p50=${fmt(status.frame_p50_ms)} p95=${fmt(status.frame_p95_ms)} p99=${fmt(status.frame_p99_ms)} max=${fmt(status.frame_max_ms)}`,
    `frame_over16=${status.frame_over16 ?? 0}`,
    `xrun capture=${status.cap_xruns ?? 0} playback=${status.play_xruns ?? 0}`,
  ].join("\n");
}

function offlineText(payload) {
  return [
    "离线音频已按实时流式路径处理。",
    `frames=${payload.frames ?? "--"}`,
    `audio_sec=${fmt(payload.audio_sec, " s")}`,
    `total_ms=${fmt(payload.total_ms, " ms")}`,
    `rtf=${fmt(payload.rtf)}`,
    `model_ms avg=${fmt(payload.model_avg)} p50=${fmt(payload.model_p50)} p95=${fmt(payload.model_p95)} p99=${fmt(payload.model_p99)} max=${fmt(payload.model_max)}`,
    payload.stdout || "",
  ].filter(Boolean).join("\n");
}

function render(status) {
  const running = Boolean(status.running);
  const recording = Boolean(status.recording);
  stateEl.textContent = recording ? "录音中" : (running ? "运行中" : "未启动");
  stateEl.classList.toggle("running", running);
  stateEl.classList.toggle("recording", recording);
  startBtn.disabled = running;
  stopBtn.disabled = !running;
  recordStartBtn.disabled = recording;
  recordStopBtn.disabled = !recording;
  modelSelect.disabled = running || recording;
  captureSelect.disabled = running || recording;
  playbackSelect.disabled = running || recording;
  applyDevicesBtn.disabled = running || recording;
  refreshDevicesBtn.disabled = running || recording;
  if (status.model_id && modelSelect.value !== status.model_id) modelSelect.value = status.model_id;
  if (status.capture && Array.from(captureSelect.options).some((option) => option.value === status.capture)) {
    captureSelect.value = status.capture;
  }
  if (status.playback && Array.from(playbackSelect.options).some((option) => option.value === status.playback)) {
    playbackSelect.value = status.playback;
  }
  recordState.textContent = recording ? "正在录制输入和降噪输出" : "未录音";
  model.textContent = status.model || "--";
  capture.textContent = status.capture || "--";
  playback.textContent = status.playback || "--";
  buffer.textContent = `periods=${status.periods ?? "--"}, prefill=${status.prefill_periods ?? "--"}`;
  frames.textContent = status.frames ?? "--";
  if (!recording) {
    updatePlayer(inputAudio, inputDownload, status.input_url, "realtime_input.wav");
    updatePlayer(outputAudio, outputDownload, status.output_url, "realtime_output.wav");
  }
  log.textContent = latencyText(status);
}

function updatePlayer(audio, link, url, fileName) {
  if (!url) {
    if (!audio.src) return;
    audio.removeAttribute("src");
    audio.load();
    link.hidden = true;
    link.removeAttribute("href");
    return;
  }
  const stamped = `${url}?t=${Date.now()}`;
  if (!audio.dataset.url || audio.dataset.url !== url) {
    audio.dataset.url = url;
    audio.src = stamped;
    link.href = url;
    link.download = fileName;
    link.hidden = false;
  }
}

async function api(path, options) {
  const res = await fetch(path, options);
  const payload = await res.json();
  if (!res.ok) throw new Error(payload.error || "请求失败");
  return payload;
}

async function refresh() {
  try {
    render(await api("/api/status"));
  } catch (err) {
    log.textContent = String(err);
  }
}

async function loadModels() {
  const payload = await api("/api/models");
  modelSelect.innerHTML = "";
  for (const item of payload.models || []) {
    const option = document.createElement("option");
    option.value = item.id;
    option.textContent = item.label;
    modelSelect.appendChild(option);
  }
}

function fillDeviceSelect(select, devices, selected) {
  select.innerHTML = "";
  for (const item of devices || []) {
    const option = document.createElement("option");
    option.value = item.id;
    option.textContent = item.label;
    select.appendChild(option);
  }
  if (selected && !Array.from(select.options).some((option) => option.value === selected)) {
    const option = document.createElement("option");
    option.value = selected;
    option.textContent = `${selected}（当前配置）`;
    select.appendChild(option);
  }
  if (selected) select.value = selected;
}

async function loadAudioDevices() {
  const payload = await api("/api/audio-devices");
  fillDeviceSelect(captureSelect, payload.capture, payload.selected_capture);
  fillDeviceSelect(playbackSelect, payload.playback, payload.selected_playback);
}

modelSelect.addEventListener("change", async () => {
  try {
    const status = await api("/api/model/select", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model_id: modelSelect.value }),
    });
    render(status);
    log.textContent = `已切换到 ${status.model}`;
  } catch (err) {
    log.textContent = String(err);
    await refresh();
  }
});

applyDevicesBtn.addEventListener("click", async () => {
  try {
    const status = await api("/api/audio-devices/select", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        capture: captureSelect.value,
        playback: playbackSelect.value,
      }),
    });
    render(status);
    log.textContent = `已应用麦克风 ${status.capture}\n已应用扬声器 ${status.playback}`;
  } catch (err) {
    log.textContent = String(err);
    await loadAudioDevices();
  }
});

refreshDevicesBtn.addEventListener("click", async () => {
  try {
    await loadAudioDevices();
    log.textContent = "已重新扫描板端 ALSA 音频设备。";
  } catch (err) {
    log.textContent = String(err);
  }
});

startBtn.addEventListener("click", async () => {
  log.textContent = "";
  render(await api("/api/start", { method: "POST" }));
});

stopBtn.addEventListener("click", async () => {
  render(await api("/api/stop", { method: "POST" }));
});

recordStartBtn.addEventListener("click", async () => {
  log.textContent = "正在启动实时录音...";
  inputAudio.removeAttribute("src");
  outputAudio.removeAttribute("src");
  delete inputAudio.dataset.url;
  delete outputAudio.dataset.url;
  inputAudio.load();
  outputAudio.load();
  inputDownload.hidden = true;
  outputDownload.hidden = true;
  try {
    render(await api("/api/record/start", { method: "POST" }));
    log.textContent = "录音中：麦克风原始输入和降噪后输出会同时保存。";
  } catch (err) {
    log.textContent = String(err);
    await refresh();
  }
});

recordStopBtn.addEventListener("click", async () => {
  log.textContent = "正在停止录音并写入 WAV...";
  try {
    const status = await api("/api/record/stop", { method: "POST" });
    render(status);
    log.textContent = "录音完成，可以在上方试听输入和降噪输出。";
  } catch (err) {
    log.textContent = String(err);
    await refresh();
  }
});

offlineForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!offlineFile.files.length) {
    log.textContent = "请选择 WAV 文件。";
    return;
  }
  offlineBtn.disabled = true;
  offlineState.textContent = "处理中";
  log.textContent = "正在上传并按 256 hop 流式推理...";
  try {
    const data = new FormData();
    data.append("audio", offlineFile.files[0]);
    const payload = await api("/api/offline/process", { method: "POST", body: data });
    updatePlayer(offlineInputAudio, offlineInputDownload, payload.input_url, "offline_input_16k_mono.wav");
    updatePlayer(offlineOutputAudio, offlineOutputDownload, payload.output_url, "offline_enhanced_stream.wav");
    offlineState.textContent = "处理完成";
    log.textContent = offlineText(payload);
  } catch (err) {
    offlineState.textContent = "处理失败";
    log.textContent = String(err);
  } finally {
    offlineBtn.disabled = false;
  }
});

Promise.all([loadModels(), loadAudioDevices()])
  .then(refresh)
  .catch((err) => { log.textContent = String(err); });
window.setInterval(refresh, 1000);
