#include <alsa/asoundlib.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <complex>
#include <csignal>
#include <cstdint>
#include <iostream>
#include <fstream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <sys/resource.h>
#include <vector>

#include "gtcrn_model.h"

namespace {

constexpr int SR = 16000;
constexpr int NFFT = 512;
constexpr int HOP = 256;
constexpr int BINS = 257;
constexpr double PI = 3.14159265358979323846;

volatile std::sig_atomic_t g_stop = 0;
void on_signal(int) { g_stop = 1; }

#if defined(__aarch64__)
void enable_fast_fp_mode() {
  uint64_t fpcr = __builtin_aarch64_get_fpcr();
  fpcr |= (1ull << 24);  // FZ: flush denormal inputs/results to zero.
  fpcr |= (1ull << 25);  // DN: use default NaN propagation.
  __builtin_aarch64_set_fpcr(fpcr);
}
#else
void enable_fast_fp_mode() {}
#endif

void check(int ret, const std::string& what) {
  if (ret < 0) throw std::runtime_error(what + ": " + snd_strerror(ret));
}

void fft(std::array<std::complex<float>, NFFT>& a, bool inverse) {
  for (int i = 1, j = 0; i < NFFT; ++i) {
    int bit = NFFT >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) std::swap(a[i], a[j]);
  }
  for (int len = 2; len <= NFFT; len <<= 1) {
    float ang = static_cast<float>((inverse ? 2.0 : -2.0) * PI / len);
    std::complex<float> wlen(std::cos(ang), std::sin(ang));
    for (int i = 0; i < NFFT; i += len) {
      std::complex<float> w(1.0f, 0.0f);
      for (int j = 0; j < len / 2; ++j) {
        auto u = a[i + j];
        auto v = a[i + j + len / 2] * w;
        a[i + j] = u + v;
        a[i + j + len / 2] = u - v;
        w *= wlen;
      }
    }
  }
  if (inverse) {
    for (auto& v : a) v /= static_cast<float>(NFFT);
  }
}

class Pcm {
 public:
  Pcm(const std::string& dev, snd_pcm_stream_t stream, int periods, int start_periods = 1) : stream_(stream) {
    check(snd_pcm_open(&pcm_, dev.c_str(), stream, 0), "snd_pcm_open " + dev);
    snd_pcm_hw_params_t* hw = nullptr;
    snd_pcm_hw_params_alloca(&hw);
    check(snd_pcm_hw_params_any(pcm_, hw), "hw_params_any");
    check(snd_pcm_hw_params_set_access(pcm_, hw, SND_PCM_ACCESS_RW_INTERLEAVED), "set_access");
    check(snd_pcm_hw_params_set_format(pcm_, hw, SND_PCM_FORMAT_S16_LE), "set_format");
    check(snd_pcm_hw_params_set_channels(pcm_, hw, 1), "set_channels");
    unsigned int rate = SR;
    check(snd_pcm_hw_params_set_rate_near(pcm_, hw, &rate, nullptr), "set_rate");
    if (rate != SR) throw std::runtime_error("device did not accept 16 kHz");
    snd_pcm_uframes_t period = HOP;
    check(snd_pcm_hw_params_set_period_size_near(pcm_, hw, &period, nullptr), "set_period_size");
    snd_pcm_uframes_t buffer = static_cast<snd_pcm_uframes_t>(HOP * periods);
    check(snd_pcm_hw_params_set_buffer_size_near(pcm_, hw, &buffer), "set_buffer_size");
    check(snd_pcm_hw_params(pcm_, hw), "hw_params");

    snd_pcm_sw_params_t* sw = nullptr;
    snd_pcm_sw_params_alloca(&sw);
    check(snd_pcm_sw_params_current(pcm_, sw), "sw_params_current");
    check(snd_pcm_sw_params_set_avail_min(pcm_, sw, HOP), "set_avail_min");
    if (stream == SND_PCM_STREAM_PLAYBACK) {
      check(snd_pcm_sw_params_set_start_threshold(pcm_, sw, HOP * std::max(1, start_periods)), "set_start_threshold");
    }
    check(snd_pcm_sw_params(pcm_, sw), "sw_params");
    check(snd_pcm_prepare(pcm_), "prepare");
  }

  ~Pcm() {
    if (pcm_) {
      snd_pcm_drop(pcm_);
      snd_pcm_close(pcm_);
    }
  }

  bool read_period(std::array<int16_t, HOP>& out) {
    int done = 0;
    while (done < HOP && !g_stop) {
      snd_pcm_sframes_t n = snd_pcm_readi(pcm_, out.data() + done, HOP - done);
      if (n == -EPIPE || n == -ESTRPIPE) {
        ++xruns_;
        snd_pcm_prepare(pcm_);
        std::fill(out.begin() + done, out.end(), 0);
        return true;
      }
      if (n < 0) {
        n = snd_pcm_recover(pcm_, static_cast<int>(n), 1);
        if (n < 0) return false;
        continue;
      }
      done += static_cast<int>(n);
    }
    return done == HOP;
  }

  bool write_period(const std::array<int16_t, HOP>& in) {
    int done = 0;
    while (done < HOP && !g_stop) {
      snd_pcm_sframes_t n = snd_pcm_writei(pcm_, in.data() + done, HOP - done);
      if (n == -EPIPE || n == -ESTRPIPE) {
        ++xruns_;
        snd_pcm_prepare(pcm_);
        continue;
      }
      if (n < 0) {
        n = snd_pcm_recover(pcm_, static_cast<int>(n), 1);
        if (n < 0) return false;
        continue;
      }
      done += static_cast<int>(n);
    }
    return done == HOP;
  }

  int xruns() const { return xruns_; }

 private:
  snd_pcm_t* pcm_ = nullptr;
  snd_pcm_stream_t stream_;
  int xruns_ = 0;
};

class StreamProcessor {
 public:
  explicit StreamProcessor(int warmup_frames) {
    for (int i = 0; i < NFFT; ++i) {
      float hann = 0.5f * (1.0f - std::cos(static_cast<float>(2.0 * PI * i / (NFFT - 1))));
      window_[i] = std::sqrt(std::max(hann, 0.0f));
    }
    std::array<float, BINS * 2> mix{};
    std::array<float, BINS * 2> enh{};
    for (int i = 0; i < warmup_frames; ++i) model_.infer(mix.data(), enh.data());
    model_.reset();
  }

  void process(const int16_t* in_pcm, int16_t* out_pcm, double* model_ms) {
    std::move(analysis_.begin() + HOP, analysis_.end(), analysis_.begin());
    for (int i = 0; i < HOP; ++i) analysis_[NFFT - HOP + i] = static_cast<float>(in_pcm[i]) / 32768.0f;

    std::array<std::complex<float>, NFFT> spec{};
    for (int i = 0; i < NFFT; ++i) spec[i] = {analysis_[i] * window_[i], 0.0f};
    fft(spec, false);

    std::array<float, BINS * 2> mix{};
    for (int i = 0; i < BINS; ++i) {
      mix[i * 2] = spec[i].real();
      mix[i * 2 + 1] = spec[i].imag();
    }

    std::array<float, BINS * 2> enh{};
    auto t0 = std::chrono::steady_clock::now();
    model_.infer(mix.data(), enh.data());
    auto t1 = std::chrono::steady_clock::now();
    *model_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::array<std::complex<float>, NFFT> inv{};
    for (int i = 0; i < BINS; ++i) inv[i] = {enh[i * 2], enh[i * 2 + 1]};
    for (int i = 1; i < BINS - 1; ++i) inv[NFFT - i] = std::conj(inv[i]);
    fft(inv, true);

    for (int i = 0; i < NFFT; ++i) ola_[i] += inv[i].real() * window_[i];
    for (int i = 0; i < HOP; ++i) {
      float v = std::clamp(ola_[i], -1.0f, 1.0f);
      out_pcm[i] = static_cast<int16_t>(std::lrint(v * 32767.0f));
    }
    std::move(ola_.begin() + HOP, ola_.end(), ola_.begin());
    std::fill(ola_.end() - HOP, ola_.end(), 0.0f);
  }

 private:
  gtcrn::Model model_;
  std::array<float, NFFT> window_{};
  std::array<float, NFFT> analysis_{};
  std::array<float, NFFT> ola_{};
};

class WavWriter {
 public:
  bool open(const std::string& path, int sample_rate) {
    out_.open(path, std::ios::binary);
    if (!out_) return false;
    sample_rate_ = sample_rate;
    samples_ = 0;
    write_header(0);
    return true;
  }

  void write_samples(const int16_t* samples, size_t count) {
    if (!out_) return;
    out_.write(reinterpret_cast<const char*>(samples), static_cast<std::streamsize>(count * sizeof(int16_t)));
    samples_ += count;
  }

  void close() {
    if (!out_) return;
    out_.seekp(0, std::ios::beg);
    write_header(static_cast<uint32_t>(samples_ * sizeof(int16_t)));
    out_.close();
  }

  ~WavWriter() { close(); }

 private:
  void write_u16(uint16_t v) {
    out_.put(static_cast<char>(v & 0xff));
    out_.put(static_cast<char>((v >> 8) & 0xff));
  }
  void write_u32(uint32_t v) {
    out_.put(static_cast<char>(v & 0xff));
    out_.put(static_cast<char>((v >> 8) & 0xff));
    out_.put(static_cast<char>((v >> 16) & 0xff));
    out_.put(static_cast<char>((v >> 24) & 0xff));
  }
  void write_header(uint32_t data_bytes) {
    out_.write("RIFF", 4);
    write_u32(36 + data_bytes);
    out_.write("WAVEfmt ", 8);
    write_u32(16);
    write_u16(1);
    write_u16(1);
    write_u32(static_cast<uint32_t>(sample_rate_));
    write_u32(static_cast<uint32_t>(sample_rate_ * sizeof(int16_t)));
    write_u16(sizeof(int16_t));
    write_u16(16);
    out_.write("data", 4);
    write_u32(data_bytes);
  }

  std::ofstream out_;
  int sample_rate_ = SR;
  size_t samples_ = 0;
};

struct Args {
  std::string capture = "plughw:0,0";
  std::string playback = "plughw:0,0";
  int seconds = 0;
  int periods = 3;
  int prefill_periods = 2;
  int warmup = 20;
  std::string record_dir;
  bool bypass = false;
  bool latency_test = false;
  bool quiet = false;
  bool status_json = false;
};

Args parse_args(int argc, char** argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--capture" && i + 1 < argc) args.capture = argv[++i];
    else if (a == "--playback" && i + 1 < argc) args.playback = argv[++i];
    else if (a == "--seconds" && i + 1 < argc) args.seconds = std::stoi(argv[++i]);
    else if (a == "--periods" && i + 1 < argc) args.periods = std::max(2, std::stoi(argv[++i]));
    else if (a == "--prefill-periods" && i + 1 < argc) args.prefill_periods = std::max(0, std::stoi(argv[++i]));
    else if (a == "--warmup" && i + 1 < argc) args.warmup = std::stoi(argv[++i]);
    else if (a == "--record-dir" && i + 1 < argc) args.record_dir = argv[++i];
    else if (a == "--bypass") args.bypass = true;
    else if (a == "--latency-test") args.latency_test = true;
    else if (a == "--quiet") args.quiet = true;
    else if (a == "--status-json") args.status_json = true;
    else {
      std::cerr << "usage: " << argv[0]
                << " [--capture dev] [--playback dev] [--seconds n] [--periods n]"
                << " [--prefill-periods n] [--warmup n] [--record-dir dir]"
                << " [--bypass] [--latency-test] [--quiet] [--status-json]\n";
      std::exit(2);
    }
  }
  return args;
}

int run_latency_test(const Args& args) {
  int seconds = args.seconds > 0 ? args.seconds : 6;
  Pcm capture(args.capture, SND_PCM_STREAM_CAPTURE, args.periods);
  Pcm playback(args.playback, SND_PCM_STREAM_PLAYBACK, args.periods, args.prefill_periods + 1);
  std::array<int16_t, HOP> in{};
  std::array<int16_t, HOP> out{};
  std::array<int16_t, HOP> silence{};
  for (int i = 0; i < args.prefill_periods; ++i) playback.write_period(silence);
  std::vector<int16_t> recorded;
  recorded.reserve(seconds * SR + HOP);
  constexpr int burst_len = 1024;
  std::array<int16_t, burst_len> burst{};
  uint32_t rng = 0x12345678u;
  for (int i = 0; i < burst_len; ++i) {
    rng = rng * 1664525u + 1013904223u;
    float taper = std::sin(static_cast<float>(PI * i / (burst_len - 1)));
    int amp = ((rng >> 31) ? 12000 : -12000);
    burst[i] = static_cast<int16_t>(std::lrint(amp * taper));
  }
  int total_frames = seconds * SR / HOP;
  int burst_start = SR;

  for (int frame = 0; frame < total_frames && !g_stop; ++frame) {
    if (!capture.read_period(in)) throw std::runtime_error("capture read failed");
    recorded.insert(recorded.end(), in.begin(), in.end());
    out.fill(0);
    int frame_start = frame * HOP;
    for (int i = 0; i < HOP; ++i) {
      int pos = frame_start + i - burst_start;
      if (pos >= 0 && pos < burst_len) out[i] = burst[pos];
    }
    if (!playback.write_period(out)) throw std::runtime_error("playback write failed");
  }

  int max_lag = std::min(SR, static_cast<int>(recorded.size()) - burst_start - burst_len - 1);
  int best_lag = 0;
  double best_score = -1.0;
  double burst_energy = 0.0;
  for (int i = 0; i < burst_len; ++i) burst_energy += static_cast<double>(burst[i]) * burst[i];
  for (int lag = 0; lag < max_lag; ++lag) {
    double dot = 0.0;
    double rec_energy = 0.0;
    int base = burst_start + lag;
    for (int i = 0; i < burst_len; ++i) {
      double r = recorded[base + i];
      dot += static_cast<double>(burst[i]) * r;
      rec_energy += r * r;
    }
    double score = rec_energy > 1.0 ? std::abs(dot) / std::sqrt(burst_energy * rec_energy) : 0.0;
    if (score > best_score) {
      best_score = score;
      best_lag = lag;
    }
  }
  int peak_abs = 0;
  for (int i = 0; i < burst_len; ++i) {
    peak_abs = std::max(peak_abs, std::abs(static_cast<int>(recorded[burst_start + best_lag + i])));
  }
  double latency_ms = best_lag * 1000.0 / SR;
  std::cout << "latency_test method=pn_correlation"
            << " recorded_samples=" << recorded.size()
            << " cap_xruns=" << capture.xruns() << " play_xruns=" << playback.xruns() << "\n";
  std::cout << "burst_start_sample=" << burst_start
            << " detected_lag_samples=" << best_lag
            << " latency_ms=" << latency_ms
            << " corr_score=" << best_score
            << " peak_abs=" << peak_abs << "\n";
  if (best_score < 0.08) {
    std::cout << "warning=low_correlation_check_headphone_volume_or_mic_coupling\n";
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  enable_fast_fp_mode();
  std::signal(SIGINT, on_signal);
  std::signal(SIGTERM, on_signal);
  setpriority(PRIO_PROCESS, 0, -10);

  try {
    Args args = parse_args(argc, argv);
    if (args.latency_test) return run_latency_test(args);

    StreamProcessor proc(args.warmup);
    Pcm capture(args.capture, SND_PCM_STREAM_CAPTURE, args.periods);
    Pcm playback(args.playback, SND_PCM_STREAM_PLAYBACK, args.periods, args.prefill_periods + 1);

    std::array<int16_t, HOP> in{};
    std::array<int16_t, HOP> out{};
    std::array<int16_t, HOP> silence{};
    for (int i = 0; i < args.prefill_periods; ++i) playback.write_period(silence);

    std::vector<double> model_times;
    std::vector<double> frame_times;
    std::vector<double> status_model_times;
    std::vector<double> status_frame_times;
    model_times.reserve(args.seconds > 0 ? args.seconds * (SR / HOP + 1) : 4096);
    frame_times.reserve(model_times.capacity());
    status_model_times.reserve(80);
    status_frame_times.reserve(80);
    auto start = std::chrono::steady_clock::now();
    int64_t frames = 0;
    int over16 = 0;
    WavWriter input_wav;
    WavWriter output_wav;
    if (!args.record_dir.empty()) {
      std::string input_path = args.record_dir + "/realtime_input.wav";
      std::string output_path = args.record_dir + "/realtime_output.wav";
      if (!input_wav.open(input_path, SR)) throw std::runtime_error("failed to open input wav");
      if (!output_wav.open(output_path, SR)) throw std::runtime_error("failed to open output wav");
    }

    while (!g_stop) {
      if (args.seconds > 0) {
        double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        if (elapsed >= args.seconds) break;
      }
      if (!capture.read_period(in)) {
        if (g_stop) break;
        throw std::runtime_error("capture read failed");
      }
      input_wav.write_samples(in.data(), in.size());
      auto ft0 = std::chrono::steady_clock::now();
      double model_ms = 0.0;
      if (args.bypass) {
        out = in;
      } else {
        proc.process(in.data(), out.data(), &model_ms);
      }
      if (!playback.write_period(out)) {
        if (g_stop) break;
        throw std::runtime_error("playback write failed");
      }
      output_wav.write_samples(out.data(), out.size());
      auto ft1 = std::chrono::steady_clock::now();
      double frame_ms = std::chrono::duration<double, std::milli>(ft1 - ft0).count();
      if (!args.bypass) {
        model_times.push_back(model_ms);
        frame_times.push_back(frame_ms);
        status_model_times.push_back(model_ms);
        status_frame_times.push_back(frame_ms);
        if (frame_ms > 16.0) ++over16;
      }
      ++frames;
      if (args.status_json && !args.bypass && frames % 63 == 0) {
        auto ms = gtcrn::summarize(status_model_times);
        auto fs = gtcrn::summarize(status_frame_times);
        std::cout << "{\"running\":true"
                  << ",\"frames\":" << frames
                  << ",\"model_avg_ms\":" << ms.avg
                  << ",\"model_p50_ms\":" << ms.p50
                  << ",\"model_p95_ms\":" << ms.p95
                  << ",\"model_p99_ms\":" << ms.p99
                  << ",\"model_max_ms\":" << ms.max
                  << ",\"frame_avg_ms\":" << fs.avg
                  << ",\"frame_p50_ms\":" << fs.p50
                  << ",\"frame_p95_ms\":" << fs.p95
                  << ",\"frame_p99_ms\":" << fs.p99
                  << ",\"frame_max_ms\":" << fs.max
                  << ",\"frame_over16\":" << over16
                  << ",\"cap_xruns\":" << capture.xruns()
                  << ",\"play_xruns\":" << playback.xruns()
                  << "}" << std::endl;
        status_model_times.clear();
        status_frame_times.clear();
      }
      if (!args.quiet && frames % 125 == 0) {
        std::cerr << "frames=" << frames << " cap_xruns=" << capture.xruns()
                  << " play_xruns=" << playback.xruns() << " over16=" << over16 << "\n";
      }
    }

    if (args.bypass || model_times.empty()) {
      std::cout << "alsa realtime frames=" << frames << " bypass=1"
                << " cap_xruns=" << capture.xruns() << " play_xruns=" << playback.xruns() << "\n";
    } else {
      auto ms = gtcrn::summarize(model_times);
      auto fs = gtcrn::summarize(frame_times);
      std::cout << "alsa realtime frames=" << frames
                << " model_avg=" << ms.avg << " model_p50=" << ms.p50
                << " model_p95=" << ms.p95 << " model_p99=" << ms.p99 << " model_max=" << ms.max
                << " frame_avg=" << fs.avg << " frame_p50=" << fs.p50
                << " frame_p95=" << fs.p95 << " frame_p99=" << fs.p99 << " frame_max=" << fs.max
                << " frame_over16=" << over16
                << " cap_xruns=" << capture.xruns() << " play_xruns=" << playback.xruns() << "\n";
    }
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}
