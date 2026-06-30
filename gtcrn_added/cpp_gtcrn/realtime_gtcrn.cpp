#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <complex>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>
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

class Pipe {
 public:
  Pipe() = default;
  Pipe(const Pipe&) = delete;
  Pipe& operator=(const Pipe&) = delete;
  ~Pipe() { close(); }

  void open_read(const std::vector<std::string>& args) { open(args, true); }
  void open_write(const std::vector<std::string>& args) { open(args, false); }

  int fd() const { return fd_; }

  void close() {
    if (fd_ >= 0) {
      ::close(fd_);
      fd_ = -1;
    }
    if (pid_ > 0) {
      int status = 0;
      waitpid(pid_, &status, 0);
      pid_ = -1;
    }
  }

 private:
  void open(const std::vector<std::string>& args, bool read_from_child) {
    int pipefd[2];
    if (pipe(pipefd) != 0) throw std::runtime_error("pipe failed");
    pid_ = fork();
    if (pid_ < 0) throw std::runtime_error("fork failed");
    if (pid_ == 0) {
      if (read_from_child) {
        dup2(pipefd[1], STDOUT_FILENO);
      } else {
        dup2(pipefd[0], STDIN_FILENO);
      }
      ::close(pipefd[0]);
      ::close(pipefd[1]);
      std::vector<char*> argv;
      argv.reserve(args.size() + 1);
      for (const auto& a : args) argv.push_back(const_cast<char*>(a.c_str()));
      argv.push_back(nullptr);
      execvp(argv[0], argv.data());
      _exit(127);
    }
    if (read_from_child) {
      ::close(pipefd[1]);
      fd_ = pipefd[0];
    } else {
      ::close(pipefd[0]);
      fd_ = pipefd[1];
    }
  }

  pid_t pid_ = -1;
  int fd_ = -1;
};

bool read_exact(int fd, void* data, size_t bytes) {
  auto* p = static_cast<uint8_t*>(data);
  size_t got = 0;
  while (got < bytes && !g_stop) {
    ssize_t n = ::read(fd, p + got, bytes - got);
    if (n == 0) return false;
    if (n < 0) {
      if (errno == EINTR) continue;
      return false;
    }
    got += static_cast<size_t>(n);
  }
  return got == bytes;
}

bool write_exact(int fd, const void* data, size_t bytes) {
  const auto* p = static_cast<const uint8_t*>(data);
  size_t sent = 0;
  while (sent < bytes && !g_stop) {
    ssize_t n = ::write(fd, p + sent, bytes - sent);
    if (n < 0) {
      if (errno == EINTR) continue;
      return false;
    }
    sent += static_cast<size_t>(n);
  }
  return sent == bytes;
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
        std::complex<float> u = a[i + j];
        std::complex<float> v = a[i + j + len / 2] * w;
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

  void process(const int16_t* in_pcm, int16_t* out_pcm, int model_stride, double* model_ms) {
    std::move(analysis_.begin() + HOP, analysis_.end(), analysis_.begin());
    for (int i = 0; i < HOP; ++i) analysis_[NFFT - HOP + i] = static_cast<float>(in_pcm[i]) / 32768.0f;

    std::array<std::complex<float>, NFFT> spec{};
    for (int i = 0; i < NFFT; ++i) spec[i] = {analysis_[i] * window_[i], 0.0f};
    fft(spec, false);

    float mix[BINS * 2];
    for (int i = 0; i < BINS; ++i) {
      mix[i * 2] = spec[i].real();
      mix[i * 2 + 1] = spec[i].imag();
    }

    float enh[BINS * 2];
    bool run_model = model_stride <= 1 || (frame_index_ % model_stride) == 0;
    if (run_model) {
      auto t0 = std::chrono::steady_clock::now();
      model_.infer(mix, enh);
      auto t1 = std::chrono::steady_clock::now();
      *model_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
      for (int i = 0; i < BINS; ++i) {
        float sr = mix[i * 2];
        float si = mix[i * 2 + 1];
        float er = enh[i * 2];
        float ei = enh[i * 2 + 1];
        float den = sr * sr + si * si + 1e-12f;
        last_mask_[i * 2] = (er * sr + ei * si) / den;
        last_mask_[i * 2 + 1] = (ei * sr - er * si) / den;
      }
    } else {
      *model_ms = 0.0;
      for (int i = 0; i < BINS; ++i) {
        float sr = mix[i * 2];
        float si = mix[i * 2 + 1];
        float mr = last_mask_[i * 2];
        float mi = last_mask_[i * 2 + 1];
        enh[i * 2] = sr * mr - si * mi;
        enh[i * 2 + 1] = si * mr + sr * mi;
      }
    }
    ++frame_index_;

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
  std::array<float, BINS * 2> last_mask_ = [] {
    std::array<float, BINS * 2> m{};
    for (int i = 0; i < BINS; ++i) m[i * 2] = 1.0f;
    return m;
  }();
  int64_t frame_index_ = 0;
};

struct Args {
  std::string capture = "default";
  std::string playback = "default";
  int seconds = 0;
  int warmup = 20;
  int buffer_frames = 64;
  int prefill_frames = 32;
  int model_stride = 1;
  bool bypass = false;
  bool quiet = false;
};

Args parse_args(int argc, char** argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--capture" && i + 1 < argc) args.capture = argv[++i];
    else if (a == "--playback" && i + 1 < argc) args.playback = argv[++i];
    else if (a == "--seconds" && i + 1 < argc) args.seconds = std::stoi(argv[++i]);
    else if (a == "--warmup" && i + 1 < argc) args.warmup = std::stoi(argv[++i]);
    else if (a == "--buffer-frames" && i + 1 < argc) args.buffer_frames = std::stoi(argv[++i]);
    else if (a == "--prefill-frames" && i + 1 < argc) args.prefill_frames = std::stoi(argv[++i]);
    else if (a == "--model-stride" && i + 1 < argc) args.model_stride = std::max(1, std::stoi(argv[++i]));
    else if (a == "--bypass") args.bypass = true;
    else if (a == "--quiet") args.quiet = true;
    else {
      std::cerr << "usage: " << argv[0]
                << " [--capture dev] [--playback dev] [--seconds n] [--warmup n]"
                << " [--buffer-frames n] [--prefill-frames n] [--model-stride n]"
                << " [--bypass] [--quiet]\n";
      std::exit(2);
    }
  }
  return args;
}

std::vector<std::string> arecord_cmd(const std::string& dev, int buffer_frames) {
  return {"arecord", "-q", "-D", dev, "-t", "raw", "-f", "S16_LE", "-r", std::to_string(SR),
          "-c", "1", "--period-size", std::to_string(HOP), "--buffer-size", std::to_string(HOP * buffer_frames)};
}

std::vector<std::string> aplay_cmd(const std::string& dev, int buffer_frames) {
  return {"aplay", "-q", "-D", dev, "-t", "raw", "-f", "S16_LE", "-r", std::to_string(SR),
          "-c", "1", "--period-size", std::to_string(HOP), "--buffer-size", std::to_string(HOP * buffer_frames)};
}

}  // namespace

int main(int argc, char** argv) {
  std::signal(SIGINT, on_signal);
  std::signal(SIGTERM, on_signal);
  std::signal(SIGPIPE, SIG_IGN);
  setpriority(PRIO_PROCESS, 0, -10);

  Args args = parse_args(argc, argv);
  try {
    StreamProcessor proc(args.warmup);
    Pipe cap;
    Pipe play;
    cap.open_read(arecord_cmd(args.capture, args.buffer_frames));
    play.open_write(aplay_cmd(args.playback, args.buffer_frames));

    std::array<int16_t, HOP> in{};
    std::array<int16_t, HOP> out{};
    std::array<int16_t, HOP> silence{};
    for (int i = 0; i < args.prefill_frames && !g_stop; ++i) {
      if (!write_exact(play.fd(), silence.data(), silence.size() * sizeof(int16_t))) {
        throw std::runtime_error("aplay stopped during prefill");
      }
    }
    std::vector<double> model_times;
    std::vector<double> frame_times;
    model_times.reserve(args.seconds > 0 ? args.seconds * (SR / HOP + 1) : 4096);
    frame_times.reserve(args.seconds > 0 ? args.seconds * (SR / HOP + 1) : 4096);

    auto start = std::chrono::steady_clock::now();
    int64_t frames = 0;
    int64_t dropped = 0;
    double max_ms = 0.0;
    while (!g_stop) {
      if (args.seconds > 0) {
        auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        if (elapsed >= args.seconds) break;
      }
      if (!read_exact(cap.fd(), in.data(), in.size() * sizeof(int16_t))) {
        throw std::runtime_error("arecord stopped");
      }
      auto frame_t0 = std::chrono::steady_clock::now();
      double model_ms = 0.0;
      if (args.bypass) {
        out = in;
      } else {
        proc.process(in.data(), out.data(), args.model_stride, &model_ms);
      }
      if (!write_exact(play.fd(), out.data(), out.size() * sizeof(int16_t))) {
        throw std::runtime_error("aplay stopped or broken pipe");
      }
      auto frame_t1 = std::chrono::steady_clock::now();
      double frame_ms = std::chrono::duration<double, std::milli>(frame_t1 - frame_t0).count();
      if (!args.bypass) {
        model_times.push_back(model_ms);
        frame_times.push_back(frame_ms);
        max_ms = std::max(max_ms, model_ms);
        if (frame_ms > 16.0) ++dropped;
      }
      ++frames;
      if (!args.quiet && frames % 125 == 0) {
        double avg = model_times.empty() ? 0.0 : std::accumulate(model_times.begin(), model_times.end(), 0.0) / model_times.size();
        std::cerr << "frames=" << frames << " avg_model_ms=" << avg << " max_model_ms=" << max_ms
                  << " over16=" << dropped << "\n";
      }
    }
    if (!args.bypass && !model_times.empty()) {
      auto stats = gtcrn::summarize(model_times);
      auto frame_stats = gtcrn::summarize(frame_times);
      std::cout << "realtime frames=" << frames << " model_avg=" << stats.avg << " model_p50=" << stats.p50
                << " p95=" << stats.p95 << " p99=" << stats.p99 << " max=" << stats.max
                << " frame_avg=" << frame_stats.avg << " frame_p50=" << frame_stats.p50
                << " frame_p95=" << frame_stats.p95 << " frame_p99=" << frame_stats.p99
                << " frame_max=" << frame_stats.max << " frame_over16=" << dropped << "\n";
    } else {
      std::cout << "realtime frames=" << frames << " bypass=1\n";
    }
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}
