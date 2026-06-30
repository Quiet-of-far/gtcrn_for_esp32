#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "gtcrn_model.h"

namespace {

constexpr int SR = 16000;
constexpr int NFFT = 512;
constexpr int HOP = 256;
constexpr int BINS = 257;
constexpr double PI = 3.14159265358979323846;

#if defined(__aarch64__)
void enable_fast_fp_mode() {
  uint64_t fpcr = __builtin_aarch64_get_fpcr();
  fpcr |= (1ull << 24);
  fpcr |= (1ull << 25);
  __builtin_aarch64_set_fpcr(fpcr);
}
#else
void enable_fast_fp_mode() {}
#endif

uint16_t read_u16(std::istream& in) {
  uint8_t b[2];
  in.read(reinterpret_cast<char*>(b), 2);
  if (!in) throw std::runtime_error("unexpected EOF");
  return static_cast<uint16_t>(b[0] | (b[1] << 8));
}

uint32_t read_u32(std::istream& in) {
  uint8_t b[4];
  in.read(reinterpret_cast<char*>(b), 4);
  if (!in) throw std::runtime_error("unexpected EOF");
  return static_cast<uint32_t>(b[0] | (b[1] << 8) | (b[2] << 16) | (b[3] << 24));
}

void write_u16(std::ostream& out, uint16_t v) {
  out.put(static_cast<char>(v & 0xff));
  out.put(static_cast<char>((v >> 8) & 0xff));
}

void write_u32(std::ostream& out, uint32_t v) {
  out.put(static_cast<char>(v & 0xff));
  out.put(static_cast<char>((v >> 8) & 0xff));
  out.put(static_cast<char>((v >> 16) & 0xff));
  out.put(static_cast<char>((v >> 24) & 0xff));
}

std::vector<int16_t> read_wav_16k_mono(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("failed to open input wav");

  char riff[4];
  in.read(riff, 4);
  if (std::string(riff, 4) != "RIFF") throw std::runtime_error("input is not RIFF WAV");
  (void)read_u32(in);
  char wave[4];
  in.read(wave, 4);
  if (std::string(wave, 4) != "WAVE") throw std::runtime_error("input is not WAVE");

  uint16_t audio_format = 0;
  uint16_t channels = 0;
  uint32_t sample_rate = 0;
  uint16_t bits_per_sample = 0;
  std::vector<int16_t> samples;

  while (in) {
    char id[4];
    in.read(id, 4);
    if (!in) break;
    uint32_t size = read_u32(in);
    std::string chunk(id, 4);
    if (chunk == "fmt ") {
      audio_format = read_u16(in);
      channels = read_u16(in);
      sample_rate = read_u32(in);
      (void)read_u32(in);
      (void)read_u16(in);
      bits_per_sample = read_u16(in);
      if (size > 16) in.seekg(size - 16, std::ios::cur);
    } else if (chunk == "data") {
      if (audio_format != 1 || channels != 1 || sample_rate != SR || bits_per_sample != 16) {
        throw std::runtime_error("input wav must be PCM 16 kHz mono S16_LE");
      }
      samples.resize(size / sizeof(int16_t));
      in.read(reinterpret_cast<char*>(samples.data()), static_cast<std::streamsize>(samples.size() * sizeof(int16_t)));
      if (!in) throw std::runtime_error("failed to read wav data");
    } else {
      in.seekg(size, std::ios::cur);
    }
    if (size & 1) in.seekg(1, std::ios::cur);
  }

  if (samples.empty()) throw std::runtime_error("empty wav data");
  return samples;
}

void write_wav_16k_mono(const std::string& path, const std::vector<int16_t>& samples) {
  std::ofstream out(path, std::ios::binary);
  if (!out) throw std::runtime_error("failed to open output wav");
  uint32_t data_bytes = static_cast<uint32_t>(samples.size() * sizeof(int16_t));
  out.write("RIFF", 4);
  write_u32(out, 36 + data_bytes);
  out.write("WAVEfmt ", 8);
  write_u32(out, 16);
  write_u16(out, 1);
  write_u16(out, 1);
  write_u32(out, SR);
  write_u32(out, SR * sizeof(int16_t));
  write_u16(out, sizeof(int16_t));
  write_u16(out, 16);
  out.write("data", 4);
  write_u32(out, data_bytes);
  out.write(reinterpret_cast<const char*>(samples.data()), static_cast<std::streamsize>(data_bytes));
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

struct Args {
  std::string input;
  std::string output;
  int warmup = 20;
};

Args parse_args(int argc, char** argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--input" && i + 1 < argc) args.input = argv[++i];
    else if (a == "--output" && i + 1 < argc) args.output = argv[++i];
    else if (a == "--warmup" && i + 1 < argc) args.warmup = std::stoi(argv[++i]);
    else {
      std::cerr << "usage: " << argv[0] << " --input in.wav --output out.wav [--warmup n]\n";
      std::exit(2);
    }
  }
  if (args.input.empty() || args.output.empty()) {
    std::cerr << "usage: " << argv[0] << " --input in.wav --output out.wav [--warmup n]\n";
    std::exit(2);
  }
  return args;
}

}  // namespace

int main(int argc, char** argv) {
  enable_fast_fp_mode();
  try {
    Args args = parse_args(argc, argv);
    auto samples = read_wav_16k_mono(args.input);
    StreamProcessor proc(args.warmup);
    std::vector<int16_t> output(samples.size());
    std::array<int16_t, HOP> in{};
    std::array<int16_t, HOP> out{};
    std::vector<double> model_times;
    model_times.reserve((samples.size() + HOP - 1) / HOP);

    auto total_t0 = std::chrono::steady_clock::now();
    size_t pos = 0;
    while (pos < samples.size()) {
      std::fill(in.begin(), in.end(), 0);
      size_t n = std::min<size_t>(HOP, samples.size() - pos);
      std::copy_n(samples.data() + pos, n, in.data());
      double model_ms = 0.0;
      proc.process(in.data(), out.data(), &model_ms);
      std::copy_n(out.data(), n, output.data() + pos);
      model_times.push_back(model_ms);
      pos += n;
    }
    auto total_t1 = std::chrono::steady_clock::now();
    write_wav_16k_mono(args.output, output);

    auto stats = gtcrn::summarize(model_times);
    double audio_sec = static_cast<double>(samples.size()) / SR;
    double total_ms = std::chrono::duration<double, std::milli>(total_t1 - total_t0).count();
    std::cout << "offline_stream frames=" << model_times.size()
              << " audio_sec=" << audio_sec
              << " total_ms=" << total_ms
              << " rtf=" << (total_ms / 1000.0 / std::max(audio_sec, 1e-9))
              << " model_avg=" << stats.avg
              << " model_p50=" << stats.p50
              << " model_p95=" << stats.p95
              << " model_p99=" << stats.p99
              << " model_max=" << stats.max << "\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}
