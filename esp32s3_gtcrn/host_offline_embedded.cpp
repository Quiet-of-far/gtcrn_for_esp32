#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/gtcrn_embedded_model.h"

namespace {

constexpr int SR = 16000;
constexpr int NFFT = 512;
constexpr int HOP = 256;
constexpr int BINS = 257;
constexpr double PI = 3.14159265358979323846;
constexpr float INPUT_GAIN = 3.0f;

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

struct Processor {
  gtcrn_esp::Workspace ws;
  gtcrn_esp::EmbeddedModel model{&ws};
  std::array<uint16_t, NFFT> bitrev{};
  std::array<std::complex<float>, NFFT / 2> twiddle_fwd{};
  std::array<std::complex<float>, NFFT / 2> twiddle_inv{};
  std::array<float, NFFT> window{};
  std::array<float, NFFT> analysis{};
  std::array<float, NFFT> ola{};
  std::array<std::complex<float>, NFFT> spec{};
  std::array<std::complex<float>, NFFT> inv{};
  std::array<float, BINS * 2> mix{};
  std::array<float, BINS * 2> enh{};
  float hp_prev_x = 0.0f;
  float hp_prev_y = 0.0f;

  explicit Processor(int warmup_frames) {
    for (int i = 0; i < NFFT; ++i) {
      unsigned x = static_cast<unsigned>(i);
      unsigned r = 0;
      for (int b = 0; b < 9; ++b) {
        r = (r << 1) | (x & 1U);
        x >>= 1U;
      }
      bitrev[i] = static_cast<uint16_t>(r);
    }
    for (int i = 0; i < NFFT / 2; ++i) {
      float ang = static_cast<float>(2.0 * PI * i / NFFT);
      twiddle_fwd[i] = {std::cos(-ang), std::sin(-ang)};
      twiddle_inv[i] = {std::cos(ang), std::sin(ang)};
    }
    for (int i = 0; i < NFFT; ++i) {
      float hann = 0.5f * (1.0f - std::cos(static_cast<float>(2.0 * PI * i / (NFFT - 1))));
      window[i] = std::sqrt(std::max(hann, 0.0f));
    }
    mix.fill(0.0f);
    enh.fill(0.0f);
    for (int i = 0; i < warmup_frames; ++i) model.infer(mix.data(), enh.data());
    model.reset();
  }

  void fft(std::array<std::complex<float>, NFFT>& a, bool inverse) {
    for (int i = 0; i < NFFT; ++i) {
      int j = bitrev[i];
      if (i < j) std::swap(a[i], a[j]);
    }
    const auto& twiddle = inverse ? twiddle_inv : twiddle_fwd;
    for (int len = 2; len <= NFFT; len <<= 1) {
      int half = len >> 1;
      int step = NFFT / len;
      for (int i = 0; i < NFFT; i += len) {
        for (int j = 0; j < half; ++j) {
          const std::complex<float>& w = twiddle[j * step];
          std::complex<float> u = a[i + j];
          std::complex<float> v = a[i + j + half] * w;
          a[i + j] = u + v;
          a[i + j + half] = u - v;
        }
      }
    }
    if (inverse) {
      for (auto& v : a) v /= static_cast<float>(NFFT);
    }
  }

  float highpass(float x) {
    constexpr float a = 0.995f;
    float y = x - hp_prev_x + a * hp_prev_y;
    hp_prev_x = x;
    hp_prev_y = y;
    return y;
  }

  void process(const int16_t* in_pcm, int16_t* out_pcm, double* model_ms) {
    std::move(analysis.begin() + HOP, analysis.end(), analysis.begin());
    for (int i = 0; i < HOP; ++i) {
      float raw = static_cast<float>(in_pcm[i]) / 32768.0f;
      float pre = std::clamp(highpass(raw) * INPUT_GAIN, -1.0f, 1.0f);
      analysis[NFFT - HOP + i] = pre;
    }

    for (int i = 0; i < NFFT; ++i) spec[i] = {analysis[i] * window[i], 0.0f};
    fft(spec, false);

    for (int i = 0; i < BINS; ++i) {
      mix[i * 2] = spec[i].real();
      mix[i * 2 + 1] = spec[i].imag();
    }

    auto t0 = std::chrono::steady_clock::now();
    model.infer(mix.data(), enh.data());
    auto t1 = std::chrono::steady_clock::now();
    *model_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    for (int i = 0; i < BINS; ++i) inv[i] = {enh[i * 2], enh[i * 2 + 1]};
    for (int i = 1; i < BINS - 1; ++i) inv[NFFT - i] = std::conj(inv[i]);
    fft(inv, true);

    for (int i = 0; i < NFFT; ++i) ola[i] += inv[i].real() * window[i];
    for (int i = 0; i < HOP; ++i) {
      float v = std::clamp(ola[i], -1.0f, 1.0f);
      out_pcm[i] = static_cast<int16_t>(std::lrint(v * 32767.0f));
    }
    std::move(ola.begin() + HOP, ola.end(), ola.begin());
    std::fill(ola.end() - HOP, ola.end(), 0.0f);
  }
};

struct Args {
  std::string input;
  std::string output;
  int warmup = 5;
};

Args parse_args(int argc, char** argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--input" && i + 1 < argc) args.input = argv[++i];
    else if (a == "--output" && i + 1 < argc) args.output = argv[++i];
    else if (a == "--warmup" && i + 1 < argc) args.warmup = std::stoi(argv[++i]);
  }
  if (args.input.empty() || args.output.empty()) {
    throw std::runtime_error("usage: gtcrn_esp_offline --input in.wav --output out.wav [--warmup n]");
  }
  return args;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    Args args = parse_args(argc, argv);
    auto samples = read_wav_16k_mono(args.input);
    std::vector<int16_t> output(samples.size(), 0);
    Processor processor(args.warmup);

    size_t frames = (samples.size() + HOP - 1) / HOP;
    double model_sum_ms = 0.0;
    double model_max_ms = 0.0;
    std::array<int16_t, HOP> in_frame{};
    std::array<int16_t, HOP> out_frame{};
    for (size_t f = 0; f < frames; ++f) {
      size_t start = f * HOP;
      size_t count = std::min<size_t>(HOP, samples.size() - start);
      std::fill(in_frame.begin(), in_frame.end(), 0);
      std::copy_n(samples.data() + start, count, in_frame.data());
      double model_ms = 0.0;
      processor.process(in_frame.data(), out_frame.data(), &model_ms);
      std::copy_n(out_frame.data(), count, output.data() + start);
      model_sum_ms += model_ms;
      model_max_ms = std::max(model_max_ms, model_ms);
    }

    write_wav_16k_mono(args.output, output);
    double avg_model_ms = frames ? (model_sum_ms / static_cast<double>(frames)) : 0.0;
    std::cout << "frames=" << frames
              << " avg_model_ms=" << avg_model_ms
              << " max_model_ms=" << model_max_ms
              << " input=" << args.input
              << " output=" << args.output
              << "\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}
