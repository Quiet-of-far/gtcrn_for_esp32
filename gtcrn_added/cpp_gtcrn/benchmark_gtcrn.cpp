#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "gtcrn_model.h"

namespace {

constexpr int MIX_FLOATS = 257 * 2;
constexpr int ENH_FLOATS = 257 * 2;
constexpr int CONV_FLOATS = 2 * 1 * 16 * 16 * 33;
constexpr int TRA_FLOATS = 2 * 3 * 1 * 1 * 16;
constexpr int INTER_FLOATS = 2 * 1 * 33 * 16;

struct Frame {
  std::vector<float> mix;
  std::vector<float> enh;
  std::vector<float> conv;
  std::vector<float> tra;
  std::vector<float> inter;
};

template <typename T>
void read_exact(std::ifstream& in, T* data, size_t count) {
  in.read(reinterpret_cast<char*>(data), static_cast<std::streamsize>(sizeof(T) * count));
  if (!in) throw std::runtime_error("unexpected EOF while reading reference file");
}

std::vector<Frame> read_ref(const std::string& path, int limit) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("failed to open reference file: " + path);
  char magic[10] = {};
  in.read(magic, 9);
  if (std::string(magic, 9) != "GTCRNREF1") throw std::runtime_error("bad reference magic");
  uint32_t frames = 0;
  read_exact(in, &frames, 1);
  if (limit > 0 && static_cast<uint32_t>(limit) < frames) frames = static_cast<uint32_t>(limit);
  std::vector<Frame> out;
  out.reserve(frames);
  for (uint32_t i = 0; i < frames; ++i) {
    Frame f;
    f.mix.resize(MIX_FLOATS);
    f.enh.resize(ENH_FLOATS);
    f.conv.resize(CONV_FLOATS);
    f.tra.resize(TRA_FLOATS);
    f.inter.resize(INTER_FLOATS);
    read_exact(in, f.mix.data(), f.mix.size());
    read_exact(in, f.enh.data(), f.enh.size());
    read_exact(in, f.conv.data(), f.conv.size());
    read_exact(in, f.tra.data(), f.tra.size());
    read_exact(in, f.inter.data(), f.inter.size());
    out.push_back(std::move(f));
  }
  return out;
}

double now_ms() {
  using clock = std::chrono::steady_clock;
  return std::chrono::duration<double, std::milli>(clock::now().time_since_epoch()).count();
}

}  // namespace

int main(int argc, char** argv) {
  std::string ref_path = "cpp_gtcrn/gtcrn_ref_1000.bin";
  int frames = 1000;
  int warmup = 20;
  std::string dump_dir;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--ref" && i + 1 < argc) ref_path = argv[++i];
    else if (arg == "--frames" && i + 1 < argc) frames = std::stoi(argv[++i]);
    else if (arg == "--warmup" && i + 1 < argc) warmup = std::stoi(argv[++i]);
    else if (arg == "--dump-dir" && i + 1 < argc) dump_dir = argv[++i];
    else {
      std::cerr << "usage: " << argv[0] << " [--ref path] [--frames n] [--warmup n] [--dump-dir path]\n";
      return 2;
    }
  }

  try {
    auto refs = read_ref(ref_path, frames);
    gtcrn::Model model;
    std::vector<float> enh(ENH_FLOATS);
    for (int i = 0; i < warmup && i < static_cast<int>(refs.size()); ++i) model.infer(refs[i].mix.data(), enh.data());
    model.reset();
    model.set_trace_dir(dump_dir);

    double max_abs = 0.0;
    double sum_abs = 0.0;
    size_t count = 0;
    std::vector<double> times;
    times.reserve(refs.size());
    for (const auto& frame : refs) {
      double t0 = now_ms();
      model.infer(frame.mix.data(), enh.data());
      double t1 = now_ms();
      times.push_back(t1 - t0);
      for (size_t i = 0; i < enh.size(); ++i) {
        double d = std::abs(static_cast<double>(enh[i]) - frame.enh[i]);
        if (d > max_abs) max_abs = d;
        sum_abs += d;
        ++count;
      }
    }
    auto stats = gtcrn::summarize(times);
    std::cout << "frames=" << refs.size() << "\n";
    std::cout << "enh_max_abs=" << max_abs << " enh_mean_abs=" << (sum_abs / count) << "\n";
    std::cout << "inference_ms avg=" << stats.avg << " p50=" << stats.p50 << " p95=" << stats.p95
              << " p99=" << stats.p99 << " max=" << stats.max << "\n";
    double mean_abs = sum_abs / count;
    return std::isfinite(max_abs) && std::isfinite(mean_abs) && max_abs <= 1e-4 && mean_abs <= 1e-5 ? 0 : 1;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}
