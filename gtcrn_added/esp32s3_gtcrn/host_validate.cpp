#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>

#include "../cpp_gtcrn/gtcrn_model.h"
#include "common/gtcrn_embedded_model.h"

namespace {

constexpr int kFrames = 100;
constexpr int kBins = gtcrn_esp::FREQ * 2;

float sample_value(int frame, int index) {
  float t = static_cast<float>(frame * 0.03125 + index * 0.017);
  return 0.35f * std::sin(t) + 0.15f * std::cos(0.37f * t);
}

}  // namespace

int main() {
  gtcrn::Model reference;
  gtcrn_esp::Workspace workspace;
  gtcrn_esp::EmbeddedModel embedded(&workspace);

  std::array<float, kBins> mix{};
  std::array<float, kBins> ref{};
  std::array<float, kBins> got{};
  gtcrn_esp::Stats total{};
  float checksum = 0.0f;
  float peak = 0.0f;

  auto t0 = std::chrono::steady_clock::now();
  for (int frame = 0; frame < kFrames; ++frame) {
    for (int i = 0; i < kBins; ++i) mix[i] = sample_value(frame, i);
    reference.infer(mix.data(), ref.data());
    embedded.infer(mix.data(), got.data());
    auto stats = gtcrn_esp::compare_buffers(ref.data(), got.data(), kBins);
    total.max_abs = std::max(total.max_abs, stats.max_abs);
    total.mean_abs += stats.mean_abs;
  }
  auto t1 = std::chrono::steady_clock::now();

  for (float v : got) {
    checksum += v;
    peak = std::max(peak, std::fabs(v));
  }
  total.mean_abs /= kFrames;
  double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  std::cout << "frames=" << kFrames
            << " max_abs=" << total.max_abs
            << " mean_abs=" << total.mean_abs
            << " total_ms=" << ms
            << " checksum=" << checksum
            << " peak=" << peak
            << "\n";
  return total.max_abs > 1e-3f ? 1 : 0;
}
