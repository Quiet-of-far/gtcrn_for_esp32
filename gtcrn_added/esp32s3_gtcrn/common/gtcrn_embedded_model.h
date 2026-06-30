#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace gtcrn_esp {

constexpr int FREQ = 257;
constexpr int ERB_LOW = 65;
constexpr int ERB_HIGH = 64;
constexpr int ERB_FULL = 129;
constexpr int BOT_F = 33;

constexpr int CONV_CACHE_SIZE = 2 * 16 * 16 * BOT_F;
constexpr int TRA_CACHE_SIZE = 2 * 3 * 16;
constexpr int INTER_CACHE_SIZE = 2 * BOT_F * 16;
constexpr int FEAT3_SIZE = 3 * FREQ;
constexpr int ERB_SIZE = 3 * ERB_FULL;
constexpr int XY_SIZE = 16 * ERB_FULL;
constexpr int EN0_SIZE = 16 * 65;
constexpr int ENX_SIZE = 16 * BOT_F;
constexpr int MASK_SIZE = 2 * FREQ;
constexpr int STREAM_X_SIZE = 8 * BOT_F;
constexpr int STREAM_S_SIZE = 24 * BOT_F;
constexpr int STREAM_H_SIZE = 16 * BOT_F;
constexpr int DP_SIZE = 16 * BOT_F;
constexpr int UPSAMPLE_UP_SIZE = 16 * 130;
constexpr int UPSAMPLE_PADDED_SIZE = 16 * 133;

struct Stats {
  float max_abs = 0.0f;
  float mean_abs = 0.0f;
};

enum class ProfileStage : int {
  kFeat = 0,
  kErb,
  kEnc01,
  kEnc234,
  kDp1,
  kDp2,
  kDec012,
  kUp1,
  kUp2,
  kMaskOut,
  kCount,
};

struct ProfileStats {
  std::array<uint64_t, static_cast<int>(ProfileStage::kCount)> total_us{};
  uint32_t calls = 0;
};

struct Workspace {
  std::array<float, CONV_CACHE_SIZE> conv_cache{};
  std::array<uint8_t, 6> conv_heads{};
  std::array<float, TRA_CACHE_SIZE> tra_cache{};
  std::array<float, INTER_CACHE_SIZE> inter_cache{};
  std::array<float, FEAT3_SIZE> feat3{};
  std::array<float, ERB_SIZE> erb{};
  std::array<float, XY_SIZE> x{};
  std::array<float, XY_SIZE> y{};
  std::array<float, EN0_SIZE> en0{};
  std::array<float, ENX_SIZE> en1{};
  std::array<float, ENX_SIZE> en2{};
  std::array<float, ENX_SIZE> en3{};
  std::array<float, ENX_SIZE> en4{};
  std::array<float, MASK_SIZE> mask{};
  std::array<float, STREAM_X_SIZE> stream_x1{};
  std::array<float, STREAM_X_SIZE> stream_x2{};
  std::array<float, STREAM_S_SIZE> stream_s{};
  std::array<float, STREAM_H_SIZE> stream_h{};
  std::array<float, STREAM_H_SIZE> stream_tmp{};
  std::array<float, DP_SIZE> dp_intra{};
  std::array<float, DP_SIZE> dp_fc{};
  std::array<float, DP_SIZE> dp_intra_out{};
  std::array<float, DP_SIZE> dp_inter_fc{};
  std::array<float, UPSAMPLE_UP_SIZE> upsample_up{};
  std::array<float, UPSAMPLE_PADDED_SIZE> upsample_padded{};
};

class EmbeddedModel {
 public:
  explicit EmbeddedModel(Workspace* workspace);

  void reset();
  void infer(const float* mix, float* enh);

 private:
  Workspace* ws_;
};

Stats compare_buffers(const float* ref, const float* got, size_t count);
void reset_profile_stats();
ProfileStats get_profile_stats();

}  // namespace gtcrn_esp
