#include "gtcrn_embedded_model.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#if __has_include("esp_timer.h")
#include "esp_timer.h"
#else
static inline int64_t esp_timer_get_time() { return 0; }
#endif

#if __has_include("esp_attr.h")
#include "esp_attr.h"
#else
#define IRAM_ATTR
#endif

#if __has_include("freertos/FreeRTOS.h")
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#define GTCRN_HAS_FREERTOS 1
#else
#define GTCRN_HAS_FREERTOS 0
#endif

#include "../../cpp_gtcrn/gtcrn_weights_dns3_finetuned.h"

namespace gtcrn_esp {

namespace {

#ifndef GTCRN_ENABLE_PROFILE
#define GTCRN_ENABLE_PROFILE 1
#endif

#ifndef GTCRN_DEBUG_DUALCORE
#define GTCRN_DEBUG_DUALCORE 0
#endif

ProfileStats g_profile_stats{};

inline void add_profile(ProfileStage stage, int64_t delta_us) {
#if GTCRN_ENABLE_PROFILE
  if (delta_us <= 0) return;
  g_profile_stats.total_us[static_cast<int>(stage)] += static_cast<uint64_t>(delta_us);
#else
  (void)stage;
  (void)delta_us;
#endif
}

using gtcrn::w_decoder_de_convs_0_depth_act_weight;
using gtcrn::w_decoder_de_convs_0_depth_bn_bias;
using gtcrn::w_decoder_de_convs_0_depth_bn_running_mean;
using gtcrn::w_decoder_de_convs_0_depth_bn_running_var;
using gtcrn::w_decoder_de_convs_0_depth_bn_weight;
using gtcrn::w_decoder_de_convs_0_depth_conv_ConvTranspose2d_bias;
using gtcrn::w_decoder_de_convs_0_depth_conv_ConvTranspose2d_weight;
using gtcrn::w_decoder_de_convs_0_point_act_weight;
using gtcrn::w_decoder_de_convs_0_point_bn1_bias;
using gtcrn::w_decoder_de_convs_0_point_bn1_running_mean;
using gtcrn::w_decoder_de_convs_0_point_bn1_running_var;
using gtcrn::w_decoder_de_convs_0_point_bn1_weight;
using gtcrn::w_decoder_de_convs_0_point_bn2_bias;
using gtcrn::w_decoder_de_convs_0_point_bn2_running_mean;
using gtcrn::w_decoder_de_convs_0_point_bn2_running_var;
using gtcrn::w_decoder_de_convs_0_point_bn2_weight;
using gtcrn::w_decoder_de_convs_0_point_conv1_bias;
using gtcrn::w_decoder_de_convs_0_point_conv1_weight;
using gtcrn::w_decoder_de_convs_0_point_conv2_bias;
using gtcrn::w_decoder_de_convs_0_point_conv2_weight;
using gtcrn::w_decoder_de_convs_0_tra_att_fc_bias;
using gtcrn::w_decoder_de_convs_0_tra_att_fc_weight;
using gtcrn::w_decoder_de_convs_0_tra_att_gru_bias_hh_l0;
using gtcrn::w_decoder_de_convs_0_tra_att_gru_bias_ih_l0;
using gtcrn::w_decoder_de_convs_0_tra_att_gru_weight_hh_l0;
using gtcrn::w_decoder_de_convs_0_tra_att_gru_weight_ih_l0;
using gtcrn::w_decoder_de_convs_1_depth_act_weight;
using gtcrn::w_decoder_de_convs_1_depth_bn_bias;
using gtcrn::w_decoder_de_convs_1_depth_bn_running_mean;
using gtcrn::w_decoder_de_convs_1_depth_bn_running_var;
using gtcrn::w_decoder_de_convs_1_depth_bn_weight;
using gtcrn::w_decoder_de_convs_1_depth_conv_ConvTranspose2d_bias;
using gtcrn::w_decoder_de_convs_1_depth_conv_ConvTranspose2d_weight;
using gtcrn::w_decoder_de_convs_1_point_act_weight;
using gtcrn::w_decoder_de_convs_1_point_bn1_bias;
using gtcrn::w_decoder_de_convs_1_point_bn1_running_mean;
using gtcrn::w_decoder_de_convs_1_point_bn1_running_var;
using gtcrn::w_decoder_de_convs_1_point_bn1_weight;
using gtcrn::w_decoder_de_convs_1_point_bn2_bias;
using gtcrn::w_decoder_de_convs_1_point_bn2_running_mean;
using gtcrn::w_decoder_de_convs_1_point_bn2_running_var;
using gtcrn::w_decoder_de_convs_1_point_bn2_weight;
using gtcrn::w_decoder_de_convs_1_point_conv1_bias;
using gtcrn::w_decoder_de_convs_1_point_conv1_weight;
using gtcrn::w_decoder_de_convs_1_point_conv2_bias;
using gtcrn::w_decoder_de_convs_1_point_conv2_weight;
using gtcrn::w_decoder_de_convs_1_tra_att_fc_bias;
using gtcrn::w_decoder_de_convs_1_tra_att_fc_weight;
using gtcrn::w_decoder_de_convs_1_tra_att_gru_bias_hh_l0;
using gtcrn::w_decoder_de_convs_1_tra_att_gru_bias_ih_l0;
using gtcrn::w_decoder_de_convs_1_tra_att_gru_weight_hh_l0;
using gtcrn::w_decoder_de_convs_1_tra_att_gru_weight_ih_l0;
using gtcrn::w_decoder_de_convs_2_depth_act_weight;
using gtcrn::w_decoder_de_convs_2_depth_bn_bias;
using gtcrn::w_decoder_de_convs_2_depth_bn_running_mean;
using gtcrn::w_decoder_de_convs_2_depth_bn_running_var;
using gtcrn::w_decoder_de_convs_2_depth_bn_weight;
using gtcrn::w_decoder_de_convs_2_depth_conv_ConvTranspose2d_bias;
using gtcrn::w_decoder_de_convs_2_depth_conv_ConvTranspose2d_weight;
using gtcrn::w_decoder_de_convs_2_point_act_weight;
using gtcrn::w_decoder_de_convs_2_point_bn1_bias;
using gtcrn::w_decoder_de_convs_2_point_bn1_running_mean;
using gtcrn::w_decoder_de_convs_2_point_bn1_running_var;
using gtcrn::w_decoder_de_convs_2_point_bn1_weight;
using gtcrn::w_decoder_de_convs_2_point_bn2_bias;
using gtcrn::w_decoder_de_convs_2_point_bn2_running_mean;
using gtcrn::w_decoder_de_convs_2_point_bn2_running_var;
using gtcrn::w_decoder_de_convs_2_point_bn2_weight;
using gtcrn::w_decoder_de_convs_2_point_conv1_bias;
using gtcrn::w_decoder_de_convs_2_point_conv1_weight;
using gtcrn::w_decoder_de_convs_2_point_conv2_bias;
using gtcrn::w_decoder_de_convs_2_point_conv2_weight;
using gtcrn::w_decoder_de_convs_2_tra_att_fc_bias;
using gtcrn::w_decoder_de_convs_2_tra_att_fc_weight;
using gtcrn::w_decoder_de_convs_2_tra_att_gru_bias_hh_l0;
using gtcrn::w_decoder_de_convs_2_tra_att_gru_bias_ih_l0;
using gtcrn::w_decoder_de_convs_2_tra_att_gru_weight_hh_l0;
using gtcrn::w_decoder_de_convs_2_tra_att_gru_weight_ih_l0;
using gtcrn::w_decoder_de_convs_3_act_weight;
using gtcrn::w_decoder_de_convs_3_bn_bias;
using gtcrn::w_decoder_de_convs_3_bn_running_mean;
using gtcrn::w_decoder_de_convs_3_bn_running_var;
using gtcrn::w_decoder_de_convs_3_bn_weight;
using gtcrn::w_decoder_de_convs_3_conv_conv_bias;
using gtcrn::w_decoder_de_convs_3_conv_conv_weight;
using gtcrn::w_decoder_de_convs_4_bn_bias;
using gtcrn::w_decoder_de_convs_4_bn_running_mean;
using gtcrn::w_decoder_de_convs_4_bn_running_var;
using gtcrn::w_decoder_de_convs_4_bn_weight;
using gtcrn::w_decoder_de_convs_4_conv_conv_bias;
using gtcrn::w_decoder_de_convs_4_conv_conv_weight;
using gtcrn::w_dpgrnn1_inter_fc_bias;
using gtcrn::w_dpgrnn1_inter_fc_weight;
using gtcrn::w_dpgrnn1_inter_ln_bias;
using gtcrn::w_dpgrnn1_inter_ln_weight;
using gtcrn::w_dpgrnn1_inter_rnn_rnn1_bias_hh_l0;
using gtcrn::w_dpgrnn1_inter_rnn_rnn1_bias_ih_l0;
using gtcrn::w_dpgrnn1_inter_rnn_rnn1_weight_hh_l0;
using gtcrn::w_dpgrnn1_inter_rnn_rnn1_weight_ih_l0;
using gtcrn::w_dpgrnn1_inter_rnn_rnn2_bias_hh_l0;
using gtcrn::w_dpgrnn1_inter_rnn_rnn2_bias_ih_l0;
using gtcrn::w_dpgrnn1_inter_rnn_rnn2_weight_hh_l0;
using gtcrn::w_dpgrnn1_inter_rnn_rnn2_weight_ih_l0;
using gtcrn::w_dpgrnn1_intra_fc_bias;
using gtcrn::w_dpgrnn1_intra_fc_weight;
using gtcrn::w_dpgrnn1_intra_ln_bias;
using gtcrn::w_dpgrnn1_intra_ln_weight;
using gtcrn::w_dpgrnn1_intra_rnn_rnn1_bias_hh_l0;
using gtcrn::w_dpgrnn1_intra_rnn_rnn1_bias_hh_l0_reverse;
using gtcrn::w_dpgrnn1_intra_rnn_rnn1_bias_ih_l0;
using gtcrn::w_dpgrnn1_intra_rnn_rnn1_bias_ih_l0_reverse;
using gtcrn::w_dpgrnn1_intra_rnn_rnn1_weight_hh_l0;
using gtcrn::w_dpgrnn1_intra_rnn_rnn1_weight_hh_l0_reverse;
using gtcrn::w_dpgrnn1_intra_rnn_rnn1_weight_ih_l0;
using gtcrn::w_dpgrnn1_intra_rnn_rnn1_weight_ih_l0_reverse;
using gtcrn::w_dpgrnn1_intra_rnn_rnn2_bias_hh_l0;
using gtcrn::w_dpgrnn1_intra_rnn_rnn2_bias_hh_l0_reverse;
using gtcrn::w_dpgrnn1_intra_rnn_rnn2_bias_ih_l0;
using gtcrn::w_dpgrnn1_intra_rnn_rnn2_bias_ih_l0_reverse;
using gtcrn::w_dpgrnn1_intra_rnn_rnn2_weight_hh_l0;
using gtcrn::w_dpgrnn1_intra_rnn_rnn2_weight_hh_l0_reverse;
using gtcrn::w_dpgrnn1_intra_rnn_rnn2_weight_ih_l0;
using gtcrn::w_dpgrnn1_intra_rnn_rnn2_weight_ih_l0_reverse;
using gtcrn::w_dpgrnn2_inter_fc_bias;
using gtcrn::w_dpgrnn2_inter_fc_weight;
using gtcrn::w_dpgrnn2_inter_ln_bias;
using gtcrn::w_dpgrnn2_inter_ln_weight;
using gtcrn::w_dpgrnn2_inter_rnn_rnn1_bias_hh_l0;
using gtcrn::w_dpgrnn2_inter_rnn_rnn1_bias_ih_l0;
using gtcrn::w_dpgrnn2_inter_rnn_rnn1_weight_hh_l0;
using gtcrn::w_dpgrnn2_inter_rnn_rnn1_weight_ih_l0;
using gtcrn::w_dpgrnn2_inter_rnn_rnn2_bias_hh_l0;
using gtcrn::w_dpgrnn2_inter_rnn_rnn2_bias_ih_l0;
using gtcrn::w_dpgrnn2_inter_rnn_rnn2_weight_hh_l0;
using gtcrn::w_dpgrnn2_inter_rnn_rnn2_weight_ih_l0;
using gtcrn::w_dpgrnn2_intra_fc_bias;
using gtcrn::w_dpgrnn2_intra_fc_weight;
using gtcrn::w_dpgrnn2_intra_ln_bias;
using gtcrn::w_dpgrnn2_intra_ln_weight;
using gtcrn::w_dpgrnn2_intra_rnn_rnn1_bias_hh_l0;
using gtcrn::w_dpgrnn2_intra_rnn_rnn1_bias_hh_l0_reverse;
using gtcrn::w_dpgrnn2_intra_rnn_rnn1_bias_ih_l0;
using gtcrn::w_dpgrnn2_intra_rnn_rnn1_bias_ih_l0_reverse;
using gtcrn::w_dpgrnn2_intra_rnn_rnn1_weight_hh_l0;
using gtcrn::w_dpgrnn2_intra_rnn_rnn1_weight_hh_l0_reverse;
using gtcrn::w_dpgrnn2_intra_rnn_rnn1_weight_ih_l0;
using gtcrn::w_dpgrnn2_intra_rnn_rnn1_weight_ih_l0_reverse;
using gtcrn::w_dpgrnn2_intra_rnn_rnn2_bias_hh_l0;
using gtcrn::w_dpgrnn2_intra_rnn_rnn2_bias_hh_l0_reverse;
using gtcrn::w_dpgrnn2_intra_rnn_rnn2_bias_ih_l0;
using gtcrn::w_dpgrnn2_intra_rnn_rnn2_bias_ih_l0_reverse;
using gtcrn::w_dpgrnn2_intra_rnn_rnn2_weight_hh_l0;
using gtcrn::w_dpgrnn2_intra_rnn_rnn2_weight_hh_l0_reverse;
using gtcrn::w_dpgrnn2_intra_rnn_rnn2_weight_ih_l0;
using gtcrn::w_dpgrnn2_intra_rnn_rnn2_weight_ih_l0_reverse;
using gtcrn::w_encoder_en_convs_0_act_weight;
using gtcrn::w_encoder_en_convs_0_bn_bias;
using gtcrn::w_encoder_en_convs_0_bn_running_mean;
using gtcrn::w_encoder_en_convs_0_bn_running_var;
using gtcrn::w_encoder_en_convs_0_bn_weight;
using gtcrn::w_encoder_en_convs_0_conv_bias;
using gtcrn::w_encoder_en_convs_0_conv_weight;
using gtcrn::w_encoder_en_convs_1_act_weight;
using gtcrn::w_encoder_en_convs_1_bn_bias;
using gtcrn::w_encoder_en_convs_1_bn_running_mean;
using gtcrn::w_encoder_en_convs_1_bn_running_var;
using gtcrn::w_encoder_en_convs_1_bn_weight;
using gtcrn::w_encoder_en_convs_1_conv_bias;
using gtcrn::w_encoder_en_convs_1_conv_weight;
using gtcrn::w_encoder_en_convs_2_depth_act_weight;
using gtcrn::w_encoder_en_convs_2_depth_bn_bias;
using gtcrn::w_encoder_en_convs_2_depth_bn_running_mean;
using gtcrn::w_encoder_en_convs_2_depth_bn_running_var;
using gtcrn::w_encoder_en_convs_2_depth_bn_weight;
using gtcrn::w_encoder_en_convs_2_depth_conv_Conv2d_bias;
using gtcrn::w_encoder_en_convs_2_depth_conv_Conv2d_weight;
using gtcrn::w_encoder_en_convs_2_point_act_weight;
using gtcrn::w_encoder_en_convs_2_point_bn1_bias;
using gtcrn::w_encoder_en_convs_2_point_bn1_running_mean;
using gtcrn::w_encoder_en_convs_2_point_bn1_running_var;
using gtcrn::w_encoder_en_convs_2_point_bn1_weight;
using gtcrn::w_encoder_en_convs_2_point_bn2_bias;
using gtcrn::w_encoder_en_convs_2_point_bn2_running_mean;
using gtcrn::w_encoder_en_convs_2_point_bn2_running_var;
using gtcrn::w_encoder_en_convs_2_point_bn2_weight;
using gtcrn::w_encoder_en_convs_2_point_conv1_bias;
using gtcrn::w_encoder_en_convs_2_point_conv1_weight;
using gtcrn::w_encoder_en_convs_2_point_conv2_bias;
using gtcrn::w_encoder_en_convs_2_point_conv2_weight;
using gtcrn::w_encoder_en_convs_2_tra_att_fc_bias;
using gtcrn::w_encoder_en_convs_2_tra_att_fc_weight;
using gtcrn::w_encoder_en_convs_2_tra_att_gru_bias_hh_l0;
using gtcrn::w_encoder_en_convs_2_tra_att_gru_bias_ih_l0;
using gtcrn::w_encoder_en_convs_2_tra_att_gru_weight_hh_l0;
using gtcrn::w_encoder_en_convs_2_tra_att_gru_weight_ih_l0;
using gtcrn::w_encoder_en_convs_3_depth_act_weight;
using gtcrn::w_encoder_en_convs_3_depth_bn_bias;
using gtcrn::w_encoder_en_convs_3_depth_bn_running_mean;
using gtcrn::w_encoder_en_convs_3_depth_bn_running_var;
using gtcrn::w_encoder_en_convs_3_depth_bn_weight;
using gtcrn::w_encoder_en_convs_3_depth_conv_Conv2d_bias;
using gtcrn::w_encoder_en_convs_3_depth_conv_Conv2d_weight;
using gtcrn::w_encoder_en_convs_3_point_act_weight;
using gtcrn::w_encoder_en_convs_3_point_bn1_bias;
using gtcrn::w_encoder_en_convs_3_point_bn1_running_mean;
using gtcrn::w_encoder_en_convs_3_point_bn1_running_var;
using gtcrn::w_encoder_en_convs_3_point_bn1_weight;
using gtcrn::w_encoder_en_convs_3_point_bn2_bias;
using gtcrn::w_encoder_en_convs_3_point_bn2_running_mean;
using gtcrn::w_encoder_en_convs_3_point_bn2_running_var;
using gtcrn::w_encoder_en_convs_3_point_bn2_weight;
using gtcrn::w_encoder_en_convs_3_point_conv1_bias;
using gtcrn::w_encoder_en_convs_3_point_conv1_weight;
using gtcrn::w_encoder_en_convs_3_point_conv2_bias;
using gtcrn::w_encoder_en_convs_3_point_conv2_weight;
using gtcrn::w_encoder_en_convs_3_tra_att_fc_bias;
using gtcrn::w_encoder_en_convs_3_tra_att_fc_weight;
using gtcrn::w_encoder_en_convs_3_tra_att_gru_bias_hh_l0;
using gtcrn::w_encoder_en_convs_3_tra_att_gru_bias_ih_l0;
using gtcrn::w_encoder_en_convs_3_tra_att_gru_weight_hh_l0;
using gtcrn::w_encoder_en_convs_3_tra_att_gru_weight_ih_l0;
using gtcrn::w_encoder_en_convs_4_depth_act_weight;
using gtcrn::w_encoder_en_convs_4_depth_bn_bias;
using gtcrn::w_encoder_en_convs_4_depth_bn_running_mean;
using gtcrn::w_encoder_en_convs_4_depth_bn_running_var;
using gtcrn::w_encoder_en_convs_4_depth_bn_weight;
using gtcrn::w_encoder_en_convs_4_depth_conv_Conv2d_bias;
using gtcrn::w_encoder_en_convs_4_depth_conv_Conv2d_weight;
using gtcrn::w_encoder_en_convs_4_point_act_weight;
using gtcrn::w_encoder_en_convs_4_point_bn1_bias;
using gtcrn::w_encoder_en_convs_4_point_bn1_running_mean;
using gtcrn::w_encoder_en_convs_4_point_bn1_running_var;
using gtcrn::w_encoder_en_convs_4_point_bn1_weight;
using gtcrn::w_encoder_en_convs_4_point_bn2_bias;
using gtcrn::w_encoder_en_convs_4_point_bn2_running_mean;
using gtcrn::w_encoder_en_convs_4_point_bn2_running_var;
using gtcrn::w_encoder_en_convs_4_point_bn2_weight;
using gtcrn::w_encoder_en_convs_4_point_conv1_bias;
using gtcrn::w_encoder_en_convs_4_point_conv1_weight;
using gtcrn::w_encoder_en_convs_4_point_conv2_bias;
using gtcrn::w_encoder_en_convs_4_point_conv2_weight;
using gtcrn::w_encoder_en_convs_4_tra_att_fc_bias;
using gtcrn::w_encoder_en_convs_4_tra_att_fc_weight;
using gtcrn::w_encoder_en_convs_4_tra_att_gru_bias_hh_l0;
using gtcrn::w_encoder_en_convs_4_tra_att_gru_bias_ih_l0;
using gtcrn::w_encoder_en_convs_4_tra_att_gru_weight_hh_l0;
using gtcrn::w_encoder_en_convs_4_tra_att_gru_weight_ih_l0;
using gtcrn::w_erb_erb_fc_weight;
using gtcrn::w_erb_ierb_fc_weight;

constexpr float kEps = 1e-12f;

inline int idx2(int c, int f, int F) { return c * F + f; }
inline int conv_cache_idx(int stream, int c, int t, int f) {
  return ((stream * 16 + c) * 16 + t) * BOT_F + f;
}
inline int tra_cache_idx(int block, int slot, int i) { return (block * 3 + slot) * 16 + i; }
inline int inter_cache_idx(int block, int f, int i) { return (block * BOT_F + f) * 16 + i; }

struct RnnWeights {
  const float* wih;
  const float* whh;
  const float* bih;
  const float* bhh;
};

struct StreamWeights {
  const float* point_conv1_w;
  const float* point_conv1_b;
  const float* point_bn1_w;
  const float* point_bn1_b;
  const float* point_bn1_mean;
  const float* point_bn1_var;
  float point_act;
  const float* depth_w;
  const float* depth_b;
  const float* depth_bn_w;
  const float* depth_bn_b;
  const float* depth_bn_mean;
  const float* depth_bn_var;
  float depth_act;
  const float* point_conv2_w;
  const float* point_conv2_b;
  const float* point_bn2_w;
  const float* point_bn2_b;
  const float* point_bn2_mean;
  const float* point_bn2_var;
  RnnWeights tra_gru;
  const float* tra_fc_w;
  const float* tra_fc_b;
};

struct DpgrnnWeights {
  RnnWeights intra_rnn[2][2];
  RnnWeights inter_rnn[2];
  const float* intra_fc_w;
  const float* intra_fc_b;
  const float* intra_ln_w;
  const float* intra_ln_b;
  const float* inter_fc_w;
  const float* inter_fc_b;
  const float* inter_ln_w;
  const float* inter_ln_b;
};

struct ConvBnActWeights {
  const float* conv_w;
  const float* conv_b;
  const float* bn_w;
  const float* bn_b;
  const float* bn_mean;
  const float* bn_var;
  float act;
};

template <int Rows, int Cols, int MaxNnz>
struct SparseMatrix {
  std::array<int16_t, Rows + 1> row_ptr{};
  std::array<int16_t, MaxNnz> col_idx{};
  std::array<float, MaxNnz> values{};
  int nnz = 0;
  bool ready = false;
};

SparseMatrix<ERB_HIGH, 192, 512> g_erb_sparse;
SparseMatrix<192, ERB_HIGH, 512> g_ierb_sparse;

#if GTCRN_HAS_FREERTOS
enum class WorkerJobType : uint8_t { kIdle = 0, kDpIntra };

struct WorkerJob {
  WorkerJobType type = WorkerJobType::kIdle;
  Workspace* ws = nullptr;
  const DpgrnnWeights* dw = nullptr;
  int inter_base = 0;
  const float* x = nullptr;
  float* intra = nullptr;
  TaskHandle_t caller = nullptr;
};

struct WorkerState {
  TaskHandle_t handle = nullptr;
  WorkerJob job{};
  bool initialized = false;
};

WorkerState g_worker{};
#endif

#if GTCRN_DEBUG_DUALCORE
struct DebugDpBuffers {
  std::array<float, ENX_SIZE> x_in{};
  std::array<float, DP_SIZE> intra{};
  std::array<float, DP_SIZE> fc{};
  std::array<float, DP_SIZE> intra_out{};
  std::array<float, DP_SIZE> inter_linear_in{};
  std::array<float, DP_SIZE> inter_linear_in_parallel{};
  std::array<float, DP_SIZE> inter_fc{};
  std::array<float, ENX_SIZE> x_out{};
  std::array<float, DP_SIZE> inter_cache_block{};
  bool printed = false;
};

DebugDpBuffers g_debug_dp{};

inline int local_inter_idx(int f, int c) { return f * 16 + c; }
#endif

#ifndef GTCRN_APPROX_ACTIVATION
#define GTCRN_APPROX_ACTIVATION 0
#endif

IRAM_ATTR inline float fast_tanh(float x) {
  if (x < -3.0f) return -0.9950548f;
  if (x > 3.0f) return 0.9950548f;
  float x2 = x * x;
  return x * (27.0f + x2) / (27.0f + 9.0f * x2);
}

IRAM_ATTR inline float tanh_approx(float x) {
#if GTCRN_APPROX_ACTIVATION
  return fast_tanh(x);
#else
  return tanhf(x);
#endif
}

IRAM_ATTR inline float sigmoid(float x) {
#if GTCRN_APPROX_ACTIVATION
  return 0.5f * (fast_tanh(0.5f * x) + 1.0f);
#else
  return 1.0f / (1.0f + expf(-x));
#endif
}

template <int InSize, int Hidden>
inline void gru_cell_fixed(const RnnWeights& rw, const float* x, float* h);

template <int OutSize, int InSize>
inline void linear_fixed(const float* weight, const float* bias, const float* x, float* y);

void layer_norm(float* x, int count, const float* gamma, const float* beta);

IRAM_ATTR void dpgrnn_intra_group(const DpgrnnWeights& dw, const float* x, int grp, float* intra) {
  for (int dir = 0; dir < 2; ++dir) {
    float hstate[4] = {};
    const RnnWeights& rw = dw.intra_rnn[grp][dir];
    for (int step = 0; step < BOT_F; ++step) {
      int f = dir == 0 ? step : (BOT_F - 1 - step);
      float xin[8];
      for (int i = 0; i < 8; ++i) xin[i] = x[idx2(grp * 8 + i, f, BOT_F)];
      gru_cell_fixed<8, 4>(rw, xin, hstate);
      for (int i = 0; i < 4; ++i) intra[idx2(grp * 8 + dir * 4 + i, f, BOT_F)] = hstate[i];
    }
  }
}

IRAM_ATTR void dpgrnn_inter_group(Workspace* ws, const DpgrnnWeights& dw, int inter_base,
                                  const float* intra_out, int grp, float* inter_buf) {
  const RnnWeights& rw = dw.inter_rnn[grp];
  for (int f = 0; f < BOT_F; ++f) {
    float xin[8];
    float hstate[8];
    for (int i = 0; i < 8; ++i) {
      xin[i] = intra_out[idx2(grp * 8 + i, f, BOT_F)];
      hstate[i] = ws->inter_cache[inter_cache_idx(inter_base, f, grp * 8 + i)];
    }
    gru_cell_fixed<8, 8>(rw, xin, hstate);
    for (int i = 0; i < 8; ++i) {
      inter_buf[idx2(grp * 8 + i, f, BOT_F)] = hstate[i];
      ws->inter_cache[inter_cache_idx(inter_base, f, grp * 8 + i)] = hstate[i];
    }
  }
}

#if GTCRN_DEBUG_DUALCORE
void dpgrnn_reference_debug(const DpgrnnWeights& dw, const float* x_in, int inter_base,
                            const float* inter_cache_src, DebugDpBuffers* dbg) {
  std::fill(dbg->intra.begin(), dbg->intra.end(), 0.0f);
  for (int grp = 0; grp < 2; ++grp) {
    for (int dir = 0; dir < 2; ++dir) {
      float hstate[4] = {};
      const RnnWeights& rw = dw.intra_rnn[grp][dir];
      for (int step = 0; step < BOT_F; ++step) {
        int f = dir == 0 ? step : (BOT_F - 1 - step);
        float xin[8];
        for (int i = 0; i < 8; ++i) xin[i] = x_in[idx2(grp * 8 + i, f, BOT_F)];
        gru_cell_fixed<8, 4>(rw, xin, hstate);
        for (int i = 0; i < 4; ++i) dbg->intra[idx2(grp * 8 + dir * 4 + i, f, BOT_F)] = hstate[i];
      }
    }
  }

  for (int f = 0; f < BOT_F; ++f) {
    float in[16];
    float out[16];
    for (int c = 0; c < 16; ++c) in[c] = dbg->intra[idx2(c, f, BOT_F)];
    linear_fixed<16, 16>(dw.intra_fc_w, dw.intra_fc_b, in, out);
    for (int c = 0; c < 16; ++c) dbg->fc[f * 16 + c] = out[c];
  }
  layer_norm(dbg->fc.data(), DP_SIZE, dw.intra_ln_w, dw.intra_ln_b);

  for (int f = 0; f < BOT_F; ++f) {
    for (int c = 0; c < 16; ++c) {
      dbg->intra_out[idx2(c, f, BOT_F)] = x_in[idx2(c, f, BOT_F)] + dbg->fc[f * 16 + c];
    }
  }

  std::fill(dbg->inter_linear_in.begin(), dbg->inter_linear_in.end(), 0.0f);
  for (int f = 0; f < BOT_F; ++f) {
    float ycat[16];
    for (int grp = 0; grp < 2; ++grp) {
      const RnnWeights& rw = dw.inter_rnn[grp];
      float xin[8];
      float hstate[8];
      for (int i = 0; i < 8; ++i) {
        xin[i] = dbg->intra_out[idx2(grp * 8 + i, f, BOT_F)];
        hstate[i] = inter_cache_src[local_inter_idx(f, grp * 8 + i)];
      }
      gru_cell_fixed<8, 8>(rw, xin, hstate);
      for (int i = 0; i < 8; ++i) {
        ycat[grp * 8 + i] = hstate[i];
      }
    }
    for (int c = 0; c < 16; ++c) dbg->inter_linear_in[local_inter_idx(f, c)] = ycat[c];
    float out[16];
    linear_fixed<16, 16>(dw.inter_fc_w, dw.inter_fc_b, ycat, out);
    for (int c = 0; c < 16; ++c) dbg->inter_fc[f * 16 + c] = out[c];
  }
  layer_norm(dbg->inter_fc.data(), DP_SIZE, dw.inter_ln_w, dw.inter_ln_b);
  for (int f = 0; f < BOT_F; ++f) {
    for (int c = 0; c < 16; ++c) {
      dbg->x_out[idx2(c, f, BOT_F)] = dbg->intra_out[idx2(c, f, BOT_F)] + dbg->inter_fc[f * 16 + c];
    }
  }
}
#endif

#if GTCRN_HAS_FREERTOS
void worker_task(void*) {
  while (true) {
    ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
    WorkerJob job = g_worker.job;
    if (job.type == WorkerJobType::kDpIntra) {
      dpgrnn_intra_group(*job.dw, job.x, 1, job.intra);
    }
    xTaskNotifyGive(job.caller);
  }
}

void ensure_worker_started() {
  if (g_worker.initialized) return;
  BaseType_t ok = xTaskCreatePinnedToCore(worker_task, "gtcrn_dp", 16384, nullptr, 5, &g_worker.handle, 1);
  g_worker.initialized = (ok == pdPASS && g_worker.handle != nullptr);
}
#else
void ensure_worker_started() {}
#endif

template <int Rows, int Cols, int MaxNnz>
void init_sparse_matrix(SparseMatrix<Rows, Cols, MaxNnz>* sm, const float* dense) {
  if (sm->ready) return;
  int nnz = 0;
  for (int r = 0; r < Rows; ++r) {
    sm->row_ptr[r] = static_cast<int16_t>(nnz);
    const float* row = dense + r * Cols;
    for (int c = 0; c < Cols; ++c) {
      float v = row[c];
      if (v == 0.0f) continue;
      sm->col_idx[nnz] = static_cast<int16_t>(c);
      sm->values[nnz] = v;
      ++nnz;
    }
  }
  sm->row_ptr[Rows] = static_cast<int16_t>(nnz);
  sm->nnz = nnz;
  sm->ready = true;
}

template <int Rows, int Cols, int MaxNnz>
void sparse_linear(const SparseMatrix<Rows, Cols, MaxNnz>& sm, const float* x, float* y) {
  for (int r = 0; r < Rows; ++r) {
    float acc = 0.0f;
    for (int i = sm.row_ptr[r]; i < sm.row_ptr[r + 1]; ++i) {
      acc += sm.values[i] * x[sm.col_idx[i]];
    }
    y[r] = acc;
  }
}

IRAM_ATTR void batch_norm(float* x, int C, int F, const float* gamma, const float* beta,
                          const float* mean, const float* var) {
  for (int c = 0; c < C; ++c) {
    float scale = gamma[c] / sqrtf(var[c] + 1e-5f);
    float bias = beta[c] - mean[c] * scale;
    for (int f = 0; f < F; ++f) x[idx2(c, f, F)] = x[idx2(c, f, F)] * scale + bias;
  }
}

IRAM_ATTR void prelu(float* x, int count, float a) {
  for (int i = 0; i < count; ++i) {
    if (x[i] < 0.0f) x[i] *= a;
  }
}

template <int OutSize, int InSize>
IRAM_ATTR inline void linear_fixed(const float* weight, const float* bias, const float* x, float* y) {
  for (int o = 0; o < OutSize; ++o) {
    float acc = bias[o];
    const float* row = weight + o * InSize;
    for (int i = 0; i < InSize; ++i) acc += row[i] * x[i];
    y[o] = acc;
  }
}

void linear(const float* weight, const float* bias, const float* x, float* y, int out_size, int in_size) {
  for (int o = 0; o < out_size; ++o) {
    float acc = bias[o];
    const float* row = weight + o * in_size;
    for (int i = 0; i < in_size; ++i) acc += row[i] * x[i];
    y[o] = acc;
  }
}

template <int InC, int OutC, int F>
IRAM_ATTR inline void conv1x1_fixed(const float* weight, const float* bias, const float* x, float* y) {
  for (int oc = 0; oc < OutC; ++oc) {
    for (int f = 0; f < F; ++f) {
      float acc = bias[oc];
      for (int ic = 0; ic < InC; ++ic) {
        acc += weight[oc * InC + ic] * x[idx2(ic, f, F)];
      }
      y[idx2(oc, f, F)] = acc;
    }
  }
}

void conv1x1(const float* weight, const float* bias, const float* x, float* y, int in_c, int out_c, int F) {
  for (int oc = 0; oc < out_c; ++oc) {
    for (int f = 0; f < F; ++f) {
      float acc = bias[oc];
      for (int ic = 0; ic < in_c; ++ic) {
        acc += weight[oc * in_c + ic] * x[idx2(ic, f, F)];
      }
      y[idx2(oc, f, F)] = acc;
    }
  }
}

IRAM_ATTR void conv1x5(const float* weight, const float* bias, const float* x, float* y, int in_c, int out_c,
                       int in_f, int out_f, int stride, int groups) {
  std::fill(y, y + out_c * out_f, 0.0f);
  int out_per_group = out_c / groups;
  int in_per_group = in_c / groups;
  for (int oc = 0; oc < out_c; ++oc) {
    int g = oc / out_per_group;
    for (int of = 0; of < out_f; ++of) {
      float acc = bias[oc];
      int center = of * stride - 2;
      for (int icg = 0; icg < in_per_group; ++icg) {
        int ic = g * in_per_group + icg;
        for (int k = 0; k < 5; ++k) {
          int inf = center + k;
          if (inf < 0 || inf >= in_f) continue;
          int widx = ((oc * in_per_group + icg) * 5) + k;
          acc += weight[widx] * x[idx2(ic, inf, in_f)];
        }
      }
      y[idx2(oc, of, out_f)] = acc;
    }
  }
}

IRAM_ATTR void upsample_conv1x5(const float* weight, const float* bias, const float* x, float* y, int in_c, int out_c,
                                int in_f, int out_f, int groups, float* up, float* padded) {
  int up_f = in_f * 2;
  std::fill(up, up + in_c * up_f, 0.0f);
  for (int c = 0; c < in_c; ++c) {
    for (int f = 0; f < in_f; ++f) up[idx2(c, f * 2, up_f)] = x[idx2(c, f, in_f)];
  }
  std::fill(padded, padded + in_c * (up_f + 3), 0.0f);
  for (int c = 0; c < in_c; ++c) {
    for (int f = 0; f < up_f; ++f) padded[idx2(c, f + 2, up_f + 3)] = up[idx2(c, f, up_f)];
  }
  std::fill(y, y + out_c * out_f, 0.0f);
  int out_per_group = out_c / groups;
  int in_per_group = in_c / groups;
  for (int oc = 0; oc < out_c; ++oc) {
    int g = oc / out_per_group;
    for (int of = 0; of < out_f; ++of) {
      float acc = bias[oc];
      for (int icg = 0; icg < in_per_group; ++icg) {
        int ic = g * in_per_group + icg;
        for (int k = 0; k < 5; ++k) {
          int pf = of + k;
          int widx = ((oc * in_per_group + icg) * 5) + k;
          acc += weight[widx] * padded[idx2(ic, pf, up_f + 3)];
        }
      }
      y[idx2(oc, of, out_f)] = acc;
    }
  }
}

IRAM_ATTR void sfe(const float* x, float* y, int C, int F) {
  for (int c = 0; c < C; ++c) {
    for (int k = 0; k < 3; ++k) {
      for (int f = 0; f < F; ++f) {
        int src_f = f + k - 1;
        y[(c * 3 + k) * F + f] = (src_f >= 0 && src_f < F) ? x[idx2(c, src_f, F)] : 0.0f;
      }
    }
  }
}

template <int InSize, int Hidden>
IRAM_ATTR inline void gru_cell_fixed(const RnnWeights& rw, const float* x, float* h) {
  float gi[3 * Hidden] = {};
  float gh[3 * Hidden] = {};
  for (int o = 0; o < 3 * Hidden; ++o) {
    float acc_i = rw.bih[o];
    float acc_h = rw.bhh[o];
    const float* row_i = rw.wih + o * InSize;
    const float* row_h = rw.whh + o * Hidden;
    for (int i = 0; i < InSize; ++i) acc_i += row_i[i] * x[i];
    for (int i = 0; i < Hidden; ++i) acc_h += row_h[i] * h[i];
    gi[o] = acc_i;
    gh[o] = acc_h;
  }
  for (int i = 0; i < Hidden; ++i) {
    float r = sigmoid(gi[i] + gh[i]);
    float z = sigmoid(gi[i + Hidden] + gh[i + Hidden]);
    float n = tanh_approx(gi[i + 2 * Hidden] + r * gh[i + 2 * Hidden]);
    h[i] = n + z * (h[i] - n);
  }
}

IRAM_ATTR void layer_norm(float* x, int count, const float* gamma, const float* beta) {
  float sum = 0.0f;
  for (int i = 0; i < count; ++i) sum += x[i];
  float mean = sum / count;
  float var_sum = 0.0f;
  for (int i = 0; i < count; ++i) {
    float d = x[i] - mean;
    var_sum += d * d;
  }
  float inv = 1.0f / sqrtf(var_sum / count + 1e-8f);
  for (int i = 0; i < count; ++i) {
    x[i] = (x[i] - mean) * inv * gamma[i] + beta[i];
  }
}

IRAM_ATTR void add_inplace(float* x, const float* y, int count) {
  for (int i = 0; i < count; ++i) x[i] += y[i];
}

const ConvBnActWeights kEnc0 = {
    w_encoder_en_convs_0_conv_weight, w_encoder_en_convs_0_conv_bias,
    w_encoder_en_convs_0_bn_weight, w_encoder_en_convs_0_bn_bias,
    w_encoder_en_convs_0_bn_running_mean, w_encoder_en_convs_0_bn_running_var,
    w_encoder_en_convs_0_act_weight[0]};
const ConvBnActWeights kEnc1 = {
    w_encoder_en_convs_1_conv_weight, w_encoder_en_convs_1_conv_bias,
    w_encoder_en_convs_1_bn_weight, w_encoder_en_convs_1_bn_bias,
    w_encoder_en_convs_1_bn_running_mean, w_encoder_en_convs_1_bn_running_var,
    w_encoder_en_convs_1_act_weight[0]};
const ConvBnActWeights kDec3 = {
    w_decoder_de_convs_3_conv_conv_weight, w_decoder_de_convs_3_conv_conv_bias,
    w_decoder_de_convs_3_bn_weight, w_decoder_de_convs_3_bn_bias,
    w_decoder_de_convs_3_bn_running_mean, w_decoder_de_convs_3_bn_running_var,
    w_decoder_de_convs_3_act_weight[0]};
const ConvBnActWeights kDec4 = {
    w_decoder_de_convs_4_conv_conv_weight, w_decoder_de_convs_4_conv_conv_bias,
    w_decoder_de_convs_4_bn_weight, w_decoder_de_convs_4_bn_bias,
    w_decoder_de_convs_4_bn_running_mean, w_decoder_de_convs_4_bn_running_var,
    0.0f};

#define STREAM_BLOCK(name, prefix)                                                \
const StreamWeights name = {                                                      \
    prefix##_point_conv1_weight, prefix##_point_conv1_bias,                       \
    prefix##_point_bn1_weight, prefix##_point_bn1_bias,                           \
    prefix##_point_bn1_running_mean, prefix##_point_bn1_running_var,              \
    prefix##_point_act_weight[0],                                                 \
    prefix##_depth_conv, prefix##_depth_bias,                                     \
    prefix##_depth_bn_weight, prefix##_depth_bn_bias,                             \
    prefix##_depth_bn_running_mean, prefix##_depth_bn_running_var,                \
    prefix##_depth_act_weight[0],                                                 \
    prefix##_point_conv2_weight, prefix##_point_conv2_bias,                       \
    prefix##_point_bn2_weight, prefix##_point_bn2_bias,                           \
    prefix##_point_bn2_running_mean, prefix##_point_bn2_running_var,              \
    {prefix##_tra_weight_ih, prefix##_tra_weight_hh, prefix##_tra_bias_ih,        \
     prefix##_tra_bias_hh},                                                       \
    prefix##_tra_fc_weight, prefix##_tra_fc_bias};                                \

const float* kEnc2DepthW = w_encoder_en_convs_2_depth_conv_Conv2d_weight;
const float* kEnc2DepthB = w_encoder_en_convs_2_depth_conv_Conv2d_bias;
const float* kEnc3DepthW = w_encoder_en_convs_3_depth_conv_Conv2d_weight;
const float* kEnc3DepthB = w_encoder_en_convs_3_depth_conv_Conv2d_bias;
const float* kEnc4DepthW = w_encoder_en_convs_4_depth_conv_Conv2d_weight;
const float* kEnc4DepthB = w_encoder_en_convs_4_depth_conv_Conv2d_bias;
const float* kDec0DepthW = w_decoder_de_convs_0_depth_conv_ConvTranspose2d_weight;
const float* kDec0DepthB = w_decoder_de_convs_0_depth_conv_ConvTranspose2d_bias;
const float* kDec1DepthW = w_decoder_de_convs_1_depth_conv_ConvTranspose2d_weight;
const float* kDec1DepthB = w_decoder_de_convs_1_depth_conv_ConvTranspose2d_bias;
const float* kDec2DepthW = w_decoder_de_convs_2_depth_conv_ConvTranspose2d_weight;
const float* kDec2DepthB = w_decoder_de_convs_2_depth_conv_ConvTranspose2d_bias;

const float* kEnc2TraWih = w_encoder_en_convs_2_tra_att_gru_weight_ih_l0;
const float* kEnc2TraWhh = w_encoder_en_convs_2_tra_att_gru_weight_hh_l0;
const float* kEnc2TraBih = w_encoder_en_convs_2_tra_att_gru_bias_ih_l0;
const float* kEnc2TraBhh = w_encoder_en_convs_2_tra_att_gru_bias_hh_l0;
const float* kEnc3TraWih = w_encoder_en_convs_3_tra_att_gru_weight_ih_l0;
const float* kEnc3TraWhh = w_encoder_en_convs_3_tra_att_gru_weight_hh_l0;
const float* kEnc3TraBih = w_encoder_en_convs_3_tra_att_gru_bias_ih_l0;
const float* kEnc3TraBhh = w_encoder_en_convs_3_tra_att_gru_bias_hh_l0;
const float* kEnc4TraWih = w_encoder_en_convs_4_tra_att_gru_weight_ih_l0;
const float* kEnc4TraWhh = w_encoder_en_convs_4_tra_att_gru_weight_hh_l0;
const float* kEnc4TraBih = w_encoder_en_convs_4_tra_att_gru_bias_ih_l0;
const float* kEnc4TraBhh = w_encoder_en_convs_4_tra_att_gru_bias_hh_l0;
const float* kDec0TraWih = w_decoder_de_convs_0_tra_att_gru_weight_ih_l0;
const float* kDec0TraWhh = w_decoder_de_convs_0_tra_att_gru_weight_hh_l0;
const float* kDec0TraBih = w_decoder_de_convs_0_tra_att_gru_bias_ih_l0;
const float* kDec0TraBhh = w_decoder_de_convs_0_tra_att_gru_bias_hh_l0;
const float* kDec1TraWih = w_decoder_de_convs_1_tra_att_gru_weight_ih_l0;
const float* kDec1TraWhh = w_decoder_de_convs_1_tra_att_gru_weight_hh_l0;
const float* kDec1TraBih = w_decoder_de_convs_1_tra_att_gru_bias_ih_l0;
const float* kDec1TraBhh = w_decoder_de_convs_1_tra_att_gru_bias_hh_l0;
const float* kDec2TraWih = w_decoder_de_convs_2_tra_att_gru_weight_ih_l0;
const float* kDec2TraWhh = w_decoder_de_convs_2_tra_att_gru_weight_hh_l0;
const float* kDec2TraBih = w_decoder_de_convs_2_tra_att_gru_bias_ih_l0;
const float* kDec2TraBhh = w_decoder_de_convs_2_tra_att_gru_bias_hh_l0;

#define prefix_enc2_point_conv1_weight w_encoder_en_convs_2_point_conv1_weight
#define prefix_enc2_point_conv1_bias w_encoder_en_convs_2_point_conv1_bias
#define prefix_enc2_point_bn1_weight w_encoder_en_convs_2_point_bn1_weight
#define prefix_enc2_point_bn1_bias w_encoder_en_convs_2_point_bn1_bias
#define prefix_enc2_point_bn1_running_mean w_encoder_en_convs_2_point_bn1_running_mean
#define prefix_enc2_point_bn1_running_var w_encoder_en_convs_2_point_bn1_running_var
#define prefix_enc2_point_act_weight w_encoder_en_convs_2_point_act_weight
#define prefix_enc2_depth_conv kEnc2DepthW
#define prefix_enc2_depth_bias kEnc2DepthB
#define prefix_enc2_depth_bn_weight w_encoder_en_convs_2_depth_bn_weight
#define prefix_enc2_depth_bn_bias w_encoder_en_convs_2_depth_bn_bias
#define prefix_enc2_depth_bn_running_mean w_encoder_en_convs_2_depth_bn_running_mean
#define prefix_enc2_depth_bn_running_var w_encoder_en_convs_2_depth_bn_running_var
#define prefix_enc2_depth_act_weight w_encoder_en_convs_2_depth_act_weight
#define prefix_enc2_point_conv2_weight w_encoder_en_convs_2_point_conv2_weight
#define prefix_enc2_point_conv2_bias w_encoder_en_convs_2_point_conv2_bias
#define prefix_enc2_point_bn2_weight w_encoder_en_convs_2_point_bn2_weight
#define prefix_enc2_point_bn2_bias w_encoder_en_convs_2_point_bn2_bias
#define prefix_enc2_point_bn2_running_mean w_encoder_en_convs_2_point_bn2_running_mean
#define prefix_enc2_point_bn2_running_var w_encoder_en_convs_2_point_bn2_running_var
#define prefix_enc2_tra_weight_ih kEnc2TraWih
#define prefix_enc2_tra_weight_hh kEnc2TraWhh
#define prefix_enc2_tra_bias_ih kEnc2TraBih
#define prefix_enc2_tra_bias_hh kEnc2TraBhh
#define prefix_enc2_tra_fc_weight w_encoder_en_convs_2_tra_att_fc_weight
#define prefix_enc2_tra_fc_bias w_encoder_en_convs_2_tra_att_fc_bias
STREAM_BLOCK(kEnc2, prefix_enc2)

#define prefix_enc3_point_conv1_weight w_encoder_en_convs_3_point_conv1_weight
#define prefix_enc3_point_conv1_bias w_encoder_en_convs_3_point_conv1_bias
#define prefix_enc3_point_bn1_weight w_encoder_en_convs_3_point_bn1_weight
#define prefix_enc3_point_bn1_bias w_encoder_en_convs_3_point_bn1_bias
#define prefix_enc3_point_bn1_running_mean w_encoder_en_convs_3_point_bn1_running_mean
#define prefix_enc3_point_bn1_running_var w_encoder_en_convs_3_point_bn1_running_var
#define prefix_enc3_point_act_weight w_encoder_en_convs_3_point_act_weight
#define prefix_enc3_depth_conv kEnc3DepthW
#define prefix_enc3_depth_bias kEnc3DepthB
#define prefix_enc3_depth_bn_weight w_encoder_en_convs_3_depth_bn_weight
#define prefix_enc3_depth_bn_bias w_encoder_en_convs_3_depth_bn_bias
#define prefix_enc3_depth_bn_running_mean w_encoder_en_convs_3_depth_bn_running_mean
#define prefix_enc3_depth_bn_running_var w_encoder_en_convs_3_depth_bn_running_var
#define prefix_enc3_depth_act_weight w_encoder_en_convs_3_depth_act_weight
#define prefix_enc3_point_conv2_weight w_encoder_en_convs_3_point_conv2_weight
#define prefix_enc3_point_conv2_bias w_encoder_en_convs_3_point_conv2_bias
#define prefix_enc3_point_bn2_weight w_encoder_en_convs_3_point_bn2_weight
#define prefix_enc3_point_bn2_bias w_encoder_en_convs_3_point_bn2_bias
#define prefix_enc3_point_bn2_running_mean w_encoder_en_convs_3_point_bn2_running_mean
#define prefix_enc3_point_bn2_running_var w_encoder_en_convs_3_point_bn2_running_var
#define prefix_enc3_tra_weight_ih kEnc3TraWih
#define prefix_enc3_tra_weight_hh kEnc3TraWhh
#define prefix_enc3_tra_bias_ih kEnc3TraBih
#define prefix_enc3_tra_bias_hh kEnc3TraBhh
#define prefix_enc3_tra_fc_weight w_encoder_en_convs_3_tra_att_fc_weight
#define prefix_enc3_tra_fc_bias w_encoder_en_convs_3_tra_att_fc_bias
STREAM_BLOCK(kEnc3, prefix_enc3)

#define prefix_enc4_point_conv1_weight w_encoder_en_convs_4_point_conv1_weight
#define prefix_enc4_point_conv1_bias w_encoder_en_convs_4_point_conv1_bias
#define prefix_enc4_point_bn1_weight w_encoder_en_convs_4_point_bn1_weight
#define prefix_enc4_point_bn1_bias w_encoder_en_convs_4_point_bn1_bias
#define prefix_enc4_point_bn1_running_mean w_encoder_en_convs_4_point_bn1_running_mean
#define prefix_enc4_point_bn1_running_var w_encoder_en_convs_4_point_bn1_running_var
#define prefix_enc4_point_act_weight w_encoder_en_convs_4_point_act_weight
#define prefix_enc4_depth_conv kEnc4DepthW
#define prefix_enc4_depth_bias kEnc4DepthB
#define prefix_enc4_depth_bn_weight w_encoder_en_convs_4_depth_bn_weight
#define prefix_enc4_depth_bn_bias w_encoder_en_convs_4_depth_bn_bias
#define prefix_enc4_depth_bn_running_mean w_encoder_en_convs_4_depth_bn_running_mean
#define prefix_enc4_depth_bn_running_var w_encoder_en_convs_4_depth_bn_running_var
#define prefix_enc4_depth_act_weight w_encoder_en_convs_4_depth_act_weight
#define prefix_enc4_point_conv2_weight w_encoder_en_convs_4_point_conv2_weight
#define prefix_enc4_point_conv2_bias w_encoder_en_convs_4_point_conv2_bias
#define prefix_enc4_point_bn2_weight w_encoder_en_convs_4_point_bn2_weight
#define prefix_enc4_point_bn2_bias w_encoder_en_convs_4_point_bn2_bias
#define prefix_enc4_point_bn2_running_mean w_encoder_en_convs_4_point_bn2_running_mean
#define prefix_enc4_point_bn2_running_var w_encoder_en_convs_4_point_bn2_running_var
#define prefix_enc4_tra_weight_ih kEnc4TraWih
#define prefix_enc4_tra_weight_hh kEnc4TraWhh
#define prefix_enc4_tra_bias_ih kEnc4TraBih
#define prefix_enc4_tra_bias_hh kEnc4TraBhh
#define prefix_enc4_tra_fc_weight w_encoder_en_convs_4_tra_att_fc_weight
#define prefix_enc4_tra_fc_bias w_encoder_en_convs_4_tra_att_fc_bias
STREAM_BLOCK(kEnc4, prefix_enc4)

#define prefix_dec0_point_conv1_weight w_decoder_de_convs_0_point_conv1_weight
#define prefix_dec0_point_conv1_bias w_decoder_de_convs_0_point_conv1_bias
#define prefix_dec0_point_bn1_weight w_decoder_de_convs_0_point_bn1_weight
#define prefix_dec0_point_bn1_bias w_decoder_de_convs_0_point_bn1_bias
#define prefix_dec0_point_bn1_running_mean w_decoder_de_convs_0_point_bn1_running_mean
#define prefix_dec0_point_bn1_running_var w_decoder_de_convs_0_point_bn1_running_var
#define prefix_dec0_point_act_weight w_decoder_de_convs_0_point_act_weight
#define prefix_dec0_depth_conv kDec0DepthW
#define prefix_dec0_depth_bias kDec0DepthB
#define prefix_dec0_depth_bn_weight w_decoder_de_convs_0_depth_bn_weight
#define prefix_dec0_depth_bn_bias w_decoder_de_convs_0_depth_bn_bias
#define prefix_dec0_depth_bn_running_mean w_decoder_de_convs_0_depth_bn_running_mean
#define prefix_dec0_depth_bn_running_var w_decoder_de_convs_0_depth_bn_running_var
#define prefix_dec0_depth_act_weight w_decoder_de_convs_0_depth_act_weight
#define prefix_dec0_point_conv2_weight w_decoder_de_convs_0_point_conv2_weight
#define prefix_dec0_point_conv2_bias w_decoder_de_convs_0_point_conv2_bias
#define prefix_dec0_point_bn2_weight w_decoder_de_convs_0_point_bn2_weight
#define prefix_dec0_point_bn2_bias w_decoder_de_convs_0_point_bn2_bias
#define prefix_dec0_point_bn2_running_mean w_decoder_de_convs_0_point_bn2_running_mean
#define prefix_dec0_point_bn2_running_var w_decoder_de_convs_0_point_bn2_running_var
#define prefix_dec0_tra_weight_ih kDec0TraWih
#define prefix_dec0_tra_weight_hh kDec0TraWhh
#define prefix_dec0_tra_bias_ih kDec0TraBih
#define prefix_dec0_tra_bias_hh kDec0TraBhh
#define prefix_dec0_tra_fc_weight w_decoder_de_convs_0_tra_att_fc_weight
#define prefix_dec0_tra_fc_bias w_decoder_de_convs_0_tra_att_fc_bias
STREAM_BLOCK(kDec0, prefix_dec0)

#define prefix_dec1_point_conv1_weight w_decoder_de_convs_1_point_conv1_weight
#define prefix_dec1_point_conv1_bias w_decoder_de_convs_1_point_conv1_bias
#define prefix_dec1_point_bn1_weight w_decoder_de_convs_1_point_bn1_weight
#define prefix_dec1_point_bn1_bias w_decoder_de_convs_1_point_bn1_bias
#define prefix_dec1_point_bn1_running_mean w_decoder_de_convs_1_point_bn1_running_mean
#define prefix_dec1_point_bn1_running_var w_decoder_de_convs_1_point_bn1_running_var
#define prefix_dec1_point_act_weight w_decoder_de_convs_1_point_act_weight
#define prefix_dec1_depth_conv kDec1DepthW
#define prefix_dec1_depth_bias kDec1DepthB
#define prefix_dec1_depth_bn_weight w_decoder_de_convs_1_depth_bn_weight
#define prefix_dec1_depth_bn_bias w_decoder_de_convs_1_depth_bn_bias
#define prefix_dec1_depth_bn_running_mean w_decoder_de_convs_1_depth_bn_running_mean
#define prefix_dec1_depth_bn_running_var w_decoder_de_convs_1_depth_bn_running_var
#define prefix_dec1_depth_act_weight w_decoder_de_convs_1_depth_act_weight
#define prefix_dec1_point_conv2_weight w_decoder_de_convs_1_point_conv2_weight
#define prefix_dec1_point_conv2_bias w_decoder_de_convs_1_point_conv2_bias
#define prefix_dec1_point_bn2_weight w_decoder_de_convs_1_point_bn2_weight
#define prefix_dec1_point_bn2_bias w_decoder_de_convs_1_point_bn2_bias
#define prefix_dec1_point_bn2_running_mean w_decoder_de_convs_1_point_bn2_running_mean
#define prefix_dec1_point_bn2_running_var w_decoder_de_convs_1_point_bn2_running_var
#define prefix_dec1_tra_weight_ih kDec1TraWih
#define prefix_dec1_tra_weight_hh kDec1TraWhh
#define prefix_dec1_tra_bias_ih kDec1TraBih
#define prefix_dec1_tra_bias_hh kDec1TraBhh
#define prefix_dec1_tra_fc_weight w_decoder_de_convs_1_tra_att_fc_weight
#define prefix_dec1_tra_fc_bias w_decoder_de_convs_1_tra_att_fc_bias
STREAM_BLOCK(kDec1, prefix_dec1)

#define prefix_dec2_point_conv1_weight w_decoder_de_convs_2_point_conv1_weight
#define prefix_dec2_point_conv1_bias w_decoder_de_convs_2_point_conv1_bias
#define prefix_dec2_point_bn1_weight w_decoder_de_convs_2_point_bn1_weight
#define prefix_dec2_point_bn1_bias w_decoder_de_convs_2_point_bn1_bias
#define prefix_dec2_point_bn1_running_mean w_decoder_de_convs_2_point_bn1_running_mean
#define prefix_dec2_point_bn1_running_var w_decoder_de_convs_2_point_bn1_running_var
#define prefix_dec2_point_act_weight w_decoder_de_convs_2_point_act_weight
#define prefix_dec2_depth_conv kDec2DepthW
#define prefix_dec2_depth_bias kDec2DepthB
#define prefix_dec2_depth_bn_weight w_decoder_de_convs_2_depth_bn_weight
#define prefix_dec2_depth_bn_bias w_decoder_de_convs_2_depth_bn_bias
#define prefix_dec2_depth_bn_running_mean w_decoder_de_convs_2_depth_bn_running_mean
#define prefix_dec2_depth_bn_running_var w_decoder_de_convs_2_depth_bn_running_var
#define prefix_dec2_depth_act_weight w_decoder_de_convs_2_depth_act_weight
#define prefix_dec2_point_conv2_weight w_decoder_de_convs_2_point_conv2_weight
#define prefix_dec2_point_conv2_bias w_decoder_de_convs_2_point_conv2_bias
#define prefix_dec2_point_bn2_weight w_decoder_de_convs_2_point_bn2_weight
#define prefix_dec2_point_bn2_bias w_decoder_de_convs_2_point_bn2_bias
#define prefix_dec2_point_bn2_running_mean w_decoder_de_convs_2_point_bn2_running_mean
#define prefix_dec2_point_bn2_running_var w_decoder_de_convs_2_point_bn2_running_var
#define prefix_dec2_tra_weight_ih kDec2TraWih
#define prefix_dec2_tra_weight_hh kDec2TraWhh
#define prefix_dec2_tra_bias_ih kDec2TraBih
#define prefix_dec2_tra_bias_hh kDec2TraBhh
#define prefix_dec2_tra_fc_weight w_decoder_de_convs_2_tra_att_fc_weight
#define prefix_dec2_tra_fc_bias w_decoder_de_convs_2_tra_att_fc_bias
STREAM_BLOCK(kDec2, prefix_dec2)

const RnnWeights kDp1Rnn11 = {w_dpgrnn1_intra_rnn_rnn1_weight_ih_l0, w_dpgrnn1_intra_rnn_rnn1_weight_hh_l0,
                              w_dpgrnn1_intra_rnn_rnn1_bias_ih_l0, w_dpgrnn1_intra_rnn_rnn1_bias_hh_l0};
const RnnWeights kDp1Rnn11R = {w_dpgrnn1_intra_rnn_rnn1_weight_ih_l0_reverse, w_dpgrnn1_intra_rnn_rnn1_weight_hh_l0_reverse,
                               w_dpgrnn1_intra_rnn_rnn1_bias_ih_l0_reverse, w_dpgrnn1_intra_rnn_rnn1_bias_hh_l0_reverse};
const RnnWeights kDp1Rnn12 = {w_dpgrnn1_intra_rnn_rnn2_weight_ih_l0, w_dpgrnn1_intra_rnn_rnn2_weight_hh_l0,
                              w_dpgrnn1_intra_rnn_rnn2_bias_ih_l0, w_dpgrnn1_intra_rnn_rnn2_bias_hh_l0};
const RnnWeights kDp1Rnn12R = {w_dpgrnn1_intra_rnn_rnn2_weight_ih_l0_reverse, w_dpgrnn1_intra_rnn_rnn2_weight_hh_l0_reverse,
                               w_dpgrnn1_intra_rnn_rnn2_bias_ih_l0_reverse, w_dpgrnn1_intra_rnn_rnn2_bias_hh_l0_reverse};
const RnnWeights kDp1Inter1 = {w_dpgrnn1_inter_rnn_rnn1_weight_ih_l0, w_dpgrnn1_inter_rnn_rnn1_weight_hh_l0,
                               w_dpgrnn1_inter_rnn_rnn1_bias_ih_l0, w_dpgrnn1_inter_rnn_rnn1_bias_hh_l0};
const RnnWeights kDp1Inter2 = {w_dpgrnn1_inter_rnn_rnn2_weight_ih_l0, w_dpgrnn1_inter_rnn_rnn2_weight_hh_l0,
                               w_dpgrnn1_inter_rnn_rnn2_bias_ih_l0, w_dpgrnn1_inter_rnn_rnn2_bias_hh_l0};
const DpgrnnWeights kDp1 = {
    {{kDp1Rnn11, kDp1Rnn11R}, {kDp1Rnn12, kDp1Rnn12R}},
    {kDp1Inter1, kDp1Inter2},
    w_dpgrnn1_intra_fc_weight, w_dpgrnn1_intra_fc_bias,
    w_dpgrnn1_intra_ln_weight, w_dpgrnn1_intra_ln_bias,
    w_dpgrnn1_inter_fc_weight, w_dpgrnn1_inter_fc_bias,
    w_dpgrnn1_inter_ln_weight, w_dpgrnn1_inter_ln_bias};

const RnnWeights kDp2Rnn11 = {w_dpgrnn2_intra_rnn_rnn1_weight_ih_l0, w_dpgrnn2_intra_rnn_rnn1_weight_hh_l0,
                              w_dpgrnn2_intra_rnn_rnn1_bias_ih_l0, w_dpgrnn2_intra_rnn_rnn1_bias_hh_l0};
const RnnWeights kDp2Rnn11R = {w_dpgrnn2_intra_rnn_rnn1_weight_ih_l0_reverse, w_dpgrnn2_intra_rnn_rnn1_weight_hh_l0_reverse,
                               w_dpgrnn2_intra_rnn_rnn1_bias_ih_l0_reverse, w_dpgrnn2_intra_rnn_rnn1_bias_hh_l0_reverse};
const RnnWeights kDp2Rnn12 = {w_dpgrnn2_intra_rnn_rnn2_weight_ih_l0, w_dpgrnn2_intra_rnn_rnn2_weight_hh_l0,
                              w_dpgrnn2_intra_rnn_rnn2_bias_ih_l0, w_dpgrnn2_intra_rnn_rnn2_bias_hh_l0};
const RnnWeights kDp2Rnn12R = {w_dpgrnn2_intra_rnn_rnn2_weight_ih_l0_reverse, w_dpgrnn2_intra_rnn_rnn2_weight_hh_l0_reverse,
                               w_dpgrnn2_intra_rnn_rnn2_bias_ih_l0_reverse, w_dpgrnn2_intra_rnn_rnn2_bias_hh_l0_reverse};
const RnnWeights kDp2Inter1 = {w_dpgrnn2_inter_rnn_rnn1_weight_ih_l0, w_dpgrnn2_inter_rnn_rnn1_weight_hh_l0,
                               w_dpgrnn2_inter_rnn_rnn1_bias_ih_l0, w_dpgrnn2_inter_rnn_rnn1_bias_hh_l0};
const RnnWeights kDp2Inter2 = {w_dpgrnn2_inter_rnn_rnn2_weight_ih_l0, w_dpgrnn2_inter_rnn_rnn2_weight_hh_l0,
                               w_dpgrnn2_inter_rnn_rnn2_bias_ih_l0, w_dpgrnn2_inter_rnn_rnn2_bias_hh_l0};
const DpgrnnWeights kDp2 = {
    {{kDp2Rnn11, kDp2Rnn11R}, {kDp2Rnn12, kDp2Rnn12R}},
    {kDp2Inter1, kDp2Inter2},
    w_dpgrnn2_intra_fc_weight, w_dpgrnn2_intra_fc_bias,
    w_dpgrnn2_intra_ln_weight, w_dpgrnn2_intra_ln_bias,
    w_dpgrnn2_inter_fc_weight, w_dpgrnn2_inter_fc_bias,
    w_dpgrnn2_inter_ln_weight, w_dpgrnn2_inter_ln_bias};

#undef STREAM_BLOCK

IRAM_ATTR void stream_gt_block(Workspace* ws, const StreamWeights& sw, int cache_stream, int cache_start,
                               int cache_t, int dilation, int cache_block, const float* x, float* y, int tra_base) {
  float* x1 = ws->stream_x1.data();
  float* x2 = ws->stream_x2.data();
  for (int c = 0; c < 8; ++c) {
    for (int f = 0; f < BOT_F; ++f) {
      x1[idx2(c, f, BOT_F)] = x[idx2(c, f, BOT_F)];
      x2[idx2(c, f, BOT_F)] = x[idx2(c + 8, f, BOT_F)];
    }
  }
  float* s = ws->stream_s.data();
  float* h = ws->stream_h.data();
  float* tmp = ws->stream_tmp.data();
  sfe(x1, s, 8, BOT_F);
  conv1x1_fixed<24, 16, BOT_F>(sw.point_conv1_w, sw.point_conv1_b, s, h);
  batch_norm(h, 16, BOT_F, sw.point_bn1_w, sw.point_bn1_b, sw.point_bn1_mean, sw.point_bn1_var);
  prelu(h, 16 * BOT_F, sw.point_act);

  std::fill(tmp, tmp + 16 * BOT_F, 0.0f);
  const int head = ws->conv_heads[cache_block];
  for (int c = 0; c < 16; ++c) {
    for (int f = 0; f < BOT_F; ++f) {
      float acc = sw.depth_b[c];
      for (int kt = 0; kt < 3; ++kt) {
        int tt = kt * dilation;
        for (int kf = 0; kf < 3; ++kf) {
          int ff = f + kf - 1;
          if (ff < 0 || ff >= BOT_F) continue;
          float v = (tt < cache_t)
                        ? ws->conv_cache[conv_cache_idx(cache_stream, c,
                                                        cache_start + ((head + tt) % cache_t), ff)]
                        : h[idx2(c, ff, BOT_F)];
          acc += sw.depth_w[((c * 3 + kt) * 3) + kf] * v;
        }
      }
      tmp[idx2(c, f, BOT_F)] = acc;
    }
    for (int f = 0; f < BOT_F; ++f) {
      ws->conv_cache[conv_cache_idx(cache_stream, c, cache_start + head, f)] = h[idx2(c, f, BOT_F)];
    }
  }
  ws->conv_heads[cache_block] = static_cast<uint8_t>((head + 1) % cache_t);
  std::memcpy(h, tmp, sizeof(float) * 16 * BOT_F);
  batch_norm(h, 16, BOT_F, sw.depth_bn_w, sw.depth_bn_b, sw.depth_bn_mean, sw.depth_bn_var);
  prelu(h, 16 * BOT_F, sw.depth_act);
  conv1x1_fixed<16, 8, BOT_F>(sw.point_conv2_w, sw.point_conv2_b, h, tmp);
  std::memcpy(h, tmp, sizeof(float) * 8 * BOT_F);
  batch_norm(h, 8, BOT_F, sw.point_bn2_w, sw.point_bn2_b, sw.point_bn2_mean, sw.point_bn2_var);

  float zt[8];
  float att_h[16];
  float att[8];
  for (int c = 0; c < 8; ++c) {
    float sum = 0.0f;
    for (int f = 0; f < BOT_F; ++f) {
      float v = h[idx2(c, f, BOT_F)];
      sum += v * v;
    }
    zt[c] = sum / BOT_F;
  }
  for (int i = 0; i < 16; ++i) att_h[i] = ws->tra_cache[tra_cache_idx(tra_base / 3, tra_base % 3, i)];
  gru_cell_fixed<8, 16>(sw.tra_gru, zt, att_h);
  for (int i = 0; i < 16; ++i) ws->tra_cache[tra_cache_idx(tra_base / 3, tra_base % 3, i)] = att_h[i];
  linear_fixed<8, 16>(sw.tra_fc_w, sw.tra_fc_b, att_h, att);
  for (float& v : att) v = sigmoid(v);
  for (int c = 0; c < 8; ++c) {
    for (int f = 0; f < BOT_F; ++f) h[idx2(c, f, BOT_F)] *= att[c];
  }

  std::fill(y, y + 16 * BOT_F, 0.0f);
  for (int c = 0; c < 8; ++c) {
    for (int f = 0; f < BOT_F; ++f) {
      y[idx2(2 * c, f, BOT_F)] = h[idx2(c, f, BOT_F)];
      y[idx2(2 * c + 1, f, BOT_F)] = x2[idx2(c, f, BOT_F)];
    }
  }
}

IRAM_ATTR void dpgrnn(Workspace* ws, const DpgrnnWeights& dw, int inter_base, float* x) {
#if GTCRN_DEBUG_DUALCORE
  float* x_original = x;
  for (int i = 0; i < ENX_SIZE; ++i) g_debug_dp.x_in[i] = x[i];
  for (int f = 0; f < BOT_F; ++f) {
    for (int c = 0; c < 16; ++c) {
      g_debug_dp.inter_cache_block[local_inter_idx(f, c)] = ws->inter_cache[inter_cache_idx(inter_base, f, c)];
    }
  }
#endif
  float* intra = ws->dp_intra.data();
  std::fill(intra, intra + DP_SIZE, 0.0f);
#if GTCRN_HAS_FREERTOS
  if (g_worker.initialized) {
    g_worker.job = {WorkerJobType::kDpIntra, ws, &dw, inter_base, x, intra, xTaskGetCurrentTaskHandle()};
    xTaskNotifyGive(g_worker.handle);
    dpgrnn_intra_group(dw, x, 0, intra);
    ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
  } else
#endif
  {
    dpgrnn_intra_group(dw, x, 0, intra);
    dpgrnn_intra_group(dw, x, 1, intra);
  }

  float* fc = ws->dp_fc.data();
  for (int f = 0; f < BOT_F; ++f) {
    float in[16];
    float out[16];
    for (int c = 0; c < 16; ++c) in[c] = intra[idx2(c, f, BOT_F)];
    linear_fixed<16, 16>(dw.intra_fc_w, dw.intra_fc_b, in, out);
    for (int c = 0; c < 16; ++c) fc[f * 16 + c] = out[c];
  }
  layer_norm(fc, DP_SIZE, dw.intra_ln_w, dw.intra_ln_b);

  float* intra_out = ws->dp_intra_out.data();
  for (int f = 0; f < BOT_F; ++f) {
    for (int c = 0; c < 16; ++c) intra_out[idx2(c, f, BOT_F)] = x[idx2(c, f, BOT_F)] + fc[f * 16 + c];
  }

  float* inter_fc = ws->dp_inter_fc.data();
  for (int f = 0; f < BOT_F; ++f) {
    float ycat[16];
    for (int grp = 0; grp < 2; ++grp) {
      const RnnWeights& rw = dw.inter_rnn[grp];
      float xin[8];
      float hstate[8];
      for (int i = 0; i < 8; ++i) {
        xin[i] = intra_out[idx2(grp * 8 + i, f, BOT_F)];
        hstate[i] = ws->inter_cache[inter_cache_idx(inter_base, f, grp * 8 + i)];
      }
      gru_cell_fixed<8, 8>(rw, xin, hstate);
      for (int i = 0; i < 8; ++i) {
        ycat[grp * 8 + i] = hstate[i];
        ws->inter_cache[inter_cache_idx(inter_base, f, grp * 8 + i)] = hstate[i];
      }
    }
#if GTCRN_DEBUG_DUALCORE
    for (int c = 0; c < 16; ++c) g_debug_dp.inter_linear_in_parallel[local_inter_idx(f, c)] = ycat[c];
#endif
    float out[16];
    linear_fixed<16, 16>(dw.inter_fc_w, dw.inter_fc_b, ycat, out);
    for (int c = 0; c < 16; ++c) inter_fc[f * 16 + c] = out[c];
  }
  layer_norm(inter_fc, DP_SIZE, dw.inter_ln_w, dw.inter_ln_b);
  for (int f = 0; f < BOT_F; ++f) {
    for (int c = 0; c < 16; ++c) x[idx2(c, f, BOT_F)] = intra_out[idx2(c, f, BOT_F)] + inter_fc[f * 16 + c];
  }

#if GTCRN_DEBUG_DUALCORE
  if (!g_debug_dp.printed) {
    dpgrnn_reference_debug(dw, g_debug_dp.x_in.data(), inter_base, g_debug_dp.inter_cache_block.data(), &g_debug_dp);
    Stats final_stats = compare_buffers(g_debug_dp.x_out.data(), x_original, ENX_SIZE);
    Stats intra_stats = compare_buffers(g_debug_dp.intra.data(), intra, DP_SIZE);
    Stats intra_out_stats = compare_buffers(g_debug_dp.intra_out.data(), intra_out, DP_SIZE);
    Stats inter_in_stats = compare_buffers(g_debug_dp.inter_linear_in.data(), g_debug_dp.inter_linear_in_parallel.data(), DP_SIZE);
    std::printf("dpdbg block=%d intra_max=%.8f intra_mean=%.8f intra_out_max=%.8f intra_out_mean=%.8f inter_in_max=%.8f inter_in_mean=%.8f final_max=%.8f final_mean=%.8f\n",
                inter_base, intra_stats.max_abs, intra_stats.mean_abs,
                intra_out_stats.max_abs, intra_out_stats.mean_abs,
                inter_in_stats.max_abs, inter_in_stats.mean_abs,
                final_stats.max_abs, final_stats.mean_abs);
    g_debug_dp.printed = true;
  }
#endif
}

}  // namespace

EmbeddedModel::EmbeddedModel(Workspace* workspace) : ws_(workspace) { reset(); }

void EmbeddedModel::reset() {
  ensure_worker_started();
  init_sparse_matrix(&g_erb_sparse, w_erb_erb_fc_weight);
  init_sparse_matrix(&g_ierb_sparse, w_erb_ierb_fc_weight);
  ws_->conv_cache.fill(0.0f);
  ws_->conv_heads.fill(0);
  ws_->tra_cache.fill(0.0f);
  ws_->inter_cache.fill(0.0f);
}

void EmbeddedModel::infer(const float* mix, float* enh) {
#if GTCRN_ENABLE_PROFILE
  int64_t t0 = esp_timer_get_time();
#endif
  float* feat3 = ws_->feat3.data();
  for (int f = 0; f < FREQ; ++f) {
    float real = mix[f * 2];
    float imag = mix[f * 2 + 1];
    feat3[idx2(0, f, FREQ)] = sqrtf(real * real + imag * imag + kEps);
    feat3[idx2(1, f, FREQ)] = real;
    feat3[idx2(2, f, FREQ)] = imag;
  }
#if GTCRN_ENABLE_PROFILE
  int64_t t1 = esp_timer_get_time();
  add_profile(ProfileStage::kFeat, t1 - t0);
#endif

  float* erb = ws_->erb.data();
  std::fill(erb, erb + ERB_SIZE, 0.0f);
  for (int c = 0; c < 3; ++c) {
    for (int f = 0; f < ERB_LOW; ++f) erb[idx2(c, f, ERB_FULL)] = feat3[idx2(c, f, FREQ)];
    sparse_linear(g_erb_sparse, feat3 + idx2(c, ERB_LOW, FREQ), erb + idx2(c, ERB_LOW, ERB_FULL));
  }
#if GTCRN_ENABLE_PROFILE
  int64_t t2 = esp_timer_get_time();
  add_profile(ProfileStage::kErb, t2 - t1);
#endif

  float* x = ws_->x.data();
  float* y = ws_->y.data();
  sfe(erb, x, 3, ERB_FULL);

  conv1x5(kEnc0.conv_w, kEnc0.conv_b, x, y, 9, 16, 129, 65, 2, 1);
  batch_norm(y, 16, 65, kEnc0.bn_w, kEnc0.bn_b, kEnc0.bn_mean, kEnc0.bn_var);
  prelu(y, 16 * 65, kEnc0.act);
  std::memcpy(ws_->en0.data(), y, sizeof(float) * EN0_SIZE);

  conv1x5(kEnc1.conv_w, kEnc1.conv_b, y, x, 16, 16, 65, 33, 2, 2);
  batch_norm(x, 16, 33, kEnc1.bn_w, kEnc1.bn_b, kEnc1.bn_mean, kEnc1.bn_var);
  prelu(x, ENX_SIZE, kEnc1.act);
  std::memcpy(ws_->en1.data(), x, sizeof(float) * ENX_SIZE);
#if GTCRN_ENABLE_PROFILE
  int64_t t3 = esp_timer_get_time();
  add_profile(ProfileStage::kEnc01, t3 - t2);
#endif

  stream_gt_block(ws_, kEnc2, 0, 0, 2, 1, 0, x, y, 0);
  std::memcpy(ws_->en2.data(), y, sizeof(float) * ENX_SIZE);
  stream_gt_block(ws_, kEnc3, 0, 2, 4, 2, 1, y, x, 1);
  std::memcpy(ws_->en3.data(), x, sizeof(float) * ENX_SIZE);
  stream_gt_block(ws_, kEnc4, 0, 6, 10, 5, 2, x, y, 2);
  std::memcpy(ws_->en4.data(), y, sizeof(float) * ENX_SIZE);
#if GTCRN_ENABLE_PROFILE
  int64_t t4 = esp_timer_get_time();
  add_profile(ProfileStage::kEnc234, t4 - t3);
#endif

  std::memcpy(x, y, sizeof(float) * ENX_SIZE);
  dpgrnn(ws_, kDp1, 0, x);
#if GTCRN_ENABLE_PROFILE
  int64_t t5 = esp_timer_get_time();
  add_profile(ProfileStage::kDp1, t5 - t4);
#endif
  dpgrnn(ws_, kDp2, 1, x);
#if GTCRN_ENABLE_PROFILE
  int64_t t6 = esp_timer_get_time();
  add_profile(ProfileStage::kDp2, t6 - t5);
#endif

  add_inplace(x, ws_->en4.data(), ENX_SIZE);
  stream_gt_block(ws_, kDec0, 1, 6, 10, 5, 3, x, y, 3);
  add_inplace(y, ws_->en3.data(), ENX_SIZE);
  stream_gt_block(ws_, kDec1, 1, 2, 4, 2, 4, y, x, 4);
  add_inplace(x, ws_->en2.data(), ENX_SIZE);
  stream_gt_block(ws_, kDec2, 1, 0, 2, 1, 5, x, y, 5);
  add_inplace(y, ws_->en1.data(), ENX_SIZE);
#if GTCRN_ENABLE_PROFILE
  int64_t t7 = esp_timer_get_time();
  add_profile(ProfileStage::kDec012, t7 - t6);
#endif

  upsample_conv1x5(kDec3.conv_w, kDec3.conv_b, y, x, 16, 16, 33, 65, 2,
                   ws_->upsample_up.data(), ws_->upsample_padded.data());
  batch_norm(x, 16, 65, kDec3.bn_w, kDec3.bn_b, kDec3.bn_mean, kDec3.bn_var);
  prelu(x, EN0_SIZE, kDec3.act);
  add_inplace(x, ws_->en0.data(), EN0_SIZE);
#if GTCRN_ENABLE_PROFILE
  int64_t t8 = esp_timer_get_time();
  add_profile(ProfileStage::kUp1, t8 - t7);
#endif

  upsample_conv1x5(kDec4.conv_w, kDec4.conv_b, x, y, 16, 2, 65, 129, 1,
                   ws_->upsample_up.data(), ws_->upsample_padded.data());
  batch_norm(y, 2, 129, kDec4.bn_w, kDec4.bn_b, kDec4.bn_mean, kDec4.bn_var);
  for (int i = 0; i < 2 * 129; ++i) y[i] = tanhf(y[i]);
#if GTCRN_ENABLE_PROFILE
  int64_t t9 = esp_timer_get_time();
  add_profile(ProfileStage::kUp2, t9 - t8);
#endif

  float* mask = ws_->mask.data();
  std::fill(mask, mask + MASK_SIZE, 0.0f);
  for (int c = 0; c < 2; ++c) {
    for (int f = 0; f < ERB_LOW; ++f) mask[idx2(c, f, FREQ)] = y[idx2(c, f, ERB_FULL)];
    sparse_linear(g_ierb_sparse, y + idx2(c, ERB_LOW, ERB_FULL), mask + idx2(c, ERB_LOW, FREQ));
  }

  for (int f = 0; f < FREQ; ++f) {
    float sr = mix[f * 2];
    float si = mix[f * 2 + 1];
    float mr = mask[idx2(0, f, FREQ)];
    float mi = mask[idx2(1, f, FREQ)];
    enh[f * 2] = sr * mr - si * mi;
    enh[f * 2 + 1] = si * mr + sr * mi;
  }
#if GTCRN_ENABLE_PROFILE
  int64_t t10 = esp_timer_get_time();
  add_profile(ProfileStage::kMaskOut, t10 - t9);
  g_profile_stats.calls += 1;
#endif
}

Stats compare_buffers(const float* ref, const float* got, size_t count) {
  Stats stats{};
  for (size_t i = 0; i < count; ++i) {
    float diff = std::fabs(ref[i] - got[i]);
    stats.max_abs = std::max(stats.max_abs, diff);
    stats.mean_abs += diff;
  }
  if (count > 0) stats.mean_abs /= static_cast<float>(count);
  return stats;
}

void reset_profile_stats() {
#if GTCRN_ENABLE_PROFILE
  g_profile_stats = {};
#endif
}

ProfileStats get_profile_stats() { return g_profile_stats; }

}  // namespace gtcrn_esp
