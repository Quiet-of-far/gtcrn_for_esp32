#pragma once

#include <array>
#include <string>
#include <unordered_map>
#include <vector>

#ifndef GTCRN_WEIGHTS_HEADER
#define GTCRN_WEIGHTS_HEADER "gtcrn_weights_dns3.h"
#endif
#include GTCRN_WEIGHTS_HEADER

namespace gtcrn {

constexpr int FREQ = 257;
constexpr int ERB_LOW = 65;
constexpr int ERB_HIGH = 64;
constexpr int ERB_FULL = 129;
constexpr int BOT_F = 33;

struct Stats {
  double avg = 0.0;
  double p50 = 0.0;
  double p95 = 0.0;
  double p99 = 0.0;
  double max = 0.0;
};

struct RnnWeights {
  const TensorView* weight_ih = nullptr;
  const TensorView* weight_hh = nullptr;
  const TensorView* bias_ih = nullptr;
  const TensorView* bias_hh = nullptr;
};

struct StreamWeights {
  const TensorView* point_conv1_w = nullptr;
  const TensorView* point_conv1_b = nullptr;
  const TensorView* point_bn1_w = nullptr;
  const TensorView* point_bn1_b = nullptr;
  const TensorView* point_bn1_mean = nullptr;
  const TensorView* point_bn1_var = nullptr;
  const TensorView* point_act = nullptr;
  const TensorView* depth_w = nullptr;
  const TensorView* depth_b = nullptr;
  const TensorView* depth_bn_w = nullptr;
  const TensorView* depth_bn_b = nullptr;
  const TensorView* depth_bn_mean = nullptr;
  const TensorView* depth_bn_var = nullptr;
  const TensorView* depth_act = nullptr;
  const TensorView* point_conv2_w = nullptr;
  const TensorView* point_conv2_b = nullptr;
  const TensorView* point_bn2_w = nullptr;
  const TensorView* point_bn2_b = nullptr;
  const TensorView* point_bn2_mean = nullptr;
  const TensorView* point_bn2_var = nullptr;
  RnnWeights tra_gru;
  const TensorView* tra_fc_w = nullptr;
  const TensorView* tra_fc_b = nullptr;
};

struct DpgrnnWeights {
  std::array<std::array<RnnWeights, 2>, 2> intra_rnn;
  std::array<RnnWeights, 2> inter_rnn;
  const TensorView* intra_fc_w = nullptr;
  const TensorView* intra_fc_b = nullptr;
  const TensorView* intra_ln_w = nullptr;
  const TensorView* intra_ln_b = nullptr;
  const TensorView* inter_fc_w = nullptr;
  const TensorView* inter_fc_b = nullptr;
  const TensorView* inter_ln_w = nullptr;
  const TensorView* inter_ln_b = nullptr;
};

struct ConvBnActWeights {
  const TensorView* conv_w = nullptr;
  const TensorView* conv_b = nullptr;
  const TensorView* bn_w = nullptr;
  const TensorView* bn_b = nullptr;
  const TensorView* bn_mean = nullptr;
  const TensorView* bn_var = nullptr;
  const TensorView* act = nullptr;
};

class Model {
 public:
  Model();
  void reset();
  void infer(const float* mix, float* enh);
  void set_trace_dir(const std::string& path);

 private:
  std::unordered_map<std::string, TensorView> weights_;
  std::vector<float> conv_cache_;
  std::vector<float> tra_cache_;
  std::vector<float> inter_cache_;
  std::vector<float> feat3_;
  std::vector<float> erb_;
  std::vector<float> x_;
  std::vector<float> y_;
  std::array<std::vector<float>, 5> en_;
  std::vector<float> mask_;
  std::vector<float> stream_x1_;
  std::vector<float> stream_x2_;
  std::vector<float> stream_s_;
  std::vector<float> stream_h_;
  std::vector<float> stream_tmp_;
  std::vector<float> dp_intra_;
  std::vector<float> dp_fc_;
  std::vector<float> dp_intra_out_;
  std::vector<float> dp_inter_fc_;
  std::vector<float> upsample_up_;
  std::vector<float> upsample_padded_;
  const TensorView* erb_fc_w_ = nullptr;
  const TensorView* ierb_fc_w_ = nullptr;
  std::array<ConvBnActWeights, 2> encoder_front_;
  std::array<ConvBnActWeights, 2> decoder_tail_;
  std::array<StreamWeights, 6> stream_weights_;
  std::array<DpgrnnWeights, 2> dpgrnn_weights_;
  std::string trace_dir_;
  int trace_frame_ = 0;

  const TensorView& w(const std::string& name) const;
  void init_scratch();
  void trace_stage(const std::string& name, const float* data, size_t size) const;
  void trace_stage(const std::string& name, const std::vector<float>& data) const;
  ConvBnActWeights make_conv_bn_act_weights(const std::string& prefix, bool nested_conv) const;
  StreamWeights make_stream_weights(const std::string& prefix, bool deconv_depth) const;
  DpgrnnWeights make_dpgrnn_weights(const std::string& prefix) const;
  void stream_gt_block(const StreamWeights& sw, int cache_stream,
                       int cache_start, int cache_t, int dilation, const std::vector<float>& x,
                       std::vector<float>& y, int tra_base);
  void dpgrnn(const DpgrnnWeights& dw, int inter_base, std::vector<float>& x);
};

Stats summarize(std::vector<double> values);

}  // namespace gtcrn
