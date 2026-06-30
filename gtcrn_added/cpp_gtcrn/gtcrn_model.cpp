#include "gtcrn_model.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <stdexcept>

namespace gtcrn {
namespace {

inline float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }
inline int idx2(int c, int f, int F) { return c * F + f; }
inline int conv_cache_idx(int stream, int c, int t, int f) {
  return ((stream * 16 + c) * 16 + t) * BOT_F + f;
}
inline int tra_cache_idx(int stream, int block, int h) {
  return (stream * 3 + block) * 16 + h;
}
inline int inter_cache_idx(int block, int f, int h) { return (block * BOT_F + f) * 16 + h; }

void batch_norm(std::vector<float>& x, int C, int F, const TensorView& gamma,
                const TensorView& beta, const TensorView& mean, const TensorView& var) {
  for (int c = 0; c < C; ++c) {
    float scale = gamma.data[c] / std::sqrt(var.data[c] + 1e-5f);
    float bias = beta.data[c] - mean.data[c] * scale;
    for (int f = 0; f < F; ++f) x[idx2(c, f, F)] = x[idx2(c, f, F)] * scale + bias;
  }
}

void prelu(std::vector<float>& x, float a) {
  for (float& v : x) {
    if (v < 0.0f) v *= a;
  }
}

void linear(const TensorView& weight, const TensorView& bias, const float* x, float* y,
            int out, int in) {
  for (int o = 0; o < out; ++o) {
    float acc = bias.size ? bias.data[o] : 0.0f;
    const float* row = weight.data + o * in;
    for (int i = 0; i < in; ++i) acc += row[i] * x[i];
    y[o] = acc;
  }
}

void conv1x1(const TensorView& weight, const TensorView& bias, const std::vector<float>& x,
             std::vector<float>& y, int in_c, int out_c, int F) {
  y.assign(out_c * F, 0.0f);
  for (int oc = 0; oc < out_c; ++oc) {
    for (int f = 0; f < F; ++f) {
      float acc = bias.data[oc];
      for (int ic = 0; ic < in_c; ++ic) acc += weight.data[oc * in_c + ic] * x[idx2(ic, f, F)];
      y[idx2(oc, f, F)] = acc;
    }
  }
}

void conv1x5(const TensorView& weight, const TensorView& bias, const std::vector<float>& x,
             std::vector<float>& y, int in_c, int out_c, int in_f, int out_f, int stride,
             int groups) {
  y.assign(out_c * out_f, 0.0f);
  int out_per_group = out_c / groups;
  int in_per_group = in_c / groups;
  for (int oc = 0; oc < out_c; ++oc) {
    int g = oc / out_per_group;
    for (int of = 0; of < out_f; ++of) {
      float acc = bias.data[oc];
      int center = of * stride - 2;
      for (int icg = 0; icg < in_per_group; ++icg) {
        int ic = g * in_per_group + icg;
        for (int k = 0; k < 5; ++k) {
          int f = center + k;
          if (f >= 0 && f < in_f) {
            int widx = ((oc * in_per_group + icg) * 5) + k;
            acc += weight.data[widx] * x[idx2(ic, f, in_f)];
          }
        }
      }
      y[idx2(oc, of, out_f)] = acc;
    }
  }
}

void upsample_conv1x5(const TensorView& weight, const TensorView& bias,
                      const std::vector<float>& x, std::vector<float>& y, int in_c,
                      int out_c, int in_f, int out_f, int groups,
                      std::vector<float>& up, std::vector<float>& padded) {
  up.assign(in_c * (in_f * 2), 0.0f);
  int up_f = in_f * 2;
  for (int c = 0; c < in_c; ++c) {
    for (int f = 0; f < in_f; ++f) up[idx2(c, f * 2, up_f)] = x[idx2(c, f, in_f)];
  }
  padded.assign(in_c * (up_f + 3), 0.0f);
  int pad_f = up_f + 3;
  for (int c = 0; c < in_c; ++c) {
    for (int f = 0; f < up_f; ++f) padded[idx2(c, f + 2, pad_f)] = up[idx2(c, f, up_f)];
  }
  y.assign(out_c * out_f, 0.0f);
  int out_per_group = out_c / groups;
  int in_per_group = in_c / groups;
  for (int oc = 0; oc < out_c; ++oc) {
    int g = oc / out_per_group;
    for (int of = 0; of < out_f; ++of) {
      float acc = bias.data[oc];
      for (int icg = 0; icg < in_per_group; ++icg) {
        int ic = g * in_per_group + icg;
        for (int k = 0; k < 5; ++k) {
          int widx = ((oc * in_per_group + icg) * 5) + k;
          acc += weight.data[widx] * padded[idx2(ic, of + k, pad_f)];
        }
      }
      y[idx2(oc, of, out_f)] = acc;
    }
  }
}

void sfe(const std::vector<float>& x, std::vector<float>& y, int C, int F) {
  y.assign(C * 3 * F, 0.0f);
  for (int c = 0; c < C; ++c) {
    for (int k = 0; k < 3; ++k) {
      for (int f = 0; f < F; ++f) {
        int src_f = f + k - 1;
        y[idx2(c * 3 + k, f, F)] = (src_f >= 0 && src_f < F) ? x[idx2(c, src_f, F)] : 0.0f;
      }
    }
  }
}

void gru_cell(const TensorView& wih, const TensorView& whh, const TensorView& bih,
              const TensorView& bhh, const float* x, float* h, int in_size, int hidden) {
  float gi[48];
  float gh[48];
  for (int g = 0; g < 3 * hidden; ++g) {
    float ai = bih.data[g];
    for (int i = 0; i < in_size; ++i) ai += wih.data[g * in_size + i] * x[i];
    gi[g] = ai;
    float ah = bhh.data[g];
    for (int i = 0; i < hidden; ++i) ah += whh.data[g * hidden + i] * h[i];
    gh[g] = ah;
  }
  for (int i = 0; i < hidden; ++i) {
    float r = sigmoid(gi[i] + gh[i]);
    float z = sigmoid(gi[hidden + i] + gh[hidden + i]);
    float n = std::tanh(gi[2 * hidden + i] + r * gh[2 * hidden + i]);
    h[i] = (1.0f - z) * n + z * h[i];
  }
}

void layer_norm(std::vector<float>& x, const TensorView& gamma, const TensorView& beta) {
  double sum = 0.0;
  for (float v : x) sum += v;
  float mean = static_cast<float>(sum / x.size());
  double var_sum = 0.0;
  for (float v : x) {
    double d = v - mean;
    var_sum += d * d;
  }
  float inv = 1.0f / std::sqrt(static_cast<float>(var_sum / x.size()) + 1e-8f);
  for (size_t i = 0; i < x.size(); ++i) x[i] = (x[i] - mean) * inv * gamma.data[i] + beta.data[i];
}

void add_inplace(std::vector<float>& x, const std::vector<float>& y) {
  for (size_t i = 0; i < x.size(); ++i) x[i] += y[i];
}

}  // namespace

#ifdef GTCRN_ENABLE_TRACE
#define GTCRN_TRACE_STAGE(...) trace_stage(__VA_ARGS__)
#else
#define GTCRN_TRACE_STAGE(...) ((void)0)
#endif

Model::Model() : weights_(make_weight_map()) {
  init_scratch();
  reset();
}

void Model::set_trace_dir(const std::string& path) {
  trace_dir_ = path;
  trace_frame_ = 0;
  if (!trace_dir_.empty()) std::filesystem::create_directories(trace_dir_);
}

void Model::trace_stage(const std::string& name, const float* data, size_t size) const {
  if (trace_dir_.empty() || trace_frame_ != 0) return;
  std::ofstream out(std::filesystem::path(trace_dir_) / (name + ".f32"), std::ios::binary);
  if (!out) throw std::runtime_error("failed to open trace output: " + name);
  out.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(size * sizeof(float)));
}

void Model::trace_stage(const std::string& name, const std::vector<float>& data) const {
  trace_stage(name, data.data(), data.size());
}

void Model::init_scratch() {
  feat3_.reserve(3 * FREQ);
  erb_.reserve(3 * ERB_FULL);
  x_.reserve(16 * ERB_FULL);
  y_.reserve(16 * ERB_FULL);
  en_[0].reserve(16 * 65);
  en_[1].reserve(16 * BOT_F);
  en_[2].reserve(16 * BOT_F);
  en_[3].reserve(16 * BOT_F);
  en_[4].reserve(16 * BOT_F);
  mask_.reserve(2 * FREQ);
  stream_x1_.reserve(8 * BOT_F);
  stream_x2_.reserve(8 * BOT_F);
  stream_s_.reserve(24 * BOT_F);
  stream_h_.reserve(16 * BOT_F);
  stream_tmp_.reserve(16 * BOT_F);
  dp_intra_.reserve(16 * BOT_F);
  dp_fc_.reserve(16 * BOT_F);
  dp_intra_out_.reserve(16 * BOT_F);
  dp_inter_fc_.reserve(16 * BOT_F);
  upsample_up_.reserve(16 * 130);
  upsample_padded_.reserve(16 * 133);
  erb_fc_w_ = &w("erb.erb_fc.weight");
  ierb_fc_w_ = &w("erb.ierb_fc.weight");
  encoder_front_[0] = make_conv_bn_act_weights("encoder.en_convs.0", false);
  encoder_front_[1] = make_conv_bn_act_weights("encoder.en_convs.1", false);
  decoder_tail_[0] = make_conv_bn_act_weights("decoder.de_convs.3", true);
  decoder_tail_[1] = make_conv_bn_act_weights("decoder.de_convs.4", true);
  stream_weights_[0] = make_stream_weights("encoder.en_convs.2", false);
  stream_weights_[1] = make_stream_weights("encoder.en_convs.3", false);
  stream_weights_[2] = make_stream_weights("encoder.en_convs.4", false);
  stream_weights_[3] = make_stream_weights("decoder.de_convs.0", true);
  stream_weights_[4] = make_stream_weights("decoder.de_convs.1", true);
  stream_weights_[5] = make_stream_weights("decoder.de_convs.2", true);
  dpgrnn_weights_[0] = make_dpgrnn_weights("dpgrnn1");
  dpgrnn_weights_[1] = make_dpgrnn_weights("dpgrnn2");
}

ConvBnActWeights Model::make_conv_bn_act_weights(const std::string& prefix, bool nested_conv) const {
  ConvBnActWeights cw;
  std::string conv_prefix = prefix + (nested_conv ? ".conv.conv" : ".conv");
  cw.conv_w = &w(conv_prefix + ".weight");
  cw.conv_b = &w(conv_prefix + ".bias");
  cw.bn_w = &w(prefix + ".bn.weight");
  cw.bn_b = &w(prefix + ".bn.bias");
  cw.bn_mean = &w(prefix + ".bn.running_mean");
  cw.bn_var = &w(prefix + ".bn.running_var");
  auto act = weights_.find(prefix + ".act.weight");
  cw.act = act == weights_.end() ? nullptr : &act->second;
  return cw;
}

StreamWeights Model::make_stream_weights(const std::string& prefix, bool deconv_depth) const {
  StreamWeights sw;
  sw.point_conv1_w = &w(prefix + ".point_conv1.weight");
  sw.point_conv1_b = &w(prefix + ".point_conv1.bias");
  sw.point_bn1_w = &w(prefix + ".point_bn1.weight");
  sw.point_bn1_b = &w(prefix + ".point_bn1.bias");
  sw.point_bn1_mean = &w(prefix + ".point_bn1.running_mean");
  sw.point_bn1_var = &w(prefix + ".point_bn1.running_var");
  sw.point_act = &w(prefix + ".point_act.weight");
  sw.depth_w = &w(prefix + (deconv_depth ? ".depth_conv.ConvTranspose2d.weight" : ".depth_conv.Conv2d.weight"));
  sw.depth_b = &w(prefix + (deconv_depth ? ".depth_conv.ConvTranspose2d.bias" : ".depth_conv.Conv2d.bias"));
  sw.depth_bn_w = &w(prefix + ".depth_bn.weight");
  sw.depth_bn_b = &w(prefix + ".depth_bn.bias");
  sw.depth_bn_mean = &w(prefix + ".depth_bn.running_mean");
  sw.depth_bn_var = &w(prefix + ".depth_bn.running_var");
  sw.depth_act = &w(prefix + ".depth_act.weight");
  sw.point_conv2_w = &w(prefix + ".point_conv2.weight");
  sw.point_conv2_b = &w(prefix + ".point_conv2.bias");
  sw.point_bn2_w = &w(prefix + ".point_bn2.weight");
  sw.point_bn2_b = &w(prefix + ".point_bn2.bias");
  sw.point_bn2_mean = &w(prefix + ".point_bn2.running_mean");
  sw.point_bn2_var = &w(prefix + ".point_bn2.running_var");
  sw.tra_gru.weight_ih = &w(prefix + ".tra.att_gru.weight_ih_l0");
  sw.tra_gru.weight_hh = &w(prefix + ".tra.att_gru.weight_hh_l0");
  sw.tra_gru.bias_ih = &w(prefix + ".tra.att_gru.bias_ih_l0");
  sw.tra_gru.bias_hh = &w(prefix + ".tra.att_gru.bias_hh_l0");
  sw.tra_fc_w = &w(prefix + ".tra.att_fc.weight");
  sw.tra_fc_b = &w(prefix + ".tra.att_fc.bias");
  return sw;
}

DpgrnnWeights Model::make_dpgrnn_weights(const std::string& prefix) const {
  DpgrnnWeights dw;
  for (int grp = 0; grp < 2; ++grp) {
    for (int dir = 0; dir < 2; ++dir) {
      std::string base = prefix + ".intra_rnn.rnn" + std::to_string(grp + 1);
      std::string suf = dir == 0 ? "" : "_reverse";
      dw.intra_rnn[grp][dir].weight_ih = &w(base + ".weight_ih_l0" + suf);
      dw.intra_rnn[grp][dir].weight_hh = &w(base + ".weight_hh_l0" + suf);
      dw.intra_rnn[grp][dir].bias_ih = &w(base + ".bias_ih_l0" + suf);
      dw.intra_rnn[grp][dir].bias_hh = &w(base + ".bias_hh_l0" + suf);
    }
    std::string base = prefix + ".inter_rnn.rnn" + std::to_string(grp + 1);
    dw.inter_rnn[grp].weight_ih = &w(base + ".weight_ih_l0");
    dw.inter_rnn[grp].weight_hh = &w(base + ".weight_hh_l0");
    dw.inter_rnn[grp].bias_ih = &w(base + ".bias_ih_l0");
    dw.inter_rnn[grp].bias_hh = &w(base + ".bias_hh_l0");
  }
  dw.intra_fc_w = &w(prefix + ".intra_fc.weight");
  dw.intra_fc_b = &w(prefix + ".intra_fc.bias");
  dw.intra_ln_w = &w(prefix + ".intra_ln.weight");
  dw.intra_ln_b = &w(prefix + ".intra_ln.bias");
  dw.inter_fc_w = &w(prefix + ".inter_fc.weight");
  dw.inter_fc_b = &w(prefix + ".inter_fc.bias");
  dw.inter_ln_w = &w(prefix + ".inter_ln.weight");
  dw.inter_ln_b = &w(prefix + ".inter_ln.bias");
  return dw;
}

void Model::reset() {
  conv_cache_.assign(2 * 16 * 16 * BOT_F, 0.0f);
  tra_cache_.assign(2 * 3 * 16, 0.0f);
  inter_cache_.assign(2 * BOT_F * 16, 0.0f);
}

const TensorView& Model::w(const std::string& name) const {
  auto it = weights_.find(name);
  if (it == weights_.end()) throw std::runtime_error("missing weight: " + name);
  return it->second;
}

void Model::stream_gt_block(const StreamWeights& sw, int cache_stream,
                            int cache_start, int cache_t, int dilation, const std::vector<float>& x,
                            std::vector<float>& y, int tra_base) {
  auto& x1 = stream_x1_;
  auto& x2 = stream_x2_;
  x1.resize(8 * BOT_F);
  x2.resize(8 * BOT_F);
  for (int c = 0; c < 8; ++c) {
    for (int f = 0; f < BOT_F; ++f) {
      x1[idx2(c, f, BOT_F)] = x[idx2(c, f, BOT_F)];
      x2[idx2(c, f, BOT_F)] = x[idx2(c + 8, f, BOT_F)];
    }
  }
  auto& s = stream_s_;
  auto& h = stream_h_;
  auto& tmp = stream_tmp_;
  sfe(x1, s, 8, BOT_F);
  conv1x1(*sw.point_conv1_w, *sw.point_conv1_b, s, h, 24, 16, BOT_F);
  batch_norm(h, 16, BOT_F, *sw.point_bn1_w, *sw.point_bn1_b, *sw.point_bn1_mean, *sw.point_bn1_var);
  prelu(h, sw.point_act->data[0]);

  tmp.assign(16 * BOT_F, 0.0f);
  const TensorView& depth_w = *sw.depth_w;
  const TensorView& depth_b = *sw.depth_b;
  for (int c = 0; c < 16; ++c) {
    for (int f = 0; f < BOT_F; ++f) {
      float acc = depth_b.data[c];
      for (int kt = 0; kt < 3; ++kt) {
        int tt = kt * dilation;
        for (int kf = 0; kf < 3; ++kf) {
          int ff = f + kf - 1;
          if (ff < 0 || ff >= BOT_F) continue;
          float v = (tt < cache_t) ? conv_cache_[conv_cache_idx(cache_stream, c, cache_start + tt, ff)]
                                   : h[idx2(c, ff, BOT_F)];
          acc += depth_w.data[((c * 3 + kt) * 3) + kf] * v;
        }
      }
      tmp[idx2(c, f, BOT_F)] = acc;
    }
    for (int t = 0; t < cache_t - 1; ++t)
      for (int f = 0; f < BOT_F; ++f)
        conv_cache_[conv_cache_idx(cache_stream, c, cache_start + t, f)] =
            conv_cache_[conv_cache_idx(cache_stream, c, cache_start + t + 1, f)];
    for (int f = 0; f < BOT_F; ++f)
      conv_cache_[conv_cache_idx(cache_stream, c, cache_start + cache_t - 1, f)] = h[idx2(c, f, BOT_F)];
  }
  h.swap(tmp);
  batch_norm(h, 16, BOT_F, *sw.depth_bn_w, *sw.depth_bn_b, *sw.depth_bn_mean, *sw.depth_bn_var);
  prelu(h, sw.depth_act->data[0]);
  conv1x1(*sw.point_conv2_w, *sw.point_conv2_b, h, tmp, 16, 8, BOT_F);
  h.swap(tmp);
  batch_norm(h, 8, BOT_F, *sw.point_bn2_w, *sw.point_bn2_b, *sw.point_bn2_mean, *sw.point_bn2_var);

  float zt[8], gru_in[8], att_h[16], att[8];
  for (int c = 0; c < 8; ++c) {
    float sum = 0.0f;
    for (int f = 0; f < BOT_F; ++f) {
      float v = h[idx2(c, f, BOT_F)];
      sum += v * v;
    }
    zt[c] = sum / BOT_F;
  }
  std::memcpy(gru_in, zt, sizeof(zt));
  for (int i = 0; i < 16; ++i) att_h[i] = tra_cache_[tra_cache_idx(tra_base / 3, tra_base % 3, i)];
  gru_cell(*sw.tra_gru.weight_ih, *sw.tra_gru.weight_hh, *sw.tra_gru.bias_ih, *sw.tra_gru.bias_hh,
           gru_in, att_h, 8, 16);
  for (int i = 0; i < 16; ++i) tra_cache_[tra_cache_idx(tra_base / 3, tra_base % 3, i)] = att_h[i];
  linear(*sw.tra_fc_w, *sw.tra_fc_b, att_h, att, 8, 16);
  for (float& v : att) v = sigmoid(v);
  for (int c = 0; c < 8; ++c)
    for (int f = 0; f < BOT_F; ++f) h[idx2(c, f, BOT_F)] *= att[c];

  y.assign(16 * BOT_F, 0.0f);
  for (int c = 0; c < 8; ++c) {
    for (int f = 0; f < BOT_F; ++f) {
      y[idx2(2 * c, f, BOT_F)] = h[idx2(c, f, BOT_F)];
      y[idx2(2 * c + 1, f, BOT_F)] = x2[idx2(c, f, BOT_F)];
    }
  }
}

void Model::dpgrnn(const DpgrnnWeights& dw, int inter_base, std::vector<float>& x) {
  auto& intra = dp_intra_;
  intra.assign(16 * BOT_F, 0.0f);
  for (int grp = 0; grp < 2; ++grp) {
    for (int dir = 0; dir < 2; ++dir) {
      float hstate[4] = {};
      const RnnWeights& rw = dw.intra_rnn[grp][dir];
      for (int step = 0; step < BOT_F; ++step) {
        int f = dir == 0 ? step : (BOT_F - 1 - step);
        float xin[8];
        for (int i = 0; i < 8; ++i) xin[i] = x[idx2(grp * 8 + i, f, BOT_F)];
        gru_cell(*rw.weight_ih, *rw.weight_hh, *rw.bias_ih, *rw.bias_hh, xin, hstate, 8, 4);
        for (int i = 0; i < 4; ++i) intra[idx2(grp * 8 + dir * 4 + i, f, BOT_F)] = hstate[i];
      }
    }
  }
  auto& fc = dp_fc_;
  fc.resize(16 * BOT_F);
  for (int f = 0; f < BOT_F; ++f) {
    float in[16], out[16];
    for (int c = 0; c < 16; ++c) in[c] = intra[idx2(c, f, BOT_F)];
    linear(*dw.intra_fc_w, *dw.intra_fc_b, in, out, 16, 16);
    for (int c = 0; c < 16; ++c) fc[f * 16 + c] = out[c];
  }
  layer_norm(fc, *dw.intra_ln_w, *dw.intra_ln_b);
  auto& intra_out = dp_intra_out_;
  intra_out.resize(16 * BOT_F);
  for (int f = 0; f < BOT_F; ++f)
    for (int c = 0; c < 16; ++c) intra_out[idx2(c, f, BOT_F)] = x[idx2(c, f, BOT_F)] + fc[f * 16 + c];

  auto& inter_fc = dp_inter_fc_;
  inter_fc.resize(16 * BOT_F);
  for (int f = 0; f < BOT_F; ++f) {
    float ycat[16];
    for (int grp = 0; grp < 2; ++grp) {
      const RnnWeights& rw = dw.inter_rnn[grp];
      float xin[8], hstate[8];
      for (int i = 0; i < 8; ++i) {
        xin[i] = intra_out[idx2(grp * 8 + i, f, BOT_F)];
        hstate[i] = inter_cache_[inter_cache_idx(inter_base, f, grp * 8 + i)];
      }
      gru_cell(*rw.weight_ih, *rw.weight_hh, *rw.bias_ih, *rw.bias_hh, xin, hstate, 8, 8);
      for (int i = 0; i < 8; ++i) {
        ycat[grp * 8 + i] = hstate[i];
        inter_cache_[inter_cache_idx(inter_base, f, grp * 8 + i)] = hstate[i];
      }
    }
    float out[16];
    linear(*dw.inter_fc_w, *dw.inter_fc_b, ycat, out, 16, 16);
    for (int c = 0; c < 16; ++c) inter_fc[f * 16 + c] = out[c];
  }
  layer_norm(inter_fc, *dw.inter_ln_w, *dw.inter_ln_b);
  for (int f = 0; f < BOT_F; ++f)
    for (int c = 0; c < 16; ++c) x[idx2(c, f, BOT_F)] = intra_out[idx2(c, f, BOT_F)] + inter_fc[f * 16 + c];
}

void Model::infer(const float* mix, float* enh) {
  GTCRN_TRACE_STAGE("00_mix", mix, FREQ * 2);
  auto& feat3 = feat3_;
  feat3.resize(3 * FREQ);
  for (int f = 0; f < FREQ; ++f) {
    float real = mix[f * 2], imag = mix[f * 2 + 1];
    feat3[idx2(0, f, FREQ)] = std::sqrt(real * real + imag * imag + 1e-12f);
    feat3[idx2(1, f, FREQ)] = real;
    feat3[idx2(2, f, FREQ)] = imag;
  }
  GTCRN_TRACE_STAGE("01_feat3", feat3);
  auto& erb = erb_;
  erb.assign(3 * ERB_FULL, 0.0f);
  for (int c = 0; c < 3; ++c) {
    for (int f = 0; f < ERB_LOW; ++f) erb[idx2(c, f, ERB_FULL)] = feat3[idx2(c, f, FREQ)];
    for (int o = 0; o < ERB_HIGH; ++o) {
      float acc = 0.0f;
      for (int i = 0; i < 192; ++i)
        acc += erb_fc_w_->data[o * 192 + i] * feat3[idx2(c, i + ERB_LOW, FREQ)];
      erb[idx2(c, o + ERB_LOW, ERB_FULL)] = acc;
    }
  }
  GTCRN_TRACE_STAGE("02_erb", erb);
  auto& x = x_;
  auto& y = y_;
  sfe(erb, x, 3, ERB_FULL);
  auto& en = en_;
  const auto& enc0 = encoder_front_[0];
  conv1x5(*enc0.conv_w, *enc0.conv_b, x, y, 9, 16, 129, 65, 2, 1);
  batch_norm(y, 16, 65, *enc0.bn_w, *enc0.bn_b, *enc0.bn_mean, *enc0.bn_var);
  prelu(y, enc0.act->data[0]);
  GTCRN_TRACE_STAGE("03_enc0", y);
  en[0] = y;
  const auto& enc1 = encoder_front_[1];
  conv1x5(*enc1.conv_w, *enc1.conv_b, y, x, 16, 16, 65, 33, 2, 2);
  batch_norm(x, 16, 33, *enc1.bn_w, *enc1.bn_b, *enc1.bn_mean, *enc1.bn_var);
  prelu(x, enc1.act->data[0]);
  GTCRN_TRACE_STAGE("04_enc1", x);
  en[1] = x;
  stream_gt_block(stream_weights_[0], 0, 0, 2, 1, x, y, 0); en[2] = y; GTCRN_TRACE_STAGE("05_enc2", y);
  stream_gt_block(stream_weights_[1], 0, 2, 4, 2, y, x, 1); en[3] = x; GTCRN_TRACE_STAGE("06_enc3", x);
  stream_gt_block(stream_weights_[2], 0, 6, 10, 5, x, y, 2); en[4] = y; GTCRN_TRACE_STAGE("07_enc4", y);
  x = y;
  dpgrnn(dpgrnn_weights_[0], 0, x);
  GTCRN_TRACE_STAGE("08_dpgrnn1", x);
  dpgrnn(dpgrnn_weights_[1], 1, x);
  GTCRN_TRACE_STAGE("09_dpgrnn2", x);
  add_inplace(x, en[4]); stream_gt_block(stream_weights_[3], 1, 6, 10, 5, x, y, 3); GTCRN_TRACE_STAGE("10_dec0", y);
  add_inplace(y, en[3]); stream_gt_block(stream_weights_[4], 1, 2, 4, 2, y, x, 4); GTCRN_TRACE_STAGE("11_dec1", x);
  add_inplace(x, en[2]); stream_gt_block(stream_weights_[5], 1, 0, 2, 1, x, y, 5); GTCRN_TRACE_STAGE("12_dec2", y);
  add_inplace(y, en[1]);
  const auto& dec3 = decoder_tail_[0];
  upsample_conv1x5(*dec3.conv_w, *dec3.conv_b, y, x, 16, 16, 33, 65, 2,
                   upsample_up_, upsample_padded_);
  batch_norm(x, 16, 65, *dec3.bn_w, *dec3.bn_b, *dec3.bn_mean, *dec3.bn_var);
  prelu(x, dec3.act->data[0]);
  GTCRN_TRACE_STAGE("13_dec3", x);
  add_inplace(x, en[0]);
  const auto& dec4 = decoder_tail_[1];
  upsample_conv1x5(*dec4.conv_w, *dec4.conv_b, x, y, 16, 2, 65, 129, 1,
                   upsample_up_, upsample_padded_);
  batch_norm(y, 2, 129, *dec4.bn_w, *dec4.bn_b, *dec4.bn_mean, *dec4.bn_var);
  for (float& v : y) v = std::tanh(v);
  GTCRN_TRACE_STAGE("14_dec4", y);

  auto& mask = mask_;
  mask.assign(2 * FREQ, 0.0f);
  for (int c = 0; c < 2; ++c) {
    for (int f = 0; f < ERB_LOW; ++f) mask[idx2(c, f, FREQ)] = y[idx2(c, f, ERB_FULL)];
    for (int o = 0; o < 192; ++o) {
      float acc = 0.0f;
      for (int i = 0; i < ERB_HIGH; ++i)
        acc += ierb_fc_w_->data[o * 64 + i] * y[idx2(c, i + ERB_LOW, ERB_FULL)];
      mask[idx2(c, o + ERB_LOW, FREQ)] = acc;
    }
  }
  GTCRN_TRACE_STAGE("15_mask", mask);
  for (int f = 0; f < FREQ; ++f) {
    float sr = mix[f * 2], si = mix[f * 2 + 1];
    float mr = mask[idx2(0, f, FREQ)], mi = mask[idx2(1, f, FREQ)];
    enh[f * 2] = sr * mr - si * mi;
    enh[f * 2 + 1] = si * mr + sr * mi;
  }
  GTCRN_TRACE_STAGE("16_enh", enh, FREQ * 2);
  ++trace_frame_;
}

#undef GTCRN_TRACE_STAGE

Stats summarize(std::vector<double> values) {
  Stats s;
  if (values.empty()) return s;
  std::sort(values.begin(), values.end());
  s.avg = std::accumulate(values.begin(), values.end(), 0.0) / values.size();
  auto at = [&](double q) { return values[std::min(values.size() - 1, static_cast<size_t>(q * (values.size() - 1)))]; };
  s.p50 = at(0.50);
  s.p95 = at(0.95);
  s.p99 = at(0.99);
  s.max = values.back();
  return s;
}

}  // namespace gtcrn
