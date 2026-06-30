#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>

#include "driver/gpio.h"
#include "driver/i2s_std.h"
#include "esp_check.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "gtcrn_embedded_model.h"

namespace {

constexpr char kTag[] = "gtcrn_rt";
constexpr int kSampleRate = 16000;
constexpr int kNfft = 512;
constexpr int kHop = 256;
constexpr int kBins = 257;
constexpr double kPi = 3.14159265358979323846;

constexpr int kRxPort = I2S_NUM_0;
constexpr int kTxPort = I2S_NUM_1;

// INMP441-class I2S mic
constexpr gpio_num_t kMicData = GPIO_NUM_6;
constexpr gpio_num_t kMicWs = GPIO_NUM_5;
constexpr gpio_num_t kMicSck = GPIO_NUM_4;
// MAX98357A-class I2S amp
constexpr gpio_num_t kSpkDin = GPIO_NUM_8;
constexpr gpio_num_t kSpkLrc = GPIO_NUM_9;
constexpr gpio_num_t kSpkBclk = GPIO_NUM_7;
constexpr gpio_num_t kSpkSd = GPIO_NUM_13;

constexpr int kDmaFrameNum = 256;
constexpr int kDmaDescNum = 6;
constexpr int kWarmupFrames = 5;
constexpr int kLogEveryFrames = 125;
constexpr bool kBypassModel = false;
constexpr float kInputGain = 3.0f;
constexpr float kBypassGain = 6.0f;

struct Runtime {
  i2s_chan_handle_t tx = nullptr;
  i2s_chan_handle_t rx = nullptr;
  gtcrn_esp::Workspace workspace;
  gtcrn_esp::EmbeddedModel model{&workspace};
  std::array<uint16_t, kNfft> bitrev{};
  std::array<std::complex<float>, kNfft / 2> twiddle_fwd{};
  std::array<std::complex<float>, kNfft / 2> twiddle_inv{};
  std::array<float, kNfft> window{};
  std::array<float, kNfft> analysis{};
  std::array<float, kNfft> ola{};
  float hp_prev_x = 0.0f;
  float hp_prev_y = 0.0f;
  std::array<int32_t, kHop> mic_i2s{};
  std::array<int16_t, kHop * 2> spk_i2s{};
  std::array<std::complex<float>, kNfft> spec{};
  std::array<std::complex<float>, kNfft> inv{};
  std::array<float, kBins * 2> mix{};
  std::array<float, kBins * 2> enh{};
  int64_t frame_index = 0;
  double model_sum_ms = 0.0;
  double frame_sum_ms = 0.0;
  double model_max_ms = 0.0;
  double frame_max_ms = 0.0;
  int64_t over_16ms = 0;
  double raw_abs_sum = 0.0;
  double pre_abs_sum = 0.0;
  double raw_peak = 0.0;
  double pre_peak = 0.0;
  int64_t pre_clip_samples = 0;
};

Runtime g_rt;

void fft(std::array<std::complex<float>, kNfft>& a, bool inverse) {
  for (int i = 0; i < kNfft; ++i) {
    int j = g_rt.bitrev[i];
    if (i < j) std::swap(a[i], a[j]);
  }
  const auto& twiddle = inverse ? g_rt.twiddle_inv : g_rt.twiddle_fwd;
  for (int len = 2; len <= kNfft; len <<= 1) {
    int half = len >> 1;
    int step = kNfft / len;
    for (int i = 0; i < kNfft; i += len) {
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
    for (auto& v : a) v /= static_cast<float>(kNfft);
  }
}

esp_err_t init_amp_enable() {
  gpio_config_t cfg{};
  cfg.pin_bit_mask = 1ULL << kSpkSd;
  cfg.mode = GPIO_MODE_OUTPUT;
  cfg.pull_up_en = GPIO_PULLUP_DISABLE;
  cfg.pull_down_en = GPIO_PULLDOWN_DISABLE;
  cfg.intr_type = GPIO_INTR_DISABLE;
  ESP_RETURN_ON_ERROR(gpio_config(&cfg), kTag, "gpio_config sd");
  ESP_RETURN_ON_ERROR(gpio_set_level(kSpkSd, 1), kTag, "gpio_set_level sd");
  return ESP_OK;
}

esp_err_t init_mic_rx(Runtime& rt) {
  i2s_chan_config_t cfg = I2S_CHANNEL_DEFAULT_CONFIG(kRxPort, I2S_ROLE_MASTER);
  cfg.dma_desc_num = kDmaDescNum;
  cfg.dma_frame_num = kDmaFrameNum;
  ESP_RETURN_ON_ERROR(i2s_new_channel(&cfg, nullptr, &rt.rx), kTag, "new rx");

  i2s_std_config_t std_cfg = {
      .clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(kSampleRate),
      .slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_32BIT, I2S_SLOT_MODE_MONO),
      .gpio_cfg =
          {
              .mclk = I2S_GPIO_UNUSED,
              .bclk = kMicSck,
              .ws = kMicWs,
              .dout = I2S_GPIO_UNUSED,
              .din = kMicData,
              .invert_flags =
                  {
                      .mclk_inv = false,
                      .bclk_inv = false,
                      .ws_inv = false,
                  },
          },
  };
  std_cfg.slot_cfg.slot_mask = I2S_STD_SLOT_LEFT;
  std_cfg.clk_cfg.mclk_multiple = I2S_MCLK_MULTIPLE_256;
  ESP_RETURN_ON_ERROR(i2s_channel_init_std_mode(rt.rx, &std_cfg), kTag, "init rx");
  ESP_RETURN_ON_ERROR(i2s_channel_enable(rt.rx), kTag, "enable rx");
  return ESP_OK;
}

esp_err_t init_spk_tx(Runtime& rt) {
  i2s_chan_config_t cfg = I2S_CHANNEL_DEFAULT_CONFIG(kTxPort, I2S_ROLE_MASTER);
  cfg.auto_clear = true;
  cfg.dma_desc_num = kDmaDescNum;
  cfg.dma_frame_num = kDmaFrameNum;
  ESP_RETURN_ON_ERROR(i2s_new_channel(&cfg, &rt.tx, nullptr), kTag, "new tx");

  i2s_std_config_t std_cfg = {
      .clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(kSampleRate),
      .slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_16BIT, I2S_SLOT_MODE_STEREO),
      .gpio_cfg =
          {
              .mclk = I2S_GPIO_UNUSED,
              .bclk = kSpkBclk,
              .ws = kSpkLrc,
              .dout = kSpkDin,
              .din = I2S_GPIO_UNUSED,
              .invert_flags =
                  {
                      .mclk_inv = false,
                      .bclk_inv = false,
                      .ws_inv = false,
                  },
          },
  };
  std_cfg.clk_cfg.mclk_multiple = I2S_MCLK_MULTIPLE_256;
  ESP_RETURN_ON_ERROR(i2s_channel_init_std_mode(rt.tx, &std_cfg), kTag, "init tx");
  ESP_RETURN_ON_ERROR(i2s_channel_enable(rt.tx), kTag, "enable tx");
  return ESP_OK;
}

void init_model(Runtime& rt) {
  for (int i = 0; i < kNfft; ++i) {
    unsigned x = static_cast<unsigned>(i);
    unsigned r = 0;
    for (int b = 0; b < 9; ++b) {
      r = (r << 1) | (x & 1U);
      x >>= 1U;
    }
    rt.bitrev[i] = static_cast<uint16_t>(r);
  }
  for (int i = 0; i < kNfft / 2; ++i) {
    float ang = static_cast<float>(2.0 * kPi * i / kNfft);
    rt.twiddle_fwd[i] = {std::cos(-ang), std::sin(-ang)};
    rt.twiddle_inv[i] = {std::cos(ang), std::sin(ang)};
  }
  for (int i = 0; i < kNfft; ++i) {
    float hann = 0.5f * (1.0f - std::cos(static_cast<float>(2.0 * kPi * i / (kNfft - 1))));
    rt.window[i] = std::sqrt(std::max(hann, 0.0f));
  }
  rt.model.reset();
  rt.mix.fill(0.0f);
  rt.enh.fill(0.0f);
  for (int i = 0; i < kWarmupFrames; ++i) rt.model.infer(rt.mix.data(), rt.enh.data());
  rt.model.reset();
}

inline float mic_slot_to_float(int32_t raw) {
  // 24-bit microphone data is usually left-justified inside 32-bit slot.
  int32_t s24 = raw >> 8;
  return static_cast<float>(s24) / 8388608.0f;
}

inline float highpass(Runtime& rt, float x) {
  constexpr float a = 0.995f;
  float y = x - rt.hp_prev_x + a * rt.hp_prev_y;
  rt.hp_prev_x = x;
  rt.hp_prev_y = y;
  return y;
}

void process_frame(Runtime& rt, const int32_t* mic_mono, int16_t* spk_stereo, double* model_ms, double* frame_ms) {
  int64_t t0 = esp_timer_get_time();

  std::move(rt.analysis.begin() + kHop, rt.analysis.end(), rt.analysis.begin());
  double raw_abs_sum = 0.0;
  double pre_abs_sum = 0.0;
  double raw_peak = 0.0;
  double pre_peak = 0.0;
  int clip_samples = 0;
  for (int i = 0; i < kHop; ++i) {
    float raw = mic_slot_to_float(mic_mono[i]);
    float hp = highpass(rt, raw);
    float pre = hp * kInputGain;
    raw_abs_sum += std::fabs(raw);
    pre_abs_sum += std::fabs(pre);
    raw_peak = std::max(raw_peak, static_cast<double>(std::fabs(raw)));
    pre_peak = std::max(pre_peak, static_cast<double>(std::fabs(pre)));
    if (std::fabs(pre) > 1.0f) ++clip_samples;
    rt.analysis[kNfft - kHop + i] = std::clamp(pre, -1.0f, 1.0f);
  }
  rt.raw_abs_sum += raw_abs_sum / kHop;
  rt.pre_abs_sum += pre_abs_sum / kHop;
  rt.raw_peak = std::max(rt.raw_peak, raw_peak);
  rt.pre_peak = std::max(rt.pre_peak, pre_peak);
  rt.pre_clip_samples += clip_samples;

  if (kBypassModel) {
    for (int i = 0; i < kHop; ++i) {
      float s = rt.analysis[kNfft - kHop + i] * kBypassGain;
      s = std::clamp(s, -1.0f, 1.0f);
      int16_t out = static_cast<int16_t>(std::lrint(s * 32767.0f));
      spk_stereo[i * 2] = out;
      spk_stereo[i * 2 + 1] = out;
    }
    int64_t t1 = esp_timer_get_time();
    *model_ms = 0.0;
    *frame_ms = static_cast<double>(t1 - t0) / 1000.0;
    return;
  }

  for (int i = 0; i < kNfft; ++i) rt.spec[i] = {rt.analysis[i] * rt.window[i], 0.0f};
  fft(rt.spec, false);

  for (int i = 0; i < kBins; ++i) {
    rt.mix[i * 2] = rt.spec[i].real();
    rt.mix[i * 2 + 1] = rt.spec[i].imag();
  }

  int64_t t_model0 = esp_timer_get_time();
  rt.model.infer(rt.mix.data(), rt.enh.data());
  int64_t t_model1 = esp_timer_get_time();
  *model_ms = static_cast<double>(t_model1 - t_model0) / 1000.0;

  for (int i = 0; i < kBins; ++i) rt.inv[i] = {rt.enh[i * 2], rt.enh[i * 2 + 1]};
  for (int i = 1; i < kBins - 1; ++i) rt.inv[kNfft - i] = std::conj(rt.inv[i]);
  fft(rt.inv, true);

  for (int i = 0; i < kNfft; ++i) rt.ola[i] += rt.inv[i].real() * rt.window[i];
  for (int i = 0; i < kHop; ++i) {
    float v = std::clamp(rt.ola[i], -1.0f, 1.0f);
    int16_t s = static_cast<int16_t>(std::lrint(v * 32767.0f));
    spk_stereo[i * 2] = s;
    spk_stereo[i * 2 + 1] = s;
  }
  std::move(rt.ola.begin() + kHop, rt.ola.end(), rt.ola.begin());
  std::fill(rt.ola.end() - kHop, rt.ola.end(), 0.0f);

  int64_t t1 = esp_timer_get_time();
  *frame_ms = static_cast<double>(t1 - t0) / 1000.0;
}

void audio_task(void*) {
  size_t bytes_read = 0;
  size_t bytes_written = 0;
  ESP_LOGI(kTag, "audio loop start");

  while (true) {
    esp_err_t err = i2s_channel_read(g_rt.rx, g_rt.mic_i2s.data(), g_rt.mic_i2s.size() * sizeof(int32_t), &bytes_read, portMAX_DELAY);
    if (err != ESP_OK) {
      ESP_LOGE(kTag, "i2s read failed: %s", esp_err_to_name(err));
      vTaskDelay(pdMS_TO_TICKS(100));
      continue;
    }
    if (bytes_read != g_rt.mic_i2s.size() * sizeof(int32_t)) {
      ESP_LOGW(kTag, "short read: %u", static_cast<unsigned>(bytes_read));
      continue;
    }

    double model_ms = 0.0;
    double frame_ms = 0.0;
    process_frame(g_rt, g_rt.mic_i2s.data(), g_rt.spk_i2s.data(), &model_ms, &frame_ms);

    err = i2s_channel_write(g_rt.tx, g_rt.spk_i2s.data(), g_rt.spk_i2s.size() * sizeof(int16_t), &bytes_written, portMAX_DELAY);
    if (err != ESP_OK) {
      ESP_LOGE(kTag, "i2s write failed: %s", esp_err_to_name(err));
      vTaskDelay(pdMS_TO_TICKS(100));
      continue;
    }

    ++g_rt.frame_index;
    g_rt.model_sum_ms += model_ms;
    g_rt.frame_sum_ms += frame_ms;
    g_rt.model_max_ms = std::max(g_rt.model_max_ms, model_ms);
    g_rt.frame_max_ms = std::max(g_rt.frame_max_ms, frame_ms);
    if (frame_ms > 16.0) ++g_rt.over_16ms;

    if ((g_rt.frame_index % kLogEveryFrames) == 0) {
      double avg_model = g_rt.model_sum_ms / g_rt.frame_index;
      double avg_frame = g_rt.frame_sum_ms / g_rt.frame_index;
      ESP_LOGI(kTag,
               "frames=%lld avg_model_ms=%.3f max_model_ms=%.3f avg_frame_ms=%.3f max_frame_ms=%.3f over16=%lld gain=%.2f raw_abs=%.4f pre_abs=%.4f raw_peak=%.4f pre_peak=%.4f pre_clip=%lld free_heap=%u internal=%u",
               static_cast<long long>(g_rt.frame_index), avg_model, g_rt.model_max_ms, avg_frame, g_rt.frame_max_ms,
               static_cast<long long>(g_rt.over_16ms), static_cast<double>(kInputGain),
               g_rt.raw_abs_sum / g_rt.frame_index, g_rt.pre_abs_sum / g_rt.frame_index,
               g_rt.raw_peak, g_rt.pre_peak, static_cast<long long>(g_rt.pre_clip_samples),
               static_cast<unsigned>(heap_caps_get_free_size(MALLOC_CAP_8BIT)),
               static_cast<unsigned>(heap_caps_get_free_size(MALLOC_CAP_INTERNAL)));
    }
  }
}

}  // namespace

extern "C" void app_main(void) {
  ESP_LOGI(kTag, "starting realtime gtcrn, mic=(sd=%d ws=%d sck=%d) spk=(din=%d lrc=%d bclk=%d sd=%d)",
           static_cast<int>(kMicData), static_cast<int>(kMicWs), static_cast<int>(kMicSck),
           static_cast<int>(kSpkDin), static_cast<int>(kSpkLrc), static_cast<int>(kSpkBclk), static_cast<int>(kSpkSd));
  ESP_ERROR_CHECK(init_amp_enable());
  ESP_ERROR_CHECK(init_mic_rx(g_rt));
  ESP_ERROR_CHECK(init_spk_tx(g_rt));
  init_model(g_rt);
  xTaskCreatePinnedToCore(audio_task, "audio_task", 24576, nullptr, 5, nullptr, 0);
}
