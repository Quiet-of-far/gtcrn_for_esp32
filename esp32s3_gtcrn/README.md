# GTCRN ESP32-S3 Prototype

This directory contains an ESP-IDF prototype for the fine-tuned DNS3 GTCRN model.

## Scope

- Offline frame-level inference only
- Float32 model path
- No I2S/audio driver integration yet

## Expected environment

- ESP-IDF 5.x with `idf.py`
- ESP32-S3 module with PSRAM

## Build

```bash
cd esp32s3_gtcrn
idf.py set-target esp32s3
idf.py build
idf.py flash monitor
```

## Host validation

Build from the repo root with:

```bash
g++ -std=c++17 -O3 \
  -Icpp_gtcrn -Iesp32s3_gtcrn/common \
  -DGTCRN_WEIGHTS_HEADER='\"gtcrn_weights_dns3_finetuned.h\"' \
  cpp_gtcrn/gtcrn_model.cpp \
  esp32s3_gtcrn/common/gtcrn_embedded_model.cpp \
  esp32s3_gtcrn/host_validate.cpp \
  -o /tmp/gtcrn_esp_validate
```

Run:

```bash
/tmp/gtcrn_esp_validate
```

The current acceptance threshold is `max_abs < 1e-3`.
