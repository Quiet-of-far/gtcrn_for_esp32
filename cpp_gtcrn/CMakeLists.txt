cmake_minimum_required(VERSION 3.16)
project(gtcrn_cpp_float32 LANGUAGES CXX)

option(GTCRN_ARM_FAST_MATH "Enable ARM -Ofast tuning" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_executable(benchmark_gtcrn benchmark_gtcrn.cpp gtcrn_model.cpp)
set(GTCRN_WEIGHTS_HEADER "gtcrn_weights_dns3.h" CACHE STRING "GTCRN generated weights header")
target_compile_definitions(benchmark_gtcrn PRIVATE
  GTCRN_WEIGHTS_HEADER="${GTCRN_WEIGHTS_HEADER}"
  GTCRN_ENABLE_TRACE=1
)
set_property(TARGET benchmark_gtcrn PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
target_compile_options(benchmark_gtcrn PRIVATE -O3)
if(GTCRN_ARM_FAST_MATH AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
  target_compile_options(benchmark_gtcrn PRIVATE -Ofast -mcpu=cortex-a55 -funroll-loops -fno-math-errno)
endif()

add_executable(realtime_gtcrn realtime_gtcrn.cpp gtcrn_model.cpp)
target_compile_definitions(realtime_gtcrn PRIVATE GTCRN_WEIGHTS_HEADER="${GTCRN_WEIGHTS_HEADER}")
set_property(TARGET realtime_gtcrn PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
target_compile_options(realtime_gtcrn PRIVATE -O3)
if(GTCRN_ARM_FAST_MATH AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
  target_compile_options(realtime_gtcrn PRIVATE -Ofast -mcpu=cortex-a55 -funroll-loops -fno-math-errno)
endif()

add_executable(realtime_gtcrn_alsa realtime_gtcrn_alsa.cpp gtcrn_model.cpp)
target_compile_definitions(realtime_gtcrn_alsa PRIVATE GTCRN_WEIGHTS_HEADER="${GTCRN_WEIGHTS_HEADER}")
set_property(TARGET realtime_gtcrn_alsa PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
target_compile_options(realtime_gtcrn_alsa PRIVATE -O3)
target_link_libraries(realtime_gtcrn_alsa PRIVATE asound)
if(GTCRN_ARM_FAST_MATH AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
  target_compile_options(realtime_gtcrn_alsa PRIVATE -Ofast -mcpu=cortex-a55 -funroll-loops -fno-math-errno)
endif()

add_executable(offline_gtcrn offline_gtcrn.cpp gtcrn_model.cpp)
target_compile_definitions(offline_gtcrn PRIVATE GTCRN_WEIGHTS_HEADER="${GTCRN_WEIGHTS_HEADER}")
set_property(TARGET offline_gtcrn PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
target_compile_options(offline_gtcrn PRIVATE -O3)
if(GTCRN_ARM_FAST_MATH AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
  target_compile_options(offline_gtcrn PRIVATE -Ofast -mcpu=cortex-a55 -funroll-loops -fno-math-errno)
endif()

add_executable(gtcrn_esp_host_validate
  gtcrn_model.cpp
  ../esp32s3_gtcrn/common/gtcrn_embedded_model.cpp
  ../esp32s3_gtcrn/host_validate.cpp
)
target_compile_definitions(gtcrn_esp_host_validate PRIVATE GTCRN_WEIGHTS_HEADER="gtcrn_weights_dns3_finetuned.h")
target_include_directories(gtcrn_esp_host_validate PRIVATE .. ../esp32s3_gtcrn/common)
target_compile_options(gtcrn_esp_host_validate PRIVATE -O3)

add_executable(gtcrn_esp_offline
  ../esp32s3_gtcrn/common/gtcrn_embedded_model.cpp
  ../esp32s3_gtcrn/host_offline_embedded.cpp
)
target_compile_definitions(gtcrn_esp_offline PRIVATE GTCRN_WEIGHTS_HEADER="gtcrn_weights_dns3_finetuned.h")
target_include_directories(gtcrn_esp_offline PRIVATE .. ../esp32s3_gtcrn/common)
target_compile_options(gtcrn_esp_offline PRIVATE -O3)
