idf_component_register(
  SRCS
    "main.cpp"
    "../common/gtcrn_embedded_model.cpp"
  INCLUDE_DIRS
    "."
    "../common"
    "../.."
  REQUIRES
    esp_driver_i2s
    esp_timer
    heap
    esp_driver_gpio
)

target_compile_options(${COMPONENT_LIB} PRIVATE
  -Ofast
  -ffast-math
  -funroll-loops
  -fomit-frame-pointer
  -fno-exceptions
  -fno-rtti
  -DGTCRN_APPROX_ACTIVATION=1
  -DGTCRN_ENABLE_PROFILE=0
  -DGTCRN_DEBUG_DUALCORE=0
)
