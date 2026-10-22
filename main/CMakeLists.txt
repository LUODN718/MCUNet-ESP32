idf_component_register(
    SRCS "main.cc"
    INCLUDE_DIRS "."
    EMBED_FILES "model/mcunet-vww1.tflite"
    REQUIRES esp_timer esp_psram esp_driver_uart led_strip
)
