#pragma once

// ESP32-S3-EYE-compatible OV3660 parallel camera wiring.
#define CAM_PIN_PWDN  (-1)
#define CAM_PIN_RESET (-1)
#define CAM_PIN_XCLK  15
#define CAM_PIN_SIOD  4
#define CAM_PIN_SIOC  5

#define CAM_PIN_D0    11
#define CAM_PIN_D1    9
#define CAM_PIN_D2    8
#define CAM_PIN_D3    10
#define CAM_PIN_D4    12
#define CAM_PIN_D5    18
#define CAM_PIN_D6    17
#define CAM_PIN_D7    16

#define CAM_PIN_VSYNC 6
#define CAM_PIN_HREF  7
#define CAM_PIN_PCLK  13

// Common onboard addressable RGB LED pin on ESP32-S3 N16R8 camera boards.
// Change this in Kconfig if your board routes the LED elsewhere.
#define STATUS_LED_GPIO CONFIG_MCUNET_STATUS_LED_GPIO
