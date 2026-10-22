#include <cstdint>
#include <cstdio>
#include <cstring>
#include "driver/uart.h"
#include "esp_check.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "led_strip.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace {
constexpr int kImageBytes = 80 * 80 * 3;
constexpr uart_port_t kUart = UART_NUM_0;
constexpr size_t kArenaBytes = CONFIG_MCUNET_TENSOR_ARENA_SIZE_KB * 1024;
extern const uint8_t kModelStart[] asm("_binary_mcunet_vww1_tflite_start");
led_strip_handle_t status_led = nullptr;

[[noreturn]] void StopHere() {
  while (true) vTaskDelay(pdMS_TO_TICKS(1000));
}

void InitLed() {
  led_strip_config_t config = {};
  config.strip_gpio_num = CONFIG_MCUNET_STATUS_LED_GPIO;
  config.max_leds = 1;
  config.led_pixel_format = LED_PIXEL_FORMAT_GRB;
  config.led_model = LED_MODEL_WS2812;
  led_strip_rmt_config_t rmt = {};
  rmt.clk_src = RMT_CLK_SRC_DEFAULT;
  rmt.resolution_hz = 10000000;
  ESP_ERROR_CHECK(led_strip_new_rmt_device(&config, &rmt, &status_led));
  ESP_ERROR_CHECK(led_strip_clear(status_led));
}

void SetDetectionLed(bool person) {
  ESP_ERROR_CHECK(led_strip_set_pixel(status_led, 0, person ? 32 : 0,
                                     0, person ? 0 : 32));
  ESP_ERROR_CHECK(led_strip_refresh(status_led));
}

void InitSerial() {
  uart_config_t config = {};
  config.baud_rate = 921600;
  config.data_bits = UART_DATA_8_BITS;
  config.parity = UART_PARITY_DISABLE;
  config.stop_bits = UART_STOP_BITS_1;
  config.flow_ctrl = UART_HW_FLOWCTRL_DISABLE;
  config.source_clk = UART_SCLK_DEFAULT;
  ESP_ERROR_CHECK(uart_driver_install(kUart, 32768, 0, 0, nullptr, 0));
  ESP_ERROR_CHECK(uart_param_config(kUart, &config));
  ESP_ERROR_CHECK(uart_set_pin(kUart, 43, 44, UART_PIN_NO_CHANGE,
                              UART_PIN_NO_CHANGE));
}

enum class ReceiveStatus { kPending, kPacket, kTimeout, kChecksumError };

struct ReceiveDiagnostics {
  uint32_t received_bytes = 0;
  uint16_t received_checksum = 0;
  uint16_t calculated_checksum = 0;
  bool receive_timeout = false;
  bool checksum_error = false;
  bool inference_started = false;
  bool response_sent = false;
};

// STM32_GUI packet: AA BB + image + little-endian XOR checksum.
ReceiveStatus ReceiveImage(int8_t* input, ReceiveDiagnostics* diagnostics) {
  static uint8_t image[kImageBytes];
  static int state = 0;
  static int index = 0;
  static uint16_t checksum = 0;
  static uint16_t received_checksum = 0;
  static int64_t last_byte_us = 0;
  // Read one byte so bytes from a subsequent packet remain in the UART queue.
  uint8_t byte;
  const int count = uart_read_bytes(kUart, &byte, 1, pdMS_TO_TICKS(20));
  const int64_t now = esp_timer_get_time();
  const bool timed_out = state != 0 && now - last_byte_us > 1000000;
  if (timed_out) {
    diagnostics->received_bytes = index;
    diagnostics->received_checksum = received_checksum;
    diagnostics->calculated_checksum = checksum;
    diagnostics->receive_timeout = true;
    state = 0;
    index = 0;
    checksum = 0;
    received_checksum = 0;
  }
  if (count != 1) {
    return timed_out ? ReceiveStatus::kTimeout : ReceiveStatus::kPending;
  }
  last_byte_us = now;
  if (timed_out) {
    if (byte == 0xaa) state = 1;
    return ReceiveStatus::kTimeout;
  }
  switch (state) {
    case 0:
      if (byte == 0xaa) {
        state = 1;
        index = 0;
        checksum = 0;
        received_checksum = 0;
        *diagnostics = ReceiveDiagnostics{};
      }
      break;
    case 1:
      if (byte == 0xbb) {
        state = 2;
        index = 0;
        checksum = 0;
        received_checksum = 0;
      } else if (byte != 0xaa) {
        state = 0;
      }
      break;
    case 2:
      image[index++] = byte;
      checksum ^= byte;
      diagnostics->received_bytes = index;
      diagnostics->calculated_checksum = checksum;
      if (index == kImageBytes) state = 3;
      break;
    case 3:
      received_checksum = byte;
      diagnostics->received_checksum = received_checksum;
      state = 4;
      break;
    case 4:
      received_checksum |= static_cast<uint16_t>(byte) << 8;
      state = 0;
      diagnostics->received_bytes = index;
      diagnostics->received_checksum = received_checksum;
      diagnostics->calculated_checksum = checksum;
      if (received_checksum == checksum) {
        memcpy(input, image, sizeof(image));
        return ReceiveStatus::kPacket;
      }
      diagnostics->checksum_error = true;
      return ReceiveStatus::kChecksumError;
  }
  return ReceiveStatus::kPending;
}

void SendDiagnostic(const char* kind, const ReceiveDiagnostics& diagnostics) {
  char line[240];
  const int length = snprintf(
      line, sizeof(line),
      "%s received_bytes=%lu received_checksum=0x%04X "
      "calculated_checksum=0x%04X receive_timeout=%u checksum_error=%u "
      "inference_started=%u response_sent=%u\n",
      kind, static_cast<unsigned long>(diagnostics.received_bytes),
      diagnostics.received_checksum, diagnostics.calculated_checksum,
      diagnostics.receive_timeout, diagnostics.checksum_error,
      diagnostics.inference_started, diagnostics.response_sent);
  if (length > 0 && length < sizeof(line)) {
    uart_write_bytes(kUart, line, length);
  }
}

bool SendResult(const TfLiteTensor* output, int64_t elapsed_us) {
  const uint16_t ms = static_cast<uint16_t>(
      elapsed_us / 1000 > 65535 ? 65535 : elapsed_us / 1000);
  // UI expects class 0 (no person) first, then class 1 (person).
  const uint8_t response[] = {
      static_cast<uint8_t>(output->data.int8[0] + 128),
      static_cast<uint8_t>(output->data.int8[1] + 128),
      static_cast<uint8_t>(ms), static_cast<uint8_t>(ms >> 8), 0xcc, 0xdd};
  return uart_write_bytes(kUart, response, sizeof(response)) == sizeof(response) &&
         uart_wait_tx_done(kUart, pdMS_TO_TICKS(1000)) == ESP_OK;
}
}  // namespace

extern "C" void app_main() {
  // UART0 is also the USB-UART bridge. Silence console output and switch it
  // from the boot console rate before any slower peripheral/model setup.
  esp_log_level_set("*", ESP_LOG_NONE);
  InitSerial();
  InitLed();
  const tflite::Model* model = tflite::GetModel(kModelStart);
  if (model->version() != TFLITE_SCHEMA_VERSION) StopHere();
#if CONFIG_MCUNET_ARENA_IN_PSRAM
  constexpr uint32_t caps = MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT;
#else
  constexpr uint32_t caps = MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT;
#endif
  auto* arena = static_cast<uint8_t*>(heap_caps_aligned_alloc(16, kArenaBytes, caps));
  if (arena == nullptr) StopHere();
  static tflite::MicroMutableOpResolver<6> resolver;
  if (resolver.AddConv2D() != kTfLiteOk ||
      resolver.AddDepthwiseConv2D() != kTfLiteOk ||
      resolver.AddAdd() != kTfLiteOk || resolver.AddPad() != kTfLiteOk ||
      resolver.AddAveragePool2D() != kTfLiteOk ||
      resolver.AddReshape() != kTfLiteOk) StopHere();
  static tflite::MicroInterpreter interpreter(model, resolver, arena, kArenaBytes);
  if (interpreter.AllocateTensors() != kTfLiteOk) StopHere();
  TfLiteTensor* input = interpreter.input(0);
  TfLiteTensor* output = interpreter.output(0);
  if (input->type != kTfLiteInt8 || input->dims->size != 4 ||
      input->dims->data[0] != 1 || input->dims->data[1] != 80 ||
      input->dims->data[2] != 80 || input->dims->data[3] != 3 ||
      input->bytes != kImageBytes || output->type != kTfLiteInt8 ||
      output->bytes != 2) StopHere();
  ReceiveDiagnostics diagnostics;
  while (true) {
    const ReceiveStatus status = ReceiveImage(input->data.int8, &diagnostics);
    if (status == ReceiveStatus::kPending) continue;
    if (status == ReceiveStatus::kTimeout ||
        status == ReceiveStatus::kChecksumError) {
      SendDiagnostic("NACK", diagnostics);
      continue;
    }
    const int64_t start = esp_timer_get_time();
    diagnostics.inference_started = true;
    if (interpreter.Invoke() != kTfLiteOk) {
      SendDiagnostic("NACK", diagnostics);
      continue;
    }
    const int64_t elapsed = esp_timer_get_time() - start;
    SetDetectionLed(output->data.int8[1] > output->data.int8[0]);
    diagnostics.response_sent = SendResult(output, elapsed);
    SendDiagnostic(diagnostics.response_sent ? "ACK" : "NACK", diagnostics);
  }
}
