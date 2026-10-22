# MCUNet VWW on ESP32-S3 N16R8

PC UI image inference using the official mcunet-vww1 TFLite model,
TensorFlow Lite Micro 1.4.1 and ESP-NN 1.4.1. No camera or display is used.
Compatible UI: `C:/Users/33581/Desktop/Mcunet/tinyengine/tutorial/TinyEngine_vww/STM32_GUI_latest/main.py`
(LUODN718/STM32_GUI).

## Use

Select COM11 and 921600 baud in MCUNet Studio, connect, then send an image.
The UI resizes RGB to 80 x 80 and subtracts 128 per channel. ESP32 runs one
inference per valid packet and returns the two scores plus inference time.
After power-on allow about two seconds for initialization before sending.
The GUI holds DTR/RTS inactive on serial open to avoid USB-UART auto-reset;
firmware configures UART0 and silences logs before model setup.

The WS2812 LED on GPIO48 stays RED for PERSON and BLUE for NO PERSON.
It starts off and holds the last result indefinitely until another valid image
is inferred. Partial packets, invalid checksums and idle time never change it.
Reset/power loss clears the state. Physical LED wiring/colors need visual
confirmation. Set `MCUNET_STATUS_LED_GPIO` in menuconfig for another GPIO.

## Wire protocol

UART0 via the CH340 bridge, TX GPIO43 / RX GPIO44, 921600 baud, 8N1.

| Direction | Format |
| --- | --- |
| PC to ESP | AA BB + 19200 int8 RGB bytes + uint16 little-endian XOR checksum |
| ESP to PC | no_person+128, person+128, time_ms low, time_ms high, CC DD |

Checksum XORs the transmitted image bytes only; its high byte is zero.
Class 1 greater than class 0 means PERSON, otherwise NO PERSON.
Scores are quantized logits, not probabilities. The response byte order follows
the actual UI parser. Wait for a response before sending the next image.
Incomplete packets expire after one second without input. Bad packets receive
no response; the UI reports a timeout. Runtime log output is suppressed once
the binary UART protocol starts; boot logs still use 115200 baud.

## Model and memory

- File: `main/model/mcunet-vww1.tflite`, 635968 bytes
- SHA256: D878916E93F59E63949CE9B8E1A585382D466D3B94D751B9F1628A7753E03C24
- Source: https://hanlab18.mit.edu/projects/tinyml/mcunet/release/mcunet-5fps_vww.tflite
- Input: int8 [1,80,80,3]; output: int8 [1,2]
- Tensor arena: 512 KB in octal PSRAM; CPU: 240 MHz; Flash: 16 MB
- Registered ops: Conv2D, DepthwiseConv2D, Add, Pad, AveragePool2D, Reshape

## Build and flash

In an ESP-IDF 5.4.4 PowerShell:

```powershell
cd C:\Users\33581\Desktop\fyp\esp32s3_mcunet_vww
idf.py build
idf.py -p COM11 flash
```

Close any UI serial connection before flashing. COM7 belongs to the STM32,
not the ESP32. The old camera pin map remains in `main/camera_pins.h` for
reference but is not included or initialized by this firmware.

## Serial verification

With pyserial installed, run `python verify_uart.py --port COM11` with the UI
disconnected. This checks valid frames, idle silence, rejected checksums,
partial-frame timeout recovery and back-to-back queued requests. It cannot
measure physical LED color. Add `--image path.jpg` with Pillow installed to
send an image using exactly the UI's preprocessing.

Hardware check on COM11 passed: valid gray input returned NO PERSON
(scores 9, -11); person_crop.png returned PERSON (scores -22, 19).
Inference took approximately 258-264 ms. Idle silence, invalid checksum
rejection, partial-frame timeout recovery and queued requests all passed.
The UI's original packet/preprocessing and response parser were also checked
against the tested sender and captured response values. Physical LED color
and hold behavior still require visual confirmation.
