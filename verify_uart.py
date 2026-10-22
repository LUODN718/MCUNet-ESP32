"""Hardware protocol checks; close the UI connection before running."""
import argparse
from functools import reduce
from operator import xor
import struct
import time

import serial


def packet(data):
    assert len(data) == 19200
    return b'\xaa\xbb' + data + struct.pack('<H', reduce(xor, data, 0))


def result(port):
    response = port.read(6)
    assert len(response) == 6 and response[4:] == b'\xcc\xdd', response.hex()
    no_person, person = response[0] - 128, response[1] - 128
    ms = int.from_bytes(response[2:4], 'little')
    print(f'{"PERSON" if person > no_person else "NO PERSON"}: '
          f'no_person={no_person}, person={person}, inference={ms} ms', flush=True)
    return response[:2]


def silence(port, seconds):
    previous = port.timeout
    port.timeout = seconds
    assert port.read(1) == b'', 'Unexpected response while idle or after bad packet'
    port.timeout = previous


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', default='COM11')
    parser.add_argument('--image')
    args = parser.parse_args()
    with serial.Serial(args.port, 921600, timeout=3, write_timeout=3) as port:
        # Match UI opening behavior and let any reset/boot output finish.
        time.sleep(2)
        port.reset_input_buffer()
        gray = packet(bytes(19200))
        port.write(gray)
        expected = result(port)
        silence(port, 1.2)
        print('PASS: idle produces no inference response')
        port.write(gray[:-1] + b'\x01')
        silence(port, 1.2)
        print('PASS: invalid checksum rejected')
        port.write(gray[:700])
        silence(port, 1.3)
        port.write(b'noise\xaa' + gray)
        assert result(port) == expected
        print('PASS: partial timeout, noise and repeated AA recovery')
        port.write(gray + gray)
        assert result(port) == expected
        assert result(port) == expected
        print('PASS: queued packets each receive one result')
        if args.image:
            from PIL import Image, ImageOps
            with Image.open(args.image) as image:
                rgb = ImageOps.exif_transpose(image).convert('RGB').resize(
                    (80, 80), Image.Resampling.BILINEAR)
                data = bytes((b - 128) & 255 for b in rgb.tobytes())
            port.write(packet(data))
            result(port)
        silence(port, 1.2)
        print('PASS: serial checks complete; LED visual check still required')


if __name__ == '__main__':
    main()
