from __future__ import division, print_function

import time
import argparse
import numpy as np

import pyaudio

from lumen import devices
from lumen import processing


def map_audio_args(args):
    kwargs = dict(
        input_device_index=args.audio_device,
        input=True,
        format=pyaudio.paInt16,
        frames_per_buffer=args.audio_chunk,
    )
    channels = 1
    args = (args.sampling_rate, channels)
    return args, kwargs


def map_serial_args(args):
    device = devices.Serial.parse_device(args.serial_device)
    speed = args.serial_speed
    args = (device, speed)
    kwargs = {}
    return args, kwargs


def build_processing_steps(args):
    steps = []

    chunk = args.audio_chunk
    fft = processing.FourierTransformer(chunk)
    steps.append(fft)

    half_rate = args.sampling_rate // 2
    half_chunk = chunk // 2
    input_freqs = np.linspace(0, half_rate, half_chunk)

    rho = np.linspace(0, 1, args.num_leds)
    output_freqs = np.exp2(8*rho + 4)
    map_ = processing.FrequencyMapper(input_freqs, output_freqs)
    steps.append(map_)

    agc = processing.GainController(window=15, amplitude=5, rho=0.9)
    steps.append(agc)

    clip = processing.Clipper(max_value=100)
    steps.append(clip)

    threshold = processing.Thresholder(min_value=100)
    steps.append(threshold)

    colorize = processing.Colorizer()
    steps.append(colorize)

    return steps


def main():
    d = "Plot samples from input audio device."
    parser = argparse.ArgumentParser(description=d)

    # Audio arguments
    parser.add_argument('--audio_device', default=0)
    parser.add_argument('--audio_chunk', type=int, default=1024)
    parser.add_argument('--sampling_rate', type=int, default=44100)

    # Serial arguments
    parser.add_argument('--serial_device', default='/dev/spidev-0.1')
    parser.add_argument('--serial_speed', type=int, default=int(500e3))

    # Processing arguments
    parser.add_argument('--num_leds', type=int, default=300)

    clargs = parser.parse_args()

    args, kwargs = map_serial_args(clargs)
    with devices.Serial(*args, **kwargs) as serial:
        steps = build_processing_steps(clargs)

        args, kwargs = map_audio_args(clargs)
        samples = devices.generate_samples(*args, **kwargs)

        for chunk in samples:
            processed = processing.apply(chunk, steps)
            serial.write(processed)
            time.sleep(15e-3)


if __name__ == '__main__':
    main()
