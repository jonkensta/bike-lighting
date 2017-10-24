from __future__ import print_function

import pyaudio


def main():
    p = pyaudio.PyAudio()
    info = p.get_host_api_info_by_index(0)
    numdevices = info.get('deviceCount')
    for index in range(numdevices):
        device = p.get_device_info_by_host_api_device_index(0, index)
        max_channels = device.get('maxInputChannels')
        if max_channels > 0:
            name = device.get('name')
            print("Input Device id {} - {}".format(index, name))


if __name__ == '__main__':
    main()
