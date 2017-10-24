import spidev
import pyaudio
import numpy as np


class PyAudioWrapper(object):

    def __init__(self):
        self._p = None

    def __enter__(self):
        self._p = pyaudio.PyAudio()
        return self

    def __exit__(self, *args):
        self._p.terminate()

    @property
    def devices(self):
        return [
            self._p._get_device_info_by_index(index).get('name')
            for index in range(self._p.get_device_count())
        ]

    def open(self, *args, **kwargs):
        return StreamWrapper(self._p, *args, **kwargs)


class StreamWrapper(object):

    def __init__(self, p, *args, **kwargs):
        self._p = p
        self._stream = None
        self._args = args
        self._kwargs = kwargs

    def __enter__(self):
        self._stream = self._p.open(*self._args, **self._kwargs)
        return self

    def __exit__(self, *args):
        self._stream.close()

    def read(self, *args, **kwargs):
        defaults = dict(exception_on_overflow=False)
        defaults.update(kwargs)
        kwargs = defaults
        return self._stream.read(*args, **kwargs)


def generate_samples(*args, **kwargs):
    chunk = kwargs['frames_per_buffer']
    with PyAudioWrapper() as p, p.open(*args, **kwargs) as stream:
        while True:
            try:
                bytes_ = stream.read(chunk)
            except IOError:
                pass
            else:
                samples = np.fromstring(bytes_, np.int16).astype(float)
                yield samples


class Serial(object):

    def __init__(self, device, speed):
        self._device = device
        self._spi = spidev.SpiDev()
        self._spi.max_speed_hz = int(speed)

    def __enter__(self):
        self._spi.open(*self._device)
        return self

    def __exit__(self):
        self._spi.close()

    @staticmethod
    def parse_device(string):
        string = string.replace('/dev/spidev-', '')
        bus, device = string.split('.')
        bus, device = int(bus), int(device)
        return (bus, device)

    def write(self, *args, **kwargs):
        return self._spi.xfer(*args, **kwargs)
