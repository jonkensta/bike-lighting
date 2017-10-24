import numpy as np
from scipy.ndimage import uniform_filter1d


class FourierTransformer(object):

    def __init__(self, size):
        self._window = np.hamming(int(size))
        self._half_size = int(size) // 2

    def __call__(self, samples):
        samples = samples * self._window
        return np.abs(np.fft.fft(samples)[:self._half_size])


class FrequencyMapper(object):

    def __init__(self, input_freqs, output_freqs):
        self._in_freqs = np.array(input_freqs)
        self._out_freqs = np.array(output_freqs)

    def __call__(self, samples):
        return np.interp(self._out_freqs, self._in_freqs, samples)


class GainController(object):

    def __init__(self, window=21, amplitude=100, rho=0.999):
        self._rho = float(rho)
        self._window = int(window)
        self._amplitude = float(amplitude)
        self._divisor = 100000

    def __call__(self, samples):
        divisor = uniform_filter1d(samples, self._window)
        self._divisor = self._rho*self._divisor + (1-self._rho)*divisor
        return self._amplitude * samples / self._divisor


class Clipper(object):

    def __init__(self, max_value=255.0):
        self._max_value = float(max_value)

    def __call__(self, samples):
        return np.clip(samples, 0, self._max_value)


class Thresholder(object):

    def __init__(self, min_value=100.0):
        self._min_value = float(min_value)

    def __call__(self, samples):
        median = np.median(samples[samples > 0])
        return np.choose(samples >= median, (0, samples))


class Colorizer(object):

    def __init__(self):
        pass

    def __call__(self, samples):
        samples = np.repeat(samples, 3)
        return [x&0xFF for x in list(samples.astype(int))]


def apply(samples, steps):
    for step in steps:
        samples = step(samples)
    return samples
