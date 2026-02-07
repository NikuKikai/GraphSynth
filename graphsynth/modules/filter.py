from functools import cache
import time
import numpy as np

from ..utils import zip_longest
from .basic import NoteSignal, Module


@cache
def _biquad_coeff_lowpass(freq: float, Q=0.707, sr=44100):
    w0 = 2 * np.pi * freq / sr
    cosw = np.cos(w0)
    alpha = np.sin(w0) / (2 * Q)

    b1 = 1 - cosw
    b2 = b0 = b1 / 2
    a0 = 1 + alpha
    a1 = -2 * cosw
    a2 = 1 - alpha

    return (
        b0 / a0, b1 / a0, b2 / a0,
        a1 / a0, a2 / a0
    )


@cache
def _biquad_coeff_highpass(freq: float, Q=0.707, sr=44100):
    w0 = 2 * np.pi * freq / sr
    cosw = np.cos(w0)
    alpha = np.sin(w0) / (2 * Q)

    b0 = (1 + cosw) / 2
    b1 = -(1 + cosw)
    b2 = b0
    a0 = 1 + alpha
    a1 = -2 * cosw
    a2 = 1 - alpha

    return (
        b0 / a0, b1 / a0, b2 / a0,
        a1 / a0, a2 / a0
    )


@cache
def _biquad_coeff_bandpass(freq: float, Q=0.707, sr=44100):
    w0 = 2 * np.pi * freq / sr
    cosw = np.cos(w0)
    alpha = np.sin(w0) / (2 * Q)

    b0 = alpha
    b1 = 0.0
    b2 = -alpha
    a0 = 1 + alpha
    a1 = -2 * cosw
    a2 = 1 - alpha

    return (
        b0 / a0, b1 / a0, b2 / a0,
        a1 / a0, a2 / a0
    )


@cache
def _biquad_coeff_notch(freq: float, Q=0.707, sr=44100):
    w0 = 2 * np.pi * freq / sr
    cosw = np.cos(w0)
    alpha = np.sin(w0) / (2 * Q)

    b0 = 1
    b1 = -2 * cosw
    b2 = 1
    a0 = 1 + alpha
    a1 = -2 * cosw
    a2 = 1 - alpha

    return (
        b0 / a0, b1 / a0, b2 / a0,
        a1 / a0, a2 / a0
    )


@cache
def _biquad_coeff_peaking(freq: float, Q=0.707, gain=0.0, sr=44100):
    A = 10 ** (gain / 40)
    w0 = 2 * np.pi * freq / sr
    alpha = np.sin(w0) / (2 * Q)
    cosw = np.cos(w0)

    b0 = 1 + alpha * A
    b1 = -2 * cosw
    b2 = 2 - b0  # 1 - alpha * A
    a0 = 1 + alpha / A
    a1 = b1
    a2 = 2 - a0  # 1 - alpha / A

    return (
        b0 / a0, b1 / a0, b2 / a0,
        a1 / a0, a2 / a0
    )


@cache
def _biquad_coeff_lowshelf(freq: float, Q=0.707, gain=0.0, sr=44100):
    A = 10 ** (gain / 40)
    w0 = 2 * np.pi * freq / sr
    alpha = np.sin(w0) / (2 * Q)
    cosw = np.cos(w0)
    sqrtA = np.sqrt(A)

    b0 = A * ((A + 1) - (A - 1) * cosw + 2 * sqrtA * alpha)
    b1 = 2 * A * ((A - 1) - (A + 1) * cosw)
    b2 = A * ((A + 1) - (A - 1) * cosw - 2 * sqrtA * alpha)
    a0 = (A + 1) + (A - 1) * cosw + 2 * sqrtA * alpha
    a1 = -2 * ((A - 1) + (A + 1) * cosw)
    a2 = (A + 1) + (A - 1) * cosw - 2 * sqrtA * alpha

    return (
        b0 / a0, b1 / a0, b2 / a0,
        a1 / a0, a2 / a0
    )


@cache
def _biquad_coeff_highshelf(freq: float, Q=0.707, gain=0.0, sr=44100):
    A = 10 ** (gain / 40)
    w0 = 2 * np.pi * freq / sr
    alpha = np.sin(w0) / (2 * Q)
    cosw = np.cos(w0)
    sqrtA = np.sqrt(A)

    b0 = A * ((A + 1) + (A - 1) * cosw + 2 * sqrtA * alpha)
    b1 = -2 * A * ((A - 1) + (A + 1) * cosw)
    b2 = A * ((A + 1) + (A - 1) * cosw - 2 * sqrtA * alpha)
    a0 = (A + 1) - (A - 1) * cosw + 2 * sqrtA * alpha
    a1 = 2 * ((A - 1) - (A + 1) * cosw)
    a2 = (A + 1) - (A - 1) * cosw - 2 * sqrtA * alpha

    return (
        b0 / a0, b1 / a0, b2 / a0,
        a1 / a0, a2 / a0
    )


def _biquad(x, z1, z2, b0, b1, b2, a1, a2):
    y = b0 * x + z1
    z1 = b1 * x - a1 * y + z2
    z2 = b2 * x - a2 * y
    return y, z1, z2


class _BiquadBase(Module):
    def __init__(self, freq=1000.0, Q=0.707, feedback=0.0, sr=44100):
        super().__init__()
        self._primitive = True
        self.sr = sr
        self.inp = self._create_inport('inp', default=0.0, is_audio=True)
        self.freq = self._create_inport('freq', default=freq)
        self.Q = self._create_inport('Q', default=Q)
        self.feedback = self._create_inport('feedback', default=feedback)
        self.out = self._create_outport('out', is_audio=True)

    def _biquad_coeff(self, freq: np.ndarray = 1000.0, Q=0.707, gain=0.0):
        raise NotImplementedError

    def _proc(self, signal: NoteSignal, inp: np.ndarray = 0.0, freq: np.ndarray = 1000.0, Q=0.707, feedback=0.0, gain=0.0) -> dict:
        # Load state
        state_key = (id(self), id(signal))
        if state_key not in signal._states:
            # z = inp if np.isscalar(inp) else inp[0]
            # signal._states[state_key] = (z, z)  # z1, z2
            signal._states[state_key] = (0, 0, 0)  # y, z1, z2
        y, z1, z2 = signal._states[state_key]

        # Process
        out = np.zeros_like(inp)
        aa = []
        for i, x_args in enumerate(zip_longest(inp, feedback, freq, Q, gain)):
            x, fb, *_args = x_args
            x = x - fb * y

            aa.append(_args[0])
            coeff = self._biquad_coeff(*_args)
            y, z1, z2 = _biquad(x, z1, z2, *coeff)
            out[i] = np.tanh(y)

        # Save state
        signal._states[state_key] = (y, z1, z2)
        return {'out': out}


class LowPass(_BiquadBase):
    def _biquad_coeff(self, freq: np.ndarray = 1000.0, Q=0.707, feedback=0.0, gain=0.0):
        return _biquad_coeff_lowpass(freq, Q=Q, sr=self.sr)


class HighPass(_BiquadBase):
    def _biquad_coeff(self, freq: np.ndarray = 1000.0, Q=0.707, feedback=0.0, gain=0.0):
        return _biquad_coeff_highpass(freq, Q=Q, sr=self.sr)


class BandPass(_BiquadBase):
    def _biquad_coeff(self, freq: np.ndarray = 1000.0, Q=0.707, feedback=0.0, gain=0.0):
        return _biquad_coeff_bandpass(freq, Q=Q, sr=self.sr)


class Notch(_BiquadBase):
    def _biquad_coeff(self, freq: np.ndarray = 1000.0, Q=0.707, feedback=0.0, gain=0.0):
        return _biquad_coeff_notch(freq, Q=Q, sr=self.sr)


class PeakingEQ(_BiquadBase):
    def __init__(self, freq=1000, Q=0.707, feedback=0.0, gain=0.0, sr=44100):
        super().__init__(freq, Q, feedback, sr)
        self.gain = self._create_inport('gain', default=gain)

    def _biquad_coeff(self, freq: np.ndarray = 1000.0, Q=0.707, feedback=0.0, gain=0.0):
        return _biquad_coeff_peaking(freq, Q=Q, gain=gain, sr=self.sr)


class LowShelf(_BiquadBase):
    def __init__(self, freq=1000, Q=0.707, feedback=0.0, gain=0.0, sr=44100):
        super().__init__(freq, Q, feedback, sr)
        self.gain = self._create_inport('gain', default=gain)

    def _biquad_coeff(self, freq: np.ndarray = 1000.0, Q=0.707, feedback=0.0, gain=0.0):
        return _biquad_coeff_lowshelf(freq, Q=Q, gain=gain, sr=self.sr)


class HighShelf(_BiquadBase):
    def __init__(self, freq=1000, Q=0.707, feedback=0.0, gain=0.0, sr=44100):
        super().__init__(freq, Q, feedback, sr)
        self.gain = self._create_inport('gain', default=gain)

    def _biquad_coeff(self, freq: np.ndarray = 1000.0, Q=0.707, feedback=0.0, gain=0.0):
        return _biquad_coeff_highshelf(freq, Q=Q, gain=gain, sr=self.sr)
