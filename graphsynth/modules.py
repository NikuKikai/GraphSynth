import enum
import numpy as np
from .core import NoteSignal, Module


class Container(Module):
    def __init__(self):
        super().__init__()
        self._primitive = False
        self.out = self._create_outport('out')


@Module.wrap_func(inports={'inp': 0.0}, outports=['out'])
def PassThru(signal: NoteSignal, inp=0.0):
    return {'out': inp}


# @Module.wrap_func(inports={'inp': 0.0, 'gain': 1.0}, outports=['out'])
# def Gain(signal: NoteSignal, inp=0.0, gain=1.0):
#     return {'out': inp * gain}

class Gain(Module):
    def __init__(self, gain=0.2):
        super().__init__()
        self._primitive = True
        self.inp = self._create_inport('inp', default=0.0)
        self.gain = self._create_inport('gain', default=gain)
        self.out = self._create_outport('out')

    def _proc(self, signal: NoteSignal, inp=0.0, gain=1.0) -> dict:
        return {'out': inp * gain}


class OSC(Module):
    class WAVETYPE(enum.Enum):
        SINE = "sine"
        SQUARE = "square"
        SAWTOOTH = "sawtooth"
        TRIANGLE = "triangle"

    def __init__(self, wavetype=WAVETYPE.SINE):
        super().__init__()
        self._primitive = True
        self.wavetype = wavetype
        self.out = self._create_outport('out')

    def _sine(self, signal: NoteSignal):
        ts = signal.local_ts
        return np.sin(2 * np.pi * signal.freq * ts)

    def _square(self, signal: NoteSignal):
        ts = signal.local_ts
        return np.sign(np.sin(2 * np.pi * signal.freq * ts))

    def _sawtooth(self, signal: NoteSignal):
        ts = signal.local_ts
        return (2 * (ts * signal.freq - np.floor(0.5 + ts * signal.freq)))

    def _triangle(self, signal: NoteSignal):
        ts = signal.local_ts
        return (2 * np.abs(2 * (ts * signal.freq - np.floor(ts * signal.freq + 0.5))) - 1)

    def _noise(self, signal: NoteSignal):
        return np.random.uniform(-1.0, 1.0, size=signal.current_ts.shape)

    def _proc(self, signal: NoteSignal, **kwargs) -> dict:
        func = {
            self.WAVETYPE.SINE: self._sine,
            self.WAVETYPE.SQUARE: self._square,
            self.WAVETYPE.SAWTOOTH: self._sawtooth,
            self.WAVETYPE.TRIANGLE: self._triangle,
        }[self.wavetype]
        return {self.out.name: func(signal)}


class Envelope(Module):
    def __init__(self, attack=0.01, decay=0.1, sustain=0.7, release=0.2):
        super().__init__()
        self._primitive = True
        self.inp = self._create_inport('inp', default=0.0)
        self.attack = self._create_inport('attack', default=attack)
        self.decay = self._create_inport('decay', default=decay)
        self.sustain = self._create_inport('sustain', default=sustain)
        self.release = self._create_inport('release', default=release)
        self.out = self._create_outport('out')

    def _proc(
        self,
        signal: NoteSignal,
        inp: np.ndarray = 0.0,
        attack: float = 0.01,
        decay: float = 0.1,
        sustain: float = 0.7,
        release: float = 0.2
    ) -> dict:
        ts = np.clip(signal.local_ts, 0, None)
        rls = signal.local_released_t

        k = np.where(
            (ts > rls) & (rls > 0),
            (1 - np.clip(ts - rls, 0, release) / release) * sustain,
            np.where(
                ts < attack,
                ts / attack,
                np.where(
                    ts < (attack + decay),
                    1 - (1 - sustain) * ((ts - attack) / decay),
                    sustain
                )
            )
        )
        return {'out': inp * k}
