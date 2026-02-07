import enum
import numpy as np
from ..core import NoteSignal, Module


class Container(Module):
    def __init__(self):
        super().__init__()
        self._primitive = False
        self.out = self._create_outport('out', is_audio=True)


@Module.wrap_func(inports={'inp': 0.0}, outports=['out'])
def PassThru(signal: NoteSignal, inp=0.0):
    return {'out': inp}


# @Module.wrap_func(inports={'inp': 0.0, 'gain': 1.0}, outports=['out'])
# def Gain(signal: NoteSignal, inp=0.0, gain=1.0):
#     return {'out': inp * gain}

class Gain(Module):
    def __init__(self, gain=0.0):
        super().__init__()
        self._primitive = True
        self.inp = self._create_inport('inp', default=0.0, is_audio=True)
        self.gain = self._create_inport('gain', default=gain)
        self.out = self._create_outport('out', is_audio=True)

    def _proc(self, signal: NoteSignal, inp=0.0, gain=0.0) -> dict:
        a = np.power(10.0, gain / 20.0)
        return {'out': inp * a}


class OSC(Module):
    class WAVETYPE(enum.Enum):
        SINE = "sine"
        SQUARE = "square"
        SAWTOOTH = "sawtooth"
        TRIANGLE = "triangle"
        WHITENOISE = "whitenoise"

    def __init__(self, wavetype=WAVETYPE.SINE):
        super().__init__()
        self._primitive = True
        self.wavetype = wavetype
        self.out = self._create_outport('out', is_audio=True)

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

    def _whitenoise(self, signal: NoteSignal):
        return np.random.uniform(-1.0, 1.0, size=signal.current_ts.shape)

    def _proc(self, signal: NoteSignal, **kwargs) -> dict:
        func = {
            self.WAVETYPE.SINE: self._sine,
            self.WAVETYPE.SQUARE: self._square,
            self.WAVETYPE.SAWTOOTH: self._sawtooth,
            self.WAVETYPE.TRIANGLE: self._triangle,
            self.WAVETYPE.WHITENOISE: self._whitenoise,
        }[self.wavetype]
        return {self.out.name: func(signal)}


class ADSHR(Module):
    """
    ADSR Envelope Module. Since the default value of inp is 1.0, it can also be used as an envelope generator for arguments of other module.
        - InPorts:
            - inp: Input audio signal (default: 1.0)
            - attack: Attack time in seconds (0.0 to inf)
            - decay: Decay time in seconds (0.0 to inf)
            - sustain: Sustain level (0.0 to 1.0)
            - release: Release time in seconds. Negative means no release.
            - scale: multiplier for the output
        - OutPorts:
            - out: Output audio signal
    """

    def __init__(self, attack=0.01, decay=0.1, sustain=0.7, hold=0.0, release=0.2, scale=1.0):
        super().__init__()
        self._primitive = True
        self.inp = self._create_inport('inp', default=1.0, is_audio=True)
        self.attack = self._create_inport('attack', default=attack)
        self.decay = self._create_inport('decay', default=decay)
        self.sustain = self._create_inport('sustain', default=sustain)
        self.hold = self._create_inport('hold', default=hold)
        self.release = self._create_inport('release', default=release)
        self.scale = self._create_inport('scale', default=scale)
        self.out = self._create_outport('out', is_audio=True)

    def _proc(
        self,
        signal: NoteSignal,
        inp: np.ndarray = 0.0,
        attack: np.ndarray = 0.01,
        decay: np.ndarray = 0.1,
        sustain: np.ndarray = 0.7,
        hold: np.ndarray = 0.0,
        release: np.ndarray = 0.2,
        scale: np.ndarray = 1.0
    ) -> dict:
        ts = np.clip(signal.local_ts, 0, None)
        rls = signal.local_released_t

        use_release = release >= 0.0

        attack = np.clip(attack, 0, None)
        decay = np.clip(decay, 0, None)
        sustain = np.clip(sustain, 0, 1.0)
        hold = np.clip(hold, 0, None)
        release = np.clip(release, 0, None)

        # k before release
        k = np.where(
            ts < attack,
            ts / attack if attack > 0 else 1.0,
            np.where(
                ts < (attack + decay),
                1 - (1 - sustain) * ((ts - attack) / decay) if decay > 0 else sustain,
                sustain
            )
        )

        # k after release
        if use_release:
            k = np.where(
                (ts > rls) & (rls > 0),
                np.where(
                    ts > rls + hold,
                    (1 - np.clip(ts - rls, 0, release) / release) * sustain if release > 0 else 0.0,
                    sustain
                ),
                k
            )

        # print("ADSR k max:", np.max(k) * scale)
        return {'out': inp * k * scale}

    @classmethod
    def DS(cls, decay=0.1, sustain=0.7, scale=1.0):
        return cls(attack=0.0, decay=decay, sustain=sustain, release=-1, scale=scale)
