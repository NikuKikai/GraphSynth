import enum
from typing import Union
import numpy as np
import time
import pygame
import sounddevice as sd


class NoteSignal:
    def __init__(self, freq=440.0):
        self.freq = freq
        self.inited_t = time.time()
        self.current_ts = np.array(0.0)
        self.released = False
        self.released_t = -1.0
        self._eval_cache = {}  # cache for one-time evaluation


class Port:
    def __init__(self, name: str, parent: 'Module'):
        self.name = name
        self.parent = parent

    def to(self, target: 'Port'):
        if target.source is not None:
            raise ValueError("Target port already has a source connected")
        target.source = self
        return target


class InPort(Port):
    def __init__(self, name: str, parent: 'Module', default=0.0):
        super().__init__(name, parent)
        self.default = default
        self.source: Port = None

    def _eval(self, signal: NoteSignal, root: 'Module', **kwargs):
        if self.parent == root:
            return kwargs.get(self.name, self.default)
        if self.source is None:
            return self.default
        if isinstance(self.source, OutPort):
            return self.source._eval(signal, root, **kwargs)
        return self.source._eval(signal, root, **kwargs)


class OutPort(Port):
    def __init__(self, name: str, parent: 'Module'):
        super().__init__(name, parent)
        self.source: Port = None

    def _eval(self, signal: NoteSignal, root: 'Module', **kwargs):
        if self.parent == root:
            raise ValueError("Cannot evaluate output port on root module")
        return self.parent._call(signal, root, **kwargs)[self.name]


class Module:
    def __init__(self, inports: dict[str, float] = {}, outports: list[str] = []):
        self._primitive = False
        self._inports: list[InPort] = []
        self._outports: list[OutPort] = []

        for ip_name, default in inports.items():
            self._create_inport(ip_name, default=default)

        for op_name in outports:
            self._create_outport(op_name)

    def __call__(self, signal: NoteSignal, **kwargs):
        signal._eval_cache.clear()
        return self._call(signal, self, **kwargs)

    def _call(self, signal: NoteSignal, root: 'Module', **kwargs):
        # cache
        cache_key = (id(self), id(root))
        if cache_key in signal._eval_cache:
            return signal._eval_cache[cache_key]

        if self._primitive:
            ipargs = dict()
            for ip in self._inports:
                ipargs[ip.name] = ip._eval(signal, root, **kwargs)
            result = self._proc(signal, **ipargs)
        else:
            res = dict()
            for op in self._outports:
                res[op.name] = op.source._eval(signal, root, **kwargs)
            result = res

        # cache
        signal._eval_cache[cache_key] = result
        return result

    def _proc(self, signal: NoteSignal, **kwargs):
        res = dict()
        for op in self._outports:
            res[op.name] = 0.0
        return res

    def _create_inport(self, name: str, default=0.0) -> InPort:
        ip = InPort(name, self, default)
        setattr(self, name, ip)
        self._inports.append(ip)
        return ip

    def _create_outport(self, name: str) -> OutPort:
        op = OutPort(name, self)
        setattr(self, name, op)
        self._outports.append(op)
        return op

    @classmethod
    def wrap_func(cls, inports: dict[str, float], outports: list[str]):
        def wrap(func):
            class FuncModule(Module):
                def __init__(self):
                    super().__init__(inports=inports, outports=outports)
                    self._primitive = True

                def _proc(self, signal: NoteSignal, **kwargs) -> dict:
                    return func(signal, **kwargs)
            return FuncModule
        return wrap


class PassThru(Module):
    def __init__(self):
        super().__init__()
        self._primitive = True
        self.inp = InPort("inp", self)
        self.out = OutPort("out", self)
        self._inports.append(self.inp)
        self._outports.append(self.out)

    def _proc(self, signal: NoteSignal, **kwargs) -> dict:
        return {self.out.name: kwargs.get(self.inp.name, 0.0)}


@Module.wrap_func(inports={'inp': 0.0, 'gain': 1.0}, outports=['out'])
def GainModule(signal: NoteSignal, inp=0.0, gain=1.0):
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
        self.out = OutPort("out", self)
        self._outports.append(self.out)

    def _sine(self, signal: NoteSignal):
        ts = signal.current_ts - signal.inited_t
        return np.sin(2 * np.pi * signal.freq * ts)

    def _square(self, signal: NoteSignal):
        ts = signal.current_ts - signal.inited_t
        return np.sign(np.sin(2 * np.pi * signal.freq * ts))

    def _sawtooth(self, signal: NoteSignal):
        ts = signal.current_ts - signal.inited_t
        return (2 * (ts * signal.freq - np.floor(0.5 + ts * signal.freq)))

    def _triangle(self, signal: NoteSignal):
        ts = signal.current_ts - signal.inited_t
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
        self.attack = attack
        self.decay = decay
        self.sustain = sustain
        self.release = release

    def proc(self):
        pass

        # Create a synth with filters


class Controller:
    def __init__(self, time_to_keep_after_release=0.5):
        self.t_keep = time_to_keep_after_release
        self.signals: dict[float, NoteSignal] = dict()

    def press(self, freq: Union[float, str], release_after: float = 0.5):
        if isinstance(freq, str):
            freq = note_str2freq(freq)
        s = NoteSignal(freq=freq)
        self.signals[freq] = s
        if release_after is not None:
            s.released_t = s.inited_t + release_after

    def release(self, freq: float):
        if freq in self.signals and self.signals[freq].released is False:
            self.signals[freq].released = True
            self.signals[freq].released_t = time.time()

    def update(self):
        now = time.time()
        to_delete = []
        for freq, signal in self.signals.items():
            if signal.released_t < 0.0:
                continue
            if signal.released_t < now:
                signal.released = True
            if now - signal.released_t > self.t_keep:
                to_delete.append(freq)
        for freq in to_delete:
            del self.signals[freq]

    def get_active_signals(self, blocksize: int, t: float = None) -> list[NoteSignal]:
        res = list(self.signals.values())
        t = t if t is not None else time.time()
        for sig in res:
            sig.current_ts = np.array([t]) + np.arange(blocksize) * (1.0 / 44100.0)
        return res


def note_str2freq(note: str) -> float:
    A4_FREQ = 440.0
    NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    octave = int(note[-1])
    key = note[:-1]
    n = NOTE_NAMES.index(key)
    semitones_from_A4 = n - 9 + (octave - 4) * 12
    return A4_FREQ * (2 ** (semitones_from_A4 / 12))


# ------ Define synth structure ------
mod = Module(inports={}, outports=['out'])
osc = OSC(OSC.WAVETYPE.SAWTOOTH)
gain = GainModule()
# env = Envelope(attack=0.05, decay=0.1, sustain=0.6, release=0.3).to(outnode)
# env.to(osc.arg1)
# osc.to(new Envelope()).to(outnode)

osc.out.to(gain.inp)
gain.gain.default = 0.05
gain.out.to(mod.out)

controller = Controller(time_to_keep_after_release=0.1)


# ------ Audio callback ------
print(sd.query_devices(sd.default.device[1]))


def callback(indata, outdata, frames, t, status):  # 240/44100 seconds per frame
    signals = controller.get_active_signals(blocksize=frames, t=t.outputBufferDacTime)
    if len(signals) == 0:
        outdata.fill(0.0)
        return

    res = 0
    for signal in signals:
        # print(frames)
        # print('latency:', t.outputBufferDacTime - t.currentTime)
        res = res + mod(signal)['out']

    outdata[:, 0] = res


if __name__ == "__main__":
    pygame.init()
    screen = pygame.display.set_mode((400, 300))

    sd.Stream(callback=callback, samplerate=44100, channels=1, blocksize=0, latency=0.05).start()

    while True:
        screen.fill((0, 0, 0))
        pygame.display.update()
        controller.update()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    exit()
                if event.key == pygame.K_a:
                    controller.press('C4')
                if event.key == pygame.K_s:
                    controller.press('C#4')
                if event.key == pygame.K_d:
                    controller.press('D4')
                if event.key == pygame.K_f:
                    controller.press('D#4')
                if event.key == pygame.K_g:
                    controller.press('E4')
                if event.key == pygame.K_h:
                    controller.press('F4')
                if event.key == pygame.K_j:
                    controller.press('F#4')
                if event.key == pygame.K_k:
                    controller.press('G4')
                if event.key == pygame.K_l:
                    controller.press('G#4')
                if event.key == pygame.K_SEMICOLON:
                    controller.press('A4')
                if event.key == pygame.K_QUOTE:
                    controller.press('A#4')
                if event.key == pygame.K_RETURN:
                    controller.press('B4')
