from typing import Union
import numpy as np
import time
import pygame
import sounddevice as sd
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from graphsynth.modules import OSC, Gain, Container, Envelope
from graphsynth.core import NoteSignal


class Controller:
    def __init__(self, time_to_keep_after_release=0.5):
        self.t_keep = time_to_keep_after_release
        self.signals: dict[float, NoteSignal] = dict()

    def press(self, freq: Union[float, str], release_after: float = None):
        if isinstance(freq, str):
            freq = note_str2freq(freq)
        s = NoteSignal(freq=freq)
        self.signals[freq] = s
        if release_after is not None:
            s.released_t_sys = s.inited_t_sys + release_after

    def release(self, freq: Union[float, str]):
        if isinstance(freq, str):
            freq = note_str2freq(freq)
        t = time.time()
        if freq in self.signals:
            sig = self.signals[freq]
            if sig.released_t_sys < 0.0 or sig.released_t_sys > t:
                sig.released_t_sys = t

    def update(self):
        now = time.time()
        to_delete = []
        for freq, signal in self.signals.items():
            if signal.released_t_sys < 0.0:
                continue
            if now - signal.released_t_sys > self.t_keep:
                to_delete.append(freq)
        for freq in to_delete:
            del self.signals[freq]

    def get_active_signals(self, blocksize: int, t: float = None) -> list[NoteSignal]:
        res = list(self.signals.values())
        t = t if t is not None else time.time()
        for sig in res:
            if sig.inited_t < 0.0:
                sig.inited_t = t
            sig.current_ts = np.array(
                [t]) + np.arange(blocksize) * (1.0 / 44100.0)
        return res


def note_str2freq(note: str) -> float:
    A4_FREQ = 440.0
    NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E',
                  'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    octave = int(note[-1])
    key = note[:-1]
    n = NOTE_NAMES.index(key)
    semitones_from_A4 = n - 9 + (octave - 4) * 12
    return A4_FREQ * (2 ** (semitones_from_A4 / 12))


# ------ Define synth structure ------
mod = Container()
osc = OSC(OSC.WAVETYPE.SAWTOOTH)
gain = Gain(gain=0.1)
env = Envelope(attack=0.05, decay=0.1, sustain=0.5, release=0.5)

osc.out.to(env.inp)
env.out.to(gain.inp)
gain.out.to(mod.out)

controller = Controller(time_to_keep_after_release=0.5)


# ------ Audio callback ------
print(sd.query_devices(sd.default.device[1]))


def callback(indata, outdata, frames, t, status):  # frames/44100 seconds per call
    signals = controller.get_active_signals(
        blocksize=frames, t=t.outputBufferDacTime)
    if len(signals) == 0:
        outdata.fill(0.0)
        return

    res = 0
    for signal in signals:
        # print(frames)
        # print('latency:', t.outputBufferDacTime - t.currentTime)
        res = res + mod(signal)['out']

    outdata[:, 0] = res


KEYMAP = {
    pygame.K_a: 'C4',
    pygame.K_s: 'C#4',
    pygame.K_d: 'D4',
    pygame.K_f: 'D#4',
    pygame.K_g: 'E4',
    pygame.K_h: 'F4',
    pygame.K_j: 'F#4',
    pygame.K_k: 'G4',
    pygame.K_l: 'G#4',
    pygame.K_SEMICOLON: 'A4',
    pygame.K_QUOTE: 'A#4',
    pygame.K_RETURN: 'B4',
}


if __name__ == "__main__":
    pygame.init()
    screen = pygame.display.set_mode((400, 300))

    sd.Stream(callback=callback, samplerate=44100,
              channels=1, blocksize=0, latency=0.05).start()

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
                if event.key in KEYMAP:
                    controller.press(KEYMAP[event.key])
            if event.type == pygame.KEYUP:
                if event.key in KEYMAP:
                    controller.release(KEYMAP[event.key])
