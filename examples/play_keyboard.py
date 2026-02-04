import numpy as np
import time
import pygame
import sounddevice as sd
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from graphsynth import Keyboard
from graphsynth.modules import OSC, Gain, Container, Envelope, LowPass


# ------ Define synth structure ------
mod = Container()
osc = OSC(OSC.WAVETYPE.SAWTOOTH)
gain = Gain(gain=0.2)
env = Envelope(attack=0.05, decay=0.1, sustain=0.5, release=0.5)
filter = LowPass(freq=300)

# Complete annotations
# osc.out.to(env.inp)
# env.out.to(gain.inp)
# gain.out.to(filter.inp)
# filter.out.to(mod.out)

# Simplified annotations
osc.to(env).to(gain).to(filter).to(mod.out)

keyboard = Keyboard(time_to_keep_after_release=0.5)

# ------ Audio callback ------
print(sd.query_devices(sd.default.device[1]))


def callback(indata, outdata, frames, t, status):  # frames/44100 seconds per call
    signals = keyboard.get_active_signals(
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
    pygame.K_q: 'C5',
    pygame.K_w: 'C#5',
    pygame.K_e: 'D5',
    pygame.K_r: 'D#5',
    pygame.K_t: 'E5',
    pygame.K_y: 'F5',
    pygame.K_u: 'F#5',
    pygame.K_i: 'G5',
    pygame.K_o: 'G#5',
    pygame.K_p: 'A5',
    pygame.K_LEFTBRACKET: 'A#5',
    pygame.K_RIGHTBRACKET: 'B5',
}


if __name__ == "__main__":
    pygame.init()
    screen = pygame.display.set_mode((400, 300))

    sd.Stream(callback=callback, samplerate=44100,
              channels=1, blocksize=0, latency=0.05).start()

    while True:
        screen.fill((0, 0, 0))
        pygame.display.update()
        keyboard.update()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    exit()
                if event.key in KEYMAP:
                    keyboard.press(KEYMAP[event.key])
            if event.type == pygame.KEYUP:
                if event.key in KEYMAP:
                    keyboard.release(KEYMAP[event.key])
