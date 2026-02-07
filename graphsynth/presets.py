from .core import Module
from .modules import OSC, Gain, ADSHR, LowPass, LowShelf


class Kick(Module):
    def __init__(self):
        super().__init__()
        self.out = self._create_outport('out', is_audio=True)

        osc = OSC(OSC.WAVETYPE.WHITENOISE)
        ds_osc = ADSHR.DS(decay=0.10, sustain=0.0, scale=1)
        ds_lowpass = ADSHR.DS(decay=0.01, sustain=0.05, scale=1000)
        lowpass = LowPass(freq=30, Q=10, feedback=0.9)
        env_amp = ADSHR(attack=0, decay=0.05, sustain=0.4, hold=0.2, release=0.3)
        lowshelf = LowShelf(freq=50, Q=3, gain=9)
        gain = Gain(gain=10)

        osc.to(ds_osc).to(lowpass).to(env_amp).to(lowshelf).to(gain).to(self.out)
        ds_lowpass.to(lowpass.freq)
