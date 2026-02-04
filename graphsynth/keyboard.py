from typing import Union
import time
import numpy as np
from .core import NoteSignal
from .utils import note_str2freq


class Keyboard:
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
            sig.current_ts = np.arange(blocksize) / 44100.0 + t
        return res
