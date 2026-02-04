import numpy as np
from typing import Union


def note_str2freq(note: str) -> float:
    A4_FREQ = 440.0
    NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E',
                  'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    octave = int(note[-1])
    key = note[:-1]
    n = NOTE_NAMES.index(key)
    semitones_from_A4 = n - 9 + (octave - 4) * 12
    return A4_FREQ * (2 ** (semitones_from_A4 / 12))


class zip_longest:
    def __init__(self, *args: tuple[Union[np.ndarray, float]], allow_loop=True):
        self.args = args
        self.allow_loop = allow_loop
        lengths = [len(a) if isinstance(a, np.ndarray) else 1 for a in args]
        self.max_len = max(lengths)
        if not allow_loop and self.max_len > 1 and any(a != self.max_len and a != 1 for a in lengths):
            raise ValueError(
                "All arrays must have the same length (except scalars).")

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index >= self.max_len:
            raise StopIteration
        tuple_i = []
        for a in self.args:
            if isinstance(a, np.ndarray):
                tuple_i.append(a[self.index % len(a)])
            else:
                tuple_i.append(a)
        self.index += 1
        return tuple(tuple_i)
