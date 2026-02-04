# Graph Synth (WIP)

A simple graph-style python framework for audio synthesizer.


```python
import numpy as np
from graphsynth import NoteSignal
from graphsynth.modules import OSC, Gain, Container, Envelope, LowPass

# Create modules
mod = Container()  # Outer module to contain others
osc = OSC(OSC.WAVETYPE.SAWTOOTH)
gain = Gain(gain=0.1)
env = Envelope(attack=0.05, decay=0.1, sustain=0.5, release=0.5)
filter = LowPass(freq=300)

# Connect them
osc.out.to(env.inp)
env.out.to(gain.inp)
gain.out.to(filter.inp)
filter.out.to(mod.out) # remember to connect to mod.out at the end.

# Run
signal = NoteSignal(freq='C4')
signal.set_current(blocksize=44100)
result = mod(signal)  # run mod to get audio waveform
```

Connecting part can be simplified, if all modules have single audio inport/outport.

```python
# Connect them
osc.to(env).to(gain).to(filter).to(mod.out)  # last one should be 'mod.out' since it's not inport.
```

## Usage

- `Module` can contain multiple input ports and output ports.
- `Module`s are connected through `InPort`/`OutPort`s, using `to` method.
    - If `Module`s have single `InPort`/`OutPort`, you can directly use `Module`'s `to` method.
    - Same `Module` object can be called with multiple `NoteSignal` simultaneously, since inner states are store in `NoteSignal` object separately.

## Examples

> Located under `\examples\`.

#### `play_keyboard.py`

Play with your keyboard like a piano. Sound streamed to your speaker in realtime.
Require `pygame`, `sounddevice`.


## Acknowledge

- Thanks to [Sprechstimme](https://github.com/Sprechstimme-lib/Sprechstimme/tree/main) as reference on mathematical implementation.
