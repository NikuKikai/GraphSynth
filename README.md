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
- `Module` can be nested. Thus you can define a custom sub-`Module`:

```python
class MyMod(Module):
    def __init__(self):
        super().__init__()
        self.out = self._create_outport('out', is_audio=True)

        osc = OSC(OSC.WAVETYPE.SAWTOOTH)
        env = Envelope(attack=0.05, decay=0.1, sustain=0.5, release=0.5)
        filter = LowPass(freq=300)

        osc.to(env).to(filter).to(self.out)  # Finally connect to self.out

# Use it just like predefined modules.
mod = mod = Container()
submod = MyMod()
gain = Gain(gain=0.1)
submod.to(mod).to(gain).to(mod.out)
...
```

- Arguments of predefined modules are mostly `Inport`s, so they can receive other modules output.

```python
mod = Container()
osc = OSC(OSC.WAVETYPE.WHITENOISE)
envelop_lowpass = ADSHR.DS(decay=0.01, sustain=0.05, scale=1000)
lowpass = LowPass()

# Main pass
osc.to(lowpass).to(mod.out)
# Modulate argument `freq` of lowpass
envelop_lowpass.to(lowpass.freq)  # Must specify `freq` since `to` search for the only audio InPort by default.
```

## Examples

> Located under `\examples\`.

#### `play_keyboard.py`

Play with your keyboard like a piano. Sound streamed to your speaker in realtime.
Require `pygame`, `sounddevice`.


## Acknowledge

- Thanks to [Sprechstimme](https://github.com/Sprechstimme-lib/Sprechstimme/tree/main) as reference on mathematical implementation.
