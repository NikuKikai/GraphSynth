# Graph Synth (WIP)

A simple graph-style python framework for audio synthesizer.


```python
import numpy as np
from graphsynth.modules import OSC, Gain, Container, Envelope

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
ts = np.arange(44100)  / 44100.0  # time series of 1 second, at sample rate 44100
result = mod(ts)  # run mod to get audio waveform
```

Connecting part can be simplified, if all modules have single audio inport/outport.

```python
# Connect them
osc.to(env).to(gain).to(filter).to(mod.out)  # last one should be 'mod.out' since it's not inport.
```

## Usage

- Module can contain multiple input ports and output ports.
- Modules are connected through ports, using `to` method.
    - If modules have single inport/outport, you can directly use Module's `to` method.