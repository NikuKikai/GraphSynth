from typing import Union
import numpy as np
import time


class NoteSignal:
    def __init__(self, freq=440.0):
        self.freq = freq
        self.inited_t_sys = time.time()
        self.released_t_sys = -1.0
        self.inited_t = -1.0
        self.current_ts = np.array(0.0)
        self._cache = {}  # cache for one-time evaluation
        self._states = {}  # storage for module states

    @property
    def local_ts(self):
        return self.current_ts - self.inited_t

    @property
    def local_released_t(self):
        return self.released_t_sys - self.inited_t_sys


class Port:
    def __init__(self, name: str, parent: 'Module', is_audio=False):
        self.name = name
        self.parent = parent
        self.is_audio = is_audio

    def to(self, target: 'Port'):
        if target.source is not None:
            raise ValueError("Target port already has a source connected")
        target.source = self
        return target


class InPort(Port):
    def __init__(self, name: str, parent: 'Module', default=0.0, is_audio=False):
        super().__init__(name, parent, is_audio=is_audio)
        self.default = default
        self.source: Port = None

    def _eval(self, signal: NoteSignal, root: 'Module', **kwargs):
        if self.parent == root:
            return kwargs.get(self.name, self.default)
        if self.source is None:
            return self.default
        return self.source._eval(signal, root, **kwargs)


class OutPort(Port):
    def __init__(self, name: str, parent: 'Module', is_audio=False):
        super().__init__(name, parent, is_audio=is_audio)
        self.source: Port = None

    def _eval(self, signal: NoteSignal, root: 'Module', **kwargs):
        if self.parent == root:
            raise ValueError("Cannot evaluate output port on root module")
        return self.parent._call(signal, root, **kwargs)[self.name]


class Module:
    def __init__(self, inports: dict[str, any] = {}, outports: list[str] = []):
        self._primitive = False
        self._inports: list[InPort] = []
        self._outports: list[OutPort] = []

        for ip_name, default in inports.items():
            self._create_inport(ip_name, default=default)

        for op_name in outports:
            self._create_outport(op_name)

    def __call__(self, signal: NoteSignal, **kwargs):
        signal._cache.clear()
        return self._call(signal, self, **kwargs)

    def _call(self, signal: NoteSignal, root: 'Module', **kwargs):
        # cache
        cache_key = (id(self), id(root))
        if cache_key in signal._cache:
            return signal._cache[cache_key]

        if self._primitive:
            ipargs = dict()
            for ip in self._inports:
                ipargs[ip.name] = ip._eval(signal, root, **kwargs)
            result = self._proc(signal, **ipargs)
        else:
            result = dict()
            for op in self._outports:
                result[op.name] = op.source._eval(signal, root, **kwargs)

        # cache
        signal._cache[cache_key] = result
        return result

    def _proc(self, signal: NoteSignal, **kwargs):
        res = dict()
        for op in self._outports:
            res[op.name] = 0.0
        return res

    def _create_inport(self, name: str, default=0.0, is_audio=False) -> InPort:
        ip = InPort(name, self, default=default, is_audio=is_audio)
        setattr(self, name, ip)
        self._inports.append(ip)
        return ip

    def _create_outport(self, name: str, is_audio=False) -> OutPort:
        op = OutPort(name, self, is_audio=is_audio)
        setattr(self, name, op)
        self._outports.append(op)
        return op

    @classmethod
    def wrap_func(cls, inports: dict[str, any], outports: list[str]):
        def wrap(func):
            class FuncModule(Module):
                def __init__(self):
                    super().__init__(inports=inports, outports=outports)
                    self._primitive = True

                def _proc(self, signal: NoteSignal, **kwargs) -> dict:
                    return func(signal, **kwargs)
            return FuncModule
        return wrap

    def to(self, target: Union['Module', Port]) -> Union['Module', Port]:
        audio_outports = [op for op in self._outports if op.is_audio]
        if len(audio_outports) != 1 and len(self._outports) != 1:
            raise ValueError(
                f'Module "{self}" has multiple output ports, cannot infer which to connect')
        if len(audio_outports) == 1:
            src_op = audio_outports[0]
        else:
            src_op = self._outports[0]

        if isinstance(target, Port):
            src_op.to(target)
        else:
            audio_inports = [ip for ip in target._inports if ip.is_audio]
            if len(audio_inports) != 1 and len(target._inports) != 1:
                raise ValueError(
                    f'Target module "{target}" has multiple input ports, cannot infer which to connect')
            if len(audio_inports) == 1:
                dest_ip = audio_inports[0]
            else:
                dest_ip = target._inports[0]
            src_op.to(dest_ip)
        return target
