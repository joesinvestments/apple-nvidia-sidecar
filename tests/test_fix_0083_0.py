#!/usr/bin/env python3
"""tests/test_fix_0083_0.py - the async copy engine must not print its setup to stdout.

sidecar-shim/tgcu.py:836 prints "[tgcu ce] engine ..." from AsyncCopy.__init__, and lines 838, 844, 846 and 848
print four more "[tgcu ce]" lines with no env-var gate and no early return. Every process that builds the native
context (the long-batch path that copies through the async engine, Context.__init__ at line 214) therefore gets
debug noise on its stdout. The shim's own profiling output is gated by TGCU_PROF and written to stderr, so the
stdout prints are leftover debug output.

These cases build the real AsyncCopy against a stub driver (no GPU, no tinygrad, no model, no weights) and watch
stdout and the construction result:
  - a default construction writes nothing to stdout;
  - the TGCU_CE_INDEX lab switch writes nothing to stdout;
  - TGCU_PROF profiling writes nothing to stdout;
  - the construction result (engine, channel handles, token) is unchanged.
Prints one PASS or FAIL line per case and exits nonzero on any FAIL. Run: python3 tests/test_fix_0083_0.py
"""
import contextlib
import ctypes
import importlib.util
import io
import os
import sys
import types
from types import SimpleNamespace

ROOT = os.getcwd()   # the tree under test: every test here runs with cwd at the copy root, never at the test file
SHIM = os.path.join(ROOT, 'sidecar-shim', 'tgcu.py')


def _round_up(x, n):
    return (x + n - 1) // n * n


def _stub_tinygrad():
    # the shim imports tinygrad at module scope; these cases never touch a device, so a plain stub is enough
    tinygrad = types.ModuleType('tinygrad')
    tinygrad.Device = SimpleNamespace()
    device = types.ModuleType('tinygrad.device')
    device.Buffer = device.BufferSpec = device.BufferStorage = object
    helpers = types.ModuleType('tinygrad.helpers')
    helpers.round_up = _round_up
    helpers.data64_le = lambda b: int.from_bytes(bytes(b), 'little')
    tinygrad.device = device
    tinygrad.helpers = helpers
    sys.modules['tinygrad'] = tinygrad
    sys.modules['tinygrad.device'] = device
    sys.modules['tinygrad.helpers'] = helpers


def load_shim():
    _stub_tinygrad()
    spec = importlib.util.spec_from_file_location('tgcu_under_test', SHIM)
    if spec is None or spec.loader is None:
        raise RuntimeError('cannot import ' + SHIM)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


class FakeNvg:
    NV2080_ENGINE_TYPE_COPY0 = 0x8000
    KEPLER_CHANNEL_GROUP_A = 0x100
    FERMI_CONTEXT_SHARE_A = 0x101
    NV_CTXSHARE_ALLOCATION_FLAGS_SUBCONTEXT_ASYNC = 0x1
    NVA06C_CTRL_CMD_GPFIFO_SCHEDULE = 0x701
    NVA06C_CTRL_CMD_SET_TIMESLICE = 0x702
    NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN = 0x801
    NVC6C0_SET_OBJECT = 0x900
    AmpereAControlGPFifo = SimpleNamespace(GPPut=SimpleNamespace(offset=0x1234))

    def __init__(self):
        self.NV_CHANNEL_GROUP_ALLOCATION_PARAMETERS = lambda **kw: kw
        self.NV_CTXSHARE_ALLOCATION_PARAMETERS = lambda **kw: kw
        self.NV_CHANNELGPFIFO_ALLOCATION_PARAMETERS = lambda **kw: kw
        self.NVA06C_CTRL_GPFIFO_SCHEDULE_PARAMS = lambda **kw: kw
        self.NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN_PARAMS = lambda **kw: kw
        self.NVA06C_CTRL_SET_TIMESLICE_PARAMS = lambda **kw: kw


class FakeOpsNv:
    def __init__(self, calls):
        self.nv_gpu = FakeNvg()
        self.ctypes = ctypes
        self.calls = calls

    def nvm(self, n, op, *args):
        self.calls.append(('nvm', op))
        return [int(op) & 0xffffffff] * n

    def nv_flags(self, name, **kw):
        return 0


class FakeIface:
    gpfifo_class = 0xc36f
    dma_class = 0xb5

    def __init__(self, calls):
        self.calls = calls
        self.next = 0x10000000

    def alloc(self, size, **kw):
        self.calls.append(('alloc', size))
        addr, self.next = self.next, self.next + max(size, 0x1000)
        return SimpleNamespace(buf=addr, host=bytearray(size), meta=SimpleNamespace(hMemory=addr))

    def rm_alloc(self, handle, cls, params=None):
        self.calls.append(('rm_alloc', cls))
        self.next += 0x1000
        return self.next

    def rm_control(self, handle, cmd, params=None):
        self.calls.append(('rm_control', cmd))
        return SimpleNamespace(workSubmitToken=0x5A5A, timesliceUs=1024)


class FakeDev:
    nvdevice = 0xdead0001
    vaspace = 0xdead0002
    channel_group = 0xdead0003

    def __init__(self):
        self.gpu_mmio = [0] * 64


def build(mod):
    calls = []
    iface = FakeIface(calls)
    dev = FakeDev()
    ctx = SimpleNamespace(dev=dev, iface=iface)
    with contextlib.redirect_stdout(io.StringIO()) as out, contextlib.redirect_stderr(io.StringIO()) as err:
        # the stub driver stands in for tinygrad's NV device: only the shim's own output is under test
        old = mod.onv._mod
        mod.onv._mod = FakeOpsNv(calls)
        try:
            ace = mod.AsyncCopy(ctx)
        finally:
            mod.onv._mod = old
    return ace, calls, out.getvalue(), err.getvalue()


def main():
    if not os.path.isfile(SHIM):
        print('FAIL run this test from the copy root: ' + SHIM + ' is not in ' + ROOT)
        return 1
    mod = load_shim()
    failures = 0

    def case(name, ok, detail=''):
        nonlocal failures
        print(('PASS ' if ok else 'FAIL ') + name + (': ' + detail if detail else ''))
        failures += 0 if ok else 1

    for k, v in (('TGCU_CE_INDEX', None), ('TGCU_PROF', None), ('TGCU_CE_PVER', None), ('TGCU_CE_CLASS', None)):
        os.environ.pop(k, None)
    ace, calls, out, err = build(mod)
    case('a default AsyncCopy construction writes nothing to stdout', out == '', repr(out[:300]))

    old_index = os.environ.get('TGCU_CE_INDEX')
    try:
        os.environ['TGCU_CE_INDEX'] = '1'
        ace_index, calls_index, out_index, err_index = build(mod)
    finally:
        if old_index is None:
            os.environ.pop('TGCU_CE_INDEX', None)
        else:
            os.environ['TGCU_CE_INDEX'] = old_index
    case('the TGCU_CE_INDEX lab switch writes nothing to stdout', out_index == '', repr(out_index[:300]))
    case('the TGCU_CE_INDEX lab switch still selects the copy engine it names',
         ace_index.engine == FakeNvg.NV2080_ENGINE_TYPE_COPY0 + 1,
         'engine %r' % (ace_index.engine,))

    old_prof = os.environ.get('TGCU_PROF')
    try:
        os.environ['TGCU_PROF'] = '1'
        ace_prof, calls_prof, out_prof, err_prof = build(mod)
    finally:
        if old_prof is None:
            os.environ.pop('TGCU_PROF', None)
        else:
            os.environ['TGCU_PROF'] = old_prof
    case('TGCU_PROF profiling writes nothing to stdout', out_prof == '', repr(out_prof[:300]))

    case('the copy engine is still set up (engine, handles, token)',
         ace.engine == FakeNvg.NV2080_ENGINE_TYPE_COPY0 and ace.token == 0x5A5A
         and isinstance(ace.group, int) and isinstance(ace.ctxshare, int) and isinstance(ace.chan, int)
         and sum(1 for c in calls if c[0] == 'rm_alloc') == 4,
         'engine %r token %r rm_alloc %d' % (ace.engine, ace.token, sum(1 for c in calls if c[0] == 'rm_alloc')))

    print('%d case(s) failed' % failures if failures else 'all cases passed')
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
