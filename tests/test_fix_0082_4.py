#!/usr/bin/env python3
"""A cooperative launch is sized from the SM's own shared memory, not from a 3090's 100 KiB.

sidecar-shim/tgcu.py:451 computes how many blocks of a kernel an SM can hold from shared memory as
(100 << 10) // round_up(0x400 + k.shared + dyn_shared, 128), a figure every architecture gets: a card whose
SM holds less than 100 KiB is credited with blocks the hardware cannot hold, so a cooperative grid that
cannot be co-resident is accepted (sidecar-shim/tgcu_api.py:602 makes that check), and a card with more is
refused grids it could run.  These cases plant a device's shared memory per SM in a stub and read back
what the context says it can hold:
  - a 100 KiB SM (the 3090) still holds 11 blocks, 22 over two SMs;
  - a 48 KiB SM holds 5 blocks, 10 over two SMs, not the 22 a 3090 would get;
  - a 228 KiB SM holds 3 blocks of 65 KiB, 6 over two SMs, not the 2 a 3090 would get;
  - a cooperative grid of 12 blocks on the 48 KiB card is refused, at 10 co-resident blocks;
  - a cooperative grid of 8 blocks there still launches.
No GPU, no model, no network: tinygrad is stubbed and the device is a two-attribute stub.
Prints one PASS or FAIL line per case and exits nonzero on any FAIL.
Run: python3 tests/test_fix_0082_4.py
"""
import importlib.util
import sys
import types
from pathlib import Path

SMS = 2                          # planted SM count, so the per-SM figure shows up as a small number
WARPS_PER_SM = 64                # neither the warp nor the register limit may bind below the case's want
REGS = 8
THREADS = 32                     # one warp
SHARED = 8192                    # a block of 0x400 + 8192 -> 9216 bytes of an SM's shared memory
BIG_SHARED = 65536               # a block of 0x400 + 65536 -> 66560 bytes
DYN = 0


def round_up(x, a):
    return (x + a - 1) // a * a


def find_root():
    """The tree under test: the copy prove_test runs us in, else the one this file sits in."""
    rel = Path('sidecar-shim') / 'tgcu.py'
    for base in (Path.cwd(), Path(__file__).resolve().parent.parent):
        if (base / rel).is_file():
            return base
    return Path.cwd()


def load_tgcu(root):
    """sidecar-shim/tgcu.py with tinygrad stubbed: the co-residency arithmetic needs no driver and no GPU."""
    tinygrad = types.ModuleType('tinygrad')
    device = types.ModuleType('tinygrad.device')
    helpers = types.ModuleType('tinygrad.helpers')
    tinygrad.Device = object
    device.Buffer = device.BufferSpec = device.BufferStorage = object
    helpers.round_up = round_up
    helpers.data64_le = lambda *xs: b''.join(int(x).to_bytes(8, 'little') for x in xs)
    sys.modules['tinygrad'] = tinygrad
    sys.modules['tinygrad.device'] = device
    sys.modules['tinygrad.helpers'] = helpers
    spec = importlib.util.spec_from_file_location('tgcu', str(root / 'sidecar-shim' / 'tgcu.py'))
    mod = importlib.util.module_from_spec(spec)
    sys.modules['tgcu'] = mod   # its dataclasses look themselves up in sys.modules by module name (3.9)
    spec.loader.exec_module(mod)
    return mod


class Dev:
    """A GPU as the context reads it: the shared memory of one SM, its warps per SM, nothing else."""

    def __init__(self, smem_per_sm):
        self.smem_per_sm = smem_per_sm
        self.max_warps_per_sm = WARPS_PER_SM


class Kernel:
    """The kernel fields the co-residency arithmetic reads."""
    name = 'coop_fill'
    regs = REGS
    shared = SHARED


class BigKernel(Kernel):
    name = 'coop_big'
    shared = BIG_SHARED


def context(tgcu, smem_per_sm):
    ctx = tgcu.Context.__new__(tgcu.Context)   # no GPU: the device and the SM count are planted
    ctx.dev = Dev(smem_per_sm)
    ctx._num_sms = SMS
    return ctx


def capacity(tgcu, smem_per_sm, kernel=Kernel()):
    return context(tgcu, smem_per_sm).coresident_blocks(kernel, THREADS, DYN)


def cooperative(tgcu, smem_per_sm, blocks):
    """encode_cooperative for a grid of `blocks`: 'refused' with the reason, or 'queued' with the kernel name."""
    ctx = context(tgcu, smem_per_sm)
    ctx.grid_ws = 0x100000   # already allocated: the launch's first use of the workspace stays out of the way
    ctx.encode = lambda k, grid, block, dyn, params: (types.SimpleNamespace(ver=3), bytearray(0x300), k.name)
    try:
        _q, _cb, name = ctx.encode_cooperative(Kernel(), (blocks, 1, 1), (THREADS, 1, 1), DYN, b'')
    except RuntimeError as e:
        return 'refused', str(e)
    return 'queued', name


def main():
    root = find_root()
    try:
        tgcu = load_tgcu(root)
    except Exception as e:
        print('FAIL sidecar-shim/tgcu.py does not load: %r' % (e,))
        return 1

    results = []

    def case(name, want, got):
        ok = got == want
        print(('PASS ' if ok else 'FAIL ') + name + ('' if ok else ': want %r, got %r' % (want, got)))
        results.append(ok)

    # a 3090: 100 KiB per SM, 9216 bytes a block -> 11 blocks an SM, 22 over the two planted SMs
    case('a 100 KiB SM holds 11 blocks of 9 KiB (22 over two SMs)',
         11 * SMS, capacity(tgcu, 100 << 10))
    # 48 KiB per SM: 5 blocks an SM; the hard-coded 100 KiB credits it with 11
    case('a 48 KiB SM holds 5 blocks of 9 KiB (10 over two SMs), not 11',
         (48 << 10) // round_up(0x400 + SHARED, 128) * SMS, capacity(tgcu, 48 << 10))
    # 228 KiB per SM: 3 blocks of 65 KiB; the hard-coded 100 KiB allows 1
    case('a 228 KiB SM holds 3 blocks of 65 KiB (6 over two SMs), not 1',
         (228 << 10) // round_up(0x400 + BIG_SHARED, 128) * SMS, capacity(tgcu, 228 << 10, BigKernel()))
    # the refusal a cooperative grid gets: 12 blocks do not fit the 48 KiB card's 10
    status, detail = cooperative(tgcu, 48 << 10, 12)
    case('a 12-block cooperative grid on the 48 KiB card is refused at 10 co-resident blocks',
         True, status == 'refused' and '12 > 10 co-resident blocks' in detail)
    # a grid the card does hold still launches
    case('an 8-block cooperative grid on the 48 KiB card still launches', 'queued', cooperative(tgcu, 48 << 10, 8)[0])

    print('%d case(s) failed' % results.count(False) if not all(results) else 'all cases passed')
    return 0 if all(results) else 1


if __name__ == '__main__':
    sys.exit(main())
