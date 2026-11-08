"""A cooperative launch is sized from the card's blocks per SM, not from a 3090's hard-coded 16.

co resident_blocks() (sidecar-shim/tgcu.py) takes the smallest of four per-SM limits and multiplies it
by the SM count.  One of the four was the literal 16, a figure only some architectures have: sm_80, sm_90
and sm_100 allow 32 resident blocks per SM, sm_89 (Ada) 24.  On such a card a cooperative grid the hardware
can hold was rejected as too large (cudaErrorCooperativeLaunchTooLarge through
sidecar-shim/tgcu_api.py:605), and on a card that allows fewer blocks the 16 credited it with more than it
has.  These cases plant a device that reports its own limit and read back what the context says it can
hold:
  - a device that reports no limit still gets the 3090's 16 blocks per SM, 32 over two SMs;
  - a 32-block SM holds 64 blocks over two SMs, not 32;
  - a 24-block SM (Ada) holds 48, not 32;
  - a device that reports 8 holds 16, so the reported limit is used, not only raised;
  - a 1024-thread block still binds at the warp limit, the block cap never overriding it;
  - a 64-block cooperative grid on the 32-block card launches where the 16 cap refused it;
  - a 66-block grid there is still refused, at 64 co-resident blocks;
  - a 32-block grid on the card that reports nothing still launches.
No GPU, no model, no network: tinygrad is stubbed and the device is a two-attribute stub.
Prints one PASS or FAIL line per case and exits nonzero on any FAIL.
Run: python3 tests/test_fix_0082_5.py
"""
import importlib.util
import sys
import types
from pathlib import Path

SMS = 2                          # planted SM count, so the per-SM figure shows up as a small number
WARPS_PER_SM = 64                # neither the warp nor the register limit may bind below the case's want
REGS = 8
THREADS = 32                     # one warp
WIDE_THREADS = 1024              # a whole SM of warps (32 of them)
SHARED = 0                       # no shared memory: the shared-memory limit stays above the block cap
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
    """A GPU as the context reads it: its warps per SM and, when the card reports one, its blocks per SM."""

    def __init__(self, max_blocks_per_sm=None):
        self.max_warps_per_sm = WARPS_PER_SM
        if max_blocks_per_sm is not None:
            self.max_blocks_per_sm = max_blocks_per_sm


class Kernel:
    """The kernel fields the co-residency arithmetic reads."""
    name = 'coop_fill'
    regs = REGS
    shared = SHARED


def context(tgcu, max_blocks_per_sm):
    ctx = tgcu.Context.__new__(tgcu.Context)   # no GPU: the device and the SM count are planted
    ctx.dev = Dev(max_blocks_per_sm)
    ctx._num_sms = SMS
    return ctx


def capacity(tgcu, max_blocks_per_sm, threads=THREADS):
    return context(tgcu, max_blocks_per_sm).coresident_blocks(Kernel(), threads, DYN)


def cooperative(tgcu, max_blocks_per_sm, blocks):
    """encode_cooperative for a grid of `blocks`: 'refused' with the reason, or 'queued' with the kernel name."""
    ctx = context(tgcu, max_blocks_per_sm)
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

    # a device with no blocks-per-SM limit of its own keeps the 3090's 16: 32 over the two planted SMs
    case('a device that reports no block limit holds 16 blocks an SM (32 over two SMs)',
         16 * SMS, capacity(tgcu, None))
    # sm_80/sm_90/sm_100: 32 blocks an SM, not the 16 the literal allowed
    case('a 32-block SM holds 32 blocks of one warp (64 over two SMs), not 16',
         32 * SMS, capacity(tgcu, 32))
    # sm_89 (Ada): 24 blocks an SM
    case('a 24-block SM holds 24 blocks (48 over two SMs), not 16',
         24 * SMS, capacity(tgcu, 24))
    # a card that allows fewer than 16: the reported limit is the cap, it is not raised to 16
    case('a device reporting 8 blocks an SM holds 8 (16 over two SMs), not 16',
         8 * SMS, capacity(tgcu, 8))
    # the warp limit still binds below the block cap: 32 warps a block, 64 warps an SM -> 2 blocks
    case('a 1024-thread block on the 32-block SM still binds at the warp limit (4 over two SMs)',
         (WARPS_PER_SM // (WIDE_THREADS // 32)) * SMS, capacity(tgcu, 32, WIDE_THREADS))
    # the refusal a cooperative grid gets: 64 blocks fit a 32-block SM pair, the literal cap of 32 refused them
    case('a 64-block cooperative grid on the 32-block card launches', 'queued', cooperative(tgcu, 32, 64)[0])
    # and a grid over the limit is still refused, at the card's own figure
    status, detail = cooperative(tgcu, 32, 66)
    case('a 66-block cooperative grid on the 32-block card is refused at 64 co-resident blocks',
         True, status == 'refused' and '66 > 64 co-resident blocks' in detail)
    # a grid the card that reports nothing does hold still launches
    case('a 32-block cooperative grid on the card that reports no limit still launches',
         'queued', cooperative(tgcu, None, 32)[0])

    print('%d case(s) failed' % results.count(False) if not all(results) else 'all cases passed')
    return 0 if all(results) else 1


if __name__ == '__main__':
    sys.exit(main())
