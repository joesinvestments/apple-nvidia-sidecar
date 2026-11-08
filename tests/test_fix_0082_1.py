#!/usr/bin/env python3
"""tests/test_fix_0082_1.py - the QMD size of a secondary constant buffer must be in 16-byte units.

sidecar-shim/tgcu.py:519 writes constant_buffer_size_shifted4_<i> for every constant bank above bank 0 as
round_up(size, 16), which is bytes. The field name says shifted4: the hardware reads size << 4 bytes, so a byte count
there tells it to read 16 times the bank (a 64-byte bank becomes 1024 bytes). These cases load the real
sidecar-shim/tgcu.py with tinygrad stubbed out (no GPU, no model, no weights), encode one launch through the real
Context._encode, and read the QMD fields it hands the hardware:
  - 64 bytes and 100 bytes on the sm_90 QMD layout (unshifted addresses) are 4 and 7 units;
  - two banks of 48 and 4112 bytes on the sm_120 QMD layout (shifted6 addresses) are 3 and 257 units;
  - a kernel with no secondary bank encodes as before: the bank-0 size, the program address and the grid are unchanged;
  - the secondary bank's address and valid flag are untouched.
Prints one PASS or FAIL line per case and exits nonzero on any FAIL. Run: python3 tests/test_fix_0082_1.py
"""
import importlib.util
import os
import sys
import types

ROOT = os.getcwd()   # the tree under test: every test here runs with cwd at the copy root, never at the test file
SHIM = os.path.join('sidecar-shim', 'tgcu.py')
UNIT = 16            # constant_buffer_size_shifted4_<i> counts 16-byte units: bytes = units << 4


def round_up(x, amt): return (x + amt - 1) // amt * amt


class FakeQMD:
  """Stands in for tinygrad's QMD: records every field write instead of packing them into a launch record."""
  grid = ('grid_x', 'grid_y', 'grid_z')
  def __init__(self, dev=None, blob=None):
    self.dev = dev
    self.ver = getattr(dev, 'ver', 3)   # the QMD layout the fake device uses: 3 (sm_90 and older) or 5 (sm_120)
    self.sz = 0x100
    self.mv = bytearray(blob) if blob is not None else bytearray(self.sz)
    self.fields = {}
  def write(self, **fields):
    self.fields.update(fields)
    if 'qmd_major_version' in fields: self.ver = fields['qmd_major_version']
    return self


class FakeDev:
  def __init__(self, ver=3): self.ver, self.slm_per_thread = ver, 0
  def __getattr__(self, name):
    if name.startswith('_'): raise AttributeError(name)
    return 0   # sass_version and the rest of the device record: any value does for a size check


class FakeNVG:  # the driver constant table: every enum the encoder reads is a plain number
  def __getattr__(self, name):
    if name.startswith('_'): raise AttributeError(name)
    return 0


def stub_tinygrad():
  """The shim imports tinygrad's Device, Buffer and helpers: give it the few names it needs and no GPU."""
  tg = types.ModuleType('tinygrad')
  tg.Device = {'NV': object()}
  helpers = types.ModuleType('tinygrad.helpers')
  helpers.round_up = lambda x, amt: (x + amt - 1) // amt * amt
  helpers.data64_le = lambda x: (x & 0xffffffff, (x >> 32) & 0xffffffff)
  device = types.ModuleType('tinygrad.device')
  device.Buffer, device.BufferSpec, device.BufferStorage = object, object, object
  runtime = types.ModuleType('tinygrad.runtime')
  ops_nv = types.ModuleType('tinygrad.runtime.ops_nv')
  ops_nv.QMD = FakeQMD
  tg.runtime = runtime
  runtime.ops_nv = ops_nv
  sys.modules.update({'tinygrad': tg, 'tinygrad.helpers': helpers, 'tinygrad.device': device,
                      'tinygrad.runtime': runtime, 'tinygrad.runtime.ops_nv': ops_nv})
  return ops_nv


def load_shim():
  if not os.path.isfile(os.path.join(ROOT, SHIM)):
    print('FAIL run this test from the copy root: ' + SHIM + ' is not in ' + ROOT)
    sys.exit(1)
  ops_nv = stub_tinygrad()
  spec = importlib.util.spec_from_file_location('tgcu_under_test', os.path.join(ROOT, SHIM))
  mod = importlib.util.module_from_spec(spec)
  sys.modules['tgcu_under_test'] = mod   # its dataclasses look their own module up while the class body runs
  spec.loader.exec_module(mod)
  mod.onv._mod = ops_nv   # pin tinygrad's NV module to the fake so no real driver is looked up
  return mod


BASE = 0x7f0000000000
def encode(tgcu, ver, constbufs, cbuf0=(0x160, 0x40)):
  """One launch through the real Context._encode: returns the QMD fields and the constant bank 0 image."""
  ctx = tgcu.Context.__new__(tgcu.Context)   # no __init__: that one boots the GPU
  ctx.dev = FakeDev(ver)
  ctx.nvg = FakeNVG()
  ctx.shmem_window, ctx.lmem_window = 0x12340000, 0x56780000
  ctx._enc_cache = {}
  k = tgcu.Kernel('k', types.SimpleNamespace(base=BASE), 0x100, 0x40)
  k.param_base, k.param_size = cbuf0
  k.constbufs = dict(constbufs)
  q, cb0, name = ctx._encode(k, (1, 1, 1), (128, 1, 1), 0, b'')
  return q.fields, cb0


def units_case(tgcu, ver, size, bank):
  fields, _ = encode(tgcu, ver, {bank: (0x80, size)})
  units = fields.get(f'constant_buffer_size_shifted4_{bank}')
  want = ((size + UNIT - 1) // UNIT * UNIT) // UNIT
  if units == want:
    return True, f'{size} bytes is {units} unit(s), so the hardware reads {units * UNIT} bytes'
  return False, (f'the QMD field says {units} 16-byte units, so the hardware reads {units * UNIT} bytes '
                 f'for the {size}-byte bank (16 times too much); it must be {want}')


def unchanged_case(tgcu):
  fields, cb0 = encode(tgcu, 3, {})
  prog = BASE + 0x100
  want = dict(constant_buffer_size_shifted4_0=round_up(max(len(cb0), 0x160 + 0x40), 256) >> 4, constant_buffer_valid_0=1,
              program_address_lower=prog & 0xffffffff, program_address_upper=prog >> 32,
              grid_x=1, grid_y=1, grid_z=1)
  wrong = [f'{k}={fields.get(k)} (want {v})' for k, v in want.items() if fields.get(k) != v]
  extra = [k for k in fields if k.startswith('constant_buffer_size_shifted4_') and k != 'constant_buffer_size_shifted4_0']
  if wrong: return False, 'a launch with no secondary bank encodes differently: ' + ', '.join(wrong)
  if extra: return False, 'a launch with no secondary bank set a secondary size field: ' + ', '.join(extra)
  return True, 'bank 0, the program address and the grid are as before'


def address_case(tgcu):
  fields, _ = encode(tgcu, 3, {2: (0x80, 64)})
  a = BASE + 0x80
  want = dict(constant_buffer_addr_lower_2=a & 0xffffffff, constant_buffer_addr_upper_2=a >> 32, constant_buffer_valid_2=1)
  wrong = [f'{k}={fields.get(k)} (want {v})' for k, v in want.items() if fields.get(k) != v]
  if wrong: return False, 'the secondary bank address or valid flag changed: ' + ', '.join(wrong)
  return True, 'bank 2 points at the image offset and is marked valid'


def two_banks(tgcu):
  fields, _ = encode(tgcu, 5, {1: (0x80, 48), 3: (0x200, 4112)})
  wrong = []
  for bank, size, want in ((1, 48, 3), (3, 4112, 257)):
    units = fields.get(f'constant_buffer_size_shifted4_{bank}')
    if units != want: wrong.append(f'bank {bank} is {units} units for {size} bytes (want {want}, '
                                   f'the hardware reads {units * UNIT if units is not None else None} bytes)')
  if wrong: return False, 'sm_120: ' + ', '.join(wrong)
  return True, 'bank 1 is 3 units (48 bytes) and bank 3 is 257 units (4112 bytes)'


def main():
  tgcu = load_shim()
  cases = [
    ('a 64-byte secondary constant buffer is 4 units, not 64', lambda: units_case(tgcu, 3, 64, 1)),
    ('a 100-byte secondary constant buffer is 7 units, not 112', lambda: units_case(tgcu, 3, 100, 1)),
    ('sm_120: two secondary constant buffers carry their own unit count', lambda: two_banks(tgcu)),
    ('a launch with no secondary constant buffer encodes as before', lambda: unchanged_case(tgcu)),
    ('the secondary bank address and valid flag are untouched', lambda: address_case(tgcu)),
  ]
  failures = 0
  for name, run in cases:
    try:
      ok, detail = run()
    except Exception as e:
      ok, detail = False, f'raised {type(e).__name__}: {e}'
    print(('PASS ' if ok else 'FAIL ') + name + (': ' + detail if detail else ''))
    failures += 0 if ok else 1
  print('%d case(s) failed' % failures if failures else 'all cases passed')
  return 1 if failures else 0


if __name__ == '__main__':
  sys.exit(main())
