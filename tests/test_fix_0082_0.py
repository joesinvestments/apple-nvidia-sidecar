#!/usr/bin/env python3
"""tests/test_fix_0082_0.py - the constant buffer size a launch hands the GPU must be in 16-byte units.

sidecar-shim/tgcu.py:514 builds the QMD of every launch and writes the constant buffer length into
constant_buffer_size_shifted4_0. The register is named shifted4 because the hardware reads it in 16-byte units, but the
code writes cb0_size, a byte count, straight into it, so the GPU is told the constant buffer is 16 times its real
length. These cases build the QMD through _encode with a stubbed tinygrad, device, driver and GPU (no GPU, no model, no
weights) and read the register back the way the hardware does:
  - an sm_120 (QMD v5) launch reports the constant buffer length in 16-byte units, not in bytes;
  - a v3 launch reports it in 16-byte units too;
  - a kernel with a larger constant block (cbuf0_init) scales the same way.
Two further cases are ordinary input that must not change: the buffer is still marked valid and the program address is
still written shifted4.
Prints one PASS or FAIL line per case and exits nonzero on any FAIL. Run: python3 tests/test_fix_0082_0.py
"""
import ctypes
import importlib.util
import os
import struct
import sys
import types

ROOT = os.getcwd()   # the tree under test: run this from the copy root
SHIM = os.path.join('sidecar-shim', 'tgcu.py')
PROG_BASE = 0x100000000
PROG_OFF = 0x40
QMD_BYTES = 256


def round_up(x, amt):
  return ((x + amt - 1) // amt) * amt


def data64_le(x):
  return struct.unpack('<II', struct.pack('<Q', x & 0xffffffffffffffff))


class FakeQMD:
  """The launch record as the hardware sees it.

  write() stores the registers the driver would set. constant_buffer_bytes reads
  constant_buffer_size_shifted4_N back the way the hardware does: the field counts 16-byte units.
  """
  grid = ('grid_dimension_x', 'grid_dimension_y', 'grid_dimension_z')

  def __init__(self, dev):
    self.ver = dev.qmd_ver
    self.mv = bytearray(QMD_BYTES)
    self.regs = {}

  def write(self, **regs):
    self.regs.update(regs)

  def constant_buffer_bytes(self, bank=0):
    return self.regs['constant_buffer_size_shifted4_%d' % bank] << 4


def load_shim():
  """Import sidecar-shim/tgcu.py against stub tinygrad modules: no tinygrad install, no GPU, no driver."""
  tinygrad = types.ModuleType('tinygrad')
  tinygrad.__path__ = []

  class Device:
    @classmethod
    def __class_getitem__(cls, key):
      return None

  tinygrad.Device = Device
  device = types.ModuleType('tinygrad.device')
  for name in ('Buffer', 'BufferSpec', 'BufferStorage'):
    setattr(device, name, type(name, (), {}))
  helpers = types.ModuleType('tinygrad.helpers')
  helpers.round_up, helpers.data64_le = round_up, data64_le
  runtime = types.ModuleType('tinygrad.runtime')
  runtime.__path__ = []
  ops_nv = types.ModuleType('tinygrad.runtime.ops_nv')
  ops_nv.QMD, ops_nv.ctypes = FakeQMD, ctypes
  runtime.ops_nv = ops_nv
  tinygrad.device, tinygrad.helpers, tinygrad.runtime = device, helpers, runtime
  sys.modules.update({'tinygrad': tinygrad, 'tinygrad.device': device, 'tinygrad.helpers': helpers,
                      'tinygrad.runtime': runtime, 'tinygrad.runtime.ops_nv': ops_nv})
  spec = importlib.util.spec_from_file_location('tgcu_shim_under_test', os.path.join(ROOT, SHIM))
  shim = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = shim
  spec.loader.exec_module(shim)
  return shim


class FakeDev:
  slm_per_thread, sass_version = 1024, 0x0c

  def __init__(self, qmd_ver):
    self.qmd_ver = qmd_ver

  def _ensure_has_local_memory(self, size):
    raise AssertionError('this kernel has no stack, the shim must not ask for local memory')


class FakeNVG:
  NVCEC0_QMDV05_00_QMD_TYPE_GRID_CTA = 0
  NVC6C0_QMDV03_00_CWD_MEMBAR_TYPE_L1_SYSMEMBAR = 0


class FakeModule:
  def __init__(self, base):
    self.base = base


def encode(shim, qmd_ver, param_size, cbuf0_init=b''):
  """Build one launch record the way a real launch does, through the shim's own _encode."""
  ctx = shim.Context.__new__(shim.Context)   # no device, driver or memory: only what _encode reads
  ctx.dev = FakeDev(qmd_ver)
  ctx._enc_cache = {}
  ctx.shmem_window, ctx.lmem_window = 0x7f0000000000, 0x7e0000000000
  ctx.qmd_sz = 1024
  ctx.nvg = FakeNVG()
  kernel = shim.Kernel(name='probe', module=FakeModule(PROG_BASE), text_off=PROG_OFF, text_size=0x100, regs=24,
                       shared=0, param_base=0x160, param_size=param_size, cbuf0_init=cbuf0_init)
  q, cb0, _ = ctx._encode(kernel, (1, 1, 1), (128, 1, 1), 0, bytes(param_size))
  return q, cb0, kernel


def check_size(shim, qmd_ver, param_size, cbuf0_init=b''):
  q, cb0, _ = encode(shim, qmd_ver, param_size, cbuf0_init)
  need = len(cb0)   # 256-byte aligned, so the 16-byte count is exact
  got = q.constant_buffer_bytes(0)
  if got == need:
    return True, ''
  times = '%.0fx' % (got / need) if need else '?'
  return False, 'the GPU is told the constant buffer is %d bytes for a %d-byte buffer (%s too large)' % (got, need, times)


CASES = [
    dict(name='an sm_120 launch reports the constant buffer size in 16-byte units', run=lambda s: check_size(s, 5, 16)),
    dict(name='a v3 launch reports the constant buffer size in 16-byte units', run=lambda s: check_size(s, 3, 16)),
    dict(name='a larger constant block scales in 16-byte units too',
         run=lambda s: check_size(s, 5, 16, bytes(1600))),
    dict(name='the constant buffer is still marked valid', run=lambda s: check_valid(s)),
    dict(name='the program address is still written shifted4', run=lambda s: check_program(s)),
]


def check_valid(shim):
  q, _, _ = encode(shim, 5, 16)
  got = q.regs.get('constant_buffer_valid_0')
  return (got == 1), '' if got == 1 else 'constant_buffer_valid_0 is %r, expected 1' % (got,)


def check_program(shim):
  q, _, kernel = encode(shim, 5, 16)
  prog = kernel.module.base + kernel.text_off
  want = ((prog >> 4) & 0xffffffff, prog >> 36)
  got = (q.regs.get('program_address_lower_shifted4'), q.regs.get('program_address_upper_shifted4'))
  return (got == want), '' if got == want else 'the program address registers are %r, expected %r' % (got, want)


def main():
  if not os.path.isfile(os.path.join(ROOT, SHIM)):
    print('FAIL run this test from the copy root: ' + SHIM + ' is not in ' + ROOT)
    return 1
  try:
    shim = load_shim()
  except Exception as e:   # a stub or import problem must be a visible failure, never a traceback
    print('FAIL cannot import ' + SHIM + ' against the stubs: %r' % (e,))
    return 1
  failures = 0
  for case in CASES:
    ok, detail = case['run'](shim)
    print(('PASS ' if ok else 'FAIL ') + case['name'] + (': ' + detail if detail else ''))
    failures += 0 if ok else 1
  print('%d case(s) failed' % failures if failures else 'all cases passed')
  return 1 if failures else 0


if __name__ == '__main__':
  sys.exit(main())
