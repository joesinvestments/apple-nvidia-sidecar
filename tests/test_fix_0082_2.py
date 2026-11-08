#!/usr/bin/env python3
"""tests/test_fix_0082_2.py - an eager launch batch larger than the VRAM ring must be refused.

sidecar-shim/tgcu.py:544 wraps the VRAM ring when the next batch does not fit the tail: head goes to 0 and the
allocation continues. A batch bigger than VRING_SIZE therefore gets offset 0 while the head is set past the end of
the ring, and the following h2d writes the whole batch into a VRING_SIZE buffer (a VRAM overflow). The allocator
must refuse such a batch before it wraps.

The module is loaded with tinygrad stubbed out (no GPU, no weights, no network). _vring_alloc runs on a Context made
with object.__new__, whose malloc and _wait are stand-ins, so only the ring arithmetic is judged. No source text is
read: every case calls the allocator and checks the offset, the head and the waits it returns.
Prints one PASS or FAIL line per case and exits nonzero on any FAIL. Run: python3 tests/test_fix_0082_2.py
"""
import importlib.util
import os
import sys
import types

ROOT = os.getcwd() if os.path.isfile(os.path.join(os.getcwd(), 'sidecar-shim', 'tgcu.py')) else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RING = 0x1000          # a small ring, so a planted batch fits in one line of thought and the real 64 MiB ring is not needed
BASE = 0x100000        # stand-in address for the ring's VRAM buffer
results = []


def round_up(x, a):
  return (x + a - 1) // a * a


def load_tgcu():
  """Import sidecar-shim/tgcu.py with tinygrad's device and helpers stubbed: only the ring arithmetic runs."""
  tinygrad = types.ModuleType('tinygrad')
  tinygrad.Device = object
  device = types.ModuleType('tinygrad.device')

  class Buffer: pass
  class BufferSpec: pass
  class BufferStorage: pass

  device.Buffer, device.BufferSpec, device.BufferStorage = Buffer, BufferSpec, BufferStorage
  helpers = types.ModuleType('tinygrad.helpers')
  helpers.round_up, helpers.data64_le = round_up, lambda *xs: b''.join(int(x).to_bytes(8, 'little') for x in xs)
  sys.modules.update({'tinygrad': tinygrad, 'tinygrad.device': device, 'tinygrad.helpers': helpers})
  path = os.path.join(ROOT, 'sidecar-shim', 'tgcu.py')
  spec = importlib.util.spec_from_file_location('tgcu_under_test', path)
  mod = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = mod   # dataclasses reads its own module's namespace back
  spec.loader.exec_module(mod)
  return mod


def fresh(mod, sem_value=7):
  """A Context with no ring yet, a stand-in malloc and a recorded _wait."""
  ctx = object.__new__(mod.Context)
  ctx.VRING_SIZE = RING
  ctx.sem_value = sem_value
  ctx.vring_peak = ctx.vring_wraps = ctx.vring_bytes = 0
  mallocs, waits = [], []

  def malloc(size):
    mallocs.append(size)
    return BASE

  ctx.malloc = malloc
  ctx._wait = lambda value, timeout_s=10.0: waits.append(value)
  return ctx, mallocs, waits


def case(name, ok, detail=''):
  results.append(bool(ok))
  print(('PASS ' if ok else 'FAIL ') + name + (': ' + detail if detail else ''))


def attempt(ctx, size):
  """Call the allocator. Returns (offset or None, exception name or None)."""
  try:
    return ctx._vring_alloc(size), None
  except Exception as e:
    return None, type(e).__name__


def main():
  mod = load_tgcu()

  ctx, mallocs, waits = fresh(mod)
  off = ctx._vring_alloc(0x400)
  case('a small batch sits at the head of a fresh ring',
       off == BASE and ctx.vring_head == 0x400 and ctx.vring_peak == 0x400 and waits == [] and ctx.vring_wraps == 0 and mallocs == [RING],
       f'offset {off - BASE:#x}, head {ctx.vring_head:#x}, waits {waits}, mallocs {mallocs}')

  ctx, mallocs, waits = fresh(mod)
  off = ctx._vring_alloc(RING)
  case('a batch that exactly fills the ring is allowed',
       off == BASE and ctx.vring_head == RING and waits == [] and ctx.vring_wraps == 0,
       f'offset {off - BASE:#x}, head {ctx.vring_head:#x}, waits {waits}')

  ctx, mallocs, waits = fresh(mod)
  first, second = ctx._vring_alloc(0x800), ctx._vring_alloc(0x800)
  wrapped = ctx._vring_alloc(0x100)
  case('a later batch wraps only when the tail does not fit, and waits first',
       (first, second, wrapped) == (BASE, BASE + 0x800, BASE) and waits == [7] and ctx.vring_wraps == 1 and ctx.vring_head == 0x100,
       f'offsets {first - BASE:#x},{second - BASE:#x},{wrapped - BASE:#x}, waits {waits}, head {ctx.vring_head:#x}')

  ctx, mallocs, waits = fresh(mod, sem_value=9)
  off, err = attempt(ctx, RING + 1)
  case('a batch one byte larger than the ring is refused, not wrapped',
       err is not None,
       f'refused with {err}' if err else
       f'it returned offset {off - BASE:#x} and set head {ctx.vring_head:#x} for a {RING + 1:#x}-byte batch in a {RING:#x}-byte ring')

  ctx, mallocs, waits = fresh(mod)
  _, err = attempt(ctx, 4 * RING)
  off = ctx._vring_alloc(0x400)
  case('a refused oversized batch leaves the ring usable',
       err is not None and off == BASE and ctx.vring_head == 0x400 and waits == [] and ctx.vring_wraps == 0,
       f'refused with {err}, then offset {off - BASE:#x}, head {ctx.vring_head:#x}, waits {waits}')

  failed = results.count(False)
  print(f'{len(results) - failed}/{len(results)} cases pass')
  return 1 if failed else 0


if __name__ == '__main__':
  sys.exit(main())
