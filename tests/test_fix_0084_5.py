#!/usr/bin/env python3
"""tests/test_fix_0084_5.py - an oversized copy-engine submission must be refused, never overrun the pushbuffer.

sidecar-shim/tgcu.py AsyncCopy._submit wraps pb_head once and then writes the whole command batch into the 1 MiB
pushbuffer. A batch larger than the 0xF0000-byte window between 0x10000 and 0x100000 still overruns it after the single
reset, which a very large _h2d_pipe batch can produce. These cases load the shim with tinygrad stubbed out, build the
copy engine without hardware and drive _submit against a fixed size host allocation:
  - a small submission lands at pb_head and moves it by the rounded length;
  - a submission that crosses the end wraps once and lands at 0x10000;
  - a submission of exactly the 0xF0000-byte window is accepted and ends at 0x100000;
  - a submission one word past the window is refused before any write and leaves pb_head and the buffer untouched.
Prints one PASS or FAIL line per case and exits nonzero on any FAIL. Run: python3 tests/test_fix_0084_5.py
"""
import array
import importlib.util
import os
import sys
import types

SHIM = os.path.join('sidecar-shim', 'tgcu.py')
PB_START = 0x10000
PB_END = 0x100000
WINDOW = PB_END - PB_START


class PushBuffer:
  """the cpu view of the 1 MiB allocation: a slice write past the end is an overrun, not a resize"""

  def __init__(self, size:int=PB_END):
    self.size = size
    self.data = bytearray(size)
    self.overruns = []

  def __setitem__(self, key, value):
    if isinstance(key, slice):
      start = 0 if key.start is None else key.start
      stop = self.size if key.stop is None else key.stop
      if start < 0 or stop > self.size:
        self.overruns.append((start, stop))
        raise IndexError('write [%#x, %#x) past the %#x-byte pushbuffer' % (start, stop, self.size))
    self.data[key] = value

  def __getitem__(self, key):
    return self.data[key]


def load_shim():
  """import sidecar-shim/tgcu.py with tinygrad stubbed out: no GPU, no model, no network"""
  if not os.path.isfile(SHIM):
    return None
  tg = types.ModuleType('tinygrad')
  tg.Device = object
  dev = types.ModuleType('tinygrad.device')
  dev.Buffer = dev.BufferSpec = dev.BufferStorage = object
  helpers = types.ModuleType('tinygrad.helpers')
  helpers.round_up = lambda x, amt: (x + amt - 1) // amt * amt
  helpers.data64_le = lambda *a, **k: b''
  sys.modules['tinygrad'] = tg
  sys.modules['tinygrad.device'] = dev
  sys.modules['tinygrad.helpers'] = helpers
  spec = importlib.util.spec_from_file_location('tgcu_0084_5', SHIM)
  mod = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = mod
  spec.loader.exec_module(mod)
  return mod


def make_copy(mod, pb_head:int=PB_START):
  """an AsyncCopy with the hardware __init__ skipped: only the fields _submit touches"""
  c = mod.AsyncCopy.__new__(mod.AsyncCopy)
  c.va = 0x7F0000000000
  c.mv = PushBuffer()
  c.put = 0
  c.pb_head = pb_head
  c.gpput_off = c.ENTRIES * 8 + 4
  c.token = 0
  c.ctx = types.SimpleNamespace(dev=types.SimpleNamespace(gpu_mmio=[0] * 0x100))
  return c


def cmds_bytes(cmds):
  return array.array('I', cmds).tobytes()


def case_small(mod):
  c = make_copy(mod)
  cmds = [0x11111111, 0x22222222, 0x33333333]
  c._submit(cmds)
  want = cmds_bytes(cmds)
  got = bytes(c.mv[PB_START:PB_START + len(want)])
  ok = got == want and c.pb_head == 0x10100 and c.put == 1
  return ok, 'head=%#x put=%d payload=%s' % (c.pb_head, c.put, 'match' if got == want else 'wrong')


def case_wrap(mod):
  c = make_copy(mod, pb_head=PB_END - 0x80)
  cmds = [0x01020304 + i for i in range(0x80)]
  c._submit(cmds)
  want = cmds_bytes(cmds)
  got = bytes(c.mv[PB_START:PB_START + len(want)])
  ok = got == want and c.pb_head == PB_START + 0x200 and c.put == 1
  return ok, 'head=%#x payload=%s' % (c.pb_head, 'match' if got == want else 'wrong')


def case_window(mod):
  c = make_copy(mod)
  cmds = [0x5A5A5A5A] * (WINDOW // 4)
  c._submit(cmds)
  want = cmds_bytes(cmds)
  got = bytes(c.mv[PB_START:PB_END])
  ok = got == want and c.pb_head == PB_END and c.put == 1 and not c.mv.overruns
  return ok, 'head=%#x payload=%s overruns=%d' % (c.pb_head, 'match' if got == want else 'wrong', len(c.mv.overruns))


def case_oversized(mod):
  c = make_copy(mod)
  cmds = [0xA5A5A5A5] * (WINDOW // 4 + 1)   # one word past the window
  try:
    c._submit(cmds)
  except RuntimeError as e:
    if c.pb_head != PB_START or c.put != 0 or c.mv.overruns:
      return False, 'refused after touching the ring: head=%#x put=%d overruns=%d' % (c.pb_head, c.put, len(c.mv.overruns))
    return True, 'refused before the write: %s' % e
  except IndexError as e:
    return False, 'the %#x-byte batch was written past the %#x-byte pushbuffer: %s' % (len(cmds) * 4, PB_END, e)
  return False, 'the %#x-byte batch was accepted into a %#x-byte window' % (len(cmds) * 4, WINDOW)


CASES = [
  ('a small submission lands at pb_head and advances it', case_small),
  ('a submission past the end wraps once and lands at 0x10000', case_wrap),
  ('a submission of exactly the 0xF0000 window is accepted', case_window),
  ('a submission one word past the window is refused before the write', case_oversized),
]


def main():
  mod = load_shim()
  if mod is None:
    print('FAIL run this test from the copy root: %s is missing' % SHIM)
    return 1
  failed = 0
  for name, fn in CASES:
    try:
      ok, detail = fn(mod)
    except Exception as e:   # an unexpected exception is a failure, not a crash of the run
      ok, detail = False, '%s: %s' % (type(e).__name__, e)
    print(('PASS ' if ok else 'FAIL ') + name + (': ' + detail if detail else ''))
    failed += 0 if ok else 1
  print('%d case(s) failed' % failed if failed else 'all cases passed')
  return 1 if failed else 0


if __name__ == '__main__':
  sys.exit(main())
