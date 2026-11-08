#!/usr/bin/env python3
"""tests/test_fix_0080_3.py - the cubin loader must read the per-function stack size (fix-0080#3).

    python3 tests/test_fix_0080_3.py

sidecar-shim/tgcu.py reads EI_MIN_STACK_SIZE, EI_MAX_STACK_SIZE and EI_FRAME_SIZE only inside
`if len(d) >= 8 and struct.unpack_from("<I", d)[0] == symidx`, the shape of a cubin-wide .nv.info entry that carries a
symbol index. nvcc also writes these three attributes per function in .nv.info.<name>, with a 4 byte payload and no
symbol index, and the .nv.info.<name> loop handled only EI_KPARAM_INFO, EI_PARAM_CBANK and EI_NUM_BARRIERS. Kernel.stack
then stayed 0, and the launch never sized the shader local memory for the kernel's stack (tgcu.py: `if k.stack:
dev._ensure_has_local_memory(k.stack + 0x240)`).

These cases build small ELF cubins by hand (no GPU, no model, no weights, no nvcc) and hand them to the real Module
loader over a stub context that only counts the image; the Kernel it registers is what the launch would size from.
Every case prints one PASS or FAIL line and any FAIL exits non-zero.
"""
import importlib.util
import struct
import sys
import types
from pathlib import Path

ROOT = Path.cwd() if (Path.cwd() / "sidecar-shim" / "tgcu.py").is_file() else Path(__file__).resolve().parents[1]
TGCU = ROOT / "sidecar-shim" / "tgcu.py"

SHT_PROGBITS, SHT_SYMTAB, SHT_STRTAB = 1, 2, 3
SHF_ALLOC, SHF_EXECINSTR = 0x2, 0x4
EIFMT_SVAL = 0x04
SYM_K, SYM_EXTERN = 1, 2          # .symtab entries: the kernel "k", then an undefined symbol
SYM_UNDEF = 0x10                  # STB_GLOBAL | STT_NOTYPE

results = []


def check(name, got, want):
  ok = got == want
  print(("PASS " if ok else "FAIL ") + name + ("" if ok else ": the loader gave %r, expected %r" % (got, want)))
  results.append(ok)


def stub_tinygrad():
  """tgcu imports tinygrad for the device it wraps; the cubin loader never touches it, so stand in for it."""
  tinygrad = types.ModuleType("tinygrad")
  tinygrad.Device = object
  device = types.ModuleType("tinygrad.device")
  device.Buffer = device.BufferSpec = device.BufferStorage = object
  helpers = types.ModuleType("tinygrad.helpers")
  helpers.round_up = lambda x, amt=1: (x + amt - 1) // amt * amt
  helpers.data64_le = lambda v: struct.pack("<II", v & 0xffffffff, v >> 32)
  sys.modules.update({"tinygrad": tinygrad, "tinygrad.device": device, "tinygrad.helpers": helpers})


def load_tgcu():
  stub_tinygrad()
  spec = importlib.util.spec_from_file_location("tgcu_under_test", str(TGCU))
  mod = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = mod
  spec.loader.exec_module(mod)
  return mod


def sval(attr_id, payload):
  """One .nv.info entry in EIFMT_SVAL form: the attribute id and its sized payload."""
  return struct.pack("<BBH", EIFMT_SVAL, attr_id, len(payload)) + payload


def word(value):
  """A 4 byte attribute payload: no symbol index, the shape nvcc writes in .nv.info.<name>."""
  return struct.pack("<I", value)


def entry(symidx, value):
  """A cubin-wide .nv.info payload: the symbol index it applies to, then the value."""
  return struct.pack("<II", symidx, value)


def build_cubin(global_info=b"", per_fn_info=b""):
  """A minimal 64 bit cubin: .shstrtab, .strtab, .symtab, .text.k, .nv.info and .nv.info.k."""
  sections = [  # (name, type, flags, body, link, info, align, entsize)
    ("", 0, 0, b"", 0, 0, 0, 0),
    (".shstrtab", SHT_STRTAB, 0, b"", 0, 0, 1, 0),          # body filled in below
    (".strtab", SHT_STRTAB, 0, b"\0k\0extern\0", 0, 0, 1, 0),
    (".symtab", SHT_SYMTAB, 0, b"", 2, 1, 8, 24),           # body filled in below
    (".text.k", SHT_PROGBITS, SHF_ALLOC | SHF_EXECINSTR, b"\x00" * 16, 0, 0, 128, 0),
    (".nv.info", SHT_PROGBITS, 0, global_info, 0, 0, 4, 0),
    (".nv.info.k", SHT_PROGBITS, 0, per_fn_info, 0, 0, 4, 0),
  ]
  shstr = b"\0" + b"\0".join(s[0].encode() for s in sections[1:]) + b"\0"
  sections[1] = (".shstrtab", SHT_STRTAB, 0, shstr, 0, 0, 1, 0)
  sections[3] = (".symtab", SHT_SYMTAB, 0,
                 b"\0" * 24 + struct.pack("<IBBHQQ", 1, 0x12, 0, 4, 0, 16) + struct.pack("<IBBHQQ", 3, SYM_UNDEF, 0, 0, 0, 0),
                 2, 1, 8, 24)
  names = {s[0]: shstr.index(s[0].encode()) for s in sections[1:]}
  out = bytearray(64)
  offs = []
  for name, typ, flags, body, link, info, align, entsize in sections[1:]:
    out += b"\0" * ((align - len(out) % align) % align)
    offs.append(len(out))
    out += body
  out += b"\0" * ((8 - len(out) % 8) % 8)
  shoff = len(out)
  out += struct.pack("<IIQQQQIIQQ", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
  for (name, typ, flags, body, link, info, align, entsize), off in zip(sections[1:], offs):
    out += struct.pack("<IIQQQQIIQQ", names[name], typ, flags, 0, off, len(body), link, info, align, entsize)
  out[0:4] = b"\x7fELF"
  out[4:7] = bytes((2, 1, 1))                              # 64 bit, little endian, version 1
  struct.pack_into("<Q", out, 0x28, shoff)
  struct.pack_into("<HHH", out, 0x3a, 64, len(sections), 1)  # shentsize, shnum, shstrndx
  return bytes(out)


class StubContext:
  """The loader lays the image out through these two calls; no GPU is opened and no VRAM exists."""
  def __init__(self):
    self.externs = {}
    self.image = None

  def code_alloc(self, size):
    return 0x10000000

  def write_vram(self, base, data):
    self.image = data


def kernel_of(tgcu, global_info=b"", per_fn_info=b""):
  return tgcu.Module(StubContext(), build_cubin(global_info, per_fn_info)).kernels["k"]


def main():
  tgcu = load_tgcu()

  # the harm: a cubin that carries its stack size only in .nv.info.<name>, the per-function section
  k = kernel_of(tgcu, per_fn_info=sval(tgcu.EI_MIN_STACK_SIZE, word(384)))
  check("a per-function min stack size reaches the kernel", k.stack, 384)

  k = kernel_of(tgcu, per_fn_info=b"".join((sval(tgcu.EI_MIN_STACK_SIZE, word(256)),
                                            sval(tgcu.EI_FRAME_SIZE, word(128)),
                                            sval(tgcu.EI_MAX_STACK_SIZE, word(384)))))
  check("the largest of the per-function min stack, frame and max stack wins", k.stack, 384)

  k = kernel_of(tgcu, per_fn_info=sval(tgcu.EI_MAX_STACK_SIZE, word(136)))
  check("a per-function max stack size alone reaches the kernel", k.stack, 136)

  # ordinary input, the same before and after: a cubin-wide entry keeps its symbol index meaning
  k = kernel_of(tgcu, global_info=b"".join((sval(tgcu.EI_MIN_STACK_SIZE, entry(SYM_K, 512)),
                                            sval(tgcu.EI_REGCOUNT, entry(SYM_K, 40)))))
  check("a cubin-wide .nv.info entry still sizes the stack and the registers", (k.stack, k.regs), (512, 40))

  # the symbol index of a cubin-wide entry is not a stack size, and another symbol's entry is not this kernel's
  k = kernel_of(tgcu, global_info=sval(tgcu.EI_MIN_STACK_SIZE, entry(SYM_EXTERN, 4096)))
  check("a cubin-wide entry for another symbol leaves the stack at zero", k.stack, 0)

  # ordinary input: a kernel with no stack attribute at all keeps the default
  k = kernel_of(tgcu)
  check("a kernel with no stack attribute keeps the stack at zero", (k.stack, k.regs, k.barriers), (0, 0, 1))

  # the per-function loop keeps parsing what it already parsed, next to the stack
  k = kernel_of(tgcu, per_fn_info=b"".join((sval(tgcu.EI_KPARAM_INFO, struct.pack("<IHHI", 0, 0, 0x210, 0x40 << 18)),
                                            sval(tgcu.EI_PARAM_CBANK, struct.pack("<IHH", 0, 0x210, 16)),
                                            sval(tgcu.EI_NUM_BARRIERS, struct.pack("<H", 8)),
                                            sval(tgcu.EI_MAX_STACK_SIZE, word(768)))))
  check("the per-function parameters, cbank and barriers still parse beside the stack",
        (k.params, k.param_base, k.param_size, k.barriers, k.stack), ([(0, 0x210, 0x40)], 0x210, 16, 8, 768))

  # ordinary input: the stack is the largest seen, never the last one read
  k = kernel_of(tgcu, global_info=sval(tgcu.EI_MIN_STACK_SIZE, entry(SYM_K, 512)),
                per_fn_info=sval(tgcu.EI_MIN_STACK_SIZE, word(384)))
  check("the stack is the larger of the cubin-wide and the per-function size", k.stack, 512)

  fails = results.count(False)
  print("%d of %d cases pass" % (len(results) - fails, len(results)))
  return 1 if fails else 0


if __name__ == "__main__":
  sys.exit(main())
