# tgcu: a minimal CUDA-runtime backend on top of tinygrad's NV device (RTX 3090 behind an ASM2464 USB3 bridge).
# tinygrad boots the GPU (GSP, VA space, channels); this module loads nvcc cubins itself, builds QMDs, chains launches into
# batches and submits them on tinygrad's compute ring. Bulk copies go through tinygrad's USB copy path.
from __future__ import annotations
import struct, re, os, time, array, sys
_WAIT_S = float(os.getenv("TGCU_WAIT_S", "0"))       # diagnostic: raise the GPU-wait timeout (the default is the caller's, 10 s)
RING_HOOK = None                                    # set by tgcu_api under TGCU_LAUNCH_RING: prints the last eager launches with their device pointers resolved
HANG_HOOK = None                                    # set by tgcu_api: prints the handshake regions
_HANG_DUMP = os.getenv("TGCU_HANG_DUMP") in ("1", "2")      # diagnostic: dump every thread + a native `sample` 8 s into a GPU wait
_FREE_DEFER = os.getenv("TGCU_FREE_DEFER", "0") != "0"   # cudaFree queues the pending work and hands the block back once it has completed, instead of a full synchronize per free (the lazy x shadows of the prompt lane: 6.9 ms x 1 per chunk-layer)
_TOP_RESERVE = int(os.getenv("TGCU_TOP_RESERVE_MIB", "0")) << 20   # the top part of the physical pool the GPU cannot use: a memset of physical 24346..24410 MiB hung the 3090 (test_vram_top.py); the pool ends at 24512 MiB, so 167 MiB.  0 = off
_BESTFIT = os.getenv("TGCU_BESTFIT", "0") != "0"       # malloc takes the smallest free hole that fits (first fit left no 112 MiB hole after a few prompts -> a new chunk every group)
from dataclasses import dataclass, field
from tinygrad import Device
from tinygrad.device import Buffer, BufferSpec, BufferStorage
from tinygrad.helpers import round_up, data64_le

class _LazyNV:  # tinygrad's NV module loads on first use: importing it at module scope would pull in more than a bare import of this shim needs
  def __init__(self): self._mod = None
  def __getattr__(self, name:str):
    if self._mod is None:
      import tinygrad.runtime.ops_nv as mod
      self._mod = mod
    return getattr(self._mod, name)
onv = _LazyNV()

DEBUG = int(os.getenv("TGCU_DEBUG", "0"))

# *** cubin loading ***

SHT_PROGBITS, SHT_SYMTAB, SHT_NOBITS, SHT_REL, SHT_RELA, SHF_ALLOC = 1, 2, 8, 9, 4, 0x2
# .nv.info attribute ids (EIATTR_*), checked against cuobjdump -elf
EI_PARAM_CBANK, EI_KPARAM_INFO, EI_MIN_STACK_SIZE, EI_FRAME_SIZE, EI_MAX_STACK_SIZE, EI_REGCOUNT = 0x0a, 0x17, 0x12, 0x11, 0x23, 0x2f
EI_NUM_BARRIERS = 0x4c

@dataclass
class Section: idx:int; name:str; type:int; flags:int; off:int; size:int; link:int; info:int; align:int; entsize:int; addr:int=-1 # noqa: E702

@dataclass
class Kernel:
  name: str
  module: "Module"
  text_off: int
  text_size: int
  regs: int = 0
  shared: int = 0                      # static shared memory
  params: list = field(default_factory=list)   # [(ordinal, offset, size)]
  param_base: int = 0x160
  param_size: int = 0
  cbuf0_init: bytes = b""
  constbufs: dict = field(default_factory=dict) # bank -> (image offset, size), banks > 0
  stack: int = 0
  barriers: int = 1
  max_dyn_shared: int = 48 << 10

class Module:
  def __init__(self, ctx:"Context", elf:bytes):
    self.ctx, self.elf = ctx, elf
    assert elf[:4] == b"\x7fELF", "not an ELF cubin"
    shoff, = struct.unpack_from("<Q", elf, 0x28); shentsize, shnum, shstrndx = struct.unpack_from("<HHH", elf, 0x3a)
    raw = [struct.unpack_from("<IIQQQQIIQQ", elf, shoff + i * shentsize) for i in range(shnum)]
    strtab = raw[shstrndx][4]
    def cstr(o): return elf[o:elf.index(0, o)].decode()
    self.secs = [Section(i, cstr(strtab + r[0]), r[1], r[2], r[4], r[5], r[6], r[7], r[8], r[9]) for i, r in enumerate(raw)]
    self.by_name = {s.name: s for s in self.secs}

    # lay out every allocated section (code, constants, globals, nobits) in one image
    img = bytearray()
    for s in self.secs:
      if not (s.flags & SHF_ALLOC) or s.type not in (SHT_PROGBITS, SHT_NOBITS) or s.size == 0: continue
      img += b"\0" * (round_up(len(img), max(s.align, 128)) - len(img))
      s.addr = len(img)
      img += elf[s.off:s.off + s.size] if s.type == SHT_PROGBITS else b"\0" * s.size
    self.image = bytearray(img.ljust(round_up(len(img), 0x1000) + 0x1000, b"\0"))  # slack for instruction prefetch

    symsec = next(s for s in self.secs if s.type == SHT_SYMTAB)
    symstr = self.secs[symsec.link]
    self.syms = []
    for i in range(symsec.size // 24):
      st_name, st_info, st_other, st_shndx, st_value, st_size = struct.unpack_from("<IBBHQQ", elf, symsec.off + i * 24)
      self.syms.append((cstr(symstr.off + st_name), st_info, st_shndx, st_value, st_size))

    # the image goes to VRAM first, so relocations can use the final addresses
    self.size = len(self.image)
    self.base = ctx.code_alloc(self.size)
    self._relocate()
    ctx.write_vram(self.base, bytes(self.image))

    self.kernels: dict[str, Kernel] = {}
    self.funcs: dict[str, int] = {}  # global functions by address (what other modules may call)
    self.globals: dict[str, tuple[int, int]] = {}  # name -> (device address, size) for __device__/__constant__ variables
    for name, info, shndx, value, size in self.syms:
      if shndx == 0 or shndx >= len(self.secs) or not name: continue
      sec = self.secs[shndx]
      if (info & 0xf) == 1 and sec.addr >= 0: self.globals[name] = (self.base + sec.addr + value, size)  # STT_OBJECT
      if (info & 0xf) == 2 and (info >> 4) in (0, 1) and sec.name == f".text.{name}": self.kernels[name] = self._kernel(name, sec)  # global FUNC
      if (info & 0xf) == 2 and (info >> 4) == 1 and sec.addr >= 0: self.funcs[name] = self.base + sec.addr + value

  def sym_addr(self, symidx:int) -> int:
    name, info, shndx, value, size = self.syms[symidx]
    if shndx == 0:  # a driver-provided function (vprintf, __assertfail): tgcu's own libc module
      if name in self.ctx.externs: return self.ctx.externs[name]
      raise RuntimeError(f"cubin references undefined symbol {name}")
    sec = self.secs[shndx]
    assert sec.addr >= 0, f"symbol {name} in unplaced section {sec.name}"
    return self.base + sec.addr + value

  def _relocate(self):
    for s in self.secs:
      if s.type not in (SHT_REL, SHT_RELA): continue
      tgt = self.secs[s.info]
      if tgt.addr < 0: continue # debug sections
      esz = 24 if s.type == SHT_RELA else 16
      for i in range(s.size // esz):
        r_off, r_info = struct.unpack_from("<QQ", self.elf, s.off + i * esz)
        addend = struct.unpack_from("<q", self.elf, s.off + i * esz + 16)[0] if s.type == SHT_RELA else 0
        typ, S = r_info & 0xffffffff, self.sym_addr(r_info >> 32) + addend
        p = tgt.addr + r_off
        if typ == 1: struct.pack_into("<I", self.image, p, S & 0xffffffff)                            # R_CUDA_32
        elif typ == 2: struct.pack_into("<Q", self.image, p, S)                                      # R_CUDA_64
        elif typ == 0x38: struct.pack_into("<I", self.image, p + 4, S & 0xffffffff)                  # R_CUDA_ABS32_LO_32
        elif typ == 0x39: struct.pack_into("<I", self.image, p + 4, (S >> 32) & 0xffffffff)          # R_CUDA_ABS32_HI_32
        elif typ == 0x3a:                                                                            # R_CUDA_ABS47_34: (S >> 2) at bits [34, 81)
          assert S < (1 << 49) and S % 4 == 0, f"ABS47_34 target {S:#x}"
          w0, w1 = struct.unpack_from("<QQ", self.image, p)
          w0 = (w0 & ~(((1 << 30) - 1) << 34)) | (((S >> 2) & ((1 << 30) - 1)) << 34)
          w1 = (w1 & ~((1 << 17) - 1)) | ((S >> 32) & ((1 << 17) - 1))
          struct.pack_into("<QQ", self.image, p, w0, w1)
        else: raise RuntimeError(f"unsupported CUDA relocation type {typ:#x} in {s.name}")

  def _info(self, sec:Section):
    o, end = sec.off, sec.off + sec.size
    while o < end:
      fmt, attr, sz = struct.unpack_from("<BBH", self.elf, o)
      if fmt == 0x04: yield attr, self.elf[o + 4:o + 4 + sz]; o += 4 + sz  # EIFMT_SVAL: sized payload
      else: yield attr, struct.pack("<H", sz); o += 4                      # EIFMT_NVAL / HVAL: value in the size field

  def _kernel(self, name:str, text:Section) -> Kernel:
    k = Kernel(name, self, text.addr, text.size)
    symidx = next(i for i, s in enumerate(self.syms) if s[0] == name and s[2] == text.idx)
    if (g:=self.by_name.get(".nv.info")) is not None:
      for attr, d in self._info(g):
        if len(d) >= 8 and struct.unpack_from("<I", d)[0] == symidx:
          v = struct.unpack_from("<I", d, 4)[0]
          if attr == EI_REGCOUNT: k.regs = v
          elif attr in (EI_MIN_STACK_SIZE, EI_MAX_STACK_SIZE, EI_FRAME_SIZE): k.stack = max(k.stack, v)
    if (s:=self.by_name.get(f".nv.info.{name}")) is not None:
      for attr, d in self._info(s):
        if attr == EI_KPARAM_INFO:
          _, ordinal, off, flags = struct.unpack_from("<IHHI", d)
          k.params.append((ordinal, off, (flags >> 18) & 0x3fff))
        elif attr == EI_PARAM_CBANK: _, k.param_base, k.param_size = struct.unpack_from("<IHH", d)
        elif attr == EI_NUM_BARRIERS and len(d) == 2: k.barriers = max(1, struct.unpack("<H", d)[0])
        elif attr in (EI_MIN_STACK_SIZE, EI_MAX_STACK_SIZE, EI_FRAME_SIZE) and len(d) >= 4: k.stack = max(k.stack, struct.unpack_from("<I", d)[0])   # per function: a 4 byte value, no symbol index
    k.params.sort()
    if (s:=self.by_name.get(f".nv.shared.{name}")) is not None: k.shared = s.size
    if (s:=self.by_name.get(f".nv.constant0.{name}")) is not None: k.cbuf0_init = self.elf[s.off:s.off + s.size]
    for s in self.secs:
      if (m:=re.fullmatch(r"\.nv\.constant(\d+)(?:\.(.+))?", s.name)) and int(m.group(1)) != 0 and m.group(2) in (None, name) and s.addr >= 0:
        k.constbufs[int(m.group(1))] = (s.addr, s.size)
    return k

# *** context: memory, submission ***

_MEMLOG = os.getenv("TGCU_MEMLOG", "0") != "0"
def _memlog(ctx, ev:str, sz:int, extra:str=""):
  # TGCU_MEMLOG=1: one stderr line per malloc/free >= 4 MiB and per new chunk: chunk bytes, free bytes inside chunks, deferred frees, physical free (tinygrad TLSF)
  import sys
  mm = ctx.iface.dev_impl.mm
  def tl(a): return sum(v[0] for v in a.blocks.values() if v[3])
  chb = sum(c["size"] + ctx.GUARD for c in ctx.chunks); frb = sum(n for c in ctx.chunks for _, n in c["free"]); lfb = max([n for c in ctx.chunks for _, n in c["free"]] or [0])
  cv = mm.cpu_visible_pa_allocator
  print(f"[tgcu-mem] {ev} {sz/1048576:.2f}MiB {extra} chunks={len(ctx.chunks)} chunkMiB={chb>>20} inchunkfreeMiB={frb>>20} largestFreeMiB={lfb>>20} deferred={len(ctx.deferred)} paFreeMiB={tl(mm.pa_allocator)>>20} cvFreeMiB={(tl(cv)>>20) if cv else -1} t={time.time():.1f}", file=sys.stderr, flush=True)

class Context:
  # Decode15: the code and mapped arenas were 256 / 48 MiB but hold 3.3 / 1.3 MiB (census); TGCU_CODE_MIB / TGCU_MAPPED_MIB size them, VRAM the expert cache gets
  CTRL_SIZE, CODE_SIZE, MAPPED_SIZE = 32 << 20, int(os.getenv("TGCU_CODE_MIB", "256")) << 20, int(os.getenv("TGCU_MAPPED_MIB", "48")) << 20   # BAR1-visible pool is small
  TRANSIENT = int(os.getenv("TGCU_TRANSIENT_BYTES", str(CTRL_SIZE // 2)))   # the transient half of the control arena (test knob: a small ring wraps often)

  def __init__(self):
    self.dev = Device["NV"]
    self.iface = self.dev.iface
    # small BAR1 (256 MiB): every cpu_access VRAM arena is taken FIRST, before tinygrad's fifos (~99 MiB) and the local memory
    # (TGCU_LMEM_STACK, ~73-115 MiB) fill the window (Mac + 4080S)
    self.mapped_mem = self.iface.alloc(self.MAPPED_SIZE, cpu_access=True, force_devmem=True)
    self.ctrl = self.iface.alloc(self.CTRL_SIZE, cpu_access=True, force_devmem=True)
    self._ace_mem = self.iface.alloc(0x100000, contiguous=True, cpu_access=True, force_devmem=True)
    _ = self.dev.fifos  # brings up the compute and copy channels
    self.fifo = self.dev.fifos["COMPUTE:0"]
    self.iface = self.dev.iface
    if _TOP_RESERVE: self._claim_top()
    # Decode15: tinygrad sizes the shader local memory (per thread x 96 warps x 41 TPCs) for the largest stack seen so far and allocates a NEW buffer each time it grows: 73 MiB at boot, 115 MiB once the
    # first attention kernel (384 B of stack) launches, mid-request.  TGCU_LMEM_STACK=N sizes it for N bytes of stack now, so the allocation happens once, before the expert cache takes the VRAM
    # (a late growth needs the new buffer while the old one is still alive, the free VRAM that a small reserve no longer has)
    if (_lm := int(os.getenv("TGCU_LMEM_STACK", "0"))): self.dev._ensure_has_local_memory(_lm + 0x240)
    self.nvg = onv.nv_gpu
    # the BAR-visible mapped arena goes first: a cpu_access VRAM allocation must sit inside the 256 MiB BAR1 window, so it is
    # taken before every other VRAM allocation
    self.mapped_va, self.mapped_mv, self.mapped_head = self.mapped_mem.buf, self.mapped_mem.host, 0
    # cpu-visible control arena: queued QMDs + constant buffers, pushbuffers, semaphores
    self.ctrl_va, self.ctrl_mv = self.ctrl.buf, self.ctrl.host
    self.sem_va, self.sem_value = self.ctrl_va, 0
    self.ctrl_mv[0:8] = bytes(8)
    self.ctrl_head = 0x1000
    self.seg_size = ((self.TRANSIENT - 0x1000) // 2) & ~0xff      # the transient ring is two segments used in turn
    self.seg, self.seg_last, self.seg_touched = 0, [0, 0], set()   # current segment; the last submission (semaphore value) that read each; segments the submission being built has written
    self.allocs: dict[int, tuple[dict, int]] = {}   # va -> (chunk, bytes)
    self.deferred: list[tuple[int, int]] = []       # TGCU_FREE_DEFER: freed blocks (va, submission that last could read them) not yet handed back to the free lists
    self.chunks: list[dict] = []
    self.externs: dict[str, int] = {}  # symbols the driver would resolve, from tgcu_libc.cubin
    self.pending: list = []  # launches not yet submitted
    self.last_launch = None
    self.shmem_window, self.lmem_window = self.dev.shared_mem_window, self.dev.local_mem_window
    self.qmd_sz = round_up(onv.QMD(self.dev).sz * 4, 256)
    self.ace = AsyncCopy(self)
    self.ace.hold_va, self.ace.hold_off = self.mapped_alloc(0x1000)   # the deferred-pull hold word (zero)
    if os.getenv("TGCU_CILP"):   # lab (5090): instruction-level compute preemption with the per-context save buffers it needs
      try:                       # (NV2080_CTRL_CMD_GR_CTXSW_PREEMPTION_BIND), so the copy engine group gets its timeslice while a kernel spins
        g = self.nvg
        info = self.iface.rm_control(self.dev.subdevice, g.NV2080_CTRL_CMD_INTERNAL_STATIC_KGR_GET_CONTEXT_BUFFERS_INFO,
          g.NV2080_CTRL_INTERNAL_STATIC_KGR_GET_CONTEXT_BUFFERS_INFO_PARAMS()).engineContextBuffersInfo[0]
        E = "NV0080_CTRL_FIFO_GET_ENGINE_CONTEXT_PROPERTIES_ENGINE_ID_GRAPHICS_"
        bp = g.NV2080_CTRL_GR_CTXSW_PREEMPTION_BIND_PARAMS(flags=1, hClient=self.iface.root, hChannel=self.dev.debug_channel,
          cilpPreemptMode=g.NV2080_CTRL_SET_CTXSW_PREEMPTION_MODE_COMPUTE_CILP)
        self._preempt_bufs = []
        for slot, name in ((0, "PREEMPT"), (1, "SPILL"), (2, "PAGEPOOL"), (3, "BETACB"), (4, "RTV")):
          eid = getattr(g, E + name); sz, al = info.engine[eid].size, max(info.engine[eid].alignment, 0x1000)
          if not sz: print(f"[tgcu]   preempt buffer {name}: size 0, skipped", flush=True); continue
          b = self.iface.alloc(round_up(sz, al)); self._preempt_bufs.append(b); bp.vMemPtrs[slot] = b.buf
          print(f"[tgcu]   preempt buffer {name}: {sz} bytes (align {al:#x}) at {b.buf:#x}", flush=True)
        self.iface.rm_control(self.dev.subdevice, g.NV2080_CTRL_CMD_GR_CTXSW_PREEMPTION_BIND, bp)
        print("[tgcu] compute preemption: CILP bound with its save buffers", flush=True)
      except Exception as e: print(f"[tgcu] compute preemption CILP refused: {e!r}", flush=True)
    if os.getenv("TGCU_TSLICE_US"):   # lab (5090): channel-group timeslices, so the copy engine group gets turns while a compute kernel spins
      g = self.nvg
      for nm, grp in (("compute", self.dev.channel_group), ("copy", self.ace.group)):
        try:
          cur = self.iface.rm_control(grp, g.NVA06C_CTRL_CMD_GET_TIMESLICE, g.NVA06C_CTRL_GET_TIMESLICE_PARAMS()).timesliceUs
          self.iface.rm_control(grp, g.NVA06C_CTRL_CMD_SET_TIMESLICE, g.NVA06C_CTRL_SET_TIMESLICE_PARAMS(timesliceUs=int(os.getenv("TGCU_TSLICE_US"))))
          new = self.iface.rm_control(grp, g.NVA06C_CTRL_CMD_GET_TIMESLICE, g.NVA06C_CTRL_GET_TIMESLICE_PARAMS()).timesliceUs
          print(f"[tgcu] {nm} group timeslice {cur} -> {new} us", flush=True)
        except Exception as e: print(f"[tgcu] {nm} group timeslice refused: {e!r}", flush=True)
    self.code = self.iface.alloc(self.CODE_SIZE)   # after every cpu_access arena: they must sit inside the 256 MiB BAR1 window
    self.code_va, self.code_head = self.code.buf, 0

  # memory, CUDA-like: 256-byte aligned blocks carved from big chunks (a chunk's tail has a 2 MB guard), so a kernel
  # that reads or writes a little past its buffer touches mapped memory as it would on CUDA instead of faulting
  CHUNK, GUARD, ALIGN = 256 << 20, 2 << 20, 256
  zero_fn = None   # set by tgcu_api once its memset kernel is loaded
  _enc_cache: dict = {}
  def _release(self, va:int):
    ch, sz = self.allocs.pop(va)
    if _MEMLOG and sz >= (4 << 20): _memlog(self, "free", sz, f"chunk={self.chunks.index(ch)} off={(va-ch['base'])>>20}")
    free, off = ch["free"], va - ch["base"]
    free.append((off, sz)); free.sort()
    merged = []
    for o, n in free:  # coalesce neighbours
      if merged and merged[-1][0] + merged[-1][1] == o: merged[-1] = (merged[-1][0], merged[-1][1] + n)
      else: merged.append((o, n))
    ch["free"][:] = merged
  def _reap(self, block:bool=False):   # hand back the deferred blocks whose submission has completed (block: after a full synchronize)
    if not self.deferred: return
    if block: self.synchronize()
    done = struct.unpack("<Q", self.ctrl_mv[0:8])[0]
    keep = []
    for va, sem in self.deferred:
      if sem <= done: self._release(va)
      else: keep.append((va, sem))
    self.deferred = keep
  def _fit(self, sz:int) -> int|None:
    best = None   # (hole size, chunk, index, offset)
    for ch in self.chunks:
      for i, (off, n) in enumerate(ch["free"]):
        if n >= sz:
          if not _BESTFIT:
            best = (n, ch, i, off); break
          if best is None or n < best[0]:
            best = (n, ch, i, off)
            if n == sz: break
      if best is not None and (not _BESTFIT or best[0] == sz): break
    if best is None: return None
    n, ch, i, off = best
    free = ch["free"]
    if n == sz: free.pop(i)
    else: free[i] = (off + sz, n - sz)
    va = ch["base"] + off
    self.allocs[va] = (ch, sz)
    if _MEMLOG and sz >= (4 << 20): _memlog(self, "malloc", sz, f"chunk={self.chunks.index(ch)} off={off>>20}")
    return va
  def _claim_top(self):
    # The top _TOP_RESERVE bytes of the physical pool are not usable (a memset of physical 24346..24410 MiB hung the 3090: test_vram_top.py) and tinygrad's TLSF is a
    # smallest-fit allocator, not a bottom-up one, so merely counting the free bytes does not keep a chunk out of them: take the highest free block's tail now, while
    # the pool is pristine, and never give it back.
    a = self.iface.dev_impl.mm.pa_allocator
    start = max(s for s, v in a.blocks.items() if v[3])
    size = a.blocks[start][0]
    assert start + size == a.size and size >= _TOP_RESERVE, f"top free block {start:#x}+{size:#x} does not reach the pool end {a.size:#x} / is smaller than the reserve"
    keep = size - _TOP_RESERVE
    if keep: a._split_block(start, size, keep); start += keep
    a._remove_block(start, _TOP_RESERVE)
    self.top_claim = (start + a.base, _TOP_RESERVE)
    print(f"[tgcu] top of the physical pool claimed: {(start + a.base) / 1048576:.0f}..{(start + a.base + _TOP_RESERVE) / 1048576:.0f} MiB, never used", file=sys.stderr, flush=True)
  def _new_chunk(self, sz:int) -> int:
    csize = max(self.CHUNK, round_up(sz, 2 << 20))
    if _TOP_RESERVE and sz >= (32 << 20): csize = round_up(sz, 2 << 20)   # a big request gets a chunk of exactly its size (a 256 MiB chunk for a 112 MiB buffer strands the tail)
    if _MEMLOG: _memlog(self, "NEWCHUNK-begin", csize + self.GUARD, f"req={sz>>20}MiB")
    try: st = self.iface.alloc(csize + self.GUARD)
    except MemoryError:   # the pool is nearly full: a small request takes just what it needs
      if csize == round_up(sz, 2 << 20): raise
      csize = round_up(sz, 2 << 20)
      st = self.iface.alloc(csize + self.GUARD)
    # as from the CUDA driver, fresh device memory reads as zeros (kernels that reset their own counters rely on it)
    # (finished before the block is handed out: copy-engine writes into it, e.g. a graph's launch chain, are not queued)
    if self.zero_fn is not None: self.zero_fn(st.buf, csize + self.GUARD)
    ch = {"st": st, "base": st.buf, "size": csize, "free": [(sz, csize - sz)] if csize > sz else []}
    self.chunks.append(ch)
    self.allocs[st.buf] = (ch, sz)
    if _MEMLOG: _memlog(self, "NEWCHUNK-done", csize + self.GUARD, f"req={sz>>20}MiB")
    return st.buf
  def malloc(self, size:int) -> int:
    sz = round_up(max(size, 1), self.ALIGN)
    if self.deferred: self._reap()
    va = self._fit(sz)
    if va is not None: return va
    if self.deferred:   # no room: wait for the freed blocks before a new chunk is carved
      self._reap(True)
      va = self._fit(sz)
      if va is not None: return va
    return self._new_chunk(sz)
  def free(self, va:int):
    if _FREE_DEFER:   # cudaFree without the stall: the work submitted so far is queued now, the block is reusable once it has completed
      self.flush()
      self.deferred.append((va, self.sem_value))
      self._reap()
      return
    self.synchronize()   # cudaFree semantics: the block may be reused right away
    self._release(va)
  # *** CUDA virtual memory (cuMem*; the engine's --kv-grow / vmm.hpp, reached through tgcudart's driver entry points): an address range is reserved with nothing behind it, 2 MiB
  # physical chunks are created on their own and mapped into / unmapped from it by address.  The chunks come straight from tinygrad's physical pool (palloc) and the ranges from its
  # VA allocator, so no tgcu chunk is involved; the chunks count in mem_info (vmm_phys).  A mapped chunk is not zeroed here (the cuMemCreate contract): the caller clears what it needs.
  # PTE writes are batched: tinygrad's map_range / unmap_range cost 1.6 / 1.5 ms per 2 MiB chunk over the USB link (every PTE is read back and written alone), so cuMemMap / cuMemUnmap only
  # queue, and the queue is written to the page tables as ONE bulk write per page-table page (512 chunks) when the caller commits (cuMemSetAccess) or touches the address space in another
  # way (create / release / free / the next map or unmap): nothing can reach a queued mapping before cuMemSetAccess (the CUDA contract), and a queued unmap leaves the old page mapped
  # until the flush, which is harmless to anything that no longer uses it.
  VMM_GRAN = 2 << 20
  vmm_phys, vmm_synced = 0, -1
  vmm_maps: dict[int, int] = {}       # chunk address -> physical address, as written to the page tables
  vmm_pend_map: dict[int, int] = {}   # chunk address -> physical address: maps not written yet
  vmm_pend_unmap: list[int] = []      # chunk addresses to unmap at the next flush
  def vmm_reserve(self, size:int) -> int:
    return self.iface.dev_impl.mm.alloc_vaddr(round_up(size, self.VMM_GRAN), align=self.VMM_GRAN)
  def vmm_free_va(self, va:int):
    self.vmm_flush()
    self.iface.dev_impl.mm.va_allocator.free(va)
  def vmm_create(self) -> int:   # a handle (physical address + 1; 0 = out of memory)
    try: pa = self.iface.dev_impl.mm.palloc(self.VMM_GRAN, align=self.VMM_GRAN, zero=False)
    except MemoryError: return 0
    self.vmm_phys += self.VMM_GRAN
    return pa + 1
  def vmm_release(self, h:int):
    self.vmm_flush()   # a page that is still mapped must not go back to the pool
    if h == 0 or h - 1 in self.vmm_maps.values(): raise AssertionError(f"vmm handle {h:#x} cannot be released (0 = out of memory, or still mapped)")
    self.iface.dev_impl.mm.pfree(h - 1)
    self.vmm_phys -= self.VMM_GRAN
  def vmm_map(self, va:int, h:int):
    if va in self.vmm_pend_map or (va in self.vmm_maps and va not in self.vmm_pend_unmap): raise AssertionError(f"vmm chunk {va:#x} is already mapped")
    self.vmm_pend_map[va] = h - 1
  def vmm_unmap(self, va:int):
    if va in self.vmm_pend_map: del self.vmm_pend_map[va]; return   # never written
    if va not in self.vmm_maps or va in self.vmm_pend_unmap: raise AssertionError(f"vmm chunk {va:#x} is not mapped")
    self.vmm_pend_unmap.append(va)
  def vmm_flush(self):
    if not self.vmm_pend_unmap and not self.vmm_pend_map: return
    if self.vmm_pend_unmap:
      if self.pending or self.vmm_synced != self.sem_value:   # work queued so far must not run on a page that goes away (the engine quiesces first: this is then one cheap check)
        self.flush(); self.synchronize(); self.vmm_synced = self.sem_value
      vas = sorted(self.vmm_pend_unmap); self.vmm_pend_unmap = []
      self._vmm_write(vas, None)
      for va in vas: del self.vmm_maps[va]
    if self.vmm_pend_map:
      items = sorted(self.vmm_pend_map.items()); self.vmm_pend_map = {}
      self._vmm_write([va for va, _ in items], [pa for _, pa in items])
      self.vmm_maps.update(items)
    self.iface.dev_impl.mm.on_range_mapped()   # one GPU TLB flush for the batch
  def _vmm_write(self, vas:list[int], pas:list[int]|None):
    # Page-table entries for the chunks `vas` (sorted, 2 MiB aligned): physical `pas` (pas None: invalid entries).  The entries of a run of chunks that share a page table are
    # encoded by tinygrad's own set_entry (a stand-in page whose writes are collected) and go to VRAM in one bulk write.
    import struct as _st
    from tinygrad.runtime.support.memory import AddrSpace, PageTableTraverseContext
    from tinygrad.runtime.support.nv.nvdev import NVPageTableEntry
    dev = self.iface.dev_impl; mm = dev.mm; G = self.VMM_GRAN
    class Cap:   # what set_entry stores, by entry index
      def __init__(s, pt): s.nvdev, s.lv, s.entries = pt.nvdev, pt.lv, {}
      def _is_dual_pde(s): return s.lv == s.nvdev.mm.level_cnt - 2
    i = 0
    while i < len(vas):
      ctx = PageTableTraverseContext(dev, mm.root_page_table, vas[i], create_pts=pas is not None)
      for _, pt, idx, _cnt, covers in ctx.next(G, paddr=pas[i] if pas is not None else None): break
      assert covers == G, f"vmm chunk {vas[i]:#x}: the 2 MiB entry is at a page table level that covers {covers:#x}"
      k = 1
      while i + k < len(vas) and idx + k < mm.pte_cnt[pt.lv] and vas[i + k] == vas[i] + k * G: k += 1
      cap = Cap(pt)
      for j in range(k):
        if pas is not None: NVPageTableEntry.set_entry(cap, idx + j, pas[i + j], uncached=False, aspace=AddrSpace.PHYS, snooped=False, frag=mm._frag_size(vas[i + j], G), valid=True)
        else: NVPageTableEntry.set_entry(cap, idx + j, 0, valid=False)
      w = 2 if cap._is_dual_pde() else 1
      lo, n = w * idx, w * k
      assert sorted(cap.entries) == list(range(lo, lo + n)), "vmm: the page-table writes are not one contiguous run"
      pt.entries[lo:lo + n] = _st.pack(f"<{n}Q", *[cap.entries[lo + j] for j in range(n)])
      i += k
  def mapped_alloc(self, size:int) -> tuple[int, int]:  # (device va, offset in the BAR-visible mapped arena)
    off = self.mapped_head
    self.mapped_head = round_up(off + size, 0x1000)
    if self.mapped_head > self.MAPPED_SIZE: raise MemoryError("tgcu mapped arena full")
    return self.mapped_va + off, off
  def code_alloc(self, size:int) -> int:
    va, self.code_head = self.code_va + self.code_head, round_up(self.code_head + size, 0x1000)
    assert self.code_head <= self.CODE_SIZE, "code arena full"
    return va
  def _buf(self, va:int, size:int) -> Buffer: # external: tinygrad must never free it
    return Buffer("NV", size, opaque=BufferStorage(va), options=BufferSpec(external_ptr=va))
  def write_vram(self, va:int, data:bytes, ordered:bool|None=True):
    # ordered=True: after everything queued so far. False: before queued (unsubmitted) launches, after submitted work.
    # None: concurrently with running GPU work (a handshake into a graph that is spinning on a flag)
    if not data: return
    if ordered: self.synchronize()
    elif ordered is False and self.sem_value: self._wait(self.sem_value)
    self.ace.h2d(va, data)
  def read_vram(self, va:int, size:int, ordered:bool=True) -> bytes:
    if ordered: self.synchronize()
    return self.ace.d2h(va, size)
  def copy_d2d(self, dst:int, src:int, size:int):  # through the host for now
    self.write_vram(dst, self.read_vram(src, size))

  # launches
  def launch(self, k:Kernel, grid:tuple, block:tuple, dyn_shared:int, params:bytes):
    self.pending.append(self.encode(k, grid, block, dyn_shared, params))

  # cooperative launches (cudaLaunchCooperativeKernel): cooperative_groups grid.sync() reads the grid workspace pointer
  # from the driver words c[0][0x8c] (high 32 bits) and c[0][0x90] (low 32 bits) (sm_86 SASS: null -> BPT.TRAP; then
  # ATOM.ADD.STRONG.GPU on [ws + 4] with a top-bit generation flip, so one zeroed word serves every serialized launch)
  # and needs every block co-resident
  GRID_WS_HI, GRID_WS_LO = 0x8c, 0x90
  def num_sms(self) -> int:
    if not hasattr(self, "_num_sms"):
      nvg, dev = self.nvg, self.dev
      self._num_sms = dev.num_sm_per_tpc * sum(bin(self.iface.rm_control(dev.subdevice, nvg.NV2080_CTRL_CMD_GR_GET_TPC_MASK,
        nvg.NV2080_CTRL_GR_GET_TPC_MASK_PARAMS(gpcId=i)).tpcMask).count("1") for i in range(dev.num_gpcs))
    return self._num_sms
  SMEM_PER_SM = 100 << 10   # the 3090's shared memory per SM: what a device that reports no size of its own gets
  def smem_per_sm(self) -> int:
    # the SM's shared memory is per-architecture (100 KiB on the 3090, 228 KiB on Hopper): a card that reports less than 100 KiB
    # holds fewer co-resident blocks than a 3090, one that reports more holds more
    return getattr(self.dev, "smem_per_sm", 0) or self.SMEM_PER_SM
  BLOCKS_PER_SM = 16   # the 3090's blocks per SM: what a device that reports no limit of its own gets
  def blocks_per_sm(self) -> int:
    # the blocks one SM holds at once is per-architecture (16 on the 3090, 24 on Ada, 32 on an A100 or an H100): a card
    # whose limit is higher gets a cooperative grid the hardware would run refused as too large, one that allows fewer
    # is credited with blocks it cannot hold
    return getattr(self.dev, "max_blocks_per_sm", 0) or self.BLOCKS_PER_SM
  def coresident_blocks(self, k:Kernel, threads:int, dyn_shared:int=0) -> int:
    warps = (threads + 31) // 32
    by_regs = 65536 // (warps * round_up(max(k.regs, 1) * 32, 256))
    by_smem = self.smem_per_sm() // round_up(0x400 + k.shared + dyn_shared, 128)
    return min(self.blocks_per_sm(), self.dev.max_warps_per_sm // warps, by_regs, by_smem) * self.num_sms()
  def encode_cooperative(self, k:Kernel, grid:tuple, block:tuple, dyn_shared:int, params:bytes) -> tuple[onv.QMD, bytes, str]:
    # the record of a cooperative launch: encode() plus the grid workspace pointer (allocated and zeroed on first use,
    # which waits for the GPU: callers with mapped regions in flight push them first)
    blocks, cap = grid[0] * grid[1] * grid[2], self.coresident_blocks(k, block[0] * block[1] * block[2], dyn_shared)
    if blocks > cap: raise RuntimeError(f"{k.name}: cooperative grid {blocks} > {cap} co-resident blocks")
    if not hasattr(self, "grid_ws"):
      self.grid_ws = self.malloc(256)
      self.write_vram(self.grid_ws, bytes(256))
    q, cb0, name = self.encode(k, grid, block, dyn_shared, params)
    cb0 = bytearray(cb0)
    for hi, lo in ([(0x1b4, 0x1b8), (0x254, 0x258)] if q.ver >= 5 else [(self.GRID_WS_HI, self.GRID_WS_LO)]):  # sm_120: c[0][0x254/0x258 - 0xa0*CgaSize]
      struct.pack_into("<I", cb0, hi, self.grid_ws >> 32)
      struct.pack_into("<I", cb0, lo, self.grid_ws & 0xffffffff)
    return q, bytes(cb0), name
  def launch_cooperative(self, k:Kernel, grid:tuple, block:tuple, dyn_shared:int, params:bytes):
    self.pending.append(self.encode_cooperative(k, grid, block, dyn_shared, params))

  def encode(self, k:Kernel, grid, block, dyn_shared:int, params:bytes) -> tuple[onv.QMD, bytes]:
    # a launch's QMD and constant buffer depend on the kernel and its dimensions only, but for the parameters: built
    # once per shape (a prompt makes ~11,000 launches per chunk), then copied with the parameters dropped in
    self.last_launch = (k.name, grid, block, dyn_shared)
    key = (id(k), grid, block, dyn_shared, self.dev.slm_per_thread)
    if (t:=self._enc_cache.get(key)) is None:
      q, cb0, _ = self._encode(k, grid, block, dyn_shared, b"")
      t = self._enc_cache[key] = (bytes(q.mv), cb0)
    cb0 = bytearray(t[1])
    cb0[k.param_base:k.param_base + len(params)] = params
    if CB_STATS is not None:
      st = CB_STATS.setdefault(k.name, [0, len(cb0), len(k.cbuf0_init), k.param_base, k.param_size]); st[0] += 1
    return onv.QMD(self.dev, blob=bytearray(t[0])), bytes(cb0), k.name
  def _encode(self, k:Kernel, grid, block, dyn_shared:int, params:bytes) -> tuple[onv.QMD, bytes]:
    dev = self.dev
    shmem = round_up(0x400 + k.shared + dyn_shared, 128)
    if shmem > 100 << 10: raise RuntimeError(f"{k.name}: {shmem} bytes of shared memory")
    if k.stack: dev._ensure_has_local_memory(k.stack + 0x240)
    smem_cfg = min(c * 1024 for c in [32, 64, 100] if c * 1024 >= shmem) // 4096 + 1
    q = onv.QMD(dev); bw = q.ver >= 5
    cb0_size = round_up(max(len(k.cbuf0_init), k.param_base + k.param_size, 896 if bw else 0), 256)
    cb0 = bytearray(k.cbuf0_init.ljust(cb0_size, b"\0"))
    if bw:  # sm_120 (nvcc 13.4 SASS): c[0][0x360] blockDim xyz, c[0][0x370] gridDim xyz, c[0][0x37c] stack; windows at 0x2f0 as tinygrad's cbuf_0[188:192]
      struct.pack_into("<3I", cb0, 0x360, *block); struct.pack_into("<4I", cb0, 0x370, *grid, 0xfffdc0)
      struct.pack_into("<4I", cb0, 0x2f0, *data64_le(self.shmem_window), *data64_le(self.lmem_window))
    else:   # CUDA driver params: c[0][0x0] blockDim xyz, c[0][0xc] gridDim xyz, then the shared/local memory windows
      struct.pack_into("<6I", cb0, 0, *block, *grid)
      struct.pack_into("<6I", cb0, 24, *data64_le(self.shmem_window), *data64_le(self.lmem_window), *data64_le(0xfffdc0))
    cb0[k.param_base:k.param_base + len(params)] = params
    if bw: q.write(qmd_major_version=5, qmd_type=self.nvg.NVCEC0_QMDV05_00_QMD_TYPE_GRID_CTA, register_count=max(k.regs, 1),
      shared_memory_size_shifted7=shmem >> 7, shader_local_memory_high_size_shifted4=dev.slm_per_thread >> 4)
    else: q.write(qmd_major_version=3, sm_global_caching_enable=1, shared_memory_size=shmem, register_count_v=max(k.regs, 1),
      shader_local_memory_high_size=dev.slm_per_thread)
    q.write(qmd_group_id=0x3f, invalidate_texture_header_cache=1,
      invalidate_texture_sampler_cache=1, invalidate_texture_data_cache=1, invalidate_shader_data_cache=1, api_visible_call_limit=1,
      sampler_index=1, barrier_count=k.barriers, cwd_membar_type=self.nvg.NVC6C0_QMDV03_00_CWD_MEMBAR_TYPE_L1_SYSMEMBAR,
      constant_buffer_invalidate_0=1, min_sm_config_shared_mem_size=smem_cfg, target_sm_config_shared_mem_size=smem_cfg,
      max_sm_config_shared_mem_size=0x1a, program_prefetch_size=min(k.text_size >> 8, 0x1ff), sass_version=dev.sass_version)
    q.write(**dict(zip(q.grid, grid)),
            cta_thread_dimension0=block[0], cta_thread_dimension1=block[1], cta_thread_dimension2=block[2])
    prog = k.module.base + k.text_off
    if bw: q.write(program_address_lower_shifted4=(prog >> 4) & 0xffffffff, program_address_upper_shifted4=prog >> 36)
    else: q.write(program_address_lower=prog & 0xffffffff, program_address_upper=prog >> 32)
    q.write(program_prefetch_addr_lower_shifted=(prog >> 8) & 0xffffffff, program_prefetch_addr_upper_shifted=prog >> 40)
    q.write(constant_buffer_size_shifted4_0=cb0_size >> 4, constant_buffer_valid_0=1)
    for i, (off, sz) in k.constbufs.items():
      a = k.module.base + off
      sh = 6 if bw else 0
      q.write(**{f"constant_buffer_addr_lower{'_shifted6' if bw else ''}_{i}": (a >> sh) & 0xffffffff, f"constant_buffer_addr_upper{'_shifted6' if bw else ''}_{i}": a >> (32 + sh),
                 f"constant_buffer_size_shifted4_{i}": round_up(sz, 16) >> 4, f"constant_buffer_valid_{i}": 1})
    return q, bytes(cb0), k.name

  def _rec_size(self, r) -> int: return self.qmd_sz + round_up(len(r[1]), 256)
  def _layout(self, recs:list[tuple[onv.QMD, bytes]], base_va:int) -> bytes:
    # launches as [QMD | cbuf0] blocks at base_va, each block as long as its own cbuf0 (256-byte aligned), chained by
    # dependent-QMD pointers
    blob, off = bytearray(), 0
    for i, (q, cb, *_) in enumerate(recs):
      qva, size = base_va + off, self._rec_size((q, cb))
      cba = qva + self.qmd_sz
      if q.ver >= 5: q.write(constant_buffer_addr_lower_shifted6_0=(cba >> 6) & 0xffffffff, constant_buffer_addr_upper_shifted6_0=cba >> 38)
      else: q.write(constant_buffer_addr_lower_0=cba & 0xffffffff, constant_buffer_addr_upper_0=cba >> 32)
      if i + 1 < len(recs):
        q.write(dependent_qmd0_pointer=(qva + size) >> 8, dependent_qmd0_action=1, dependent_qmd0_prefetch=1, dependent_qmd0_enable=1)
      else: q.write(dependent_qmd0_enable=0)
      blob += bytes(q.mv).ljust(self.qmd_sz, b"\0") + cb.ljust(size - self.qmd_sz, b"\0")
      off += size
    return bytes(blob)

  VRING_SIZE, BAR_BATCH = int(os.getenv("TGCU_VRING_MIB", "64")) << 20, 4096
  vring_peak, vring_wraps, vring_bytes = 0, 0, 0   # census: the high-water mark, the wraps (each waits for everything submitted) and the bytes launched through it
  def _vring_alloc(self, size:int) -> int:
    # VRAM ring for eager launch batches too big for BAR writes; wrapping waits for everything submitted
    assert size <= self.VRING_SIZE, f"vring allocation of {size} bytes does not fit the {self.VRING_SIZE} byte ring"
    if not hasattr(self, "vring_va"): self.vring_va, self.vring_head = self.malloc(self.VRING_SIZE), 0
    if self.vring_head + size > self.VRING_SIZE:
      self._wait(self.sem_value)
      self.vring_head = 0
      self.vring_wraps += 1
    off, self.vring_head = self.vring_head, round_up(self.vring_head + size, 0x100)
    self.vring_peak = max(self.vring_peak, self.vring_head); self.vring_bytes += size
    return self.vring_va + off

  def alloc_ctrl(self, size:int) -> int:
    if not hasattr(self, "persist_head"): self.persist_head = self.CTRL_SIZE // 2
    off, self.persist_head = self.persist_head, round_up(self.persist_head + size, 0x100)
    if self.persist_head > self.CTRL_SIZE: raise MemoryError(f"tgcu control arena full ({self.persist_head >> 20} MB of graphs)")
    return off

  def _pushbuf(self, first_qmd:int|None) -> list[int]:
    nvg, nvm = self.nvg, onv.nvm
    cmds = []
    if first_qmd is not None:
      # constants, globals and code may have been written by copies since the last batch
      cmds += nvm(1, nvg.NVC6C0_INVALIDATE_SHADER_CACHES_NO_WFI,
                  onv.nv_flags("NVC6C0_INVALIDATE_SHADER_CACHES_NO_WFI", instruction="true", global_data="true", constant="true"))
      cmds += nvm(1, nvg.NVC6C0_SEND_PCAS_A, first_qmd >> 8)
      cmds += nvm(1, nvg.NVC6C0_SEND_SIGNALING_PCAS2_B, nvg.NVC6C0_SEND_SIGNALING_PCAS2_B_PCAS_ACTION_PREFETCH_SCHEDULE)
    self.sem_value += 1
    flags = onv.nv_flags("NVC56F_SEM_EXECUTE", operation="release", release_wfi="en", payload_size="64bit")
    cmds += nvm(0, nvg.NVC56F_SEM_ADDR_LO, self.sem_va & 0xffffffff, self.sem_va >> 32, self.sem_value & 0xffffffff, self.sem_value >> 32, flags)
    return cmds

  def _transient(self, size:int) -> int:
    # The transient ring is two segments used in turn.  Entering a segment waits only for the submissions that last read it - all queued a whole segment ago, so
    # never for the submission being built (its pushbuffer allocation lands here AFTER _pushbuf took its semaphore value: waiting for `sem_value` was a wait for
    # a release that cannot be queued until this returns, a 10 s GPU timeout whenever a pushbuffer crossed the end of the ring) and never for a graph that
    # still spins on a host handshake.
    off = round_up(self.ctrl_head, 0x100)
    if off + size > 0x1000 + (self.seg + 1) * self.seg_size:
      assert size <= self.seg_size, f"transient allocation of {size} bytes does not fit a {self.seg_size} byte segment"
      nxt = self.seg ^ 1
      self._wait(self.seg_last[nxt])
      self.seg, off = nxt, 0x1000 + nxt * self.seg_size
    self.seg_touched.add(self.seg)
    self.ctrl_head = off + round_up(size, 0x100)
    return off

  def _submit(self, cmds:list[int]):
    pb = array.array("I", cmds).tobytes()
    off = self._transient(len(pb))
    self.ctrl_mv[off:off + len(pb)] = pb
    fifo = self.fifo
    put = fifo.put_value.as_memoryview(allow_zero_copy=True).cast("Q")
    p = put[0]
    entry = (self.ctrl_va + off) | ((len(pb) // 4) << 42) | (1 << 41)
    ring = fifo.ring.host
    ring[(p % fifo.entries) * 8:(p % fifo.entries) * 8 + 8] = struct.pack("<Q", entry)
    put[0] = p + 1
    fifo.gpput.host[0:4] = struct.pack("<I", (p + 1) % fifo.entries)
    self.dev.gpu_mmio[0x90 // 4] = fifo.token # usermode doorbell in BAR0, a plain store
    for s in self.seg_touched: self.seg_last[s] = self.sem_value   # this submission (or a later one) is the last to read the segments it wrote
    self.seg_touched.clear()

  def flush(self, extra:list[int]|None=None):
    if not self.pending:
      if extra: self._submit(extra)
      return
    recs, self.pending = self.pending, []
    if SPLIT_BATCH and len(recs) > 1:   # debug: one pushbuffer (and a wait for idle) per launch
      for r in recs[:-1]:
        self.pending = [r]; self.flush()
      recs = recs[-1:]
    size = sum(self._rec_size(r) for r in recs)
    if size <= self.BAR_BATCH:   # a few launches: straight into the BAR-visible arena
      off = self._transient(size)
      first = self.ctrl_va + off
      self.ctrl_mv[off:off + size] = self._layout(recs, first)
    else:                        # a long batch: VRAM ring, through the async copy engine
      first = self._vring_alloc(size)
      self.ace.h2d(first, self._layout(recs, first))
    self._submit(self._pushbuf(first) + (extra or []))

  # GPU-timed events: a semaphore release that also writes the GPU's timer (ns) after all work submitted before it
  EV_SLOTS = 4096
  EVENT_SUBMIT = os.getenv("TGCU_EVENT_TIMING") == "1"   # profiling: a GPU stamp for every event
  # (on the last queued launch's own QMD release when there is one: no submission; else a pushbuffer of its own)
  ev_last = None
  def event_record(self) -> int:
    if not hasattr(self, "ev_off"): self.ev_off, self.ev_next = self.alloc_ctrl(16 * self.EV_SLOTS), 0
    if self.pending and self.ev_last is not None and self.ev_last[0] is self.pending[-1][0]: return self.ev_last[1]
    slot, self.ev_next = self.ev_next, (self.ev_next + 1) % self.EV_SLOTS
    va = self.ctrl_va + self.ev_off + 16 * slot
    if self.pending and self._qmd_timestamp(self.pending[-1][0], va):
      self.ev_last = (self.pending[-1][0], slot)
      return slot
    # nothing queued (or the launch's two releases are taken): host time - a submission per event costs ~0.65 ms
    if self.pending or not self.EVENT_SUBMIT: return -1
    flags = onv.nv_flags("NVC56F_SEM_EXECUTE", operation="release", release_wfi="en", payload_size="64bit", release_timestamp="en")
    self.flush(onv.nvm(0, self.nvg.NVC56F_SEM_ADDR_LO, va & 0xffffffff, va >> 32, 1, 0, flags))
    return slot
  @staticmethod
  def _qmd_timestamp(q, va:int) -> bool:   # a release (payload + GPU timer) when the grid completes, in a free slot
    for i in range(2):
      if q.read(f"release{i}_enable"): continue
      if q.ver >= 5: q.write(**{f"release_semaphore{i}_addr_lower": va & 0xffffffff, f"release_semaphore{i}_addr_upper": va >> 32,
                 f"release_semaphore{i}_payload_lower": 1, f"release_semaphore{i}_payload_upper": 0, f"release{i}_enable": 1,
                 f"release_structure_size_{i}": 0, f"release_payload64b_{i}": 1})
      else: q.write(**{f"release{i}_address_lower": va & 0xffffffff, f"release{i}_address_upper": va >> 32,
                 f"release{i}_payload_lower": 1, f"release{i}_payload_upper": 0, f"release{i}_enable": 1,
                 f"release{i}_structure_size": 0, f"release{i}_payload64b": 1})
      return True
    return False
  def event_time(self, slot:int) -> int: return struct.unpack_from("<Q", self.ctrl_mv[self.ev_off + 16 * slot + 8:self.ev_off + 16 * slot + 16])[0]

  def _hang_dump(self, value:int):   # TGCU_HANG_DUMP=1 (diagnostic): the Python stacks of every thread, the semaphores, and a native `sample` of this process
    import faulthandler, subprocess, sys
    self.hang_n = getattr(self, "hang_n", 0) + 1
    try:
      ce = struct.unpack("<I", self.ace.mv[self.ace.sem_off:self.ace.sem_off + 4])[0]
      print(f"[tgcu HANG] compute sem {struct.unpack('<Q', self.ctrl_mv[0:8])[0]} want {value} (submitted {self.sem_value}); CE sem {ce} (submitted {self.ace.sem_value}); last {self.last_launch}", file=sys.stderr, flush=True)
    except Exception as e: print(f"[tgcu HANG] state read failed: {e}", file=sys.stderr, flush=True)
    if os.getenv("TGCU_HANG_DUMP") == "2":   # light: semaphores + Python stacks only (no USB probes, no RM fault readout, no native sample): the full dump coincided with the mini dropping off the network
      faulthandler.dump_traceback(file=sys.stderr, all_threads=True)
      return
    try:
      off = self.ace.ENTRIES * 8 + getattr(onv.nv_gpu.AmpereAControlGPFifo, 'GPGet').offset
      print(f"[tgcu HANG] CE gpget {struct.unpack_from('<I', self.ace.mv, off)[0]} put {self.ace.put % self.ace.ENTRIES}", file=sys.stderr, flush=True)
    except Exception as e: print(f"[tgcu HANG] CE gpget failed: {e!r}", file=sys.stderr, flush=True)
    try:
      f = self.fifo; gp = onv.nv_gpu.AmpereAControlGPFifo
      goff = f.gpput.offset - getattr(gp, 'GPPut').offset + getattr(gp, 'GPGet').offset
      print(f"[tgcu HANG] compute gpget {struct.unpack_from('<I', self.dev.gpfifo_buf.host, goff)[0]} gpput {struct.unpack_from('<I', f.gpput.host, 0)[0]} put_value {f.put_value.as_memoryview(allow_zero_copy=True).cast('Q')[0]}", file=sys.stderr, flush=True)
    except Exception as e: print(f"[tgcu HANG] compute gpget failed: {e!r}", file=sys.stderr, flush=True)
    try:
      t0 = time.perf_counter(); self.ace.d2h(self.ctrl_va, 16)
      print(f"[tgcu HANG] CE probe (d2h 16 B of the control arena) ok in {(time.perf_counter() - t0) * 1e3:.1f} ms", file=sys.stderr, flush=True)
    except Exception as e: print(f"[tgcu HANG] CE probe failed: {e!r}", file=sys.stderr, flush=True)
    try:
      nvg, dev = onv.nv_gpu, self.dev
      se = dev.iface.rm_control(dev.debugger, nvg.NV83DE_CTRL_CMD_DEBUG_READ_ALL_SM_ERROR_STATES, nvg.NV83DE_CTRL_DEBUG_READ_ALL_SM_ERROR_STATES_PARAMS(hTargetChannel=dev.debug_channel, numSMsToRead=100))
      print(f"[tgcu HANG] mmuFault.valid {se.mmuFault.valid}", file=sys.stderr, flush=True)
      if se.mmuFault.valid:
        mm = dev.iface.rm_control(dev.debugger, nvg.NV83DE_CTRL_CMD_DEBUG_READ_MMU_FAULT_INFO, nvg.NV83DE_CTRL_DEBUG_READ_MMU_FAULT_INFO_PARAMS())
        for i in range(mm.count):
          pf = mm.mmuFaultInfoList[i]
          print(f"[tgcu HANG] MMU fault {i}: addr {pf.faultAddress:#x} type {pf.faultType} access {pf.accessType} engine {getattr(pf, 'faultEngine', '?')}", file=sys.stderr, flush=True)
      for i, e in enumerate(se.smErrorStateArray):
        if e.hwwGlobalEsr or e.hwwWarpEsr: print(f"[tgcu HANG] SM {i}: esr {e.hwwGlobalEsr} warp_esr {e.hwwWarpEsr:#x} pc {e.hwwWarpEsrPc64:#x}", file=sys.stderr, flush=True)
    except Exception as e: print(f"[tgcu HANG] fault readout failed: {e!r}", file=sys.stderr, flush=True)
    if HANG_HOOK is not None: HANG_HOOK()
    faulthandler.dump_traceback(file=sys.stderr, all_threads=True)
    subprocess.Popen(["/usr/bin/sample", str(os.getpid()), "4", "-file", f"/tmp/tgcu_hang_{os.getpid()}_{self.hang_n}.sample"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

  def _wait(self, value:int, timeout_s:float=10.0):
    timeout_s = max(timeout_s, _WAIT_S)
    t0 = time.perf_counter()
    dumped = False
    while struct.unpack("<Q", self.ctrl_mv[0:8])[0] < value:
      el = time.perf_counter() - t0
      if el > 8.0 and not dumped and _HANG_DUMP:
        dumped = True
        self._hang_dump(value)
      if el > timeout_s:
        report = ""
        if RING_HOOK is not None:
          try: RING_HOOK()
          except Exception as e: print(f"[tgcu RING] dump failed: {e!r}", flush=True)
        try: self.dev.on_device_hang()   # raises with the MMU fault / SM error report read through GSP
        except Exception as e: report = str(e)
        raise RuntimeError(f"GPU timeout waiting for semaphore {value} (last {self.last_launch}): {report}")

  def synchronize(self, timeout_s:float=10.0):
    self.flush()
    if self.sem_value: self._wait(self.sem_value, timeout_s)
    if KPROF and _kp_launched: kprof_collect()   # every launched graph has finished: read its completion timestamps

  # graphs: a fixed chain of launches written once, replayed with one pushbuffer
  def graph_instantiate(self, recs:list[tuple[onv.QMD, bytes]]) -> int:
    # a graph's launch chain lives in plain VRAM, written once through the async copy engine
    if KPROF:   # lab: every launch of the graph writes a GPU timestamp on completion into its own 16-byte slot of Mac DMA memory
      kp = self.iface.alloc(max((16 * len(recs) + 0x3fff) & ~0x3fff, 0x4000), host=True)
      ok = [Context._qmd_timestamp(r[0], kp.buf + 16 * i) for i, r in enumerate(recs)]
    first = self.malloc(sum(self._rec_size(r) for r in recs))
    self.ace.h2d(first, self._layout(recs, first))
    if KPROF: _kp_graphs[first] = (kp, [r[2] if len(r) > 2 else "?" for r in recs], ok)
    return first
  def graph_launch(self, first_qmd:int):
    self.flush()
    self._submit(self._pushbuf(first_qmd))
    if KPROF and first_qmd in _kp_graphs: _kp_launched.add(first_qmd)

CB_STATS: dict | None = {} if os.getenv("TGCU_CB_STATS") else None
# *** lab: TGCU_KPROF=1 per-kernel GPU time (graphs only): after a synchronize, consecutive completion timestamps of each launched graph ***
KPROF = os.getenv("TGCU_KPROF") == "1"
_kp_graphs: dict = {}; _kp_launched: set = set(); _kp_stats: dict = {}; _kp_last = [time.time()]
def kprof_collect():
  import re as _re
  for first in list(_kp_launched):
    kp, names, ok = _kp_graphs[first]
    ts = struct.unpack_from(f"<{2 * len(names)}Q", bytes(kp.host[0:16 * len(names)]))[1::2]
    for i in range(1, len(names)):
      if not (ok[i] and ok[i - 1]) or not ts[i] or not ts[i - 1] or ts[i] < ts[i - 1]: continue
      n = _re.sub(r"_GLOBAL__N__\w+?_cu_\w{8}", "", names[i])[:90]
      st = _kp_stats.setdefault(n, [0, 0]); st[0] += 1; st[1] += ts[i] - ts[i - 1]
  _kp_launched.clear()
  if time.time() - _kp_last[0] > float(os.getenv("TGCU_KPROF_EVERY", "5")) and _kp_stats:
    _kp_last[0] = time.time(); tot = sum(v[1] for v in _kp_stats.values())
    with open(os.getenv("TGCU_KPROF_OUT", "/tmp/tgcu_kprof.txt"), "w") as f:
      f.write(f"total {tot / 1e6:.1f} ms over {sum(v[0] for v in _kp_stats.values())} kernel completions\n")
      for n, (c, t) in sorted(_kp_stats.items(), key=lambda x: -x[1][1])[:80]:
        f.write(f"{100 * t / tot:6.2f}%  {t / 1e6:9.2f} ms  {c:8d} x  {t / c / 1e3:8.2f} us  {n}\n")
SPLIT_BATCH = os.getenv("TGCU_SPLIT_BATCH") == "1"

# *** fatbin ***

def fatbin_cubins(blob:bytes) -> list[tuple[int, bytes]]:
  magic, ver, hsz, fsz = struct.unpack_from("<IHHQ", blob)
  assert magic == 0xBA55ED50, f"bad fatbin magic {magic:#x}"
  out, o, end = [], hsz, hsz + fsz
  while o < end:
    kind, _, ehsz, psz = struct.unpack_from("<HHIQ", blob, o)
    arch = struct.unpack_from("<I", blob, o + 28)[0]
    flags = struct.unpack_from("<Q", blob, o + 40)[0] if ehsz >= 48 else 0
    payload = blob[o + ehsz:o + ehsz + psz]
    if kind == 2:
      if payload[:4] != b"\x7fELF": raise RuntimeError(f"compressed fatbin entry (flags {flags:#x}); build with nvcc -no-compress")
      out.append((arch, payload))
    o += ehsz + psz
  return out

def fatbin_size(blob_ptr_bytes:bytes) -> int:
  _, _, hsz, fsz = struct.unpack_from("<IHHQ", blob_ptr_bytes)
  return hsz + fsz

# Host stubs (clang, macOS) and device code (nvcc, Linux) name the same kernel differently in two ways:
#  - anonymous namespaces: clang "12_GLOBAL__N_1", nvcc "41_GLOBAL__N__<hash>_<file>_<hash>"
#  - 64-bit integers: uint64_t/int64_t are unsigned long long/long long on macOS ("y"/"x"), unsigned long/long on Linux ("m"/"l")
# canon() rewrites both into one form; identifiers (length-prefixed) are copied verbatim.
def canon(mangled:str) -> str:
  if not mangled.startswith("_Z"): return mangled
  out, i, n = ["_Z"], 2, len(mangled)
  if mangled.startswith("_ZL") and len(mangled) > 3 and mangled[3].isdigit(): i = 3  # clang marks static kernels _ZL, nvcc does not
  remap = {"y": "m", "x": "l"}
  while i < n:
    c = mangled[i]
    if c.isdigit():
      j = i
      while j < n and mangled[j].isdigit(): j += 1
      k = int(mangled[i:j])
      ident = mangled[j:j + k]
      out.append("@A@" if ident.startswith("_GLOBAL__N_") else f"{k}{ident}")
      i = j + k
    elif c in "ST" and i + 1 < n and (mangled[i + 1] == "_" or mangled[i + 1].isdigit() or mangled[i + 1].isupper()):
      j = i + 1                                              # back-reference S<seq-id>_ / template parameter T<seq-id>_
      while j < n and mangled[j] != "_": j += 1
      out.append(mangled[i:j + 1]); i = j + 1
    elif c == "L" and i + 1 < n and mangled[i + 1] != "_":  # template literal: L <type> <value> E
      j = i + 1
      t = mangled[j]; out.append("L" + remap.get(t, t)); j += 1
      while j < n and mangled[j] != "E": out.append(mangled[j]); j += 1
      i = j
    else:
      out.append(remap.get(c, c)); i += 1
  return "".join(out)

def anon_key(mangled:str) -> str: return canon(mangled)

# *** async copy engine: a channel group of its own, so copies run while a compute graph spins on a host flag ***

_PROF = bool(os.getenv("TGCU_PROF"))
_PROF_ACC = [[0.0, 0.0, 0], [0.0, 0.0, 0]]   # [h2d, d2h]: submit s, copy-engine wait s, calls
_SUB_ACC = [[0.0, 0.0, 0.0, 0.0, 0.0]]
def _prof_dump():
  import sys
  for name, p in zip(("h2d", "d2h"), _PROF_ACC):
    if p[2]: print(f"[tgcu prof] {name}: submit {p[0]/p[2]*1e3:.3f} ms, ce-wait {p[1]/p[2]*1e3:.3f} ms", file=sys.stderr, flush=True)
    for i in range(3): p[i] = 0 if i == 2 else 0.0

class AsyncCopy:
  # host memory <-> VRAM with a copy engine outside the graphics channel group: the host stages every transfer in one host
  # allocation, the copy engine moves it to or from VRAM, and the job's semaphore release is the only completion signal
  ENTRIES, CHUNK = 0x400, 0x40000   # gpfifo entries; bytes per handshake round
  STAGE_BYTES = 64 << 20            # the staging allocation: the pipelined upload uses two halves of it

  def __init__(self, ctx:"Context", engine_type:int|None=None):
    self.ctx, dev, iface, nvg = ctx, ctx.dev, ctx.iface, onv.nv_gpu
    # COPY0 in a channel group of its own: time-sliced against the graphics group (the async CE runlists do not start here)
    self.engine = engine_type if engine_type is not None else (nvg.NV2080_ENGINE_TYPE_COPY0 + int(os.getenv("TGCU_CE_INDEX", "0")))   # lab: TGCU_CE_INDEX picks COPYn
    self.stage = iface.alloc(self.STAGE_BYTES, host=True)   # host DMA memory: .buf is the GPU VA, .host the cpu view
    self.stage_va, self.stage_mv = self.stage.buf, self.stage.host
    # ring + USERD + pushbuffers in one cpu-visible allocation
    self.mem = getattr(ctx, '_ace_mem', None) or iface.alloc(0x100000, contiguous=True, cpu_access=True, force_devmem=True)  # taken early by Context
    self.va, self.mv, self.paddr = self.mem.buf, self.mem.host, self.mem.meta.hMemory
    self.mv[0:0x10000] = bytes(0x10000)
    self.notifier = iface.alloc(0x1000, uncached=True)
    # a channel group of its own on the copy engine, sharing the device's VA space, like tinygrad's graphics group
    self.group = iface.rm_alloc(dev.nvdevice, nvg.KEPLER_CHANNEL_GROUP_A, nvg.NV_CHANNEL_GROUP_ALLOCATION_PARAMETERS(engineType=self.engine))
    self.ctxshare = iface.rm_alloc(self.group, nvg.FERMI_CONTEXT_SHARE_A,
      nvg.NV_CTXSHARE_ALLOCATION_PARAMETERS(hVASpace=dev.vaspace, flags=nvg.NV_CTXSHARE_ALLOCATION_FLAGS_SUBCONTEXT_ASYNC))
    params = nvg.NV_CHANNELGPFIFO_ALLOCATION_PARAMETERS(gpFifoOffset=self.va, gpFifoEntries=self.ENTRIES,
      hObjectError=self.notifier.meta.hMemory, hObjectBuffer=self.paddr, hUserdMemory=(onv.ctypes.c_uint32*8)(self.paddr),
      userdOffset=(onv.ctypes.c_uint64*8)(self.ENTRIES * 8), engineType=self.engine, hContextShare=self.ctxshare)
    self.chan = iface.rm_alloc(self.group, iface.gpfifo_class, params)
    class _CEAllocParams(onv.ctypes.Structure): _fields_ = [("version", onv.ctypes.c_uint32), ("engineType", onv.ctypes.c_uint32)]   # NVB0B5_ALLOCATION_PARAMETERS (clb0b5sw.h)
    _pv = os.getenv("TGCU_CE_PVER", "1:type").split(":")   # lab: NVB0B5 params version and engineType form (type | inst | none)
    _ep = None if _pv[1] == "none" else _CEAllocParams(int(_pv[0]), self.engine if _pv[1] == "type" else self.engine - nvg.NV2080_ENGINE_TYPE_COPY0)
    _cls = int(os.getenv("TGCU_CE_CLASS", "0"), 0) or iface.dma_class
    iface.rm_alloc(self.chan, _cls, _ep)
    iface.rm_control(self.group, nvg.NVA06C_CTRL_CMD_GPFIFO_SCHEDULE, nvg.NVA06C_CTRL_GPFIFO_SCHEDULE_PARAMS(bEnable=1))
    self.token = iface.rm_control(self.chan, nvg.NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN,
      nvg.NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN_PARAMS(workSubmitToken=-1)).workSubmitToken
    self.gpput_off = self.ENTRIES * 8 + getattr(nvg.AmpereAControlGPFifo, 'GPPut').offset
    self.put, self.pb_head = 0, 0x10000
    self.sem_off = 0x8000                                   # semaphore word inside the same allocation
    self.sem_value = 0
    self.gate_seq = 0
    self._rd = 0                                           # bytes of the pull queued by d2h_arm / h2d_pre(nxt=...), collected by d2h_collect
    self.hold_va, self.hold_off, self.hold_tok = 0, 0, 0   # set by Context once its mapped arena exists
    self._submit(onv.nvm(4, nvg.NVC6C0_SET_OBJECT, _cls))
    # TGCU_TS_G / TGCU_TS_C (microseconds): the graphics/compute group's and the copy group's RM time slice (both 1024 by default).  A short compute slice lets the copy
    # engine run at full rate while a kernel runs (prompt lane: the y push beside the next chunk's front)
    for _grp, _ev in ((dev.channel_group, "TGCU_TS_G"), (self.group, "TGCU_TS_C")):
      if os.getenv(_ev): iface.rm_control(_grp, nvg.NVA06C_CTRL_CMD_SET_TIMESLICE, nvg.NVA06C_CTRL_SET_TIMESLICE_PARAMS(timesliceUs=int(os.environ[_ev])))

  def _submit(self, cmds:list[int]):
    pb = array.array("I", cmds).tobytes()
    if len(pb) > 0x100000 - 0x10000: raise RuntimeError(f"a {len(pb)}-byte copy-engine submission does not fit the {0x100000 - 0x10000}-byte pushbuffer window")  # the ring wraps once, so a bigger batch can never fit
    if self.pb_head + len(pb) > 0x100000: self.pb_head = 0x10000  # callers wait on each submission, so the ring is free
    off, self.pb_head = self.pb_head, round_up(self.pb_head + len(pb), 0x100)
    _s0 = time.perf_counter()
    self.mv[off:off + len(pb)] = pb
    _s1 = time.perf_counter()
    entry = (self.va + off) | ((len(pb) // 4) << 42) | (1 << 41)
    slot = self.put % self.ENTRIES
    self.mv[slot * 8:slot * 8 + 8] = struct.pack("<Q", entry)
    _s2 = time.perf_counter()
    self.put += 1
    self.mv[self.gpput_off:self.gpput_off + 4] = struct.pack("<I", self.put % self.ENTRIES)
    self.ctx.dev.gpu_mmio[0x90 // 4] = self.token
    if _PROF:
      q = _SUB_ACC[0]; q[0] += _s1 - _s0; q[1] += _s2 - _s1; q[2] += time.perf_counter() - _s2; q[3] += 1
      if q[3] % 400 == 0:
        import sys; print(f"[tgcu prof] submit parts (ms, avg of {int(q[3])}): pushbuf {q[0]/q[3]*1e3:.3f} ring {q[1]/q[3]*1e3:.3f} doorbell {q[2]/q[3]*1e3:.3f}", file=sys.stderr, flush=True)
        for i in range(4): q[i] = 0.0

  def _copy_and_signal(self, dst:int, src:int, n:int, gate:tuple[int, int]|None=None, down:bool=False):
    # gate = (staging address, 64-bit value): the copy engine channel first acquires that word (equal), so the job can be queued
    # before the payload exists; the payload's own last bytes release it.
    nvg, nvm = onv.nv_gpu, onv.nvm
    self.sem_value += 1
    sem = self.va + self.sem_off
    cmds = []
    if gate is not None:
      ga, gv = gate
      cmds += nvm(0, nvg.NVC56F_SEM_ADDR_LO, ga & 0xffffffff, ga >> 32, gv & 0xffffffff, gv >> 32,
                  onv.nv_flags("NVC56F_SEM_EXECUTE", operation="acquire", payload_size="64bit"))
    cmds += nvm(4, nvg.NVC6B5_OFFSET_IN_UPPER, src >> 32, src & 0xffffffff, dst >> 32, dst & 0xffffffff)
    cmds += nvm(4, nvg.NVC6B5_LINE_LENGTH_IN, n)
    cmds += nvm(4, nvg.NVC6B5_LAUNCH_DMA, onv.nv_flags("NVC6B5_LAUNCH_DMA", data_transfer_type="non_pipelined",
                                                         src_memory_layout="pitch", dst_memory_layout="pitch"))
    cmds += nvm(4, nvg.NVC6B5_SET_SEMAPHORE_A, sem >> 32, sem & 0xffffffff, self.sem_value & 0xffffffff)
    cmds += nvm(4, nvg.NVC6B5_LAUNCH_DMA, onv.nv_flags("NVC6B5_LAUNCH_DMA", flush_enable="true", semaphore_type="release_one_word_semaphore"))
    _t1 = time.perf_counter()
    self._submit(cmds)
    _t2 = time.perf_counter()
    t0 = time.perf_counter()
    while struct.unpack("<I", self.mv[self.sem_off:self.sem_off + 4])[0] != self.sem_value:
      if time.perf_counter() - t0 > 5: raise RuntimeError("async copy timeout")
    if _PROF:
      _t3 = time.perf_counter(); p = _PROF_ACC[1 if down else 0]
      p[0] += _t2 - _t1; p[1] += _t3 - _t2; p[2] += 1
      if p[2] % 200 == 0: _prof_dump()

  # ---- pipelined transfers. ONE pushbuffer holds the copy-engine jobs of the whole message (one doorbell), and the uploads
  # alternate between the two halves of the staging allocation, so the copy engine drains one half while the CPU fills the other.
  # Downloads queue one hold-gated job per round (a single doorbell for the message; the host releases a job with one store after
  # the previous round has been read). TGCU_PIPE_MIN (bytes, default 1 MiB; 0 = off).
  PIPE_MIN = int(os.getenv("TGCU_PIPE_MIN", str(1 << 20)))

  def _wait_sem(self, value:int, timeout_s:float=5.0):
    t0 = time.perf_counter()
    while ((struct.unpack("<I", self.mv[self.sem_off:self.sem_off + 4])[0] - value) & 0xffffffff) >= 0x80000000:
      if time.perf_counter() - t0 > timeout_s: raise RuntimeError("async copy timeout (pipelined)")

  def h2d(self, dst:int, data:bytes):
    if self.PIPE_MIN and len(data) >= self.PIPE_MIN: return self._h2d_pipe(dst, data)
    return self._h2d_rounds(dst, data)

  def d2h(self, src:int, n:int) -> bytes:
    if self.PIPE_MIN and n >= self.PIPE_MIN: return self._d2h_pipe(src, n)
    return self._d2h_rounds(src, n)

  def _h2d_pipe(self, dst:int, data:bytes):
    nvg, nvm = onv.nv_gpu, onv.nvm
    cap = self.STAGE_BYTES // 2                            # bytes per staging half
    rb = cap - 512                                         # payload per round (the gate tail rounds the wire up to the half's end)
    n = len(data)
    rounds = (n + rb - 1) // rb
    mv = memoryview(data) if not isinstance(data, memoryview) else data
    sem = self.va + self.sem_off
    cmds, plan = [], []
    for i in range(rounds):
      o = i * rb
      m = min(rb, n - o)
      wire = round_up(m + 8, 512)
      reg = i & 1
      base = self.stage_va + reg * cap                     # half 0 = the staging allocation's first half, half 1 = the second
      self.gate_seq = (self.gate_seq + 1) & 0xffffffff
      gate = 0x5A17C0DE00000000 | self.gate_seq
      self.sem_value += 1
      cmds += nvm(0, nvg.NVC56F_SEM_ADDR_LO, (base + wire - 8) & 0xffffffff, (base + wire - 8) >> 32, gate & 0xffffffff, gate >> 32,
                  onv.nv_flags("NVC56F_SEM_EXECUTE", operation="acquire", payload_size="64bit"))
      d = dst + o
      cmds += nvm(4, nvg.NVC6B5_OFFSET_IN_UPPER, base >> 32, base & 0xffffffff, d >> 32, d & 0xffffffff)
      cmds += nvm(4, nvg.NVC6B5_LINE_LENGTH_IN, m)
      cmds += nvm(4, nvg.NVC6B5_LAUNCH_DMA, onv.nv_flags("NVC6B5_LAUNCH_DMA", data_transfer_type="non_pipelined",
                                                           src_memory_layout="pitch", dst_memory_layout="pitch"))
      cmds += nvm(4, nvg.NVC6B5_SET_SEMAPHORE_A, sem >> 32, sem & 0xffffffff, self.sem_value & 0xffffffff)
      cmds += nvm(4, nvg.NVC6B5_LAUNCH_DMA, onv.nv_flags("NVC6B5_LAUNCH_DMA", flush_enable="true", semaphore_type="release_one_word_semaphore"))
      plan.append((o, m, wire, reg, gate, self.sem_value))
    self._submit(cmds)                                      # the jobs wait on their sentinels; the pushbuffer is free again once the last one has run
    smv = self.stage_mv
    for i, (o, m, wire, reg, gate, sv) in enumerate(plan):
      if i >= 2: self._wait_sem(plan[i - 2][5])             # this half's previous job has drained it
      off = reg * cap
      smv[off:off + m] = mv[o:o + m]                        # plain store into the host staging view; the copy engine reads it from there
      smv[off + wire - 8:off + wire] = struct.pack("<Q", gate)
    if plan: self._wait_sem(plan[-1][5])

  def _d2h_pipe(self, src:int, n:int, into=None) -> bytes|None:
    nvg, nvm = onv.nv_gpu, onv.nvm
    R = self.STAGE_BYTES // 2                              # download payload bytes per round (each round is read before the next one is released)
    rounds = (n + R - 1) // R
    sem, ha = self.va + self.sem_off, self.hold_va
    base_tok = self.hold_tok
    cmds, plan = [], []
    for i in range(rounds):
      o = i * R
      m = min(R, n - o)
      self.sem_value += 1
      tok = base_tok + 1 + i
      cmds += nvm(0, nvg.NVC56F_SEM_ADDR_LO, ha & 0xffffffff, ha >> 32, tok & 0xffffffff, 0,
                  onv.nv_flags("NVC56F_SEM_EXECUTE", operation="acq_strict_geq", payload_size="32bit"))   # released by the host below
      s = src + o
      cmds += nvm(4, nvg.NVC6B5_OFFSET_IN_UPPER, s >> 32, s & 0xffffffff, self.stage_va >> 32, self.stage_va & 0xffffffff)
      cmds += nvm(4, nvg.NVC6B5_LINE_LENGTH_IN, m)
      cmds += nvm(4, nvg.NVC6B5_LAUNCH_DMA, onv.nv_flags("NVC6B5_LAUNCH_DMA", data_transfer_type="non_pipelined",
                                                           src_memory_layout="pitch", dst_memory_layout="pitch"))
      cmds += nvm(4, nvg.NVC6B5_SET_SEMAPHORE_A, sem >> 32, sem & 0xffffffff, self.sem_value & 0xffffffff)
      cmds += nvm(4, nvg.NVC6B5_LAUNCH_DMA, onv.nv_flags("NVC6B5_LAUNCH_DMA", flush_enable="true", semaphore_type="release_one_word_semaphore"))
      plan.append((o, m, tok, self.sem_value))
    self.hold_tok = base_tok + rounds
    self._submit(cmds)
    hmv = self.ctx.mapped_mv
    out = into if into is not None else bytearray(n)
    for o, m, tok, sv in plan:
      hmv[self.hold_off:self.hold_off + 4] = struct.pack("<I", tok & 0xffffffff)   # release the round: the copy engine fills the staging buffer
      self._wait_sem(sv)                                    # the job's semaphore release says the staging bytes are final
      out[o:o + m] = self.stage_mv[0:m]
    return bytes(out) if into is None else None

  def d2h_into(self, src:int, n:int, into):   # download n bytes straight into a writable buffer (one copy fewer than d2h)
    self._d2h_pipe(src, n, into)

  def _h2d_rounds(self, dst:int, data:bytes):
    # the copy engine job is queued FIRST, gated by an acquire on a sentinel (the last 8 bytes of the payload in the staging
    # buffer), then the payload is stored into the staging host view; the payload's own last bytes release the copy engine.
    # (On the old bridge path the arm had to follow the submission; with plain host memory the write-then-submit order is fine.)
    step = self.STAGE_BYTES - 512                          # payload bytes per round: the staging allocation minus the gate block
    for o in range(0, len(data), step):
      chunk = data[o:o + step]
      wire = round_up(len(chunk) + 8, 512)
      self.gate_seq = (self.gate_seq + 1) & 0xffffffff
      gate = 0x5A17C0DE00000000 | self.gate_seq
      self.stage_mv[:len(chunk)] = chunk
      self.stage_mv[wire - 8:wire] = struct.pack("<Q", gate)
      self._copy_and_signal(dst + o, self.stage_va, len(chunk), gate=(self.stage_va + wire - 8, gate))

  def _d2h_rounds(self, src:int, n:int) -> bytes:
    out = bytearray()
    for o in range(0, n, self.CHUNK):
      m = min(self.CHUNK, n - o)
      self._copy_and_signal(self.stage_va, src + o, m, down=True)   # the semaphore release says the staging bytes are final
      out += self.stage_mv[0:m]
    return bytes(out)

  # ---- pre-armed handshake transfers (STRATA_HS_PRE): the copy engine pulls the doorbell payload the moment the GPU raises
  # the ring word, with no host poll / arm / submit after the ring; the y push + flag is one gated job submitted before the
  # CPU computes, so only the arm and the bulk OUT are left once the rows exist.
  def _d2h_cmds(self, src, n:int, gate_va:int, want:int, hold:int|None=None) -> list[int]:
    # src: a device VA (one copy of n bytes) or, with TGCU_PULL_PACK, a list of (va, nbytes) gathered back to back into the staging buffer (n = their total)
    nvg, nvm = onv.nv_gpu, onv.nvm
    self.sem_value += 1
    sem = self.va + self.sem_off
    cmds = []
    if hold is not None:   # the host releases this job by writing `hold` to the word
      ha = self.hold_va
      cmds += nvm(0, nvg.NVC56F_SEM_ADDR_LO, ha & 0xffffffff, ha >> 32, hold & 0xffffffff, 0,
                  onv.nv_flags("NVC56F_SEM_EXECUTE", operation="acquire", payload_size="32bit"))
    cmds += nvm(0, nvg.NVC56F_SEM_ADDR_LO, gate_va & 0xffffffff, gate_va >> 32, want & 0xffffffff, 0,
                onv.nv_flags("NVC56F_SEM_EXECUTE", operation="acq_strict_geq", payload_size="32bit"))
    pos = 0
    for sva, sn in (src if isinstance(src, list) else [(src, n)]):
      dva = self.stage_va + pos; pos += sn
      cmds += nvm(4, nvg.NVC6B5_OFFSET_IN_UPPER, sva >> 32, sva & 0xffffffff, dva >> 32, dva & 0xffffffff)
      cmds += nvm(4, nvg.NVC6B5_LINE_LENGTH_IN, sn)
      cmds += nvm(4, nvg.NVC6B5_LAUNCH_DMA, onv.nv_flags("NVC6B5_LAUNCH_DMA", data_transfer_type="non_pipelined",
                                                           src_memory_layout="pitch", dst_memory_layout="pitch"))
    cmds += nvm(4, nvg.NVC6B5_SET_SEMAPHORE_A, sem >> 32, sem & 0xffffffff, self.sem_value & 0xffffffff)
    cmds += nvm(4, nvg.NVC6B5_LAUNCH_DMA, onv.nv_flags("NVC6B5_LAUNCH_DMA", flush_enable="true", semaphore_type="release_one_word_semaphore"))
    return cmds

  def d2h_arm(self, src, n:int, gate_va:int, want:int):
    assert n <= self.CHUNK
    self._submit(self._d2h_cmds(src, n, gate_va, want))
    self._rd = n

  def d2h_release(self):
    # open the hold word: the queued (deferred) pull may now run
    self.hold_tok += 1
    self.ctx.mapped_mv[self.hold_off:self.hold_off + 4] = struct.pack("<I", self.hold_tok)

  def d2h_collect(self, timeout:int=20000) -> memoryview:
    self._wait_sem(self.sem_value, timeout_s=timeout / 1e3)
    return self.stage_mv[:self._rd]   # a view of the staging buffer, valid until the next h2d_go or d2h_collect: the caller copies it out at once

  def h2d_pre(self, dsts:list[tuple[int, int]], nxt:tuple|None=None) -> list[int]:
    # dsts = [(vram va, nbytes multiple of 4)]; the payload is their concatenation. Queue ONE gated job (acquire on the sentinel,
    # then one copy per destination, ordered by a flushing release in between); returns each part's offset in the payload.
    nvg, nvm = onv.nv_gpu, onv.nvm
    pos, o = [], 0
    for _, n in dsts: pos.append(o); o += n
    assert o <= self.STAGE_BYTES - 512
    wire = round_up(o + 8, 512)
    self.gate_seq = (self.gate_seq + 1) & 0xffffffff
    gate = 0x5A17C0DE00000000 | self.gate_seq
    self.sem_value += 1
    sem = self.va + self.sem_off
    cmds = nvm(0, nvg.NVC56F_SEM_ADDR_LO, (self.stage_va + wire - 8) & 0xffffffff, (self.stage_va + wire - 8) >> 32, gate & 0xffffffff, gate >> 32,
               onv.nv_flags("NVC56F_SEM_EXECUTE", operation="acquire", payload_size="64bit"))
    for idx, ((dva, n), p) in enumerate(zip(dsts, pos)):
      src = self.stage_va + p
      cmds += nvm(4, nvg.NVC6B5_OFFSET_IN_UPPER, src >> 32, src & 0xffffffff, dva >> 32, dva & 0xffffffff)
      cmds += nvm(4, nvg.NVC6B5_LINE_LENGTH_IN, n)
      cmds += nvm(4, nvg.NVC6B5_LAUNCH_DMA, onv.nv_flags("NVC6B5_LAUNCH_DMA", data_transfer_type="non_pipelined",
                                                           src_memory_layout="pitch", dst_memory_layout="pitch"))
      if idx + 1 < len(dsts):   # the next copy (the flag) must not pass this one (the rows)
        d = sem + 8
        cmds += nvm(4, nvg.NVC6B5_SET_SEMAPHORE_A, d >> 32, d & 0xffffffff, idx + 1)
        cmds += nvm(4, nvg.NVC6B5_LAUNCH_DMA, onv.nv_flags("NVC6B5_LAUNCH_DMA", flush_enable="true", semaphore_type="release_one_word_semaphore"))
    cmds += nvm(4, nvg.NVC6B5_SET_SEMAPHORE_A, sem >> 32, sem & 0xffffffff, self.sem_value & 0xffffffff)
    cmds += nvm(4, nvg.NVC6B5_LAUNCH_DMA, onv.nv_flags("NVC6B5_LAUNCH_DMA", flush_enable="true", semaphore_type="release_one_word_semaphore"))
    if nxt is not None:   # (src va or [(va, nbytes)], nbytes, seq gate va, want): the next ring's pull, held until d2h_release()
      src, n, gva, want = nxt
      assert n <= self.CHUNK
      cmds += self._d2h_cmds(src, n, gva, want, hold=self.hold_tok + 1)
      self._rd = n
    self._wr = (wire, gate, o)
    self._submit(cmds)
    return pos

  def h2d_go(self, parts:list[bytes]):
    wire, gate, o = self._wr
    buf = self.stage_mv
    p = 0
    for b in parts: buf[p:p + len(b)] = b; p += len(b)
    buf[wire - 8:wire] = struct.pack("<Q", gate)   # the sentinel releases the already queued copy job

# *** TGCU_HSPROF=1 (diagnostic, HostTailAndPrefill 2026-10-12): per-call wall time of the handshake path, printed every 480 rings ***
if os.getenv("TGCU_HSPROF"):
  import sys as _sys
  _HP: dict[str, list] = {}
  _ring_n = [0]
  def _wrap(obj, name, label=None):
    fn = getattr(obj, name); lab = label or name
    def w(*a, **k):
      t = time.perf_counter()
      try: return fn(*a, **k)
      finally:
        e = _HP.setdefault(lab, [0.0, 0]); e[0] += time.perf_counter() - t; e[1] += 1
    setattr(obj, name, w)
  for _n in ("h2d_pre", "h2d_go", "d2h_release", "d2h_arm", "_submit"): _wrap(AsyncCopy, _n)
  def _collect(self, timeout=20000, _f=AsyncCopy.d2h_collect):
    t = time.perf_counter()
    try: return _f(self, timeout)
    finally:
      e = _HP.setdefault("d2h_collect", [0.0, 0]); e[0] += time.perf_counter() - t; e[1] += 1
      _ring_n[0] += 1
      if _ring_n[0] % 480 == 0:
        print("[tgcu hsprof] per call ms (n): " + " | ".join(f"{k} {v[0]/v[1]*1e3:.3f} ({v[1]})" for k, v in sorted(_HP.items())), file=_sys.stderr, flush=True)
        _HP.clear()
  AsyncCopy.d2h_collect = _collect
