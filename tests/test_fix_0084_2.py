#!/usr/bin/env python3
"""tests/test_fix_0084_2.py - the copy engine binds the object class it allocated.

sidecar-shim/tgcu.py:850-863 (AsyncCopy.__init__): TGCU_CE_CLASS picks the class of the copy object and rm_alloc is
called with that class (_cls), but the NVC6C0_SET_OBJECT command submitted on the channel used iface.dma_class.  With
the override set the class the channel decodes later NVC6B5_* methods under (SET_OBJECT) is not the class the copy
object was allocated with.  With the override unset _cls is iface.dma_class, so the command is the same as before.

No GPU, no tinygrad, no model: tinygrad is stubbed, the NV interface records allocations and the command stream, and
the AsyncCopy constructor runs against a fake device.  Every case prints one PASS or FAIL line and any FAIL exits
nonzero.  Run: python3 tests/test_fix_0084_2.py
"""
from __future__ import annotations

import ctypes
import importlib.util
import os
import sys
import types
from pathlib import Path

ROOT = Path.cwd() if (Path.cwd() / "sidecar-shim" / "tgcu.py").is_file() else Path(__file__).resolve().parents[1]
TGCU = ROOT / "sidecar-shim" / "tgcu.py"

DMA_CLASS = 0xB5          # the interface's default copy object class
OVERRIDE = 0xB8           # TGCU_CE_CLASS: a different class, the case the finding names
SET_OBJECT = 0x6C0        # NVC6C0_SET_OBJECT, as the fake nv_gpu exposes it
FAILS = 0
SUBMITTED: list[list[int]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    global FAILS
    print(f"{'PASS' if ok else 'FAIL'} {name}" + ("" if ok else f": {detail}"))
    if not ok:
        FAILS += 1


def stub_tinygrad() -> None:
    """The shim imports tinygrad at module scope; the NV device is never touched here."""
    tg = types.ModuleType("tinygrad")
    tg.Device = {}
    device = types.ModuleType("tinygrad.device")
    device.Buffer = type("Buffer", (), {})
    device.BufferSpec = type("BufferSpec", (), {})
    device.BufferStorage = type("BufferStorage", (), {})
    helpers = types.ModuleType("tinygrad.helpers")
    helpers.round_up = lambda x, a: (x + a - 1) // a * a
    helpers.data64_le = lambda v, n=8: int(v).to_bytes(n, "little")
    sys.modules.update({"tinygrad": tg, "tinygrad.device": device, "tinygrad.helpers": helpers})


def load_shim():
    stub_tinygrad()
    spec = importlib.util.spec_from_file_location("tgcu_ce_class_under_test", str(TGCU))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


class Mem:
    """An allocation: a device VA, a host view and the RM handle the shim reads."""

    def __init__(self, size: int):
        self.buf = 0x100000000 + size
        self.host = bytearray(size)
        self.meta = types.SimpleNamespace(hMemory=0x200000000 + size)


def nvm(nargs: int, method: int, *args) -> list[int]:
    """One NV method: the length, the method code, then its words (the shape the shim submits)."""
    return [nargs, method] + [int(a) & 0xFFFFFFFF for a in args]


def nv_gpu() -> types.SimpleNamespace:
    g = types.SimpleNamespace(
        NV2080_ENGINE_TYPE_COPY0=0x1F,
        KEPLER_CHANNEL_GROUP_A=0xA0,
        FERMI_CONTEXT_SHARE_A=0xA1,
        NV_CTXSHARE_ALLOCATION_FLAGS_SUBCONTEXT_ASYNC=1,
        NVA06C_CTRL_CMD_GPFIFO_SCHEDULE=0x6C,
        NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN=0x36F,
        NVC6C0_SET_OBJECT=SET_OBJECT,
        AmpereAControlGPFifo=types.SimpleNamespace(GPPut=types.SimpleNamespace(offset=0x88)),
    )
    g.NV_CHANNEL_GROUP_ALLOCATION_PARAMETERS = lambda **kw: dict(kw)
    g.NV_CTXSHARE_ALLOCATION_PARAMETERS = lambda **kw: dict(kw)
    g.NV_CHANNELGPFIFO_ALLOCATION_PARAMETERS = lambda **kw: dict(kw)
    g.NVA06C_CTRL_GPFIFO_SCHEDULE_PARAMS = lambda **kw: dict(kw)
    g.NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN_PARAMS = lambda **kw: dict(kw)
    return g


class Iface:
    """The NV interface: records the rm_alloc calls, hands out allocations and a submit token."""

    dma_class = DMA_CLASS
    gpfifo_class = 0xC0

    def __init__(self):
        self.rm_allocs: list[tuple[object, int, object]] = []

    def alloc(self, size: int, **kw) -> Mem:
        return Mem(size)

    def rm_alloc(self, parent, cls: int, params):
        self.rm_allocs.append((parent, cls, params))
        return 0x100 + len(self.rm_allocs)

    def rm_control(self, obj, cmd: int, params):
        return types.SimpleNamespace(workSubmitToken=7)


def install_fakes(tgcu) -> None:
    """Point the shim at the fake NV surface and record every command list it submits."""
    tgcu.onv = types.SimpleNamespace(ctypes=ctypes, nv_gpu=nv_gpu(), nvm=nvm, nv_flags=lambda *a, **k: 0)
    original_submit = tgcu.AsyncCopy._submit

    def recording_submit(self, cmds):
        SUBMITTED.append(list(cmds))
        return original_submit(self, cmds)

    tgcu.AsyncCopy._submit = recording_submit


ENV_KEYS = ("TGCU_CE_CLASS", "TGCU_CE_INDEX", "TGCU_CE_PVER", "TGCU_TS_G", "TGCU_TS_C")


def run_ctor(tgcu, ce_class: str | None) -> dict:
    """Build one AsyncCopy with TGCU_CE_CLASS set (or unset) and report the class it allocated and bound."""
    saved = {k: os.environ.get(k) for k in ENV_KEYS}
    for k in ENV_KEYS:
        os.environ.pop(k, None)
    if ce_class is not None:
        os.environ["TGCU_CE_CLASS"] = ce_class
    SUBMITTED.clear()
    try:
        iface = Iface()
        dev = types.SimpleNamespace(nvdevice=0x10, vaspace=0x20, channel_group=0x30, gpu_mmio=[0] * 64)
        ctx = types.SimpleNamespace(dev=dev, iface=iface, _ace_mem=Mem(0x100000))
        tgcu.AsyncCopy(ctx)
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    allocated = iface.rm_allocs[-1][1] if iface.rm_allocs else None
    bound = None
    for cmds in SUBMITTED:
        if len(cmds) >= 3 and cmds[1] == SET_OBJECT:
            bound = cmds[2]
    return {"allocated": allocated, "bound": bound, "submits": len(SUBMITTED)}


def main() -> int:
    tgcu = load_shim()
    install_fakes(tgcu)

    cases = [
        ("the default object class is allocated and bound", None, DMA_CLASS),
        ("the TGCU_CE_CLASS override is the class the channel binds", f"{OVERRIDE:#x}", OVERRIDE),
        ("an override equal to the interface class keeps the default command", f"{DMA_CLASS:#x}", DMA_CLASS),
        ("a second override is bound as allocated too", "0xbb", 0xBB),
    ]
    for name, env_value, want in cases:
        got = run_ctor(tgcu, env_value)
        ok = got["allocated"] == want and got["bound"] == want
        check(name, ok, f"allocated {got['allocated']!r}, SET_OBJECT {got['bound']!r}, want {want:#x}")

    print(f"{len(cases)} cases, {FAILS} failed")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
