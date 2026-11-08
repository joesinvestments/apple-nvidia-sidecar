"""tests/test_fix_0087_1.py - a static VRAM mirror must fit the card (fix-0087#1).

    python3 tests/test_fix_0087_1.py

sidecar-shim/tgcu_api.py map_host() copies a mapped host region larger than MAPPED_MAX into VRAM once and returns
the device address; above the size limit it returns 0 and the program keeps its no-alias path. The limit used to be
the fixed constant STATIC_MAX = int(os.getenv("TGCU_STATIC_MAX", str(1536 << 20))) built at import, so on a card with
less VRAM (or less free VRAM) a region under that limit reached ctx.malloc(size) and the allocation failed or
thrashed. The fix keeps TGCU_STATIC_MAX as an override and otherwise derives the default from the card's reported
VRAM (a quarter of it, capped at 1536 MiB).

The module is loaded from the tree under test with a stub tgcu package, and the card, its allocator and host memory
are fakes: no GPU, no model, no network, temp state only. Every case prints one PASS or FAIL line; any FAIL exits
non-zero.

tgcu_api.py itself needs Python 3.12 or newer (its debug prints use a backslash inside an f-string, PEP 701), so when
the caller is older this file re-runs itself under the first python3.12+ on PATH that can import numpy.
"""
from __future__ import annotations

import contextlib
import ctypes
import importlib.util
import mmap
import os
import shutil
import subprocess
import sys
import types
from pathlib import Path

ROOT = Path.cwd() if (Path.cwd() / "sidecar-shim" / "tgcu_api.py").is_file() else Path(__file__).resolve().parents[1]
API = ROOT / "sidecar-shim" / "tgcu_api.py"
MIB = 1 << 20
RIG_VRAM_GIB = 32                   # TGCU_DEV_MEM_GIB in the selected config
ENV_KEYS = ("TGCU_DEV_MEM_GIB", "TGCU_MAPPED_MAX", "TGCU_STATIC_MAX")
results: list[bool] = []


def case(name, ok, detail=""):
    results.append(bool(ok))
    print(("PASS " if ok else "FAIL ") + name + (f": {detail}" if detail else ""))


def newer_python():
    """None when this interpreter can load tgcu_api.py, else the exit code of a run under python3.12+"""
    if sys.version_info >= (3, 12):
        return None
    if os.environ.get("SCE0087_REEXEC"):
        case("run under Python 3.12 or newer", False, f"re-exec landed on {sys.version.split()[0]} again")
        return 1
    for name in ("python3.12", "python3.13", "python3.14", "python3.15", "python3.16"):
        exe = shutil.which(name)
        if exe is None:
            continue
        probe = subprocess.run([exe, "-c", "import sys, numpy; sys.exit(0 if sys.version_info >= (3, 12) else 1)"],
                               capture_output=True)
        if probe.returncode:
            continue
        env = dict(os.environ, SCE0087_REEXEC="1")
        return subprocess.run([exe, os.path.abspath(__file__)], env=env).returncode
    case("find Python 3.12 or newer with numpy", False, f"this is {sys.version.split()[0]} and no newer python3.x worked")
    return 1


class FakeGpu:
    """the parts of tgcu.Context map_host() and mem_info() touch, with a VRAM budget that refuses too large a request"""

    def __init__(self, avail):
        self.avail = avail
        self.mallocs = []
        self.writes = []
        self.freed = []
        self.chunks = []            # mem_info() sums these; it reports the card's total, which is what the default uses
        self.dev = None             # the shim asks the device first and falls back to TGCU_DEV_MEM_GIB
        self.GUARD = 0
        self.vmm_phys = 0
        self.mapped_va = 1 << 40
        self.next_va = 1 << 44
        self.mapped_mv = bytearray()
        self.ace = types.SimpleNamespace(PIPE_MIN=0, h2d=self._h2d)

    def _h2d(self, va, data):
        self.writes.append((va, len(data)))

    def malloc(self, size):
        self.mallocs.append(size)
        if size > self.avail:
            raise MemoryError(f"fake card cannot allocate {size} bytes with {self.avail} free")
        self.avail -= size
        va, self.next_va = self.next_va, self.next_va + size
        return va

    def write_vram(self, va, data):
        if va <= 0:
            raise ValueError(f"write_vram to a null device address ({len(data)} bytes)")
        self.writes.append((va, len(data)))

    def free(self, va):
        self.freed.append(va)

    def mapped_alloc(self, size):
        off = len(self.mapped_mv)
        self.mapped_mv.extend(b"\0" * size)
        return self.mapped_va + off, off


class HostRegion:
    """anonymous host memory the module can map: enough pages for _mv() to read, no file touched"""

    def __init__(self, size):
        self.buf = mmap.mmap(-1, size)
        self.addr = ctypes.addressof(ctypes.c_char.from_buffer(self.buf))
        self.size = size

    def close(self):
        self.buf.close()


def load_api():
    sys.path.insert(0, str(API.parent))     # the shim imports its own folder's modules (tgcu, geom) beside it
    tgcu = types.ModuleType("tgcu")
    tgcu.Context = object       # the module only annotates ctx with it, so the stub never builds a device
    sys.modules["tgcu"] = tgcu
    spec = importlib.util.spec_from_file_location("tgcu_api_under_test", API)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@contextlib.contextmanager
def fake_card(dev_gib, avail, static_max=None, mapped_max=8 * MIB):
    saved = {k: os.environ.get(k) for k in ENV_KEYS}
    try:
        os.environ["TGCU_DEV_MEM_GIB"] = str(dev_gib)
        os.environ["TGCU_MAPPED_MAX"] = str(mapped_max)
        os.environ.pop("TGCU_STATIC_MAX", None)
        if static_max is not None:
            os.environ["TGCU_STATIC_MAX"] = str(static_max)
        mod = load_api()        # loaded here so an import-time constant sees this case's environment too
        mod.ctx = FakeGpu(avail)
        yield mod, mod.ctx
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def small_card_oversized_mirror():
    # a 320 MiB table on a 1 GiB card: under the old fixed 1536 MiB limit, over a quarter of the card, and more
    # than the 128 MiB the engine still has free, so the mirror allocation the old default allowed fails outright
    with fake_card(1, 128 * MIB) as (mod, gpu):
        host = HostRegion(320 * MIB)
        try:
            dva = mod.map_host(host.addr, 320 * MIB)
            if dva != 0:
                return False, f"map_host returned {dva:#x}, expected 0 (the no-alias path)"
            if gpu.mallocs:
                return False, f"tried VRAM allocations {gpu.mallocs} on a 1 GiB card"
            if host.addr in mod.static_mirrors:
                return False, "registered a static mirror"
            return True, ""
        finally:
            host.close()


def small_region_still_mapped():
    # ordinary input: a region under MAPPED_MAX keeps the BAR-visible mapped path before and after the fix
    with fake_card(1, 128 * MIB) as (mod, gpu):
        host = HostRegion(4 * MIB)
        try:
            dva = mod.map_host(host.addr, 4 * MIB)
            ok = dva != 0 and host.addr in mod.mapped and host.addr in mod.shadow and not gpu.mallocs
            return ok, "" if ok else "a region under MAPPED_MAX no longer takes the mapped (BAR) path"
        finally:
            host.close()


def large_table_on_rig_card():
    # ordinary input on the selected 32 GiB card: a 64 MiB table still gets one static mirror and one allocation
    with fake_card(RIG_VRAM_GIB, 20 * 1024 * MIB) as (mod, gpu):
        host = HostRegion(64 * MIB)
        try:
            dva = mod.map_host(host.addr, 64 * MIB)
            ok = dva != 0 and host.addr in mod.static_mirrors and gpu.mallocs == [64 * MIB]
            return ok, "" if ok else (f"a 64 MiB table on a {RIG_VRAM_GIB} GiB card lost its static mirror "
                                      f"(dva={dva:#x}, mallocs={gpu.mallocs})")
        finally:
            host.close()


def override_raises_default():
    # TGCU_STATIC_MAX is still an override: 400 MiB lets the same 320 MiB table through on the small card
    with fake_card(1, 512 * MIB, static_max=400 * MIB) as (mod, gpu):
        host = HostRegion(320 * MIB)
        try:
            dva = mod.map_host(host.addr, 320 * MIB)
            ok = dva != 0 and host.addr in mod.static_mirrors and gpu.mallocs == [320 * MIB]
            return ok, "" if ok else "TGCU_STATIC_MAX no longer overrides the device-derived default"
        finally:
            host.close()


def override_lowers_default():
    # TGCU_STATIC_MAX is still an override: 16 MiB rejects the 64 MiB table on the big card
    with fake_card(RIG_VRAM_GIB, 20 * 1024 * MIB, static_max=16 * MIB) as (mod, gpu):
        host = HostRegion(64 * MIB)
        try:
            dva = mod.map_host(host.addr, 64 * MIB)
            ok = dva == 0 and not gpu.mallocs
            return ok, "" if ok else f"TGCU_STATIC_MAX=16 MiB still allocated {gpu.mallocs}"
        finally:
            host.close()


def run_case(name, fn):
    try:
        ok, detail = fn()
    except Exception as e:      # the harm itself can be the exception: report it as a failing case, keep going
        ok, detail = False, f"{type(e).__name__}: {e}"
    case(name, ok, detail)


def main():
    if not API.is_file():
        case("sidecar-shim/tgcu_api.py is present", False, str(API))
    else:
        run_case("small card refuses a 320 MiB mirror it cannot hold", small_card_oversized_mirror)
        run_case("region under MAPPED_MAX keeps its mapped path", small_region_still_mapped)
        run_case(f"{RIG_VRAM_GIB} GiB card still mirrors a 64 MiB table", large_table_on_rig_card)
        run_case("TGCU_STATIC_MAX can raise the default", override_raises_default)
        run_case("TGCU_STATIC_MAX can lower the default", override_lowers_default)
    print(f"{sum(results)}/{len(results)} cases passed")
    return 0 if results and all(results) else 1


if __name__ == "__main__":
    code = newer_python()
    sys.exit(main() if code is None else code)
