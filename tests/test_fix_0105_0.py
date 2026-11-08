"""tests/test_fix_0105_0.py - the shim hands 32-bit x86 builds the compiler's own <smmintrin.h> (fix-0105#0).

    python3 tests/test_fix_0105_0.py

sidecar-shim/x86compat/smmintrin.h stands in for the compiler's <smmintrin.h> only on a host that has none: on x86
it must include_next the compiler header, and only a non-x86 host takes the SIMDe fallback. sidecar-shim/macbuild.py
puts that folder first on the include path of every Strata compile in the macOS build. The guard used to name only
the 64-bit x86 macros, so a 32-bit x86 build (__i386__ or _M_IX86) fell through to the shim's own immintrin.h, and
every intrinsic in that translation unit came from the fallback instead of the compiler header that exists there.

Each case plants two small header trees in a temp folder: a compiler <smmintrin.h> that sets the origin to 0 and a
fallback <simde/x86/avx512.h> plus <stdint.h> that set it to 1. The program returns the origin, so a built program
says which header the shim actually selected. The architecture macros are forced on the command line and -nostdinc
keeps every real system header out, so no case depends on the host's architecture and nothing outside the temp
folder is read. No server, model or hardware is touched. Every case prints one PASS or FAIL line; any FAIL exits
non-zero.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path.cwd() if (Path.cwd() / "sidecar-shim" / "x86compat" / "smmintrin.h").is_file() else Path(__file__).resolve().parents[1]
INC = ROOT / "sidecar-shim" / "x86compat"
CC = shutil.which("cc") or shutil.which("clang") or shutil.which("gcc")
FORCE_OFF = ["-U__x86_64__", "-U_M_X64", "-U__i386__", "-U_M_IX86"]   # no case inherits the host's architecture
ORIGIN = "#define SCE_SMMINTRIN_ORIGIN"

# the compiler header the shim must reach on x86, and the fallback it must reach everywhere else
COMPILER_HEADER = """\
#pragma once
#define SCE_SMMINTRIN_ORIGIN 0
"""
FALLBACK_HEADERS = {
    "simde/x86/avx512.h": """\
#pragma once
#define SCE_SMMINTRIN_ORIGIN 1
typedef int simde__m512i;
typedef int simde__m128i;
typedef unsigned long long simde__mmask64;
static inline void simde_mm512_storeu_si512(void* p, simde__m512i v) { (void) p; (void) v; }
static inline simde__m512i simde_mm512_loadu_si512(void const* p) { (void) p; return 0; }
static inline void simde_mm_storeu_si128(void* p, simde__m128i v) { (void) p; (void) v; }
""",
    "simde/x86/fma.h": "#pragma once\n",
    "simde/x86/f16c.h": "#pragma once\n",
    "simde/x86/clmul.h": "#pragma once\n",
    "stdint.h": """\
#pragma once
typedef unsigned long long uint64_t;
typedef unsigned short uint16_t;
typedef int int32_t;
typedef long long int64_t;
""",
}
PROGRAM = """\
#include <smmintrin.h>

int main(void) { return SCE_SMMINTRIN_ORIGIN; }
"""

results: list[bool] = []


def case(name, ok, detail=""):
    results.append(bool(ok))
    print(("PASS " if ok else "FAIL ") + name + (f": {detail}" if detail else ""))


def first_line(out):
    lines = [l.strip() for l in (out or "").splitlines() if l.strip()]
    return lines[0][:220] if lines else ""


def plant(tmp):
    """Write the compiler tree and the fallback tree; return both roots."""
    compiler = Path(tmp) / "compiler"
    fallback = Path(tmp) / "fallback"
    compiler.mkdir(exist_ok=True)
    (compiler / "smmintrin.h").write_text(COMPILER_HEADER, encoding="utf-8")
    for rel, text in FALLBACK_HEADERS.items():
        p = fallback / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    return compiler, fallback


def origin(tmp, name, defines):
    """Build and run one translation unit; return (origin the program returned, first error line)."""
    compiler, fallback = plant(tmp)
    src = Path(tmp) / f"{name}.c"
    src.write_text(PROGRAM, encoding="utf-8")
    exe = Path(tmp) / f"{name}.bin"
    cmd = [CC, "-std=c11", "-nostdinc", *FORCE_OFF, *defines, "-I", str(INC), "-I", str(compiler), "-I", str(fallback),
           str(src), "-o", str(exe)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        return None, first_line(r.stderr or r.stdout)
    r = subprocess.run([str(exe)], capture_output=True, text=True)
    return r.returncode, first_line(r.stderr or r.stdout)


def main():
    if not CC:
        case("a C compiler is available", False, "no cc, clang or gcc on PATH")
        return 1
    if not (INC / "smmintrin.h").is_file():
        case("the smmintrin shim is present", False, str(INC / "smmintrin.h"))
        return 1
    with tempfile.TemporaryDirectory(prefix="fix-0105-0-") as tmp:
        # 1 and 2: a 32-bit x86 build has a compiler <smmintrin.h>, so the shim must include_next it.
        for name, define in (("gcc32", ["-D__i386__"]), ("msvc32", ["-D_M_IX86"])):
            got, err = origin(tmp, name, define)
            case(f"a 32-bit x86 build ({define[0][2:]}) gets the compiler <smmintrin.h>",
                 got == 0, err or f"origin={got}")

        # 3 and 4: the x86-64 builds the guard already named keep the compiler header (the same answer before and
        # after the fix).
        for name, define in (("gcc64", ["-D__x86_64__"]), ("msvc64", ["-D_M_X64"])):
            got, err = origin(tmp, name, define)
            case(f"an x86-64 build ({define[0][2:]}) still gets the compiler <smmintrin.h>",
                 got == 0, err or f"origin={got}")

        # 5: a non-x86 host has no compiler <smmintrin.h>, so the SIMDe fallback stays the one that runs (the same
        # answer before and after the fix).
        got, err = origin(tmp, "nonx86", [])
        case("a non-x86 build still takes the SIMDe fallback", got == 1, err or f"origin={got}")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
