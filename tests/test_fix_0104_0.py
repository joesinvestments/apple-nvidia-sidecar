"""tests/test_fix_0104_0.py - the x86 SSE3 compat header is native on 32-bit x86 (fix-0104#0).

    python3 tests/test_fix_0104_0.py

sidecar-shim/x86compat/pmmintrin.h stands in for the system <pmmintrin.h> on a non-x86 host, and
sidecar-shim/macbuild.py puts that folder on the include path of every Strata compile in the macOS build.
Its guard used to read

    #if defined(__x86_64__) || defined(_M_X64)

so a 32-bit x86 build (__i386__ or _M_IX86) fell through to the shim's own emulation header instead of the
system SSE3 header. Each case here compiles and runs a small program that includes <pmmintrin.h> against the
shim in this tree, with a stand-in system header and stand-in SIMDe headers in a temp folder. The program
prints which branch the include took. No server, model or hardware is touched. Every case prints one PASS or
FAIL line; any FAIL exits non-zero.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path.cwd() if (Path.cwd() / "sidecar-shim" / "x86compat" / "pmmintrin.h").is_file() else Path(__file__).resolve().parents[1]
INC = ROOT / "sidecar-shim" / "x86compat"
CC = shutil.which("cc") or shutil.which("clang") or shutil.which("gcc")
# Undo whatever the host defines, then each case sets only the macro it is about.
FORCE = ["-U__x86_64__", "-U_M_X64", "-U__i386__", "-U_M_IX86"]
results: list[bool] = []

# The stand-in system SSE3 header: reaching it is the whole behaviour under test.
NATIVE_HEADER = """\
#pragma once
#define SYSTEM_SSE3_HEADER 1
"""

# The stand-in SIMDe headers the shim's emulation path includes. They only have to compile the wrapper
# functions in sidecar-shim/x86compat/immintrin.h, not do SIMD.
SIMDE_AVX512 = """\
#pragma once
#include <stdint.h>
#define SIMDE_EMULATION_HEADER 1
typedef struct { int lane; } simde__m512i;
typedef struct { int lane; } simde__m128i;
typedef uint64_t simde__mmask64;
static inline void simde_mm512_storeu_si512(void *p, simde__m512i v) { (void) p; (void) v; }
static inline simde__m512i simde_mm512_loadu_si512(const void *p) { (void) p; simde__m512i v = {0}; return v; }
static inline void simde_mm_storeu_si128(simde__m128i *p, simde__m128i v) { (void) p; (void) v; }
"""

SIMDE_MARKER = """\
#pragma once
#define SIMDE_EMULATION_HEADER 1
"""

SOURCE = """\
#include <pmmintrin.h>
#include <stdio.h>

int main(void) {
#if defined(SYSTEM_SSE3_HEADER)
    puts("native-sse3");
#elif defined(SIMDE_EMULATION_HEADER)
    puts("shim-emulation");
#else
    puts("neither");
#endif
    return 0;
}
"""


def case(name, ok, detail=""):
    results.append(bool(ok))
    print(("PASS " if ok else "FAIL ") + name + (f": {detail}" if detail else ""))


def first_line(out):
    lines = [l.strip() for l in (out or "").splitlines() if l.strip()]
    return lines[0][:220] if lines else ""


def include_headers(tmp):
    native = Path(tmp) / "native"
    simde = Path(tmp) / "simde" / "simde" / "x86"
    native.mkdir(parents=True)
    simde.mkdir(parents=True)
    (native / "pmmintrin.h").write_text(NATIVE_HEADER, encoding="utf-8")
    (simde / "avx512.h").write_text(SIMDE_AVX512, encoding="utf-8")
    for name in ("fma.h", "f16c.h", "clmul.h"):
        (simde / name).write_text(SIMDE_MARKER, encoding="utf-8")
    return native, Path(tmp) / "simde"


def run_case(tmp, name, defines, native, simde):
    """Compile and run one program; return the word it printed, or None with the first error line."""
    src = Path(tmp) / f"{name}.c"
    src.write_text(SOURCE, encoding="utf-8")
    exe = Path(tmp) / f"{name}.bin"
    cmd = [CC, "-std=c11", *FORCE, *defines, "-I", str(INC), "-I", str(native), "-I", str(simde),
           str(src), "-o", str(exe)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        return None, first_line(r.stderr or r.stdout)
    r = subprocess.run([str(exe)], capture_output=True, text=True)
    return r.stdout.strip(), first_line(r.stderr)


def main():
    if not CC:
        case("a C compiler is available", False, "no cc, clang or gcc on PATH")
        return 1
    if not (INC / "pmmintrin.h").is_file():
        case("the pmmintrin shim is present", False, str(INC / "pmmintrin.h"))
        return 1
    with tempfile.TemporaryDirectory(prefix="fix-0104-0-") as tmp:
        native, simde = include_headers(tmp)
        cases = [
            # Ordinary configurations: the same answer before and after the fix.
            ("non-x86 host", [], "shim-emulation"),
            ("64-bit x86 host (__x86_64__)", ["-D__x86_64__"], "native-sse3"),
            ("64-bit x86 host (_M_X64)", ["-D_M_X64"], "native-sse3"),
            # The harm: 32-bit x86 is native too, and used to get the emulation header.
            ("32-bit x86 host (__i386__)", ["-D__i386__"], "native-sse3"),
            ("32-bit x86 host (_M_IX86)", ["-D_M_IX86"], "native-sse3"),
        ]
        for label, defines, want in cases:
            got, err = run_case(tmp, label.split(" (")[0].replace(" ", "_"), defines, native, simde)
            case(f"{label} takes the {want} header", got == want, f"got {got!r}" + (f", {err}" if err else ""))
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
