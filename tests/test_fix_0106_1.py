"""tests/test_fix_0106_1.py - a 32-bit x86 build reaches the real tmmintrin.h (fix-0106#1).

    python3 tests/test_fix_0106_1.py

sidecar-shim/x86compat/tmmintrin.h is a wrapper on the include path of every Strata compile. Its condition used
to read

    #if defined(__x86_64__) || defined(_M_X64)

so a 32-bit x86 build, where the compiler defines __i386__ or _M_IX86 instead, took the #else branch and included
the emulated immintrin.h (SIMDe) rather than the real system <tmmintrin.h>. The 64-bit and non-x86 branches were
already right.

No server, model or hardware is touched: each case copies the wrapper into a temp folder next to a fake emulated
immintrin.h, puts a fake real tmmintrin.h after it on the include path, compiles a tiny C file with the system
compiler under forced target macros, and runs it. The program returns 0 when the real header was reached and 7
when the emulated one was; the fake headers only define that value, never the wrapper's condition. Every case
prints one PASS or FAIL line; any FAIL exits non-zero.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path.cwd() if (Path.cwd() / "sidecar-shim" / "x86compat" / "tmmintrin.h").is_file() else Path(__file__).resolve().parents[1]
SRC = ROOT / "sidecar-shim" / "x86compat" / "tmmintrin.h"
CC = shutil.which("cc") or shutil.which("clang") or shutil.which("gcc")

REAL = 0        # the fake real system tmmintrin.h
EMULATED = 7    # the fake emulated immintrin.h
results: list[bool] = []

PROGRAM = """\
#include <tmmintrin.h>

int main(void) {
    return TMMINTRIN_SOURCE;
}
"""


def case(name, ok, detail=""):
    results.append(bool(ok))
    print(("PASS " if ok else "FAIL ") + name + (f": {detail}" if detail else ""))


def first_line(out):
    lines = [l.strip() for l in (out or "").splitlines() if l.strip()]
    return lines[0][:220] if lines else ""


def compile_c(tmp, name, flags):
    """Compile the program with forced target macros; return (binary or None, first line of the error)."""
    src = Path(tmp) / f"{name}.c"
    src.write_text(PROGRAM, encoding="utf-8")
    exe = Path(tmp) / f"{name}.bin"
    cmd = [CC, "-std=c11", *flags, "-I", str(Path(tmp) / "shim"), "-I", str(Path(tmp) / "real"), str(src), "-o", str(exe)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    return (exe if r.returncode == 0 else None), first_line(r.stderr or r.stdout)


def check(tmp, name, flags, want):
    exe, err = compile_c(tmp, name, flags)
    if exe is None:
        case(name, False, err or "did not compile")
        return
    r = subprocess.run([str(exe)], capture_output=True, text=True)
    case(name, r.returncode == want,
         f"got exit {r.returncode} ({'real' if r.returncode == REAL else 'emulated' if r.returncode == EMULATED else 'neither'}),"
         f" want {want} ({'real' if want == REAL else 'emulated'})")


def main():
    if not CC:
        case("a C compiler is available", False, "no cc, clang or gcc on PATH")
        return 1
    if not SRC.is_file():
        case("the tmmintrin wrapper is present", False, str(SRC))
        return 1
    with tempfile.TemporaryDirectory(prefix="fix-0106-1-") as tmp:
        shim = Path(tmp) / "shim"
        real = Path(tmp) / "real"
        shim.mkdir()
        real.mkdir()
        (shim / "tmmintrin.h").write_bytes(SRC.read_bytes())
        # The fake neighbours carry only the routing answer, so the case tests the wrapper, not SIMDe.
        (shim / "immintrin.h").write_text(f"#define TMMINTRIN_SOURCE {EMULATED}\n", encoding="utf-8")
        (real / "tmmintrin.h").write_text(f"#define TMMINTRIN_SOURCE {REAL}\n", encoding="utf-8")

        # 1 and 2: a 32-bit x86 build reaches the real system header by either compiler macro. Both fail today.
        check(tmp, "i386 with __i386__ reaches the real tmmintrin.h",
              ["-U__x86_64__", "-U_M_X64", "-D__i386__"], REAL)
        check(tmp, "i386 with _M_IX86 reaches the real tmmintrin.h",
              ["-U__x86_64__", "-U_M_X64", "-D_M_IX86"], REAL)

        # 3 and 4: ordinary x86_64 input keeps reaching the real header, as it did before the fix.
        check(tmp, "x86_64 with __x86_64__ keeps the real tmmintrin.h",
              ["-U__i386__", "-U_M_IX86", "-D__x86_64__"], REAL)
        check(tmp, "x86_64 with _M_X64 keeps the real tmmintrin.h",
              ["-U__i386__", "-U_M_IX86", "-D_M_X64"], REAL)

        # 5: ordinary non-x86 input keeps the emulated header, as it did before the fix.
        check(tmp, "a non-x86 host keeps the emulated immintrin.h",
              ["-U__x86_64__", "-U_M_X64", "-U__i386__", "-U_M_IX86"], EMULATED)
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
