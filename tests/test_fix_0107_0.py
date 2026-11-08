"""tests/test_fix_0107_0.py - the x86intrin shim must not send MSVC x64 to #include_next (fix-0107#0).

    python3 tests/test_fix_0107_0.py

sidecar-shim/x86compat/x86intrin.h stands in for the compiler's <x86intrin.h> on the non-x86 macOS build, and
sidecar-shim/macbuild.py puts that folder on the include path of every Strata compile there. Line 2 reads

    #if defined(__x86_64__) || defined(_M_X64)

so a target that defines _M_X64, MSVC x64 included, takes the `#include_next <x86intrin.h>` branch. #include_next is
a GCC and Clang extension: MSVC has no such directive, and no next x86intrin.h to find, so an MSVC x64 translation
unit that includes this shim dies in the preprocessor before any of its source is read.

No server, model, hardware or MSVC is touched: each case writes a few lines of C and a small stand-in header tree
into a temp folder, sets the macro set the compiler really defines (_MSC_VER and _M_X64 for MSVC x64, __x86_64__ for
GCC and Clang x64, none of them for a non-x86 host), and builds the header with the system C compiler. The stand-in
headers return a different value each, so the program's exit code says which header the shim reached:
  - an MSVC x64 unit with only MSVC's <intrin.h> beside it must build and read that header (FAILS today: no next
    x86intrin.h);
  - an MSVC x64 unit must read MSVC's <intrin.h>, not a foreign x86intrin.h chain even when one is on the path
    (FAILS today: it reads the chain that returns 4);
  - an x64 Clang or GCC unit still takes the next x86intrin.h (ordinary input, same before and after);
  - a non-x86 unit still takes the SIMDe immintrin shim (ordinary input, same before and after).
Every case prints one PASS or FAIL line; any FAIL exits non-zero.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path.cwd() if (Path.cwd() / "sidecar-shim" / "x86compat" / "x86intrin.h").is_file() else Path(__file__).resolve().parents[1]
INC = ROOT / "sidecar-shim" / "x86compat"
CC = shutil.which("cc") or shutil.which("clang") or shutil.which("gcc")

# The macro sets each compiler defines for the target, with the other targets' macros cleared.
MSVC = ["-D_MSC_VER=1938", "-D_M_X64=100", "-U__x86_64__"]
CLANG_X64 = ["-D__x86_64__", "-U_MSC_VER", "-U_M_X64"]
NON_X86 = ["-U__x86_64__", "-U_M_X64", "-U_MSC_VER"]

# Each stand-in header answers under the same name, so the value the program returns names the header it reached.
NEXT_HEADER = "static inline int x86intrin_source(void) { return 4; }\n"
MSVC_HEADER = "static inline int x86intrin_source(void) { return 7; }\n"

HARNESS = """\
#include <x86intrin.h>

int main(void) {
    return x86intrin_source();
}
"""

SIMDE_HEADER = "static inline int simde_avx512_shim_seen(void) { return 0; }\n"

results: list[bool] = []


def case(name, ok, detail=""):
    results.append(bool(ok))
    print(("PASS " if ok else "FAIL ") + name + (f": {detail}" if detail and not ok else ""))


def first_line(out):
    lines = [l.strip() for l in (out or "").splitlines() if l.strip()]
    errs = [l for l in lines if "error" in l.lower() or "fatal" in l.lower()]
    return (errs or lines or [""])[0][:220]


def build(tmp, name, macros, dirs, preprocess=False):
    """Build one translation unit that includes the shim; return (output or None, first line of the error)."""
    src = tmp / f"{name}.c"
    src.write_text(HARNESS, encoding="utf-8")
    out = tmp / (f"{name}.i" if preprocess else f"{name}.bin")
    cmd = [CC, "-std=c11", "-Wno-everything", "-nostdinc", *macros, "-I", str(INC)]
    for d in dirs:
        cmd += ["-I", str(d)]
    cmd += (["-E", str(src), "-o", str(out)] if preprocess else [str(src), "-o", str(out)])
    r = subprocess.run(cmd, capture_output=True, text=True)
    return (out if r.returncode == 0 else None), first_line(r.stderr or r.stdout)


def run(exe):
    return subprocess.run([str(exe)], capture_output=True, text=True).returncode


def main():
    if not CC:
        case("a C compiler is available", False, "no cc, clang or gcc on PATH")
        return 1
    if not (INC / "x86intrin.h").is_file():
        case("the x86intrin shim is present", False, str(INC / "x86intrin.h"))
        return 1
    with tempfile.TemporaryDirectory(prefix="fix-0107-0-") as tmp:
        tmp = Path(tmp)
        nxt = tmp / "next"
        nxt.mkdir()
        (nxt / "x86intrin.h").write_text(NEXT_HEADER, encoding="utf-8")
        msvc = tmp / "msvc"
        msvc.mkdir()
        (msvc / "intrin.h").write_text(MSVC_HEADER, encoding="utf-8")
        simde = tmp / "simde"
        (simde / "simde" / "x86").mkdir(parents=True)
        (simde / "simde" / "x86" / "avx512.h").write_text(SIMDE_HEADER, encoding="utf-8")
        for h in ("fma", "f16c", "clmul"):
            (simde / "simde" / "x86" / f"{h}.h").write_text("#pragma once\n", encoding="utf-8")
        (simde / "stdint.h").write_text("#pragma once\n", encoding="utf-8")

        # 1: an MSVC x64 unit has only MSVC's <intrin.h> to reach. Today the shim reaches for a next x86intrin.h.
        exe, err = build(tmp, "msvc_only", MSVC, [msvc])
        if exe is None:
            case("an MSVC x64 unit including the shim builds and reads MSVC intrin.h", False, err)
        else:
            rc = run(exe)
            case("an MSVC x64 unit including the shim builds and reads MSVC intrin.h", rc == 7,
                 f"the program returned {rc}, MSVC intrin.h returns 7")

        # 2: a foreign x86intrin.h chain is on the path too. Today the shim takes it instead of MSVC's header.
        exe, err = build(tmp, "msvc_pick", MSVC, [nxt, msvc])
        if exe is None:
            case("an MSVC x64 unit reads MSVC intrin.h, not a foreign x86intrin.h chain", False, err)
        else:
            rc = run(exe)
            case("an MSVC x64 unit reads MSVC intrin.h, not a foreign x86intrin.h chain", rc == 7,
                 f"the shim reached the header that returns {rc}, MSVC intrin.h returns 7")

        # 3: an x64 Clang or GCC unit keeps the next system x86intrin.h (this case passes before and after the fix).
        exe, err = build(tmp, "x64_chain", CLANG_X64, [nxt])
        if exe is None:
            case("an x64 Clang or GCC unit still takes the next x86intrin.h", False, err)
        else:
            rc = run(exe)
            case("an x64 Clang or GCC unit still takes the next x86intrin.h", rc == 4,
                 f"the shim reached the header that returns {rc}, the next x86intrin.h returns 4")

        # 4: a non-x86 unit keeps the SIMDe immintrin shim (this case passes before and after the fix).
        out, err = build(tmp, "non_x86", NON_X86, [simde], preprocess=True)
        if out is None:
            case("a non-x86 unit still takes the SIMDe immintrin shim", False, err)
        else:
            text = out.read_text(encoding="utf-8", errors="replace")
            case("a non-x86 unit still takes the SIMDe immintrin shim", "simde_avx512_shim_seen" in text,
                 "the preprocessed header does not reach the SIMDe stand-in")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
