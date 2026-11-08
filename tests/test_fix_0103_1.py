"""tests/test_fix_0103_1.py - the x86compat nmmintrin.h shim lets 32-bit x86 reach the compiler's header (fix-0103#1).

    python3 tests/test_fix_0103_1.py

sidecar-shim/macbuild.py puts sidecar-shim/x86compat first on the include path of every Strata source in the macOS
build, so this nmmintrin.h stands in front of the compiler's own <nmmintrin.h>. Its line 2 chose between the two:

    #if defined(__x86_64__) || defined(_M_X64)
    #include_next <nmmintrin.h>
    #else
    #include "immintrin.h"

The guard named only x86-64, so a 32-bit x86 build (__i386__ or _M_IX86) fell into the emulation branch even though
that target has a real nmmintrin.h: the shim silently shadowed the platform header and the program compiled against
the emulation chain instead of the intrinsics the compiler provides.

No server, model or hardware is touched. Each case copies the header under test into a temp folder, stands in two
one-line neighbors for the emulation header and for the compiler's own nmmintrin.h (each defines a distinct backend
number), compiles a probe against them with the system C compiler, and reads the selected backend from the
program's exit code. The x86 target macros are unset first and then set per case, so the cases do not depend on the
host architecture. Every case prints one PASS or FAIL line; any FAIL exits non-zero.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path.cwd() if (Path.cwd() / "sidecar-shim" / "x86compat" / "nmmintrin.h").is_file() else Path(__file__).resolve().parents[1]
HEADER = ROOT / "sidecar-shim" / "x86compat" / "nmmintrin.h"
CC = shutil.which("cc") or shutil.which("clang") or shutil.which("gcc")
UNSET = ["-U__x86_64__", "-U_M_X64", "-U__i386__", "-U_M_IX86"]   # each case sets the target it compiles for
PLATFORM = 32    # the compiler's own nmmintrin.h, reached through #include_next
EMULATION = 16   # sidecar-shim/x86compat/immintrin.h and the chain behind it
results: list[bool] = []

PROBE = """\
#include <nmmintrin.h>

int main(void) {
#ifdef SCE_NMMINTRIN
    return SCE_NMMINTRIN;    /* the backend the shim selected */
#else
    return 99;
#endif
}
"""


def case(name, ok, detail=""):
    results.append(bool(ok))
    print(("PASS " if ok else "FAIL ") + name + (f": {detail}" if detail else ""))


def first_line(out):
    lines = [l.strip() for l in (out or "").splitlines() if l.strip()]
    return lines[0][:220] if lines else ""


def backend(tmp, defines):
    """Compile the probe against the header under test for one target; return (backend or None, why)."""
    src = Path(tmp) / "probe.c"
    src.write_text(PROBE, encoding="utf-8")
    exe = Path(tmp) / "probe.bin"
    cmd = [CC, "-std=c11", "-nostdinc", *UNSET, *defines,
           "-I", str(Path(tmp) / "shim"), "-I", str(Path(tmp) / "plat"), str(src), "-o", str(exe)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        return None, first_line(r.stderr or r.stdout) or f"did not compile, exit {r.returncode}"
    p = subprocess.run([str(exe)], capture_output=True, text=True)
    return p.returncode, f"exit {p.returncode}"


def main():
    if not CC:
        case("a C compiler is available", False, "no cc, clang or gcc on PATH")
        return 1
    if not HEADER.is_file():
        case("the nmmintrin.h shim is present", False, str(HEADER))
        return 1
    with tempfile.TemporaryDirectory(prefix="fix-0103-1-") as tmp:
        shim = Path(tmp) / "shim"
        plat = Path(tmp) / "plat"
        shim.mkdir()
        plat.mkdir()
        # the header under test is this tree's, byte for byte; only its two neighbors are stand-ins
        (shim / "nmmintrin.h").write_bytes(HEADER.read_bytes())
        (shim / "immintrin.h").write_text(f"#define SCE_NMMINTRIN {EMULATION}\n", encoding="utf-8")
        (plat / "nmmintrin.h").write_text(f"#define SCE_NMMINTRIN {PLATFORM}\n", encoding="utf-8")

        cases = [
            ("32-bit x86 through __i386__ reaches the compiler's nmmintrin.h", ["-D__i386__"], PLATFORM),
            ("32-bit x86 through _M_IX86 reaches the compiler's nmmintrin.h", ["-D_M_IX86"], PLATFORM),
            ("x86-64 through __x86_64__ still reaches the compiler's nmmintrin.h", ["-D__x86_64__"], PLATFORM),
            ("x86-64 through _M_X64 still reaches the compiler's nmmintrin.h", ["-D_M_X64"], PLATFORM),
            ("a non-x86 host still gets the emulation header", [], EMULATION),
        ]
        for name, defines, want in cases:
            got, why = backend(tmp, defines)
            case(name, got == want, f"got {got}, want {want} ({why})")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
