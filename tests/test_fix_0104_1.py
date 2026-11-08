"""tests/test_fix_0104_1.py - the x86 compat shim builds on a compiler without include_next (fix-0104#1).

    python3 tests/test_fix_0104_1.py

sidecar-shim/x86compat/pmmintrin.h routes every x86 host to the compiler's own header with

    #if defined(__x86_64__) || defined(_M_X64)
    #include_next <pmmintrin.h>

`_M_X64` is the MSVC x64 macro, and MSVC has no `#include_next`, so a translation unit that includes the
shim cannot be built at all on MSVC x64: the preprocessor enters the first branch and has no fallback.
sidecar-shim/x86compat/immintrin.h, the header that every other shim header falls back to, carries the same
`_M_X64` branch, so the portable path is broken from both ends.

No server, model or hardware is touched: each case compiles a few lines of C with the system C compiler and
a planted include root (stub SIMDe headers, a stub <stdint.h>, and a stub compiler-provided x86 header), so
the cases do not depend on the host's architecture or on the SDK. The compiler is told which world it is in
only by macros, which is how the header decides: `_M_X64` with no GNU macro is the MSVC x64 shape, and no
x86 macro at all is the arm64 shape this shim is written for. The compiler's own headers are switched off
(-nostdinc -nobuiltininc), so the shim's `#include_next` can only resolve to a header the case planted, and
a case that plants no next header sees exactly what MSVC sees: no such header to fall through to. Every case
prints one PASS or FAIL line; any FAIL exits non-zero.
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
# The MSVC x64 shape: the macro MSVC defines for the architecture, its version, and none of the GNU ones.
MSVC = ["-D_M_X64", "-D_MSC_VER=1938", "-U__x86_64__", "-U__GNUC__", "-U__clang__"]
# The GNU x86_64 shape, where the compiler does have #include_next and its own header.
GNU = ["-D__x86_64__"]
# The arm64 shape this shim is mainly written for: no x86 macro at all.
ARM = ["-U__x86_64__", "-U_M_X64"]
results: list[bool] = []

# The planted include root: what a portable intrinsics provider and the C library look like to a shim build.
STDINT_H = """\
#ifndef SCE_PLANT_STDINT_H
#define SCE_PLANT_STDINT_H
typedef signed int int32_t;
typedef unsigned short uint16_t;
typedef unsigned long long uint64_t;
typedef long long int64_t;
#endif
"""

SIMDE_AVX512_H = """\
#ifndef SCE_PLANT_SIMDE_AVX512_H
#define SCE_PLANT_SIMDE_AVX512_H
#include <stdint.h>
#define SCE_PORTABLE_INTRINSICS 1
typedef unsigned long long simde__mmask64;
typedef struct { int32_t v[16]; } simde__m512i;
typedef struct { int32_t v[4]; } simde__m128i;
static inline void simde_mm512_storeu_si512(void *p, simde__m512i v) { (void) p; (void) v; }
static inline simde__m512i simde_mm512_loadu_si512(void const *p) { simde__m512i r; (void) p; r.v[0] = 0; return r; }
static inline void simde_mm_storeu_si128(simde__m128i *p, simde__m128i v) { *p = v; }
#endif
"""

SIMDE_EMPTY_H = """\
#ifndef SCE_PLANT_SIMDE_EMPTY_H
#define SCE_PLANT_SIMDE_EMPTY_H
#endif
"""

# A compiler that ships the intrinsics itself, as MSVC and GCC do, offers <intrin.h> as well.
INTRIN_H = """\
#ifndef SCE_PLANT_INTRIN_H
#define SCE_PLANT_INTRIN_H
#define SCE_PORTABLE_INTRINSICS 1
#endif
"""

VENDOR_PMMINTRIN_H = """\
#ifndef SCE_VENDOR_PMMINTRIN_H
#define SCE_VENDOR_PMMINTRIN_H
#define SCE_VENDOR_PMMINTRIN 1
#endif
"""


def case(name, ok, detail=""):
    results.append(bool(ok))
    print(("PASS " if ok else "FAIL ") + name + (f": {detail}" if detail else ""))


def first_line(out):
    lines = [l.strip() for l in (out or "").splitlines() if l.strip()]
    errs = [l for l in lines if "error" in l.lower()]
    return (errs[0] if errs else (lines[0] if lines else ""))[:220]


def plant(plant_dir, vendor_dir):
    """Write the planted include roots: portable providers under plant_dir, a compiler's own headers in vendor_dir."""
    (plant_dir / "simde" / "x86").mkdir(parents=True, exist_ok=True)
    (plant_dir / "stdint.h").write_text(STDINT_H, encoding="utf-8")
    (plant_dir / "intrin.h").write_text(INTRIN_H, encoding="utf-8")
    (plant_dir / "simde" / "x86" / "avx512.h").write_text(SIMDE_AVX512_H, encoding="utf-8")
    for name in ("fma", "f16c", "clmul"):
        (plant_dir / "simde" / "x86" / f"{name}.h").write_text(SIMDE_EMPTY_H, encoding="utf-8")
    vendor_dir.mkdir(parents=True, exist_ok=True)
    (vendor_dir / "pmmintrin.h").write_text(VENDOR_PMMINTRIN_H, encoding="utf-8")


def build(tmp, name, source, macros, extra_include=None):
    """Compile one C file against the shim in this tree with the compiler's own headers off; return (ok, first error)."""
    src = Path(tmp) / f"{name}.c"
    src.write_text(source, encoding="utf-8")
    exe = Path(tmp) / f"{name}.bin"
    cmd = [CC, "-w", "-std=c11", "-x", "c", "-nostdinc", "-nobuiltininc", "-I", str(INC), "-I", str(Path(tmp) / "plant")]
    if extra_include is not None:
        cmd += ["-I", str(extra_include)]
    cmd += [*macros, str(src), "-o", str(exe)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.returncode == 0, first_line(r.stderr or r.stdout)


MSVC_PMMINTRIN = """\
#include <pmmintrin.h>
#ifndef SCE_PORTABLE_INTRINSICS
#error "the intrinsics were not provided"
#endif
#ifdef SCE_VENDOR_PMMINTRIN
#error "a compiler-provided header was used"
#endif
int main(void) { return 0; }
"""

MSVC_IMMINTRIN = """\
#include <immintrin.h>
#ifndef SCE_PORTABLE_INTRINSICS
#error "the intrinsics were not provided"
#endif
int main(void) { return 0; }
"""

GNU_PMMINTRIN = """\
#include <pmmintrin.h>
#ifndef SCE_VENDOR_PMMINTRIN
#error "the compiler's own pmmintrin.h was not used"
#endif
#ifdef SCE_PORTABLE_INTRINSICS
#error "the shim's fallback shadowed the compiler's own header"
#endif
int main(void) { return 0; }
"""

ARM_PMMINTRIN = """\
#include <pmmintrin.h>
#ifndef SCE_PORTABLE_INTRINSICS
#error "the intrinsics were not provided"
#endif
#ifdef SCE_VENDOR_PMMINTRIN
#error "a compiler-provided header was reached from a non-x86 build"
#endif
int main(void) { return 0; }
"""


def main():
    if not CC:
        case("a C compiler is available", False, "no cc, clang or gcc on PATH")
        return 1
    if not (INC / "pmmintrin.h").is_file():
        case("the x86 compat shim is present", False, str(INC / "pmmintrin.h"))
        return 1
    with tempfile.TemporaryDirectory(prefix="fix-0104-1-") as tmp:
        tmp = Path(tmp)
        plant(tmp / "plant", tmp / "vendor")

        # 1: the MSVC x64 case the finding names. There is no #include_next and no next header, so today the
        # preprocessor dies at the shim; the shim has to supply the intrinsics from its own portable branch.
        ok, err = build(tmp, "msvc_pmmintrin", MSVC_PMMINTRIN, MSVC)
        case("a translation unit including the shim builds on MSVC x64", ok, err)

        # 2: the same for the header every other shim header falls back to; a fix that only repoints
        # pmmintrin.h lands here and fails again.
        ok, err = build(tmp, "msvc_immintrin", MSVC_IMMINTRIN, MSVC)
        case("the fallback header itself builds on MSVC x64", ok, err)

        # 3 and 4: ordinary x86_64 input keeps its answer. A GNU-shaped compiler still defers to its own
        # header, with the portable fallback out of the way, whether or not MSVC's macro is also defined.
        for name, macros in (("gnu", GNU), ("gnu_msvc_macro", GNU + ["-D_M_X64"])):
            ok, err = build(tmp, name, GNU_PMMINTRIN, macros, extra_include=tmp / "vendor")
            case(f"x86_64 {name} still uses the compiler's own pmmintrin.h", ok, err)

        # 5: ordinary arm64 input keeps its answer: no x86 macro, the portable provider, and no reach for a
        # next header even though one is on the include path.
        ok, err = build(tmp, "arm", ARM_PMMINTRIN, ARM, extra_include=tmp / "vendor")
        case("arm64 still uses the portable intrinsics", ok, err)
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
