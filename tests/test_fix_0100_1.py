"""tests/test_fix_0100_1.py - an MSVC x64 build reaches <intrin.h> (fix-0100#1).

    python3 tests/test_fix_0100_1.py

sidecar-shim/x86compat/cpuid.h forwards to the compiler's own <cpuid.h> on x86 and falls back to a zeroing stub
everywhere else. The guard admitted MSVC through _M_X64 and _M_IX86:

    #if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    #include_next <cpuid.h>

but #include_next is a GCC/Clang extension MSVC does not implement, and MSVC has no <cpuid.h>; its CPUID
intrinsics live in <intrin.h>. An MSVC x64 or x86 build that included this header therefore failed to compile.

No server, model or hardware is touched. Each case writes a few lines of C and a planted <intrin.h> (the header
MSVC supplies, holding the __cpuid and __cpuidex intrinsics) into a temp folder, then compiles the probe with
-nostdinc and the target's macros against the shim in this tree. -nostdinc is what the MSVC build has: no GCC
<cpuid.h> to reach. The architecture macros are forced on the command line, so the cases do not depend on the
host's architecture. Every case prints one PASS or FAIL line; any FAIL exits non-zero.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path.cwd() if (Path.cwd() / "sidecar-shim" / "x86compat" / "cpuid.h").is_file() else Path(__file__).resolve().parents[1]
INC = ROOT / "sidecar-shim" / "x86compat"
CC = shutil.which("cc") or shutil.which("clang") or shutil.which("gcc")
REAL = 42      # exit status the planted compiler cpuid header gives the probe
results: list[bool] = []

# <intrin.h> as MSVC supplies it: the CPUID intrinsics, with answers a probe can tell apart from the zeroing stub.
PLANTED_INTRIN = """\
#ifndef SCE_FAKE_INTRIN_H
#define SCE_FAKE_INTRIN_H
#define SCE_FAKE_INTRIN 1
static inline void __cpuidex(int out[4], int leaf, int sub) {
    out[0] = 0x100 + leaf + sub;
    out[1] = 0x200 + sub;
    out[2] = 0x300 + leaf;
    out[3] = 0x400;
}
static inline void __cpuid(int out[4], int leaf) {
    out[0] = 0x500 + leaf;
    out[1] = 0x600 + leaf;
    out[2] = 0x700;
    out[3] = 0x800;
}
#endif
"""

# The compiler header the shim forwards to on GCC and Clang x86. The marker makes the forwarding visible.
PLANTED_CPUID = """\
#ifndef SCE_FAKE_CPUID_H
#define SCE_FAKE_CPUID_H
#define SCE_FAKE_CPUID 42
#define __cpuid_count(leaf, sub, a, b, c, d) ((a) = 0, (b) = 0, (c) = 0, (d) = 0)
#define __cpuid(leaf, a, b, c, d) ((a) = 0, (b) = 0, (c) = 0, (d) = 0)
static inline int __get_cpuid_max(unsigned ext, unsigned *sig) { if (sig) *sig = 0; return 0; }
#endif
"""

# 1: the header alone must compile for an MSVC target. Today the only path it takes is #include_next <cpuid.h>.
MSVC_HEADER_ONLY = """\
#include <cpuid.h>
#ifndef SCE_FAKE_INTRIN
#error the shim did not reach MSVC's <intrin.h>
#endif
int main(void) { return 0; }
"""

# 2: the GCC spellings the shim promises must read the MSVC intrinsics, not the zeroing stub.
MSVC_QUERY = """\
#include <cpuid.h>
#ifndef SCE_FAKE_INTRIN
#define SCE_FAKE_INTRIN 0
#endif
int main(void) {
    unsigned a = 11, b = 12, c = 13, d = 14;
    __cpuid_count(0x10, 2, a, b, c, d);
    if (a != 0x112 || b != 0x202 || c != 0x310 || d != 0x400) return 1;
    unsigned sig = 0xDEADBEEFu;
    unsigned max = __get_cpuid_max(0, &sig);
    if (max != 0x500 || sig != 0x600) return 2;
    unsigned e = 0, f = 0, g = 0, h = 0;
    __cpuid(1, e, f, g, h);
    if (e != 0x501 || f != 0x601 || g != 0x700 || h != 0x800) return 3;
    return SCE_FAKE_INTRIN ? 0 : 4;
}
"""

# 3: an ordinary non-x86 build still takes the zeroing stub, before and after.
ORDINARY = """\
#include <cpuid.h>
int main(void) {
    unsigned a = 11, b = 12, c = 13, d = 14;
    __cpuid_count(7, 0, a, b, c, d);
    __cpuid(0, a, b, c, d);
    unsigned sig = 0xDEADBEEFu;
    unsigned max = __get_cpuid_max(0, &sig);
    return (a == 0 && b == 0 && c == 0 && d == 0 && sig == 0 && max == 0) ? 0 : 1;
}
"""

# 4: the GCC and Clang x86 builds still forward, so the MSVC branch only adds a target and removes none.
CPUID_PROBE = """\
#include <cpuid.h>
#ifndef SCE_FAKE_CPUID
#define SCE_FAKE_CPUID 0
#endif
int main(void) { return SCE_FAKE_CPUID; }
"""


def case(name, ok, detail=""):
    results.append(bool(ok))
    print(("PASS " if ok else "FAIL ") + name + (f": {detail}" if detail else ""))


def first_line(out):
    lines = [l.strip() for l in (out or "").splitlines() if l.strip()]
    for line in lines:
        if "error:" in line:
            return line[:220]
    return lines[0][:220] if lines else ""


def build(tmp, name, source, macros, planted):
    """Compile source with -nostdinc against the shim plus one planted include folder; return (exe or None, error)."""
    src = Path(tmp) / f"{name}.c"
    src.write_text(source, encoding="utf-8")
    exe = Path(tmp) / f"{name}.bin"
    cmd = [CC, "-nostdinc", *macros, "-I", str(INC), "-I", str(Path(tmp) / planted), str(src), "-o", str(exe)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    return (exe if r.returncode == 0 else None), first_line(r.stderr or r.stdout)


def run_exe(exe):
    r = subprocess.run([str(exe)], capture_output=True, text=True)
    return r.returncode


def main():
    if not CC:
        case("a C compiler is available", False, "no cc, clang or gcc on PATH")
        return 1
    if not (INC / "cpuid.h").is_file():
        case("the cpuid shim is present", False, str(INC / "cpuid.h"))
        return 1
    with tempfile.TemporaryDirectory(prefix="fix-0100-1-") as tmp:
        (Path(tmp) / "msvc").mkdir()
        (Path(tmp) / "msvc" / "intrin.h").write_text(PLANTED_INTRIN, encoding="utf-8")
        (Path(tmp) / "gcc").mkdir()
        (Path(tmp) / "gcc" / "cpuid.h").write_text(PLANTED_CPUID, encoding="utf-8")
        msvc64 = ["-D_MSC_VER=1938", "-D_M_X64", "-U__x86_64__"]
        msvc32 = ["-D_MSC_VER=1938", "-D_M_IX86", "-U__i386__"]
        plain = ["-U__x86_64__", "-U_M_X64", "-U__i386__", "-U_M_IX86"]

        # 1 and 2: an MSVC x64 or x86 build including the shim must compile and reach <intrin.h>.
        for macros, label in ((msvc64, "MSVC x64"), (msvc32, "MSVC x86")):
            exe, err = build(tmp, f"header_{label.split()[-1]}", MSVC_HEADER_ONLY, macros, "msvc")
            if exe is None:
                case(f"an {label} build including the shim compiles", False, err or "did not compile")
            else:
                rc = run_exe(exe)
                case(f"an {label} build including the shim compiles", rc == 0, f"exit {rc}")

        # 3: the feature queries read the intrinsics: __cpuid_count leaves, __cpuid leaves and the signature.
        exe, err = build(tmp, "msvc_query", MSVC_QUERY, msvc64, "msvc")
        if exe is None:
            case("the MSVC feature queries read <intrin.h>", False, err or "did not compile")
        else:
            rc = run_exe(exe)
            case("the MSVC feature queries read <intrin.h>", rc == 0,
                 f"exit {rc}, expected 0 from the intrinsics, 4 means the zeroing stub answered")

        # 4: an ordinary non-x86 build still zeroes every query (this case passes before and after).
        exe, err = build(tmp, "ordinary", ORDINARY, plain, "msvc")
        if exe is None:
            case("an ordinary non-x86 build still uses the zeroing stub", False, err or "did not compile")
        else:
            rc = run_exe(exe)
            case("an ordinary non-x86 build still uses the zeroing stub", rc == 0,
                 f"exit {rc}, expected 0")

        # 5 and 6: the GCC and Clang x86 builds still reach the compiler header, before and after.
        for macro, label in (("-D__x86_64__", "__x86_64__"), ("-D__i386__", "__i386__")):
            exe, err = build(tmp, f"gcc_{label}", CPUID_PROBE, [macro], "gcc")
            if exe is None:
                case(f"{label} still reaches the compiler cpuid header", False, err or "did not compile")
            else:
                rc = run_exe(exe)
                case(f"{label} still reaches the compiler cpuid header", rc == REAL,
                     f"exit {rc}, expected {REAL}")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
