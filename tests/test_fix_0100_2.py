"""tests/test_fix_0100_2.py - the non-x86 cpuid shim is valid C and writes the signature (fix-0100#2).

    python3 tests/test_fix_0100_2.py

sidecar-shim/x86compat/cpuid.h stands in for GCC's <cpuid.h> on a non-x86 host, and sidecar-shim/macbuild.py puts
that folder on the include path of every Strata compile in the macOS build. Its #else branch used to read

    static inline int __get_cpuid_max(unsigned, unsigned*) { return 0; }

so a C translation unit that includes the header failed to compile before C23, because an unnamed parameter in a
function definition is invalid C there, and a caller that passed a signature pointer read whatever the slot held
before the call, because the definition never wrote it.

No server, model or hardware is touched: each case writes a few lines of C into a temp folder, compiles them with
the system C compiler against the header in this tree, and runs the program where the answer is a value. The
non-x86 branch is forced with -U__x86_64__ -U_M_X64, so the cases do not depend on the host's architecture. Every
case prints one PASS or FAIL line; any FAIL exits non-zero.
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
FORCE = ["-U__x86_64__", "-U_M_X64"]     # the header's #else branch is the one a non-x86 host compiles
results: list[bool] = []

HEADER_ONLY = """\
#include <cpuid.h>

int main(void) {
    unsigned sig = 0;
    return __get_cpuid_max(0, &sig);
}
"""

SIGNATURE = """\
#include <cpuid.h>
#include <stdio.h>

int main(void) {
    unsigned sig = 0xDEADBEEFu;          /* planted: the signature slot starts dirty */
    unsigned max = __get_cpuid_max(0, &sig);
    printf("max=%u sig=%08x\\n", max, sig);
    return 0;
}
"""

ORDINARY = """\
#include <cpuid.h>
#include <stdio.h>

int main(void) {
    unsigned a = 11, b = 12, c = 13, d = 14;
    __cpuid_count(7, 0, a, b, c, d);
    __cpuid(0, a, b, c, d);
    unsigned max = __get_cpuid_max(0, 0);   /* a caller that asks for no signature */
    printf("leaves=%u,%u,%u,%u max=%u\\n", a, b, c, d, max);
    return 0;
}
"""


def case(name, ok, detail=""):
    results.append(bool(ok))
    print(("PASS " if ok else "FAIL ") + name + (f": {detail}" if detail else ""))


def first_line(out):
    lines = [l.strip() for l in (out or "").splitlines() if l.strip()]
    return lines[0][:220] if lines else ""


def compile_c(tmp, name, source, std, strict=True):
    """Compile one C file against the shim in this tree; return (binary or None, first line of the error)."""
    src = Path(tmp) / f"{name}.c"
    src.write_text(source, encoding="utf-8")
    exe = Path(tmp) / f"{name}.bin"
    cmd = [CC, f"-std={std}", *FORCE, "-I", str(INC)]
    if strict:
        cmd.append("-pedantic-errors")
    cmd += [str(src), "-o", str(exe)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    return (exe if r.returncode == 0 else None), first_line(r.stderr or r.stdout)


def main():
    if not CC:
        case("a C compiler is available", False, "no cc, clang or gcc on PATH")
        return 1
    if not (INC / "cpuid.h").is_file():
        case("the cpuid shim is present", False, str(INC / "cpuid.h"))
        return 1
    with tempfile.TemporaryDirectory(prefix="fix-0100-2-") as tmp:
        # 1 and 2: a C translation unit that includes the shim must compile as C11 and as C99.
        for std in ("c11", "c99"):
            exe, err = compile_c(tmp, f"header_{std}", HEADER_ONLY, std)
            case(f"a C translation unit including the shim compiles as {std}", exe is not None, err)

        # 3: the signature out-parameter is written, so the caller never reads the dirty slot.
        exe, err = compile_c(tmp, "signature", SIGNATURE, "c11", strict=False)
        if exe is None:
            case("the signature out-parameter is written", False, err or "did not compile")
        else:
            r = subprocess.run([str(exe)], capture_output=True, text=True)
            got = r.stdout.strip()
            case("the signature out-parameter is written", got == "max=0 sig=00000000",
                 f"got {got!r}, exit {r.returncode}")

        # 4: ordinary input keeps its answer: the feature macros and the max leaf still read zero, and a null
        # signature pointer is accepted (this case passes before and after the fix).
        exe, err = compile_c(tmp, "ordinary", ORDINARY, "c11", strict=False)
        if exe is None:
            case("the zeroed leaves and the null signature call behave as before", False, err or "did not compile")
        else:
            r = subprocess.run([str(exe)], capture_output=True, text=True)
            got = r.stdout.strip()
            case("the zeroed leaves and the null signature call behave as before",
                 got == "leaves=0,0,0,0 max=0", f"got {got!r}, exit {r.returncode}")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
