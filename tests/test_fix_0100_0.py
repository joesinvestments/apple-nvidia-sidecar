"""tests/test_fix_0100_0.py - an i386 build reaches the real cpuid header (fix-0100#0).

    python3 tests/test_fix_0100_0.py

sidecar-shim/x86compat/cpuid.h forwards to the compiler's own <cpuid.h> on x86 and falls back to a zeroing
stub everywhere else. The guard used to name only x86-64:

    #if defined(__x86_64__) || defined(_M_X64)

so a 32-bit x86 build (__i386__ or _M_IX86) took the stub and every feature query read as absent, which sends
the kernels down their slow fallback paths.

No server, model or hardware is touched. Each case writes a small probe program plus a planted compiler
<cpuid.h> into a temp folder, and compiles the probe against the shim in this tree with that folder as the
next include directory. The planted header is what a real x86 compiler supplies: it marks itself with
SCE_FAKE_CPUID, so the probe's exit status says whether the shim forwarded to the real header (42) or used
the zeroing stub (0). The architecture macros are forced on the command line, so the cases do not depend on
the host's architecture. Every case prints one PASS or FAIL line; any FAIL exits non-zero.
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
REAL = 42      # exit status the planted compiler header gives the probe
STUB = 0       # exit status the shim's zeroing stub gives the probe
results: list[bool] = []

# The compiler header the shim forwards to. A real x86 compiler supplies the same three intrinsics and its own
# feature macros; the marker makes the forwarding visible to the probe.
PLANTED_HEADER = """\
#ifndef SCE_FAKE_CPUID_H
#define SCE_FAKE_CPUID_H
#define SCE_FAKE_CPUID 42
#define __cpuid_count(leaf, sub, a, b, c, d) ((a) = 0, (b) = 0, (c) = 0, (d) = 0)
#define __cpuid(leaf, a, b, c, d) ((a) = 0, (b) = 0, (c) = 0, (d) = 0)
static inline int __get_cpuid_max(unsigned ext, unsigned *sig) { if (sig) *sig = 0; return 0; }
#endif
"""

PROBE = """\
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
    return lines[0][:220] if lines else ""


def probe_exit(tmp, name, macros):
    """Compile the probe against the shim and return (exit status or None, first line of a compile error)."""
    src = Path(tmp) / f"{name}.c"
    src.write_text(PROBE, encoding="utf-8")
    exe = Path(tmp) / f"{name}.bin"
    cmd = [CC, *macros, "-I", str(INC), "-I", str(Path(tmp) / "sys"), str(src), "-o", str(exe)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        return None, first_line(r.stderr or r.stdout)
    run = subprocess.run([str(exe)], capture_output=True, text=True)
    return run.returncode, ""


def main():
    if not CC:
        case("a C compiler is available", False, "no cc, clang or gcc on PATH")
        return 1
    if not (INC / "cpuid.h").is_file():
        case("the cpuid shim is present", False, str(INC / "cpuid.h"))
        return 1
    with tempfile.TemporaryDirectory(prefix="fix-0100-0-") as tmp:
        sysdir = Path(tmp) / "sys"
        sysdir.mkdir()
        (sysdir / "cpuid.h").write_text(PLANTED_HEADER, encoding="utf-8")

        # 1 and 2: the two 32-bit x86 macros the guard used to miss must reach the compiler header. On today's
        # shim the probe exits with the stub's 0; with the fix it exits with the planted header's 42.
        for macro, label in (("-D__i386__", "__i386__"), ("-D_M_IX86", "_M_IX86")):
            rc, err = probe_exit(tmp, f"i386_{label}", [macro])
            if rc is None:
                case(f"{label} reaches the compiler cpuid header", False, err or "did not compile")
            else:
                case(f"{label} reaches the compiler cpuid header", rc == REAL,
                     f"exit {rc}, expected {REAL} from the compiler header, {STUB} means the zeroing stub")

        # 3: an ordinary non-x86 build (the host's own macros) still takes the zeroing stub, before and after.
        rc, err = probe_exit(tmp, "ordinary", [])
        if rc is None:
            case("an ordinary non-x86 build still uses the zeroing stub", False, err or "did not compile")
        else:
            case("an ordinary non-x86 build still uses the zeroing stub", rc == STUB,
                 f"exit {rc}, expected {STUB}")

        # 4: the existing x86-64 guard still forwards, so the fix only adds targets and removes none.
        rc, err = probe_exit(tmp, "x86_64", ["-D__x86_64__"])
        if rc is None:
            case("an x86-64 build still reaches the compiler cpuid header", False, err or "did not compile")
        else:
            case("an x86-64 build still reaches the compiler cpuid header", rc == REAL,
                 f"exit {rc}, expected {REAL}; {STUB} means the fix broke x86-64")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
