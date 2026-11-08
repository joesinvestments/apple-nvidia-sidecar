"""tests/test_fix_0101_0.py - the x86_64 branch of the emmintrin shim must not use #include_next on a compiler
that does not implement it (fix-0101#0).

    python3 tests/test_fix_0101_0.py

sidecar-shim/x86compat/emmintrin.h guards its forwarding branch with

    #if defined(__x86_64__) || defined(_M_X64)

and the branch body is `#include_next <emmintrin.h>`, which is a GCC/Clang extension. MSVC x64 defines `_M_X64`, so it
enters that branch and hits a directive its preprocessor does not have: the toolchain header is never reached and the
build stops. A folder can only forward past itself with `#include_next`, so for MSVC the shim folder has to be searched
after the toolchain folders and the header has to reach them with a plain `#include`.

The cases below plant an MSVC shaped macro set (`_M_X64` and `_MSC_VER`, no `__GNUC__`, no `__clang__`) and a toolchain
folder that holds the real `emmintrin.h` after the shim folder, the include order the fix requires for MSVC. A
translation unit that reaches the shim header then has to end up in the toolchain header and read its value; today it
dies on the `#include_next` line. The GCC/Clang and the arm64 shapes are the ordinary input: they must answer the same
before and after the fix.

No server, model or hardware is touched: each case writes a few lines of C and a stand in toolchain folder into a temp
folder, runs the system C compiler with -nostdinc so only the planted folders are searched, and checks the file the
preprocessor read or whether the translation unit compiles. Every case prints one PASS or FAIL line; any FAIL exits
non-zero.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path.cwd() if (Path.cwd() / "sidecar-shim" / "x86compat" / "emmintrin.h").is_file() else Path(__file__).resolve().parents[1]
SHIM = ROOT / "sidecar-shim" / "x86compat"
CC = shutil.which("cc") or shutil.which("clang") or shutil.which("gcc")
MSVC = ["-U__x86_64__", "-U__GNUC__", "-U__clang__", "-D_M_X64=1", "-D_MSC_VER=1940"]   # the macros an MSVC x64 build has
CLANG_CL = ["-U__x86_64__", "-D_M_X64=1", "-D_MSC_VER=1940"]                            # MSVC macros, but clang implements the extension

# the toolchain header the shim has to forward to: it holds a value, so reaching it is observable at compile time
TOOLCHAIN_EMMINTRIN = """\
#pragma once
#define SCE0101_TOOLCHAIN_EMMINTRIN 1
#define SCE0101_TOOLCHAIN_VALUE 7
typedef struct { int lane; } __m128i;
"""
TOOLCHAIN_STDINT = "#pragma once\ntypedef unsigned long long uint64_t;\n"
SIMDE_STUBS = ("avx512", "fma", "f16c", "clmul")

# a translation unit that reaches the shim header by its path in the tree: a build that puts the shim folder after the
# toolchain folders still reaches it, and on MSVC the shim has to forward to the toolchain header without #include_next
BY_PATH = """\
#include "sidecar-shim/x86compat/emmintrin.h"
#ifndef SCE0101_TOOLCHAIN_EMMINTRIN
#error "the toolchain emmintrin.h was not reached"
#endif
_Static_assert(SCE0101_TOOLCHAIN_VALUE == 7, "the toolchain emmintrin.h was not read");
"""
# the ordinary lookup: the shim folder comes first, so <emmintrin.h> names the shim header
BY_NAME = """\
#include <emmintrin.h>
#ifndef SCE0101_TOOLCHAIN_EMMINTRIN
#error "the toolchain emmintrin.h was not reached"
#endif
_Static_assert(SCE0101_TOOLCHAIN_VALUE == 7, "the toolchain emmintrin.h was not read");
"""
# the arm64 shim branch reaches SIMDe instead of a toolchain header, so that case only reads the include graph
INCLUDE_ONLY = "#include <emmintrin.h>\n"

results: list[bool] = []


def case(name, ok, detail=""):
    results.append(bool(ok))
    print(("PASS " if ok else "FAIL ") + name + (f": {detail}" if detail else ""))


def first_line(out):
    lines = [l.strip() for l in (out or "").splitlines() if l.strip()]
    return lines[0][:220] if lines else ""


def plant(tmp):
    """The planted world: a toolchain folder with the real emmintrin.h and the SIMDe stubs the arm64 branch includes."""
    sys_dir = Path(tmp) / "toolchain"
    (sys_dir / "simde" / "x86").mkdir(parents=True)
    (sys_dir / "emmintrin.h").write_text(TOOLCHAIN_EMMINTRIN, encoding="utf-8")
    (sys_dir / "stdint.h").write_text(TOOLCHAIN_STDINT, encoding="utf-8")
    for name in SIMDE_STUBS:
        (sys_dir / "simde" / "x86" / f"{name}.h").write_text("#pragma once\n", encoding="utf-8")
    return sys_dir


def cc_cmd(path_order, macros, graph=False):
    """One compiler run: -nostdinc, so the planted toolchain folder and the tree are the whole search path."""
    cmd = [CC, "-nostdinc", "-std=c11", "-M" if graph else "-c", *macros]
    for p in path_order:
        cmd += ["-I", str(p)]
    return cmd


def compiles(tmp, name, source, path_order, macros):
    """Build one plant to an object; return (True, "") when the translation unit compiles."""
    src = Path(tmp) / f"{name}.c"
    src.write_text(source, encoding="utf-8")
    cmd = cc_cmd(path_order, macros) + [str(src), "-o", str(Path(tmp) / f"{name}.o")]
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.returncode == 0, first_line(r.stderr or r.stdout)


def reads(tmp, name, source, path_order, macros):
    """The files the preprocessor reads for one plant; return (set of names, first line of the error)."""
    src = Path(tmp) / f"{name}.c"
    src.write_text(source, encoding="utf-8")
    r = subprocess.run(cc_cmd(path_order, macros, graph=True) + [str(src)], capture_output=True, text=True)
    if r.returncode != 0:
        return None, first_line(r.stderr or r.stdout)
    body = r.stdout.split(":", 1)[1] if ":" in r.stdout else r.stdout
    return set(Path(p).as_posix() for p in body.split()), ""


def main():
    if not CC:
        case("a C compiler is available", False, "no cc, clang or gcc on PATH")
        return 1
    if not (SHIM / "emmintrin.h").is_file():
        case("the emmintrin shim is present", False, str(SHIM / "emmintrin.h"))
        return 1
    with tempfile.TemporaryDirectory(prefix="fix-0101-0-") as tmp:
        toolchain = plant(tmp)
        after = [toolchain, ROOT]                # the shim folder after the toolchain folders: the order MSVC needs
        before = [SHIM, toolchain]               # the shim folder first: the ordinary GCC/Clang order

        # 1: an MSVC x64 build reaches the toolchain emmintrin.h and reads its value. Today the header dies on
        # #include_next, so this is the case that shows the harm.
        ok, err = compiles(tmp, "msvc", BY_PATH, after, MSVC)
        case("an MSVC x64 compile forwards to the toolchain emmintrin.h", ok, err)

        # 2: the same MSVC plant reads the toolchain header and none of the SIMDe headers, so the forwarding lands on
        # the real header rather than on the arm64 branch. Today the preprocessor fails before naming any file.
        files, err = reads(tmp, "msvc_graph", BY_PATH, after, MSVC)
        if files is None:
            case("the MSVC x64 include graph holds the toolchain header", False, err)
        else:
            ok = any(f.endswith("sidecar-shim/x86compat/emmintrin.h") for f in files) and \
                 any(f.endswith("toolchain/emmintrin.h") for f in files) and \
                 not any("/simde/" in f for f in files)
            case("the MSVC x64 include graph holds the toolchain header", ok, sorted(files))

        # 3: ordinary input, GCC/Clang on x86_64: the shim still forwards with #include_next to the toolchain header.
        ok, err = compiles(tmp, "gnu", BY_NAME, before, ["-D__x86_64__"])
        case("a GCC/Clang x86_64 compile still forwards with #include_next", ok, err)

        # 4: ordinary input, arm64: the header still takes the SIMDe branch and never touches the toolchain one.
        files, err = reads(tmp, "arm64", INCLUDE_ONLY, before + [toolchain / "simde"], [])
        if files is None:
            case("an arm64 compile still takes the SIMDe branch", False, err)
        else:
            ok = any(f.endswith("sidecar-shim/x86compat/immintrin.h") for f in files) and \
                 all(any(f.endswith(f"simde/x86/{n}.h") for f in files) for n in SIMDE_STUBS) and \
                 not any(f.endswith("toolchain/emmintrin.h") for f in files)
            case("an arm64 compile still takes the SIMDe branch", ok, sorted(files))

        # 5: ordinary input, clang-cl: it has the MSVC macros but also implements #include_next, so it keeps the
        # forwarding branch and the answer does not change.
        ok, err = compiles(tmp, "clang_cl", BY_NAME, before, CLANG_CL)
        case("clang with the MSVC macros keeps the #include_next branch", ok, err)
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
