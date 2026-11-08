"""tests/test_fix_0079_5.py - macbuild.py takes the SIMDe root from its caller (fix-0079#5).

    python3 tests/test_fix_0079_5.py

sidecar-shim/macbuild.py writes one -isystem $HOME/simde into every Strata-owned C/C++ host rule of host.ninja:

    fl = ["-isystem", str(SHIM / "x86compat"), "-isystem", str(HOME / "simde"), ...] + fl

$HOME/simde is hard-coded: no flag, no environment variable, no existence check. sidecar-shim/x86compat/immintrin.h
includes <simde/x86/avx512.h>, so the first clang invocation of a build on a machine that keeps SIMDe anywhere else
fails on a header path that is not there. tools/build-sidecar.sh already takes SIDECAR_SIMDE as required input and
symlinks it to $ROOT/simde, so the run works on that layout only by accident of the symlink. The fix accepts --simde
and SIDECAR_SIMDE, keeps $HOME/simde as the default so tools/build-sidecar.sh and the Dockerfile still work, and an
explicit command-line value wins over the environment.

Every case is a small fake: a temp HOME, a temp configure command list in the shape `ninja -t commands strata`
writes, temp SDK and SIMDe header trees, and macbuild.py run as a subprocess. Only the two ninja files it writes are
read, and every -isystem root it names for SIMDe is checked against a real simde/x86/avx2.h planted there. Nothing
is compiled, fetched or served, and nothing is written outside a temp folder. One PASS or FAIL line per case; any
FAIL exits non-zero.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path.cwd() if (Path.cwd() / "sidecar-shim" / "macbuild.py").is_file() else Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "sidecar-shim" / "macbuild.py"
RESULTS: list[bool] = []


def case(name, ok, detail=""):
    RESULTS.append(bool(ok))
    print(("PASS " if ok else "FAIL ") + name + (f": {detail}" if detail else ""))


def one_line(text):
    lines = [l.strip() for l in (text or "").splitlines() if l.strip()]
    return lines[-1][:220] if lines else ""


def configure_cmds(lin, cuda_dev):
    """The commands sidecar-build/Dockerfile captures, in the shape macbuild.py parses: nvcc, c++, cc, ar, link."""
    return "\n".join([
        f"/usr/bin/nvcc -c {lin}/src/kernel.cu -o {lin}/CMakeFiles/strata.dir/src/kernel.cu.o"
        f" -I{cuda_dev}/include -DSTRATA_TGCU_BUILTIN=1 -std=c++17 -MD -MT kernel -MF kernel.d",
        f"c++ -c {lin}/src/main.cpp -o {lin}/CMakeFiles/strata.dir/src/main.cpp.o"
        f" -I{cuda_dev}/include -O2",
        f"cc -c {lin}/src/fast.c -o {lin}/CMakeFiles/strata.dir/src/fast.c.o"
        f" -I{cuda_dev}/include -O2",
        f"ar qc {lin}/libstrata.a {lin}/CMakeFiles/strata.dir/src/kernel.cu.o"
        f" {lin}/CMakeFiles/strata.dir/src/main.cpp.o {lin}/CMakeFiles/strata.dir/src/fast.c.o",
        f"c++ {lin}/CMakeFiles/strata.dir/src/kernel.cu.o"
        f" {lin}/CMakeFiles/strata.dir/src/main.cpp.o {lin}/CMakeFiles/strata.dir/src/fast.c.o -o strata",
    ]) + "\n"


def planted_simde(root):
    """A SIMDe header install: <root>/simde/x86/avx2.h, the header sidecar-shim/x86compat/immintrin.h includes."""
    root = Path(root)
    write = root / "simde" / "x86" / "avx2.h"
    write.parent.mkdir(parents=True, exist_ok=True)
    write.write_text("#define SIMDE_VERSION_PLANTED 1\n", encoding="utf-8")
    return root


def planted_sdk(root, name="MacOSX26.4.sdk"):
    """A search root holding one installed 26.x SDK, so macbuild.py never reads the host's own SDK folder."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    (root / name).mkdir()
    return root


def macbuild(tmp, env_extra=None, args=()):
    """Run macbuild.py over a temp HOME and temp command list. Returns (process, host text, home)."""
    tmp = Path(tmp)
    home = tmp / "home"
    lin = tmp / "strata-linux"
    home.mkdir(parents=True)
    lin.mkdir(parents=True)
    cmds = lin / "strata_cmds.txt"
    cmds.write_text(configure_cmds(lin, home / "cuda-13.4"), encoding="utf-8")
    drop = ("SDK", "SIDECAR_SDK_DIRS", "SIDECAR_CUDA_HOST", "SIDECAR_CUDA_DEV", "SIDECAR_SIMDE")
    env = {k: v for k, v in os.environ.items() if k not in drop}
    env["HOME"] = str(home)
    env.update(env_extra or {})
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--lin", str(lin), "--cmds", str(cmds), *args],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=120,
    )
    out = home / "strata-mac"
    host = (out / "host.ninja").read_text(encoding="utf-8") if (out / "host.ninja").is_file() else ""
    return proc, host, home


def host_compiles(host):
    """(rule, source, flags) for every C/C++ host compile in host.ninja, in file order."""
    out, lines = [], host.splitlines()
    for i, line in enumerate(lines):
        m = re.fullmatch(r"build (\S+): (cxx|cc) (\S+)", line)
        if not m:
            continue
        flags = ""
        for nxt in lines[i + 1:i + 3]:
            if nxt.startswith("  flags = "):
                flags = nxt[len("  flags = "):]
                break
        out.append((m.group(2), m.group(3), flags))
    return out


def isystems(flags):
    """Every directory one host compile passes to -isystem, in flag order."""
    return [m.group(1) for m in re.finditer(r"-isystem (\S+)", flags)]


def simde_roots(flags):
    """The -isystem roots of one compile that really hold a SIMDe install (simde/x86/avx2.h is there)."""
    return [p for p in isystems(flags) if (Path(p) / "simde" / "x86" / "avx2.h").is_file()]


def simde_used(proc, host, want, stale=()):
    """Every Strata-owned C/C++ host rule compiles against the SIMDe install this case chose, and no other one."""
    if proc.returncode != 0:
        return False, "macbuild.py exited %d: %s" % (proc.returncode, one_line(proc.stderr or proc.stdout))
    rules = host_compiles(host)
    if len(rules) < 2:
        return False, "host.ninja has %d C/C++ compile rules, expected the cxx and cc sources: %s" % (
            len(rules), [src for _, src, _ in rules])
    for rule, src, flags in rules:
        roots = simde_roots(flags)
        if str(want) not in roots:
            return False, "%s rule for %s compiles against SIMDe roots %s, the case installed %s" % (
                rule, src, roots, want)
        other = [p for p in roots if p != str(want)]
        if other:
            return False, "%s rule for %s also passes %s as a SIMDe root" % (rule, src, other)
        if "-isystem " + str(want) not in flags:
            return False, "%s rule for %s does not pass %s through -isystem: %s" % (rule, src, want, isystems(flags))
    leaked = [str(p) for p in stale if str(p) in host]
    if leaked:
        return False, "a default root is still in host.ninja: " + ", ".join(leaked)
    return True, ""


def main():
    if not SCRIPT.is_file():
        print("FAIL run this test from the copy root: sidecar-shim/macbuild.py is not in " + str(ROOT))
        return 1
    tmp = Path(tempfile.mkdtemp(prefix="macbuild-simde-"))
    try:
        sdk_dirs = {"SIDECAR_SDK_DIRS": str(planted_sdk(tmp / "sdks"))}

        env_simde = planted_simde(tmp / "env" / "simde-headers")
        proc, host, home = macbuild(tmp / "env-case", dict(sdk_dirs, SIDECAR_SIMDE=str(env_simde)))
        ok, detail = simde_used(proc, host, env_simde, stale=(home / "simde",))
        case("the environment chooses the SIMDe root", ok, detail)

        cli_simde = planted_simde(tmp / "cli" / "simde-headers")
        proc, host, home = macbuild(
            tmp / "cli-case", sdk_dirs, args=("--simde", str(cli_simde)))
        ok, detail = simde_used(proc, host, cli_simde, stale=(home / "simde",))
        case("the command line chooses the SIMDe root", ok, detail)

        proc, host, home = macbuild(
            tmp / "precedence-case",
            dict(sdk_dirs, SIDECAR_SIMDE=str(env_simde)),
            args=("--simde", str(cli_simde)),
        )
        ok, detail = simde_used(proc, host, cli_simde, stale=(home / "simde", env_simde))
        case("a command-line SIMDe root wins over the environment", ok, detail)

        proc, host, home = macbuild(tmp / "empty-env-case", dict(sdk_dirs, SIDECAR_SIMDE=""))
        want = planted_simde(home / "simde")
        ok, detail = simde_used(proc, host, want)
        case("an empty variable keeps the HOME default", ok, detail)

        proc, host, home = macbuild(tmp / "default-case", sdk_dirs)
        want = planted_simde(home / "simde")
        ok, detail = simde_used(proc, host, want)
        case("no override keeps the HOME default", ok, detail)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    failed = RESULTS.count(False)
    print(f"{failed} case(s) failed" if failed else "all cases passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
