"""tests/test_fix_0079_2.py - macbuild.py finds the newest installed 26.x macOS SDK (fix-0079#2).

    python3 tests/test_fix_0079_2.py

sidecar-shim/macbuild.py baked one absolute SDK path into every host compile and link:

    SDK = os.environ.get("SDK", "/Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk")

That path exists on the machine it was written on and nowhere else. On a Mac whose CommandLineTools ship 26.1,
or only 27.x, a run with SDK unset writes -isysroot /Library/.../MacOSX26.5.sdk into host.ninja and the first
clang invocation fails against a path that is not there, while nothing in the script names the variable that
would have fixed it. The script's own comment says the 27.0 SDK breaks ld, so discovery must stay on 26.x.

The fix reads the caller's SDK when it is set, and otherwise takes the newest installed 26.x SDK from the search
roots (CommandLineTools and Xcode by default, SIDECAR_SDK_DIRS when set so a caller or a test can name them); when
no 26.x SDK is installed it exits naming SDK instead of writing a fixed path.

Every case runs macbuild.py as a subprocess over a temp HOME, a temp configure command list in the shape
`ninja -t commands strata` writes, and a temp folder of planted SDK directories. Only the two ninja files the
script writes are read. Nothing is compiled, fetched or served, and nothing is written outside a temp folder.
One PASS or FAIL line per case; any FAIL exits non-zero.
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
    """The commands sidecar-build/Dockerfile captures, in the shape macbuild.py parses: nvcc, c++, ar, link."""
    return "\n".join([
        f"/usr/bin/nvcc -c {lin}/src/kernel.cu -o {lin}/CMakeFiles/strata.dir/src/kernel.cu.o"
        f" -I{cuda_dev}/include -DSTRATA_TGCU_BUILTIN=1 -std=c++17 -MD -MT kernel -MF kernel.d",
        f"c++ -c {lin}/src/main.cpp -o {lin}/CMakeFiles/strata.dir/src/main.cpp.o"
        f" -I{cuda_dev}/include -O2",
        f"ar qc {lin}/libstrata.a {lin}/CMakeFiles/strata.dir/src/kernel.cu.o"
        f" {lin}/CMakeFiles/strata.dir/src/main.cpp.o",
        f"c++ {lin}/CMakeFiles/strata.dir/src/kernel.cu.o"
        f" {lin}/CMakeFiles/strata.dir/src/main.cpp.o -o strata",
    ]) + "\n"


def planted_sdks(root, *names):
    """A search root holding SDK directories with the given names, as CommandLineTools/SDKs does."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    for name in names:
        (root / name).mkdir()
    return root


def macbuild(tmp, env_extra=None):
    """Run macbuild.py over a temp HOME and temp command list. Returns (process, device text, host text, home)."""
    tmp = Path(tmp)
    home = tmp / "home"
    lin = tmp / "strata-linux"
    home.mkdir(parents=True)
    lin.mkdir(parents=True)
    cmds = lin / "strata_cmds.txt"
    cmds.write_text(configure_cmds(lin, home / "cuda-13.4"), encoding="utf-8")
    drop = ("SDK", "SIDECAR_SDK_DIRS", "SIDECAR_CUDA_HOST", "SIDECAR_CUDA_DEV")
    env = {k: v for k, v in os.environ.items() if k not in drop}
    env["HOME"] = str(home)
    env.update(env_extra or {})
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--lin", str(lin), "--cmds", str(cmds)],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=120,
    )
    out = home / "strata-mac"
    device = (out / "device.ninja").read_text(encoding="utf-8") if (out / "device.ninja").is_file() else ""
    host = (out / "host.ninja").read_text(encoding="utf-8") if (out / "host.ninja").is_file() else ""
    return proc, device, host, home


def ninja_token(token):
    """Undo ninja's $ escaping ($$ for $, $  for a space, $: for a colon) in one path."""
    return re.sub(r"\$(.)", lambda m: m.group(1), token)


def isysroots(host):
    """Every SDK path host.ninja passes to -isysroot, in file order (one host compile, one link)."""
    return [ninja_token(t) for t in re.findall(r"-isysroot (\S+)", host)]


def sdk_used(proc, device, host, want):
    """host.ninja is built against exactly the SDK this case planted, and device.ninja names no SDK at all."""
    if proc.returncode != 0:
        return False, "macbuild.py exited %d: %s" % (proc.returncode, one_line(proc.stderr or proc.stdout))
    roots = isysroots(host)
    if len(roots) < 2:
        return False, "host.ninja passes %d -isysroot values, expected one per host compile plus the link: %s" % (
            len(roots), roots)
    wrong = sorted(set(r for r in roots if r != str(want)))
    if wrong:
        return False, "host.ninja builds against %s, the case installed %s" % (wrong, want)
    if "-isysroot" in device:
        return False, "device.ninja carries the macOS SDK, it runs in the Linux container"
    return True, ""


def main():
    if not SCRIPT.is_file():
        print("FAIL run this test from the copy root: sidecar-shim/macbuild.py is not in " + str(ROOT))
        return 1
    tmp = Path(tempfile.mkdtemp(prefix="macbuild-sdk-"))
    try:
        clt = planted_sdks(tmp / "sdks", "MacOSX26.1.sdk", "MacOSX26.4.sdk")
        proc, device, host, home = macbuild(tmp / "newest-case", {"SIDECAR_SDK_DIRS": str(clt)})
        ok, detail = sdk_used(proc, device, host, clt / "MacOSX26.4.sdk")
        case("the newest installed 26.x SDK is the default", ok, detail)

        planted_sdks(clt, "MacOSX27.0.sdk")
        proc, device, host, home = macbuild(tmp / "no-27-case", {"SIDECAR_SDK_DIRS": str(clt)})
        ok, detail = sdk_used(proc, device, host, clt / "MacOSX26.4.sdk")
        case("an installed 27.x SDK is never chosen over a 26.x", ok, detail)

        roots = tmp / "two-roots"
        planted_sdks(roots / "clt", "MacOSX26.0.sdk")
        planted_sdks(roots / "xcode", "MacOSX26.6.sdk")
        proc, device, host, home = macbuild(
            tmp / "two-roots-case",
            {"SIDECAR_SDK_DIRS": os.pathsep.join([str(roots / "clt"), str(roots / "xcode")])},
        )
        ok, detail = sdk_used(proc, device, host, roots / "xcode" / "MacOSX26.6.sdk")
        case("the newest 26.x SDK wins across search roots", ok, detail)

        chosen = planted_sdks(tmp / "chosen", "MacOSX26.9.sdk") / "MacOSX26.9.sdk"
        proc, device, host, home = macbuild(
            tmp / "explicit-case",
            {"SDK": str(chosen), "SIDECAR_SDK_DIRS": str(clt)},
        )
        ok, detail = sdk_used(proc, device, host, chosen)
        case("a caller's SDK still wins over discovery", ok, detail)

        only27 = planted_sdks(tmp / "only-27", "MacOSX27.0.sdk")
        proc, device, host, home = macbuild(tmp / "no-sdk-case", {"SIDECAR_SDK_DIRS": str(only27)})
        text = (proc.stdout or "") + (proc.stderr or "")
        ok = proc.returncode != 0 and "SDK" in text and "-isysroot" not in host
        case("no installed 26.x SDK names SDK instead of a fixed path", ok,
             "" if ok else "exited %d, output %r, host.ninja isysroot %s" % (
                 proc.returncode, one_line(text), isysroots(host)))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    failed = RESULTS.count(False)
    print(f"{failed} case(s) failed" if failed else "all cases passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
