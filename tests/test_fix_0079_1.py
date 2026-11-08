"""tests/test_fix_0079_1.py - macbuild.py takes both CUDA roots from its caller (fix-0079#1).

    python3 tests/test_fix_0079_1.py

sidecar-shim/macbuild.py writes device.ninja and host.ninja with two CUDA roots baked in: $HOME/cuda-12.8 for the
headers clang reads and $HOME/cuda-13.4 for nvcc. A machine that keeps its toolkits anywhere else cannot use the
script. The fix accepts --cuda-host/--cuda-dev and SIDECAR_CUDA_HOST/SIDECAR_CUDA_DEV, keeps the HOME paths as the
default so tools/build-sidecar.sh (HOME=$ROOT) and the Dockerfile still work, and an explicit command-line value wins
over the environment.

Every case is a small fake: a temp HOME, a temp configure command list in the shape `ninja -t commands strata` writes,
and temp CUDA roots. macbuild.py runs as a subprocess and only the two ninja files it writes are read. Nothing is
compiled, fetched or served, and nothing is written outside a temp folder. One PASS or FAIL line per case; any FAIL
exits non-zero.
"""
from __future__ import annotations

import os
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


def macbuild(tmp, env_extra=None, args=(), cmds_dev=None):
    """Run macbuild.py over a temp HOME and temp command list. Returns (process, device text, host text, home)."""
    tmp = Path(tmp)
    home = tmp / "home"
    lin = tmp / "strata-linux"
    home.mkdir(parents=True)
    lin.mkdir(parents=True)
    dev = Path(cmds_dev) if cmds_dev is not None else home / "cuda-13.4"
    cmds = lin / "strata_cmds.txt"
    cmds.write_text(configure_cmds(lin, dev), encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if k not in ("SIDECAR_CUDA_HOST", "SIDECAR_CUDA_DEV")}
    env["HOME"] = str(home)
    env.update(env_extra or {})
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--lin", str(lin), "--cmds", str(cmds), *args],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=120,
    )
    out = home / "strata-mac"
    device = (out / "device.ninja").read_text(encoding="utf-8") if (out / "device.ninja").is_file() else ""
    host = (out / "host.ninja").read_text(encoding="utf-8") if (out / "host.ninja").is_file() else ""
    return proc, device, host, home


def roots_ok(proc, device, host, cuda_host, cuda_dev, stale=()):
    """Both ninja files name the roots this case chose, and no default root leaks into a file that should not hold it."""
    if proc.returncode != 0:
        return False, "macbuild.py exited %d: %s" % (proc.returncode, (proc.stderr or proc.stdout).strip()[:200])
    want = [
        ("device.ninja runs nvcc from the dev root", device, str(cuda_dev / "bin" / "nvcc")),
        ("host.ninja reads the host headers", host, "--cuda-path=" + str(cuda_host)),
        ("host.ninja has the host include path", host, "-isystem " + str(cuda_host / "include")),
        ("host.ninja maps the dev include to the host root", host, "-I" + str(cuda_host / "include")),
    ]
    missing = [name for name, text, needle in want if needle not in text]
    if missing:
        return False, "missing: " + ", ".join(missing)
    leaked = [str(p) for p in stale if str(p) in device or str(p) in host]
    if leaked:
        return False, "a default root is still in the output: " + ", ".join(leaked)
    return True, ""


def main():
    if not SCRIPT.is_file():
        print("FAIL run this test from the copy root: sidecar-shim/macbuild.py is not in " + str(ROOT))
        return 1
    tmp = Path(tempfile.mkdtemp(prefix="macbuild-roots-"))
    try:
        env_host = tmp / "env" / "cuda-12.8-headers"
        env_dev = tmp / "env" / "cuda-13.4-toolkit"
        proc, device, host, home = macbuild(
            tmp / "env-case",
            {"SIDECAR_CUDA_HOST": str(env_host), "SIDECAR_CUDA_DEV": str(env_dev)},
            cmds_dev=env_dev,
        )
        ok, detail = roots_ok(proc, device, host, env_host, env_dev, stale=(home / "cuda-12.8", home / "cuda-13.4"))
        case("the environment chooses both CUDA roots", ok, detail)

        cli_host = tmp / "cli" / "cuda-12.8-headers"
        cli_dev = tmp / "cli" / "cuda-13.4-toolkit"
        proc, device, host, home = macbuild(
            tmp / "cli-case",
            args=("--cuda-host", str(cli_host), "--cuda-dev", str(cli_dev)),
            cmds_dev=cli_dev,
        )
        ok, detail = roots_ok(proc, device, host, cli_host, cli_dev, stale=(home / "cuda-12.8", home / "cuda-13.4"))
        case("the command line chooses both CUDA roots", ok, detail)

        proc, device, host, home = macbuild(
            tmp / "precedence-case",
            {"SIDECAR_CUDA_HOST": str(env_host), "SIDECAR_CUDA_DEV": str(env_dev)},
            args=("--cuda-host", str(cli_host), "--cuda-dev", str(cli_dev)),
            cmds_dev=cli_dev,
        )
        ok, detail = roots_ok(proc, device, host, cli_host, cli_dev, stale=(home / "cuda-12.8", home / "cuda-13.4"))
        case("a command-line root wins over the environment", ok, detail)

        proc, device, host, home = macbuild(
            tmp / "empty-env-case",
            {"SIDECAR_CUDA_HOST": "", "SIDECAR_CUDA_DEV": ""},
        )
        ok, detail = roots_ok(proc, device, host, home / "cuda-12.8", home / "cuda-13.4")
        case("an empty variable keeps the HOME default", ok, detail)

        proc, device, host, home = macbuild(tmp / "default-case")
        ok, detail = roots_ok(proc, device, host, home / "cuda-12.8", home / "cuda-13.4")
        case("no override keeps the HOME defaults", ok, detail)
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)

    failed = RESULTS.count(False)
    print(f"{failed} case(s) failed" if failed else "all cases passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
