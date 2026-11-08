"""tests/test_build_vm_stop.py - tools/build-sidecar.sh stops the build VM it started (F7).

    python3 tests/test_build_vm_stop.py

The build runs in a Docker container on the Mac. tools/build-sidecar.sh used to leave the Colima VM that holds
that container running after every build, and on 11-05 that VM held 7.0 to 8.2 GB on both 64 GB test machines.
The script must record whether the VM is already running: when it is down the script starts it, stops it in the
same EXIT trap that removes its container, and says so on one line; when it is already up the script leaves it
alone and says so.

No VM, no container and no build is real here. The script under test is copied into a temp repo; fake colima,
docker, rsync, ninja, zsh, python3.12 and cp scripts sit first on PATH and append every call to a log file, and
the CUDA, SIMDe, SDK and Python prerequisites are temp paths. Every case prints one PASS or FAIL line; any
FAIL exits non-zero.
"""
from __future__ import annotations

import os
import shutil
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path.cwd() if (Path.cwd() / "tools" / "build-sidecar.sh").is_file() else Path(__file__).resolve().parents[1]

STARTED = "build VM: started by this build, stopped on exit"
LEFT = "build VM: already running, left running"
results: list[bool] = []


def case(name, ok, detail=""):
    results.append(bool(ok))
    print(("PASS " if ok else "FAIL ") + name + (f": {detail}" if detail else ""))


def write(path, text, mode=0o644):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    path.chmod(mode)


def make_stubs(bindir):
    """Fake tools on PATH. They append their own call to $CALL_LOG, never touch a real VM or build."""
    write(bindir / "colima", """#!/bin/sh
printf 'colima %s\\n' "$*" >> "$CALL_LOG"
case "$1" in
    status) [ "${VM_UP:-0}" = 1 ] && exit 0 || exit 1 ;;
    start) exit 0 ;;
    stop) exit 0 ;;
esac
exit 0
""", 0o755)
    write(bindir / "docker", """#!/bin/sh
printf 'docker %s\\n' "$*" >> "$CALL_LOG"
case "$1" in
    build) [ "${DOCKER_BUILD_FAIL:-0}" = 1 ] && exit 1; exit 0 ;;
    create) printf 'fake-container\\n'; exit 0 ;;
esac
exit 0
""", 0o755)
    for name in ("rsync", "ninja", "zsh", "cp", "python3.12"):
        write(bindir / name, """#!/bin/sh
printf '%s %s\\n' "$(basename "$0")" "$*" >> "$CALL_LOG"
exit 0
""", 0o755)


def make_prereqs(work):
    """Temp CUDA, SIMDe, Python and SDK trees the script accepts as installed headers."""
    cuda = work / "cuda"
    write(cuda / "include" / "cuda_runtime.h", "")
    simde = work / "simde"
    write(simde / "simde" / "x86" / "avx2.h", "")
    prefix = work / "python"
    write(prefix / "include" / "python3.12" / "Python.h", "")
    sdk = work / "sdk"
    sdk.mkdir(parents=True)
    return cuda, simde, prefix, sdk


def make_repo(work):
    """A temp repo holding the real script under test and stubs for the heavy steps it calls."""
    repo = work / "repo"
    write(repo / "tools" / "build-sidecar.sh", (ROOT / "tools" / "build-sidecar.sh").read_text(encoding="utf-8"), 0o755)
    write(repo / "sidecar-build" / "Dockerfile", "FROM scratch\n")
    write(repo / "sidecar-shim" / "tgcu.py", "")
    write(repo / "exl3-metal" / "build.sh", "#!/bin/sh\nexit 0\n", 0o755)
    write(repo / "tools" / "stamp-build.sh", "#!/bin/sh\nexit 0\n", 0o755)
    # The script copies the built binary here and then runs it; the fake cp leaves this stub in place.
    write(repo / "build" / "sidecar" / "strata", "#!/bin/sh\nexit 0\n", 0o755)
    return repo


def run_case(tmp, name, vm_up, build_fail):
    work = Path(tmp) / name
    work.mkdir(parents=True)
    bindir = work / "bin"
    make_stubs(bindir)
    cuda, simde, prefix, sdk = make_prereqs(work)
    repo = make_repo(work)
    call_log = work / "calls.log"
    call_log.write_text("", encoding="utf-8")
    env = dict(os.environ)
    env.update({
        "PATH": f"{bindir}:{os.environ.get('PATH', '/usr/bin:/bin')}",
        "SIDECAR_CUDA_HOST": str(cuda),
        "SIDECAR_SIMDE": str(simde),
        "SIDECAR_PY_PREFIX": str(prefix),
        "SIDECAR_SDK": str(sdk),
        "CALL_LOG": str(call_log),
        "VM_UP": "1" if vm_up else "0",
        "DOCKER_BUILD_FAIL": "1" if build_fail else "0",
        "TMPDIR": str(work),
    })
    for key in ("SIDECAR_DOCKER", "SIDECAR_COLIMA", "SIDECAR_PYTHON"):
        env.pop(key, None)
    # The job shell exports a docker guard function that shadows PATH; the fake docker on PATH is the only
    # engine this test starts, so the child shell must resolve the name through PATH.
    env.pop("BASH_FUNC_docker%%", None)
    r = subprocess.run(["bash", str(repo / "tools" / "build-sidecar.sh")], cwd=str(repo), env=env,
                       capture_output=True, text=True, timeout=120)
    calls = [ln.strip() for ln in call_log.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return r.returncode, r.stdout + r.stderr, calls


def colima_calls(calls):
    return [call for call in calls if call.startswith("colima ")]


def index(calls, prefix):
    for i, call in enumerate(calls):
        if call.startswith(prefix):
            return i
    return -1


def main():
    tmp = tempfile.mkdtemp(prefix="test-build-vm-stop-")
    try:
        rc, out, calls = run_case(tmp, "down", vm_up=False, build_fail=False)
        start, stop = index(calls, "colima start"), index(calls, "colima stop")
        case("VM down: colima start called", start >= 0, f"exit {rc}, colima calls {colima_calls(calls)}")
        case("VM down: colima stop called on exit after start", stop >= 0 and start >= 0 and stop > start,
             f"start {start}, stop {stop}")
        case("VM down: the started line is printed once", out.count("build VM:") == 1 and STARTED in out,
             f"exit {rc}, build VM lines: {out.count('build VM:')}")
        case("VM down: the container is removed", index(calls, "docker rm -f") >= 0)

        rc, out, calls = run_case(tmp, "up", vm_up=True, build_fail=False)
        case("VM up: colima is not started or stopped",
             index(calls, "colima start") < 0 and index(calls, "colima stop") < 0, f"colima calls {colima_calls(calls)}")
        case("VM up: the left-running line is printed once", out.count("build VM:") == 1 and LEFT in out,
             f"exit {rc}, build VM lines: {out.count('build VM:')}")
        case("VM up: the container is removed", index(calls, "docker rm -f") >= 0)

        rc, out, calls = run_case(tmp, "fail", vm_up=False, build_fail=True)
        start, stop = index(calls, "colima start"), index(calls, "colima stop")
        case("build fails: the script exits non-zero", rc != 0, f"exit {rc}")
        case("build fails: colima stop is still called", start >= 0 and stop > start,
             f"exit {rc}, start {start}, stop {stop}")
        case("build fails: the started line is printed once", out.count("build VM:") == 1 and STARTED in out,
             f"exit {rc}, build VM lines: {out.count('build VM:')}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    failed = results.count(False)
    print(f"{len(results) - failed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
