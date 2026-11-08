"""tests/test_fix_0088_0.py - a malformed TGCU_FATBIN_SWAP token must not crash the shim at import (fix-0088#0).

    python3 tests/test_fix_0088_0.py

sidecar-shim/tgcu_api.py:527 builds the lab-only fatbin swap table straight from the environment:

    _SWAP = dict(kv.split("=", 1) for kv in re.split(r"[;,]", os.getenv("TGCU_FATBIN_SWAP", "")) if kv)

A token without "=" splits into one field, so dict() raises ValueError while the module is still being imported and the
whole shim fails to load. This test imports the real module in a child interpreter over a stub tgcu driver (no GPU, no
model, no weights) and reads the swap table the module exposes: a token without "=" must be skipped so the shim loads
with an empty table, a well formed pair must still map exactly as before, and a broken token next to good ones must
leave the good pairs alone.

The shim targets Python 3.12 or newer (an f-string in it holds a backslash), so the module is loaded in the first
interpreter tried that can import it, starting with the one running this test; the tree under test is only read.
Every case prints one PASS or FAIL line; any FAIL exits non-zero.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

SHIM = "sidecar-shim/tgcu_api.py"
ROOT = Path.cwd() if (Path.cwd() / SHIM).is_file() else Path(__file__).resolve().parents[1]

# the child: a stub tgcu (any module attribute is a placeholder), the shim's own folder on sys.path the way tgcudart
# loads it (so its `import geom` finds sidecar-shim/geom.py), then the real shim loaded from its path
BOOT = '''import importlib.util, json, os, sys, types


class _Any:
    def __getattr__(self, name):
        return _Any()

    def __call__(self, *args, **kwargs):
        return _Any()


stub = types.ModuleType("tgcu")
stub.__getattr__ = lambda name: _Any()
sys.modules["tgcu"] = stub
sys.path.insert(0, os.path.dirname(os.path.abspath(sys.argv[1])))
spec = importlib.util.spec_from_file_location("tgcu_api_under_test", sys.argv[1])
module = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(module)
except Exception as e:
    json.dump({"error": type(e).__name__, "detail": str(e)[:200]}, open(sys.argv[2], "w"))
    raise SystemExit(2)
json.dump({"swap": {str(k): str(v) for k, v in module._SWAP.items()}}, open(sys.argv[2], "w"))
'''

GOOD = {"0123456789abcdef": "/tmp/rebuilt.fatbin"}
CASES = [
    ("the shim loads with an empty swap table when TGCU_FATBIN_SWAP is unset", None, {}),
    ("a well formed pair still maps its digest to the rebuilt fatbin", "0123456789abcdef=/tmp/rebuilt.fatbin", GOOD),
    ("a value that holds = keeps everything after the first one", "0123456789abcdef=/tmp/a=b.fatbin",
     {"0123456789abcdef": "/tmp/a=b.fatbin"}),
    ("a token with no = is skipped instead of crashing the import", "abc", {}),
    ("a token with no = next to a good pair leaves the good pair", "abc;0123456789abcdef=/tmp/rebuilt.fatbin", GOOD),
    ("a token with no = between two good pairs is skipped", "a=1,broken,b=2", {"a": "1", "b": "2"}),
]
fails = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global fails
    print(("PASS " if ok else "FAIL ") + name + (": " + detail if detail else ""))
    if not ok:
        fails += 1


def child_env(value: str | None) -> dict:
    """The caller's environment without any TGCU_ switch, plus the one value under test."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("TGCU_")}
    env["PYTHONDONTWRITEBYTECODE"] = "1"   # the tree under test stays byte for byte as it is
    if value is not None:
        env["TGCU_FATBIN_SWAP"] = value
    return env


def import_shim(exe: str, boot: str, work: str, value: str | None) -> dict:
    """Load the shim in exe and return the swap table it exposes, or the error that stopped the import."""
    out = os.path.join(work, "result.json")
    if os.path.exists(out):
        os.remove(out)
    run = subprocess.run([exe, boot, SHIM, out], cwd=str(ROOT), env=child_env(value),
                         capture_output=True, text=True, timeout=300)
    if not os.path.isfile(out):
        tail = ((run.stderr or "") + (run.stdout or "")).strip().splitlines()
        return {"error": "no result", "detail": (tail[-1] if tail else "exit %d" % run.returncode)[-200:]}
    with open(out) as f:
        return json.load(f)


def interpreters() -> list:
    found = [sys.executable]
    for name in ("python3.12", "python3.13", "python3.14", "python3.15", "python3.11", "python3.10", "python3"):
        path = shutil.which(name)
        if path:
            found.append(path)
    seen, unique = set(), []
    for path in found:
        real = os.path.realpath(path)
        if real not in seen:
            seen.add(real)
            unique.append(path)
    return unique


def main() -> int:
    if not (ROOT / SHIM).is_file():
        print("FAIL run this test from the copy root: " + SHIM + " is not in " + str(ROOT))
        return 1
    work = tempfile.mkdtemp(prefix="fix-0088-")
    try:
        boot = os.path.join(work, "boot.py")
        with open(boot, "w") as f:
            f.write(BOOT)
        exe, first, tried = None, None, []
        for path in interpreters():
            result = import_shim(path, boot, work, None)
            if "swap" in result:
                exe, first = path, result
                break
            tried.append(os.path.basename(path) + ": " + str(result.get("error")) + " " + str(result.get("detail", "")))
        if exe is None:
            print("FAIL no interpreter could import " + SHIM + ": " + " | ".join(tried[-3:]))
            return 1
        print("loading " + SHIM + " with " + exe)
        for name, value, want in CASES:
            result = first if value is None else import_shim(exe, boot, work, value)
            if "error" in result:
                check(name, False, "the import raised " + str(result["error"]) + ": " + str(result.get("detail", "")))
                continue
            got = result["swap"]
            check(name, got == want, "" if got == want else "the swap table is %r, expected %r" % (got, want))
    finally:
        shutil.rmtree(work, ignore_errors=True)
    print("%d case(s) failed" % fails if fails else "all cases passed")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
