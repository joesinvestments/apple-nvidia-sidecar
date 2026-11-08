#!/usr/bin/env python3
"""src/core/expert_source.cpp: the Metal lane's dlopen failure never names STRATA_EXL3_METAL_LIB.

ml_init builds the library path from STRATA_EXL3_METAL_LIB, else from
$HOME/exl3/metal/libexl3metal.dylib, which is a developer's own build directory and exists on no other
machine.  When the dlopen fails the message prints the resolved path and dlerror() but not the override
that would fix it, so an operator who turned the lane on with STRATA_EXL3_METAL=1 has no hint about how
to point it at their own build.  ml_init then returns false and every caller (strata_pf_mac_init and the
rest) silently skips the lane.

This test compiles the real ml_init prologue (its signature through the dlopen failure guard) out of the
tree under test against a small fake MetalLane and runs it with the system's own dlopen, no GPU, no model
and no network:
  - no override, an empty home: the failure names STRATA_EXL3_METAL_LIB;
  - no override and no home at all: the failure names STRATA_EXL3_METAL_LIB;
  - an override that points at a missing file: the failure names that path and STRATA_EXL3_METAL_LIB;
  - config/sidecar.json's own value, whose build is absent here: the failure names STRATA_EXL3_METAL_LIB;
  - an override that points at a loadable library: it loads with no message (the ordinary path, the same
    answer before and after the fix).
Run from the tree root: python3 tests/test_fix_0117_2.py
Prints one PASS or FAIL line per case and exits non-zero on any FAIL.  Temp folders only.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

SRC_REL = os.path.join("src", "core", "expert_source.cpp")
CONFIG_REL = os.path.join("config", "sidecar.json")
SIG = "bool ml_init(MetalLane& M) {"
OVERRIDE = "STRATA_EXL3_METAL_LIB"

# the fake lane carries only what the prologue touches, and the probe library gives a real dlopen a real file
PRELUDE = r"""
#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>
#include <string>

struct MetalLane {
    bool tried = false;
    bool ok = false;
    void* lib = nullptr;
};

static MetalLane& ml() { static MetalLane m; return m; }

@@BLOCK@@
    return M.lib != nullptr;
}

int main() {
    MetalLane& M = ml();
    const bool loaded = ml_init(M);
    std::printf("loaded=%d tried=%d lib=%d\n", loaded ? 1 : 0, M.tried ? 1 : 0, M.lib != nullptr ? 1 : 0);
    return 0;
}
"""

PROBE = "int exl3m_prologue_probe() { return 1; }\n"

fails = 0


def check(name, ok, detail=""):
    global fails
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else ": " + detail))
    if not ok:
        fails += 1


def find_root():
    """The tree under test: the copy prove_test runs us in, else the one this file sits in."""
    for base in (os.getcwd(), os.path.dirname(os.path.dirname(os.path.abspath(__file__)))):
        if os.path.isfile(os.path.join(base, SRC_REL)):
            return base
    return os.getcwd()


def matching_brace(text, i):
    """The index of the brace that closes the one at i, with comments and literals skipped."""
    depth, n = 0, len(text)
    while i < n:
        c = text[i]
        if text.startswith("//", i):
            j = text.find("\n", i)
            i = n if j < 0 else j
        elif text.startswith("/*", i):
            j = text.find("*/", i + 2)
            i = n if j < 0 else j + 2
        elif c in "\"'":
            quote = c
            i += 1
            while i < n and text[i] != quote:
                i += 2 if text[i] == "\\" else 1
            i += 1
        else:
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return i
            i += 1
    return -1


def prologue(src):
    """ml_init from its signature through the dlopen failure guard, exactly as the tree writes it."""
    start = src.find(SIG)
    if start < 0:
        return None, "ml_init is gone from " + SRC_REL
    load = src.find("M.lib = dlopen(", start)
    if load < 0:
        return None, "ml_init no longer loads the Metal library with dlopen"
    guard = src.find("if (M.lib == nullptr)", load)
    if guard < 0:
        return None, "ml_init has no dlopen failure guard"
    brace = src.find("{", guard)
    if brace < 0:
        return None, "the dlopen failure guard has no body"
    close = matching_brace(src, brace)
    if close < 0:
        return None, "the dlopen failure guard has no closing brace"
    return src[start:close + 1], None


def compiler():
    for name in (os.environ.get("CXX"), "clang++", "g++", "c++"):
        if name and shutil.which(name):
            return name
    return None


def build(cxx, work, block):
    src = os.path.join(work, "ml_init_prologue.cpp")
    exe = os.path.join(work, "ml_init_prologue")
    with open(src, "w", encoding="utf-8") as f:
        f.write(PRELUDE.replace("@@BLOCK@@", block))
    r = subprocess.run([cxx, "-std=c++17", "-O0", "-o", exe, src], capture_output=True, text=True, timeout=300)
    if r.returncode != 0:
        errors = [l.strip() for l in (r.stdout + r.stderr).splitlines() if " error: " in l]
        return None, (errors[0] if errors else "see the compile log")
    return exe, ""


def probe_library(cxx, work):
    """A real loadable library the lane can dlopen: the branch that must stay quiet."""
    src = os.path.join(work, "probe.cpp")
    lib = os.path.join(work, "libexl3metal.dylib")
    with open(src, "w", encoding="utf-8") as f:
        f.write(PROBE)
    shape = "-dynamiclib" if sys.platform == "darwin" else "-shared"
    r = subprocess.run([cxx, "-std=c++17", "-O0", shape, "-o", lib, src], capture_output=True, text=True, timeout=300)
    if r.returncode != 0:
        return None, (r.stdout + r.stderr).strip()[:300]
    return lib, ""


def run(exe, cwd, env_extra):
    """The probe with a clean environment: no inherited override, HOME or lane switch."""
    e = dict(os.environ)
    for k in ("HOME", OVERRIDE, "STRATA_EXL3_METAL"):
        e.pop(k, None)
    e.update(env_extra)
    r = subprocess.run([exe], capture_output=True, text=True, timeout=120, env=e, cwd=cwd)
    return r.returncode, r.stdout.strip(), r.stderr


def parse(stdout):
    out = {}
    for part in stdout.split():
        k, _, v = part.partition("=")
        out[k] = int(v)
    return out


def observe(rc, out, err):
    """The program's line, or why the fake lane did not run at all."""
    if rc != 0:
        return None, "exit %d: %s" % (rc, (out + err).strip()[-200:])
    try:
        return parse(out), ""
    except ValueError:
        return None, "cannot read the observation: " + out


def main():
    root = find_root()
    src_path = os.path.join(root, SRC_REL)
    if not os.path.isfile(src_path):
        check("the tree holds " + SRC_REL, False, src_path + " is missing")
        return
    with open(src_path, encoding="utf-8", errors="replace") as f:
        src = f.read()
    block, err = prologue(src)
    if block is None:
        check("ml_init's dlopen prologue is readable", False, err)
        return

    cxx = compiler()
    if cxx is None:
        check("a C++ compiler is available", False, "no clang++, g++, c++ on PATH")
        return
    work = tempfile.mkdtemp(prefix="fix-0117-2-")
    try:
        exe, detail = build(cxx, work, block)
        if exe is None:
            check("the faked ml_init prologue compiles", False, detail)
            return
        check("the faked ml_init prologue compiles", True)
        probe, detail = probe_library(cxx, work)
        if probe is None:
            check("a loadable probe library builds", False, detail)
            return

        home = os.path.join(work, "home")
        os.makedirs(home, exist_ok=True)
        missing = os.path.join(work, "absent", "libexl3metal.dylib")

        # the production sample: the library path the selected config itself sets, read only
        with open(os.path.join(root, CONFIG_REL), encoding="utf-8") as f:
            configured = (json.load(f).get("env") or {}).get(OVERRIDE)
        if not isinstance(configured, str) or not configured:
            check(CONFIG_REL + " names " + OVERRIDE, False, "the config carries no Metal library path")
            return
        if os.path.exists(os.path.join(root, configured)):
            check(CONFIG_REL + " names " + OVERRIDE, False,
                  "%s is present in the tree, so the failure case cannot be planted" % configured)
            return

        cases = [
            ("no override and an empty home names the override in the failure",
             {"HOME": home},
             lambda o, e: (o.get("lib") == 0 and OVERRIDE in e, e.strip()[-240:])),
            ("no override and no home at all names the override in the failure",
             {},
             lambda o, e: (o.get("lib") == 0 and OVERRIDE in e, e.strip()[-240:])),
            ("an override at a missing file names that path and the override",
             {"HOME": home, OVERRIDE: missing},
             lambda o, e: (o.get("lib") == 0 and missing in e and OVERRIDE in e, e.strip()[-240:])),
            ("the configured library path, whose build is absent here, names the override",
             {"HOME": home, OVERRIDE: configured},
             lambda o, e: (o.get("lib") == 0 and configured in e and OVERRIDE in e, e.strip()[-240:])),
        ]

        for name, env_extra, want in cases:
            rc, out, err = run(exe, root, env_extra)
            obs, why = observe(rc, out, err)
            if obs is None:
                check(name, False, why)
                continue
            ok, detail = want(obs, err)
            check(name, ok, "message: " + detail)

        # the ordinary path: a library that is there loads, and the lane says nothing
        rc, out, err = run(exe, root, {"HOME": home, OVERRIDE: probe})
        obs, why = observe(rc, out, err)
        if obs is None:
            check("a loadable library at the override still loads without a message", False, why)
        else:
            ok = obs.get("lib") == 1 and obs.get("loaded") == 1 and not err.strip()
            check("a loadable library at the override still loads without a message", ok,
                  "loaded=%s lib=%s stderr=%r" % (obs.get("loaded"), obs.get("lib"), err.strip()[-240:]))
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
    sys.exit(1 if fails else 0)
