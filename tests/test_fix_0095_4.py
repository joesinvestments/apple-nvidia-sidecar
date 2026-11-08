#!/usr/bin/env python3
"""tests/test_fix_0095_4.py - a failed sidecar free must not be reported as success.

sidecar-shim/tgcudart.cpp cudaFree() validates the pointer against g_dev and then calls the Python sidecar:

    call_i("free", "(K)", (unsigned long long)(uintptr_t)p);
    return cudaSuccess;

The return value of call_i is dropped, so when the device-side free fails cudaFree still answers cudaSuccess.
The caller believes the allocation is gone while the device memory is still held, and the bookkeeping entry
has already been erased, so nothing can retry or report the leak. The same file checks the analogous
sysmem_free return in tgcuSysmemFree(), so the sidecar call yields a status that can be checked.

These cases take the real cudaFree function out of sidecar-shim/tgcudart.cpp, compile it against a stub
runtime (g_dev, API_ENTER, fail, call_i) and call it. No GPU, no CUDA headers and no Python headers are
needed, and the sidecar free result is the planted input:
  - a sidecar free that returns nonzero must arrive at the caller as cudaErrorInvalidValue, not cudaSuccess;
    this case FAILS on today's code;
  - a second nonzero result is surfaced the same way;
  - a sidecar free that returns zero still reports cudaSuccess, as before;
  - a pointer the shim never allocated is still refused without reaching the sidecar;
  - a null pointer still reports cudaSuccess without reaching the sidecar.
Prints one PASS or FAIL line per case and exits nonzero on any FAIL.
Run from the copy root: python3 tests/test_fix_0095_4.py
"""
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.getcwd()   # the tree under test: every test here runs with cwd at the copy root, never at the test file
SHIM = os.path.join("sidecar-shim", "tgcudart.cpp")

PRELUDE = """// built by tests/test_fix_0095_4.py: the stub runtime cudaFree needs, then the real function below
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdarg>
#include <map>
#include <string>

typedef int cudaError_t;

enum {
  cudaSuccess = 0,
  cudaErrorInvalidValue = 1,
  cudaErrorMemoryAllocation = 2,
  cudaErrorInitializationError = 3
};

#define API_ENTER(name) ((void)(name))

static std::map<uint64_t, size_t> g_dev;

static int free_calls = 0;
static int fail_calls = 0;
static long long free_rc = 0;

static cudaError_t fail(cudaError_t e, const char* what) {
  (void)what;
  fail_calls++;
  return e;
}

static long long call_i(const char* fn, const char* fmt, ...) {
  (void)fmt;
  if (std::string(fn) == "free") {
    free_calls++;
    return free_rc;
  }
  return -1;
}
"""

MAIN = """
static void run_probe(const char* name, void* p, bool known, long long rc) {
  g_dev.clear();
  if (known) g_dev[(uint64_t)p] = 4096;
  free_calls = 0;
  fail_calls = 0;
  free_rc = rc;
  cudaError_t e = cudaFree(p);
  std::printf("probe %s rc=%d free_calls=%d fail_calls=%d\\n", name, (int)e, free_calls, fail_calls);
}

int main() {
  // the device-side free failed with a nonzero status, planted on each case
  run_probe("free_fails", (void*)(uintptr_t)0x1000, true, -1);
  run_probe("free_fails_again", (void*)(uintptr_t)0x2000, true, -7);
  // ordinary input: the sidecar free succeeded and behaves as before
  run_probe("free_ok", (void*)(uintptr_t)0x3000, true, 0);
  // ordinary input: the pointer was never allocated here, so the sidecar is not reached
  run_probe("unknown_pointer", (void*)(uintptr_t)0x4000, false, 0);
  run_probe("null_pointer", nullptr, false, 0);
  return 0;
}
"""

# name, probe, want (error code, sidecar free calls, fail calls)
CASES = [
    dict(name="a failed sidecar free is reported as cudaErrorInvalidValue, not success", probe="free_fails",
         want=(1, 1, 1)),
    dict(name="a second nonzero sidecar free result is reported as an error", probe="free_fails_again",
         want=(1, 1, 1)),
    dict(name="a successful sidecar free still reports cudaSuccess", probe="free_ok", want=(0, 1, 0)),
    dict(name="an unknown pointer is still refused without a sidecar free", probe="unknown_pointer",
         want=(1, 0, 1)),
    dict(name="a null pointer still reports cudaSuccess without a sidecar free", probe="null_pointer",
         want=(0, 0, 0)),
]


def compiler():
    for name in ("g++", "clang++", "c++"):
        path = shutil.which(name)
        if path:
            return path
    return None


def function_source(text, signature):
    """The real function body, brace matched, so the cases call the shim's own code."""
    start = text.find(signature)
    if start < 0:
        return None
    brace = text.find("{", start)
    if brace < 0:
        return None
    depth = 0
    i = brace
    n = len(text)
    while i < n:
        c = text[i]
        if c in ('"', "'"):
            quote = c
            i += 1
            while i < n and text[i] != quote:
                i += 2 if text[i] == "\\" else 1
        elif text.startswith("//", i):
            j = text.find("\n", i)
            i = n if j < 0 else j
            continue
        elif text.startswith("/*", i):
            j = text.find("*/", i + 2)
            i = n if j < 0 else j + 2
            continue
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
        i += 1
    return None


def build_and_run(work):
    shim = os.path.join(ROOT, SHIM)
    try:
        with open(shim, encoding="utf-8") as f:
            text = f.read()
    except OSError as e:
        return None, "cannot read %s: %s" % (SHIM, e)
    source = function_source(text, "cudaError_t cudaFree(void* p)")
    if not source:
        return None, "%s does not hold cudaError_t cudaFree(void* p) to compile" % SHIM
    cxx = compiler()
    if not cxx:
        return None, "no C++ compiler (g++, clang++, c++) on PATH"
    src = os.path.join(work, "cuda_free_probe.cpp")
    exe = os.path.join(work, "cuda_free_probe")
    with open(src, "w", encoding="utf-8") as f:
        f.write(PRELUDE + "\n" + source + "\n" + MAIN)
    built = subprocess.run([cxx, "-std=c++17", "-O0", "-o", exe, src], capture_output=True, text=True, timeout=180)
    if built.returncode != 0:
        errors = [l.strip() for l in (built.stdout + built.stderr).splitlines() if " error: " in l]
        return None, "the cudaFree probe does not build (" + (errors[0] if errors else "see the compile log") + ")"
    run = subprocess.run([exe], capture_output=True, text=True, timeout=120)
    if run.returncode != 0:
        return None, "the cudaFree probe exited %d: %s" % (run.returncode, run.stdout.strip()[-200:])
    seen = {}
    for line in run.stdout.splitlines():
        parts = line.split()
        if len(parts) == 5 and parts[0] == "probe" and parts[2].startswith("rc=") and \
                parts[3].startswith("free_calls=") and parts[4].startswith("fail_calls="):
            seen[parts[1]] = (int(parts[2][3:]), int(parts[3][11:]), int(parts[4][11:]))
    return seen, ""


def main():
    work = tempfile.mkdtemp(prefix="fix-0095-4-")
    try:
        seen, error = build_and_run(work)
        bad = 0
        for case in CASES:
            if seen is None:
                ok, detail = False, error
            else:
                got = seen.get(case["probe"])
                if got is None:
                    ok, detail = False, "the probe printed no result for " + case["probe"]
                elif got != case["want"]:
                    ok, detail = False, "got rc=%d free_calls=%d fail_calls=%d, expected rc=%d free_calls=%d fail_calls=%d" % (
                        got[0], got[1], got[2], case["want"][0], case["want"][1], case["want"][2])
                else:
                    ok, detail = True, "rc=%d free_calls=%d fail_calls=%d" % got
            if not ok:
                bad += 1
            print(("PASS " if ok else "FAIL ") + case["name"] + (": " + detail if detail else ""))
        return 1 if bad else 0
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
