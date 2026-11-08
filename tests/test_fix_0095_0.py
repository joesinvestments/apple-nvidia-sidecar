#!/usr/bin/env python3
"""tests/test_fix_0095_0.py - a block whose thread count wraps in 32 bits must still be refused.

sidecar-shim/tgcudart.cpp:302-303 reads
    if (g.x == 0 || g.y == 0 || g.z == 0 || b.x * b.y * b.z == 0) return fail(cudaErrorInvalidConfiguration, "launch dims");
    if (b.x * b.y * b.z > 1024) return fail(cudaErrorInvalidConfiguration, "block too large");
dim3 holds three unsigned int, so the product is computed in 32 bits and can wrap: the block (1024, 4194305, 1)
multiplies to 2**32 + 1024, wraps to 1024, passes both guards and reaches the launch call. The card cannot run that
block (maxThreadsPerBlock is 1024 and maxThreadsDim is 1024/1024/64), and no other guard in the shim checks the
individual dimensions.

No GPU, no CUDA headers and no Python headers are needed: these cases take the real do_launch function out of
sidecar-shim/tgcudart.cpp, compile it against a stub runtime (dim3, fail, call_i, captured) and run it.
  - a block whose product wraps to 1024, and one that wraps to 4, are refused as cudaErrorInvalidConfiguration and
    never reach the launch call;
  - an ordinary block and the 1024 thread block at the limit still launch;
  - blocks plainly over the limit, a block over it with no wrap, and a zero grid are still refused.
Prints one PASS or FAIL line per case and exits nonzero on any FAIL.
Run: python3 tests/test_fix_0095_0.py
"""
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.getcwd()   # the tree under test: every test here runs with cwd at the copy root, never at the test file
SHIM = os.path.join("sidecar-shim", "tgcudart.cpp")

PRELUDE = """// built by tests/test_fix_0095_0.py: the stub runtime do_launch needs, then the real function below
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdarg>
#include <string>

typedef int cudaError_t;
typedef void* cudaStream_t;

enum {
  cudaSuccess = 0,
  cudaErrorInvalidValue = 1,
  cudaErrorMemoryAllocation = 2,
  cudaErrorInitializationError = 3,
  cudaErrorLaunchFailure = 4,
  cudaErrorNotReady = 5,
  cudaErrorInvalidConfiguration = 9,
  cudaErrorInvalidDeviceFunction = 98,
  cudaErrorCooperativeLaunchTooLarge = 1030
};

struct dim3 {
  unsigned int x, y, z;
  dim3(unsigned int vx = 1, unsigned int vy = 1, unsigned int vz = 1) : x(vx), y(vy), z(vz) {}
};

struct Func { const void* fat; std::string name; int kid; };

static int launch_calls = 0;
static int fail_calls = 0;
static long long launch_rc = 0;
static bool g_trace = false;

static cudaError_t fail(cudaError_t e, const char* what) { (void)what; fail_calls++; return e; }
static bool captured(cudaStream_t s) { (void)s; return false; }

static long long call_i(const char* fn, const char* fmt, ...) {
  (void)fmt;
  if (std::string(fn) == "launch") launch_calls++;
  return launch_rc;
}
"""

MAIN = """
static const char* err_name(cudaError_t e) {
  switch (e) {
    case cudaSuccess: return "success";
    case cudaErrorInvalidConfiguration: return "invalid_configuration";
    case cudaErrorLaunchFailure: return "launch_failure";
    case cudaErrorInvalidDeviceFunction: return "invalid_device_function";
    case cudaErrorCooperativeLaunchTooLarge: return "cooperative_too_large";
    default: return "other";
  }
}

static Func g_fn;

static void run_probe(const char* name, dim3 g, dim3 b) {
  launch_calls = 0;
  fail_calls = 0;
  launch_rc = 0;
  cudaError_t e = do_launch(&g_fn, g, b, 0, nullptr, 0, nullptr);
  std::printf("probe %s rc=%s launches=%d\\n", name, err_name(e), launch_calls);
}

int main() {
  g_fn.fat = (const void*)1;
  g_fn.name = "probe_kernel";
  g_fn.kid = 0;
  // 1024 * 4194305 = 2**32 + 1024, so the 32-bit product wraps to 1024 and passes the size guard
  run_probe("wrap_to_1024", dim3(64, 1, 1), dim3(1024, 4194305, 1));
  // 4 * 1073741825 = 2**32 + 4, so the 32-bit product wraps to 4 and passes the size guard
  run_probe("wrap_to_4", dim3(64, 1, 1), dim3(4, 1073741825, 1));
  run_probe("ordinary_256x1x1", dim3(64, 1, 1), dim3(256, 1, 1));
  run_probe("full_block_1024x1x1", dim3(64, 1, 1), dim3(1024, 1, 1));
  run_probe("oversize_2048x1x1", dim3(64, 1, 1), dim3(2048, 1, 1));
  run_probe("over_limit_512x3x1", dim3(64, 1, 1), dim3(512, 3, 1));
  run_probe("no_wrap_1024x1024x1", dim3(64, 1, 1), dim3(1024, 1024, 1));
  run_probe("wrap_to_zero_65536x65536x1", dim3(1, 1, 1), dim3(65536, 65536, 1));
  run_probe("zero_grid_0x1x1", dim3(0, 1, 1), dim3(256, 1, 1));
  return 0;
}
"""

# want = (error name the guard must return, number of launch calls that may reach the driver)
CASES = [
    dict(name="a block of 1024 x 4194305 x 1 wraps to 1024 and must be refused, not launched", probe="wrap_to_1024",
         want=("invalid_configuration", 0)),
    dict(name="a block of 4 x 1073741825 x 1 wraps to 4 and must be refused, not launched", probe="wrap_to_4",
         want=("invalid_configuration", 0)),
    dict(name="an ordinary block of 256 x 1 x 1 still launches", probe="ordinary_256x1x1", want=("success", 1)),
    dict(name="the full block of 1024 x 1 x 1 at the limit still launches", probe="full_block_1024x1x1",
         want=("success", 1)),
    dict(name="a block of 2048 x 1 x 1 is still refused", probe="oversize_2048x1x1",
         want=("invalid_configuration", 0)),
    dict(name="a block of 512 x 3 x 1 over the limit is still refused", probe="over_limit_512x3x1",
         want=("invalid_configuration", 0)),
    dict(name="a block of 1024 x 1024 x 1 with no wrap is still refused", probe="no_wrap_1024x1024x1",
         want=("invalid_configuration", 0)),
    dict(name="a block of 65536 x 65536 x 1 that wraps to zero is still refused", probe="wrap_to_zero_65536x65536x1",
         want=("invalid_configuration", 0)),
    dict(name="a zero grid is still refused", probe="zero_grid_0x1x1", want=("invalid_configuration", 0)),
]


def compiler():
    for name in ("g++", "clang++", "c++"):
        path = shutil.which(name)
        if path:
            return path
    return None


def launch_source(text):
    """The real cudaError_t do_launch body, brace matched, so the cases run the shim's own code."""
    start = text.find("cudaError_t do_launch(")
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
    source = launch_source(text)
    if not source:
        return None, "%s does not hold cudaError_t do_launch(...) to compile" % SHIM
    cxx = compiler()
    if not cxx:
        return None, "no C++ compiler (g++, clang++, c++) on PATH"
    src = os.path.join(work, "do_launch_probe.cpp")
    exe = os.path.join(work, "do_launch_probe")
    with open(src, "w", encoding="utf-8") as f:
        f.write(PRELUDE + "\n" + source + "\n" + MAIN)
    built = subprocess.run([cxx, "-std=c++17", "-O0", "-o", exe, src], capture_output=True, text=True, timeout=180)
    if built.returncode != 0:
        errors = [l.strip() for l in (built.stdout + built.stderr).splitlines() if " error: " in l]
        return None, "the do_launch probe does not build (" + (errors[0] if errors else "see the compile log") + ")"
    run = subprocess.run([exe], capture_output=True, text=True, timeout=120)
    if run.returncode != 0:
        return None, "the do_launch probe exited %d: %s" % (run.returncode, run.stdout.strip()[-200:])
    seen = {}
    for line in run.stdout.splitlines():
        parts = line.split()
        if len(parts) == 4 and parts[0] == "probe" and parts[2].startswith("rc=") and parts[3].startswith("launches="):
            seen[parts[1]] = (parts[2][3:], int(parts[3][9:]))
    return seen, ""


def main():
    work = tempfile.mkdtemp(prefix="fix-0095-")
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
                    ok, detail = False, "got rc=%s launches=%d, expected rc=%s launches=%d" % (
                        got[0], got[1], case["want"][0], case["want"][1])
                else:
                    ok, detail = True, "rc=%s launches=%d" % got
            if not ok:
                bad += 1
            print(("PASS " if ok else "FAIL ") + case["name"] + (": " + detail if detail else ""))
        return 1 if bad else 0
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
