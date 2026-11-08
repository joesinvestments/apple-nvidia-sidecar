#!/usr/bin/env python3
"""tests/test_fix_0097_2.py - cudaMemcpyFromSymbol must infer a device destination.

sidecar-shim/tgcudart.cpp cudaMemcpyFromSymbol ends with

    return memcpy_impl(dst, (void*)(a + off), n, kind == cudaMemcpyDefault ? cudaMemcpyDeviceToHost : kind);

so cudaMemcpyFromSymbol(dst, sym, n, off, cudaMemcpyDefault) labels every copy device to host, even when dst is
a device pointer (a legal CUDA call that copies the symbol to device memory). The symmetric ToSymbol path above
already infers the direction from is_dev(src). These cases cut the real cudaMemcpyFromSymbol out of the shim, build
it against toy helpers (no GPU, no model, no CPython), and print the memcpy_impl call it makes:
  - default kind with a device destination is device to device. This case FAILS on today's code;
  - default kind with a host destination is still device to host, as before;
  - explicit kinds are passed through unchanged, and off and n reach memcpy_impl;
  - an unknown symbol is refused and calls no copy.
The ordinary cases give the same result before and after the fix.

Run from the copy root: python3 tests/test_fix_0097_2.py
Prints one PASS or FAIL line per case and exits nonzero on any FAIL.
"""
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

REL = Path("sidecar-shim") / "tgcudart.cpp"

# the harness's address space: no real memory, only the shim's is_dev bookkeeping
DEV_BASE = 0x7000001000      # a symbol's device address
DEV_DST = 0x7000002000       # a device destination pointer
DEV_SIZE = 0x100000

FAKE_HELPERS = r"""
#define EXPORT extern "C"
typedef int cudaError_t;

enum cudaMemcpyKind {
  cudaMemcpyHostToHost = 0, cudaMemcpyHostToDevice = 1, cudaMemcpyDeviceToHost = 2,
  cudaMemcpyDeviceToDevice = 3, cudaMemcpyDefault = 4
};
enum { cudaSuccess = 0, cudaErrorInvalidSymbol = 201 };

static const uintptr_t DEV_BASE = %d;
static const uintptr_t DEV_DST = %d;
static const uintptr_t DEV_SIZE = %d;
static char g_host[64];
static const int g_marker = 7;

static uint64_t g_sym = 0;
static int g_fail_calls = 0;
static int g_calls = 0;
static void* g_dst = nullptr;
static const void* g_src = nullptr;
static size_t g_n = 0;
static int g_kind = -1;

static cudaError_t fail(cudaError_t e, const char*) { ++g_fail_calls; return e; }

#define API_ENTER(name) do { (void) (name); } while (0)

static bool is_dev(const void* p) {
  uintptr_t a = (uintptr_t) p;
  return a >= DEV_BASE && a < DEV_BASE + DEV_SIZE;
}

static uint64_t var_addr(const void* sym, size_t* size) {
  (void) sym;
  if (size) *size = 64;
  return g_sym;
}

static cudaError_t memcpy_impl(void* dst, const void* src, size_t n, cudaMemcpyKind kind) {
  ++g_calls;
  g_dst = dst; g_src = src; g_n = n; g_kind = (int) kind;
  return cudaSuccess;
}
""" % (DEV_BASE, DEV_DST, DEV_SIZE)

HARNESS = r"""// built by tests/test_fix_0097_2.py
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include "helpers.h"

// ---- cudaMemcpyFromSymbol, cut out of sidecar-shim/tgcudart.cpp ----
@EXTRACTED@
// ---- end of the cut out function ----

static void run(const char* id, int dest, size_t n, size_t off, int kind, uint64_t sym) {
  g_sym = sym;
  g_calls = 0; g_dst = nullptr; g_src = nullptr; g_n = 0; g_kind = -1; g_fail_calls = 0;
  void* dst = dest ? (void*) (uintptr_t) DEV_DST : (void*) g_host;
  cudaError_t rc = cudaMemcpyFromSymbol(dst, &g_marker, n, off, (cudaMemcpyKind) kind);
  std::printf("case=%s rc=%d calls=%d kind=%d dstdev=%d dsthost=%d src=%llu n=%llu fails=%d\n", id, (int) rc,
              g_calls, g_kind, g_dst == (void*) (uintptr_t) DEV_DST, g_dst == (void*) g_host,
              (unsigned long long) (uintptr_t) g_src, (unsigned long long) g_n, g_fail_calls);
}

int main() {
  run("default_device_dst", 1, 16, 8, cudaMemcpyDefault, DEV_BASE);
  run("default_host_dst", 0, 16, 8, cudaMemcpyDefault, DEV_BASE);
  run("explicit_device_to_device", 1, 16, 8, cudaMemcpyDeviceToDevice, DEV_BASE);
  run("explicit_device_to_host", 0, 16, 8, cudaMemcpyDeviceToHost, DEV_BASE);
  run("default_offset_and_length", 1, 4, 24, cudaMemcpyDefault, DEV_BASE);
  run("unknown_symbol", 1, 16, 8, cudaMemcpyDefault, 0);
  return 0;
}
"""

# id, destination, kind, n, off, symbol, and what the call to memcpy_impl must be
CASES = [
  dict(id="default_device_dst", dest=1, kind=4, n=16, off=8, sym=DEV_BASE, want_kind=3,
       why="default kind with a device destination is copied device to device"),
  dict(id="default_host_dst", dest=0, kind=4, n=16, off=8, sym=DEV_BASE, want_kind=2,
       why="default kind with a host destination is copied device to host"),
  dict(id="explicit_device_to_device", dest=1, kind=3, n=16, off=8, sym=DEV_BASE, want_kind=3,
       why="an explicit device to device kind is passed through"),
  dict(id="explicit_device_to_host", dest=0, kind=2, n=16, off=8, sym=DEV_BASE, want_kind=2,
       why="an explicit device to host kind is passed through"),
  dict(id="default_offset_and_length", dest=1, kind=4, n=4, off=24, sym=DEV_BASE, want_kind=3,
       why="the offset and length reach the copy with a device destination"),
  dict(id="unknown_symbol", dest=1, kind=4, n=16, off=8, sym=0, want_kind=-1, want_rc=201, want_calls=0,
       why="an unknown symbol is refused and copies nothing"),
]


def find_root():
    """The tree under test: the copy prove_test runs us in, else the one this file sits in."""
    for base in (Path.cwd(), Path(__file__).resolve().parent.parent):
        if (base / REL).is_file():
            return base
    return Path.cwd()


def strip_noise(text):
    """Blank comments and literals in place, so an index into the result is an index into the source."""
    out = list(text)

    def blank(a, b):
        for k in range(a, min(b, len(out))):
            if out[k] != "\n":
                out[k] = " "

    i, n = 0, len(text)
    while i < n:
        if text.startswith("//", i):
            j = text.find("\n", i)
            j = n if j < 0 else j
            blank(i, j)
            i = j
        elif text.startswith("/*", i):
            j = text.find("*/", i + 2)
            j = n if j < 0 else j + 2
            blank(i, j)
            i = j
        elif text[i] in "\"'":
            start = i
            quote = text[i]
            i += 1
            while i < n:
                if text[i] == "\\":
                    i += 2
                    continue
                if text[i] == quote or text[i] == "\n":
                    i += 1
                    break
                i += 1
            blank(start, i)
        else:
            i += 1
    return "".join(out)


def extract(path):
    """The real cudaMemcpyFromSymbol definition, from its signature to its matching closing brace."""
    text = path.read_text(encoding="utf-8", errors="replace")
    clean = strip_noise(text)
    m = re.search(r"^[ \t]*(?:EXPORT[ \t]+)?cudaError_t[ \t]+cudaMemcpyFromSymbol[ \t]*\(", clean, re.M)
    if not m:
        return None
    start = clean.find("{", m.end())
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(clean)):
        if clean[i] == "{":
            depth += 1
        elif clean[i] == "}":
            depth -= 1
            if depth == 0:
                return text[m.start():i + 1].strip()
    return None


def compiler():
    for name in ("clang++", "g++", "c++"):
        path = shutil.which(name)
        if path:
            return path
    return None


def build_and_run(cxx, body, work):
    src = Path(work) / "from_symbol.cpp"
    exe = Path(work) / "from_symbol"
    (Path(work) / "helpers.h").write_text(FAKE_HELPERS, encoding="utf-8")
    src.write_text(HARNESS.replace("@EXTRACTED@", body), encoding="utf-8")
    cmd = [cxx, "-std=c++17", "-O0", "-Wno-unused-function", "-I", work, "-o", str(exe), str(src)]
    built = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    if built.returncode != 0:
        errors = [l.strip() for l in (built.stdout + built.stderr).splitlines() if " error: " in l]
        return None, "does not build (" + (errors[0] if errors else "see the compile log").replace('"', "'") + ")"
    run = subprocess.run([str(exe)], capture_output=True, text=True, timeout=120)
    if run.returncode != 0:
        return None, "the built program exited %d: %s" % (run.returncode, (run.stdout + run.stderr).strip()[-200:])
    rows = {}
    for line in run.stdout.splitlines():
        if line.startswith("case="):
            rows[line.split()[0][5:]] = dict(part.split("=", 1) for part in line.split() if "=" in part)
    return rows, ""


def check(case, row):
    """None when the row is right, else what it got wrong."""
    if row is None:
        return "the harness printed no line for this case"
    calls = int(row["calls"])
    want_calls = case.get("want_calls", 1)
    if calls != want_calls:
        return "called memcpy_impl %d time(s), want %d" % (calls, want_calls)
    if case["id"] == "unknown_symbol":
        if int(row["rc"]) != case["want_rc"]:
            return "rc=%s, want cudaErrorInvalidSymbol (%d)" % (row["rc"], case["want_rc"])
        if row["dstdev"] != "0" or row["dsthost"] != "0":
            return "a copy ran for an unknown symbol"
        if int(row["fails"]) != 1:
            return "did not go through fail() (%s call(s))" % row["fails"]
        return None
    if int(row["rc"]) != 0:
        return "rc=%s, want cudaSuccess" % row["rc"]
    if int(row["kind"]) != case["want_kind"]:
        return "memcpy_impl kind=%s, want %d" % (row["kind"], case["want_kind"])
    if case["dest"]:
        if row["dstdev"] != "1" or row["dsthost"] != "0":
            return "the device destination did not reach memcpy_impl"
    elif row["dsthost"] != "1" or row["dstdev"] != "0":
        return "the host destination did not reach memcpy_impl"
    if int(row["src"]) != case["sym"] + case["off"]:
        return "src=%s, want the symbol address plus %d" % (row["src"], case["off"])
    if int(row["n"]) != case["n"]:
        return "n=%s, want %d" % (row["n"], case["n"])
    if int(row["fails"]) != 0:
        return "went through fail() on a copy that must succeed"
    return None


def main():
    root = find_root()
    source = root / REL
    if not source.is_file():
        print("FAIL run this test from the copy root: " + str(REL) + " is not in " + str(root))
        return 1
    body = extract(source)
    if not body:
        print("FAIL cannot find the cudaMemcpyFromSymbol definition in " + str(REL))
        return 1
    cxx = compiler()
    if cxx is None:
        print("FAIL no C++ compiler was found (looked for clang++, g++, c++)")
        return 1
    work = tempfile.mkdtemp(prefix="fix-0097-fromsymbol-")
    failures = 0
    try:
        rows, detail = build_and_run(cxx, body, work)
        if rows is None:
            print("FAIL the cut out cudaMemcpyFromSymbol " + detail)
            return 1
        for case in CASES:
            wrong = check(case, rows.get(case["id"]))
            print(("PASS " if wrong is None else "FAIL ") + case["why"] + (": " + wrong if wrong else ""))
            failures += 0 if wrong is None else 1
    finally:
        shutil.rmtree(work, ignore_errors=True)
    print("%d case(s) failed" % failures if failures else "all cases passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
