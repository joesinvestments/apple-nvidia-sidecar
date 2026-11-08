#!/usr/bin/env python3
"""tests/test_fix_0097_1.py - cudaMemGetInfo must not trust what mem_info hands back.

sidecar-shim/tgcudart.cpp:851 reads

    *free_b = PyLong_AsSize_t(PyTuple_GetItem(r, 0)); *total = PyLong_AsSize_t(PyTuple_GetItem(r, 1));

after only checking that the call itself succeeded (line 850). A result that is not a 2-tuple passes
NULL into PyLong_AsSize_t, and a 2-tuple of non-integers yields (size_t)-1 with a Python exception left
pending, so cudaMemGetInfo reports cudaSuccess with bogus sizes. These cases cut the real cudaMemGetInfo
out of the shim, build it against a toy Python C API (no GPU, no model, no CPython needed) and plant
hostile mem_info results:
  - a 2-tuple of integers still reports those sizes and succeeds;
  - anything else (a non-tuple, strings, one item, three items, a mixed tuple) is refused: a nonzero
    CUDA error, the caller's sizes untouched and no pending Python exception.
The stock reply (two integers) gives the same sizes before and after the fix.

Run: python3 tests/test_fix_0097_1.py
Prints one PASS or FAIL line per case and exits nonzero on any FAIL.
"""
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

REL = Path("sidecar-shim") / "tgcudart.cpp"


def find_root():
    """The tree under test: the copy prove_test runs us in, else the one this file sits in."""
    for base in (Path.cwd(), Path(__file__).resolve().parent.parent):
        if (base / REL).is_file():
            return base
    return Path.cwd()


ROOT = find_root()
SOURCE = ROOT / REL

# the toy Python C API: only the calls cudaMemGetInfo makes, with CPython's failure results (a NULL from
# PyTuple_GetItem or PyLong_AsSize_t is (size_t)-1 plus a pending exception; CPython would crash on the NULL)
FAKE_PY = r"""
#pragma once
#include <cstddef>

typedef long Py_ssize_t;

typedef struct _PyObj {
  int refcnt;
  int kind;   /* 1 tuple, 2 int, 3 str */
  long long ival;
  int nitems;
  struct _PyObj* items[4];
} PyObject;

static int py_error = 0;
static void py_fail() { py_error = 1; }
static int PyErr_Occurred(void) { return py_error; }
static void PyErr_Clear(void) { py_error = 0; }
static void Py_DECREF(PyObject* o) { if (o && o->refcnt > 0) o->refcnt--; }

static int PyTuple_Check(PyObject* o) { return o != nullptr && o->kind == 1; }
static Py_ssize_t PyTuple_Size(PyObject* o) {
  if (!PyTuple_Check(o)) { py_fail(); return -1; }
  return o->nitems;
}
static PyObject* PyTuple_GetItem(PyObject* o, Py_ssize_t i) {
  if (!PyTuple_Check(o)) { py_fail(); return nullptr; }
  if (i < 0 || i >= o->nitems) { py_fail(); return nullptr; }
  return o->items[i];
}
static int PyLong_Check(PyObject* o) { return o != nullptr && o->kind == 2; }
static size_t PyLong_AsSize_t(PyObject* o) {
  if (!PyLong_Check(o)) { py_fail(); return (size_t) -1; }
  if (o->ival < 0) { py_fail(); return (size_t) -1; }
  return (size_t) o->ival;
}
"""

# the harness: stubs for the shim's own helpers, then the real function cut out of the source above
HARNESS = r"""// built by tests/test_fix_0097_1.py
#include <cstddef>
#include <cstdio>
#include <cstring>
#include "Python.h"

#define EXPORT extern "C"
typedef int cudaError_t;
enum { cudaSuccess = 0, cudaErrorUnknown = 999 };

static int g_fail_calls = 0;
static int g_call_calls = 0;
static PyObject* g_return = nullptr;

static cudaError_t fail(cudaError_t e, const char*) { ++g_fail_calls; return e; }

static bool call_obj(PyObject** out, const char* fn, const char*, ...) {
  (void) fn;
  ++g_call_calls;
  *out = g_return;
  return g_return != nullptr;
}

#define API_ENTER(name) do { (void) (name); } while (0)

// ---- cudaMemGetInfo, cut out of sidecar-shim/tgcudart.cpp ----
@EXTRACTED@
// ---- end of the cut out function ----

static PyObject g_pool[32];
static int g_pool_n = 0;
static const unsigned long long SENTINEL = 0x5A5A5A5A5A5A5A5AULL;

static PyObject* obj_int(long long v) {
  PyObject* o = &g_pool[g_pool_n++];
  std::memset(o, 0, sizeof *o);
  o->refcnt = 1; o->kind = 2; o->ival = v;
  return o;
}
static PyObject* obj_str() {
  PyObject* o = &g_pool[g_pool_n++];
  std::memset(o, 0, sizeof *o);
  o->refcnt = 1; o->kind = 3;
  return o;
}
static PyObject* obj_tuple(int n) {
  PyObject* o = &g_pool[g_pool_n++];
  std::memset(o, 0, sizeof *o);
  o->refcnt = 1; o->kind = 1; o->nitems = n;
  return o;
}

static void run_case(const char* id, PyObject* planted) {
  g_return = planted; g_call_calls = 0; g_fail_calls = 0;
  PyErr_Clear();
  size_t free_b = (size_t) SENTINEL, total = (size_t) SENTINEL;
  cudaError_t rc = cudaMemGetInfo(&free_b, &total);
  std::printf("case=%s rc=%d free=%llu total=%llu err=%d refs=%d calls=%d fails=%d\n", id, (int) rc,
              (unsigned long long) free_b, (unsigned long long) total, PyErr_Occurred(),
              planted ? planted->refcnt : -1, g_call_calls, g_fail_calls);
}

int main() {
  PyObject* good = obj_tuple(2);
  good->items[0] = obj_int(274877906944LL);
  good->items[1] = obj_int(34359738368LL);
  run_case("good", good);

  run_case("scalar", obj_int(7));

  PyObject* strings = obj_tuple(2);
  strings->items[0] = obj_str();
  strings->items[1] = obj_str();
  run_case("strings", strings);

  PyObject* one = obj_tuple(1);
  one->items[0] = obj_int(4096);
  run_case("one", one);

  PyObject* three = obj_tuple(3);
  three->items[0] = obj_int(1);
  three->items[1] = obj_int(2);
  three->items[2] = obj_int(3);
  run_case("three", three);

  PyObject* mixed = obj_tuple(2);
  mixed->items[0] = obj_int(1024);
  mixed->items[1] = obj_str();
  run_case("mixed", mixed);
  return 0;
}
"""

SENTINEL = 0x5A5A5A5A5A5A5A5A
GOOD_FREE = 274877906944
GOOD_TOTAL = 34359738368

# ok=True: mem_info answered properly, so the sizes are reported and the call succeeds.
# ok=False: a planted hostile reply must be refused, with the caller's sizes untouched.
CASES = [
    dict(id="good", ok=True, free=GOOD_FREE, total=GOOD_TOTAL),
    dict(id="scalar", ok=False),
    dict(id="strings", ok=False),
    dict(id="one", ok=False),
    dict(id="three", ok=False),
    dict(id="mixed", ok=False),
]


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
    """The real cudaMemGetInfo definition, from its signature to its matching closing brace."""
    text = path.read_text(encoding="utf-8", errors="replace")
    clean = strip_noise(text)
    m = re.search(r"^[ \t]*(?:EXPORT[ \t]+)?cudaError_t[ \t]+cudaMemGetInfo[ \t]*\(", clean, re.M)
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
    for name in ("g++", "clang++", "c++"):
        path = shutil.which(name)
        if path:
            return path
    return None


def build_and_run(cxx, body, work):
    src = Path(work) / "mem_info.cpp"
    exe = Path(work) / "mem_info"
    (Path(work) / "Python.h").write_text(FAKE_PY, encoding="utf-8")
    src.write_text(HARNESS.replace("@EXTRACTED@", body), encoding="utf-8")
    cmd = [cxx, "-std=c++17", "-O0", "-I", work, "-o", str(exe), str(src)]
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
    got_free, got_total = int(row["free"]), int(row["total"])
    if case["ok"]:
        if row["rc"] != "0":
            return "rc=%s, want cudaSuccess" % row["rc"]
        if (got_free, got_total) != (case["free"], case["total"]):
            return "reported free=%d total=%d, want free=%d total=%d" % (got_free, got_total, case["free"], case["total"])
    else:
        if row["rc"] == "0":
            return "cudaSuccess with free=%d total=%d from a hostile mem_info reply" % (got_free, got_total)
        if (got_free, got_total) != (SENTINEL, SENTINEL):
            return "wrote free=%d total=%d into the caller's sizes on failure" % (got_free, got_total)
        if row["fails"] != "1":
            return "did not go through fail() (%s call(s))" % row["fails"]
    if row["err"] != "0":
        return "left a Python exception pending"
    if row["refs"] != "0":
        return "leaked the mem_info reply (%s reference(s) left)" % row["refs"]
    if row["calls"] != "1":
        return "called mem_info %s time(s), want 1" % row["calls"]
    return None


def main():
    if not SOURCE.is_file():
        print("FAIL run this test from the copy root: " + str(REL) + " is not in " + str(ROOT))
        return 1
    body = extract(SOURCE)
    if not body:
        print("FAIL cannot find the cudaMemGetInfo definition in " + str(REL))
        return 1
    cxx = compiler()
    if cxx is None:
        print("FAIL no C++ compiler was found (looked for g++, clang++, c++)")
        return 1
    work = tempfile.mkdtemp(prefix="fix-0097-meminfo-")
    failures = 0
    try:
        rows, detail = build_and_run(cxx, body, work)
        if rows is None:
            print("FAIL the cut out cudaMemGetInfo " + detail)
            return 1
        for case in CASES:
            if case["ok"]:
                why = "a 2-tuple of integers reports free %d and total %d" % (case["free"], case["total"])
            else:
                why = "a %s reply is refused, sizes untouched, no exception pending" % (
                    {"scalar": "non-tuple", "strings": "tuple of strings", "one": "1-tuple",
                     "three": "3-tuple", "mixed": "mixed"}[case["id"]])
            wrong = check(case, rows.get(case["id"]))
            print(("PASS " if wrong is None else "FAIL ") + why + (": " + wrong if wrong else ""))
            failures += 0 if wrong is None else 1
    finally:
        shutil.rmtree(work, ignore_errors=True)
    print("%d case(s) failed" % failures if failures else "all cases passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
