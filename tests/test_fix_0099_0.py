#!/usr/bin/env python3
"""tests/test_fix_0099_0.py - a failed hs_push_pre_next must not look like Python's own -1, nor leave an error set.

sidecar-shim/tgcudart.cpp:1343 (line 1312 of the reviewed source; this copy carries earlier fixes above it) reads
    long long v = r && PyLong_Check(r) ? PyLong_AsLongLong(r) : -1;
inside tgcuHsPushPreNext. Four outcomes all come back as -1: Python's own -1, a return that is not an int, a call that
failed with r null, and an int too large for long long (PyLong_AsLongLong sets OverflowError and returns -1). Nothing
clears that OverflowError, so the next Python call the shim makes sees a stale exception. The caller
(src/core/verify.cpp:4009) compares the result with 0 only, so a failed handshake is already read as "not merged" and
nothing tells the failure apart from a value Python chose to return.

No GPU, no model, no weights and no Python headers: these cases take the real tgcuHsPushPreNext out of the shim,
compile it against a stub CPython surface (one refcounted object, one int type, one pending exception) and run it with
a planted return:
  - an int too large for long long comes back as -4 and leaves no exception set;
  - a return that is not an int comes back as -4, not as Python's -1;
  - a Python call that failed comes back as -4 and its exception is cleared;
  - Python's own 0, -1 and -2 come back unchanged, so the legitimate -1 is not mistaken for a failure;
  - every object the shim was handed is released, and the shim calls the Python function once per handshake.
Prints one PASS or FAIL line per case and exits nonzero on any FAIL.
Run: python3 tests/test_fix_0099_0.py
"""
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.getcwd()   # the tree under test: every test here runs with cwd at the copy root, never at the test file
SHIM = os.path.join("sidecar-shim", "tgcudart.cpp")

PRELUDE = r"""
// built by tests/test_fix_0099_0.py: a stub CPython surface, then the shim's own tgcuHsPushPreNext below
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define EXPORT
#define API_ENTER(name) ((void) 0)

enum PyKind { PY_INT = 1, PY_OTHER = 2 };   // PY_INT is the int type; any other kind is not a PyLong

struct PyObject {
  int refs;       // the reference count Py_DECREF and Py_XDECREF adjust
  int kind;
  long long value;
  bool too_big;   // an int the C API cannot convert to long long
};

static PyObject g_exc;                  // the one pending exception, the way CPython keeps one per thread
static PyObject* g_error = nullptr;
static int g_live = 0;                  // live objects: a shim that forgets to release one leaves this above zero
static int g_calls = 0;                 // Python calls the shim made
static int g_mode = 0;                  // 0: return the int g_ret, 1: an int too big, 2: not an int, 3: the call failed
static long long g_ret = 0;

static PyObject* py_object(int kind, long long value, bool too_big) {
  PyObject* o = (PyObject*) std::calloc(1, sizeof(PyObject));
  o->refs = 1; o->kind = kind; o->value = value; o->too_big = too_big;
  ++g_live;
  return o;
}
static void py_error() { g_error = &g_exc; }   // the exception object is not refcounted, as CPython's is not
static void Py_DECREF(PyObject* o) { if (--o->refs == 0) { --g_live; std::free(o); } }
static void Py_XDECREF(PyObject* o) { if (o) Py_DECREF(o); }
static int PyLong_Check(PyObject* o) { return o && o->kind == PY_INT; }
static PyObject* PyErr_Occurred() { return g_error; }
static void PyErr_Clear() { g_error = nullptr; }
static long long PyLong_AsLongLong(PyObject* o) {
  if (!o || o->kind != PY_INT) { py_error(); return -1; }   // TypeError, as CPython raises for a non-int
  if (o->too_big) { py_error(); return -1; }                // OverflowError, as CPython raises past long long
  return o->value;
}

// the Python function the shim calls, standing in for tgcu_api.hs_push_pre_next (it returns 0 or -2 in production)
static bool call_obj(PyObject** out, const char* fn, const char* fmt, ...) {
  (void) fmt;
  if (std::strcmp(fn, "hs_push_pre_next") != 0) { std::fprintf(stderr, "unexpected python call: %s\n", fn); std::exit(2); }
  ++g_calls;
  *out = nullptr;
  if (g_mode == 3) { py_error(); return false; }                            // the call raised, so r stays null
  if (g_mode == 2) { *out = py_object(PY_OTHER, 0, false); return true; }   // a return that is not an int
  *out = py_object(PY_INT, g_ret, g_mode == 1);
  return true;
}

// the mapped-region lookup and segment builder the handshake uses: this probe plants one region and counts release
static uintptr_t region_base(const void* p) { return p ? (uintptr_t) 0x1000 : 0; }
static PyObject* hs_segs(void* const* ps, const size_t* ns, int cnt) {
  (void) ps; (void) ns;
  if (cnt < 0) return nullptr;
  return py_object(PY_OTHER, cnt, false);
}
"""

MAIN = r"""
static void probe(const char* name, int mode, long long ret) {
  g_mode = mode; g_ret = ret; g_calls = 0; g_error = nullptr;
  void* ps[2] = {(void*) 0x2000, (void*) 0x2004};
  size_t ns[2] = {8, 4};
  void* nps[3] = {(void*) 0x2008, (void*) 0x200c, (void*) 0x2010};
  size_t nns[3] = {4, 4, 4};
  int rc = tgcuHsPushPreNext(ps, ns, 2, nps, nns, 3, (const void*) 0x3000, 5u);
  std::printf("probe %s rc=%d err=%d calls=%d live=%d\n", name, rc, g_error ? 1 : 0, g_calls, g_live);
}

int main() {
  probe("big", 1, 0);          // Python returned an int too large for long long
  probe("not_an_int", 2, 0);   // Python returned something that is not an int
  probe("call_failed", 3, 0);  // the Python call raised, so r is null
  probe("zero", 0, 0);         // Python returned 0: the handshake worked
  probe("minus_one", 0, -1);   // Python returned -1 itself
  probe("minus_two", 0, -2);   // Python returned -2, its chunk-limit code
  return 0;
}
"""

# want = (return code, pending exception after the call, Python calls, live objects)
CASES = [
    dict(name="an int too large for long long comes back as -4 and leaves no exception set",
         probe="big", want=(-4, 0, 1, 0)),
    dict(name="a return that is not an int comes back as -4, not Python's -1",
         probe="not_an_int", want=(-4, 0, 1, 0)),
    dict(name="a Python call that failed comes back as -4 and clears its exception",
         probe="call_failed", want=(-4, 0, 1, 0)),
    dict(name="Python's own 0 still comes back as 0", probe="zero", want=(0, 0, 1, 0)),
    dict(name="Python's own -1 still comes back as -1, not as the failure code",
         probe="minus_one", want=(-1, 0, 1, 0)),
    dict(name="Python's own -2 (the chunk limit) still comes back as -2",
         probe="minus_two", want=(-2, 0, 1, 0)),
]


def compiler():
    for name in ("g++", "clang++", "c++"):
        path = shutil.which(name)
        if path:
            return path
    return None


def push_pre_next_source(text):
    """The real tgcuHsPushPreNext body, brace matched, so the cases run the shim's own code."""
    start = text.find("int tgcuHsPushPreNext(")
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
    source = push_pre_next_source(text)
    if not source:
        return None, "%s does not hold tgcuHsPushPreNext(...) to compile" % SHIM
    cxx = compiler()
    if not cxx:
        return None, "no C++ compiler (g++, clang++, c++) on PATH"
    src = os.path.join(work, "hs_push_pre_next_probe.cpp")
    exe = os.path.join(work, "hs_push_pre_next_probe")
    with open(src, "w", encoding="utf-8") as f:
        f.write(PRELUDE + "\n" + source + "\n" + MAIN)
    built = subprocess.run([cxx, "-std=c++17", "-O0", "-o", exe, src], capture_output=True, text=True, timeout=180)
    if built.returncode != 0:
        errors = [l.strip() for l in (built.stdout + built.stderr).splitlines() if " error: " in l]
        return None, "the tgcuHsPushPreNext probe does not build (" + (errors[0] if errors else "see the compile log") + ")"
    run = subprocess.run([exe], capture_output=True, text=True, timeout=120)
    if run.returncode != 0:
        return None, "the tgcuHsPushPreNext probe exited %d: %s" % (run.returncode, run.stdout.strip()[-200:])
    seen = {}
    for line in run.stdout.splitlines():
        parts = line.split()
        if len(parts) == 6 and parts[0] == "probe":
            try:
                seen[parts[1]] = (int(parts[2][3:]), int(parts[3][4:]), int(parts[4][6:]), int(parts[5][5:]))
            except ValueError:
                return None, "the probe printed a line this test cannot read: " + line
    return seen, ""


def main():
    work = tempfile.mkdtemp(prefix="fix-0099-")
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
                    ok, detail = False, "got rc=%d err=%d calls=%d live=%d, expected rc=%d err=%d calls=%d live=%d" % (
                        got + case["want"])
                else:
                    ok, detail = True, "rc=%d err=%d calls=%d live=%d" % got
            if not ok:
                bad += 1
            print(("PASS " if ok else "FAIL ") + case["name"] + (": " + detail if detail else ""))
        return 1 if bad else 0
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
