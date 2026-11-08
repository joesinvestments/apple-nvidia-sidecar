#!/usr/bin/env python3
"""sidecar-shim/tgcudart.cpp: a kernel_params reply that is not a list must fail the launch, not launch empty.

pack_and_launch asks the Python side for the kernel's parameter layout and reads it with PyList_Size. When the reply
is not a list, PyList_Size returns -1 and sets a Python exception; the packing loop is skipped, buf stays empty and
do_launch runs the kernel with zero parameters, so a broken kernel launches silently instead of failing.

This test does not read the answer out of the source text. It takes the real pack_and_launch function out of the tree
under test and compiles it against a small fake CPython C API (PyObject, PyList_Size, PyList_GetItem, PyTuple_GetItem,
PyLong_AsSize_t, Py_DECREF, PyErr_Occurred) and a recording do_launch, then runs the built program once per planted
kernel_params reply:
  - a non-list reply (a plain object) fails the launch and never calls do_launch;
  - a tuple reply fails the launch the same way;
  - an ordinary list of (offset, size) pairs still packs every argument and launches with the packed buffer;
  - an empty list (a kernel with no parameters) still launches with no parameters, so the guard is not too wide;
  - a kernel_params call that already fails still fails before the packing starts.
The first two cases are the harm: on today's code they launch with psize 0. No GPU, no model, no network.
Prints one PASS or FAIL line per case and exits nonzero on any FAIL. Run: python3 tests/test_fix_0095_2.py
"""
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

SHIM = Path("sidecar-shim") / "tgcudart.cpp"


def find_root():
    """The tree under test: the copy prove_test runs us in, else the one this file sits in."""
    for base in (Path.cwd(), Path(__file__).resolve().parent.parent):
        if (base / SHIM).is_file():
            return base
    return Path.cwd()


ROOT = find_root()

# The compiled harness: a fake CPython object model with the documented semantics (PyList_Size on a non-list sets an
# error and returns -1), a recording do_launch, and one run per planted kernel_params reply.
PRELUDE = r"""
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <vector>

typedef int cudaError_t;
static const cudaError_t cudaSuccess = 0;
static const cudaError_t cudaErrorInvalidConfiguration = 1;
static const cudaError_t cudaErrorLaunchFailure = 4;
static const cudaError_t cudaErrorCooperativeLaunchTooLarge = 12;
typedef long Py_ssize_t;
struct dim3 { unsigned x, y, z; };
typedef void* cudaStream_t;
struct Func { int kid; };

enum PyKind { PY_LIST, PY_TUPLE, PY_LONG, PY_OTHER };
struct PyObject { PyKind kind; long long ival; std::vector<PyObject*> items; int refs; };
static PyObject* py_object(PyKind kind, long long ival = 0) {
  PyObject* o = new PyObject();
  o->kind = kind;
  o->ival = ival;
  o->refs = 1;
  return o;
}

static bool g_py_err = false;
static int g_decrefs = 0;
static void py_decref(PyObject* o) { ++g_decrefs; if (o) --o->refs; }
#define Py_DECREF(o) py_decref(o)
#define Py_XDECREF(o) py_decref(o)

static Py_ssize_t PyList_Size(PyObject* o) {
  if (!o || o->kind != PY_LIST) { g_py_err = true; return -1; }
  return (Py_ssize_t)o->items.size();
}
static PyObject* PyList_GetItem(PyObject* o, Py_ssize_t i) { return o->items[(size_t)i]; }
static PyObject* PyTuple_GetItem(PyObject* o, Py_ssize_t i) { return o->items[(size_t)i]; }
static Py_ssize_t PyTuple_Size(PyObject* o) { return (Py_ssize_t)o->items.size(); }
static size_t PyLong_AsSize_t(PyObject* o) { return (size_t)o->ival; }
static long long PyLong_AsLongLong(PyObject* o) { return o->ival; }
static int PyErr_Occurred(void) { return g_py_err ? 1 : 0; }
static void PyErr_Clear(void) { g_py_err = false; }
static PyObject* PyExc_TypeError = nullptr;
static void PyErr_SetString(PyObject*, const char*) { g_py_err = true; }
static int PyList_Check(PyObject* o) { return o && o->kind == PY_LIST; }
static int PyList_CheckExact(PyObject* o) { return PyList_Check(o); }
static int PyTuple_Check(PyObject* o) { return o && o->kind == PY_TUPLE; }

static cudaError_t fail(cudaError_t e, const char* what);
static cudaError_t do_launch(Func* fn, dim3 g, dim3 b, size_t shmem, const void* params, size_t psize,
                             cudaStream_t st, bool coop = false);
static bool call_obj(PyObject** out, const char* fn, const char* fmt, ...);
"""

EPILOGUE = r"""
static size_t g_launch_calls = 0;
static size_t g_launch_psize = 0;
static std::vector<unsigned char> g_launch_buf;
static cudaError_t g_launch_rc = cudaSuccess;
static int g_scenario = 0;

static cudaError_t fail(cudaError_t e, const char* what) { (void)what; return e; }

static cudaError_t do_launch(Func* fn, dim3 g, dim3 b, size_t shmem, const void* params, size_t psize,
                             cudaStream_t st, bool coop) {
  (void)fn; (void)g; (void)b; (void)shmem; (void)st; (void)coop;
  ++g_launch_calls;
  g_launch_psize = psize;
  g_launch_buf.clear();
  if (psize) g_launch_buf.assign((const unsigned char*)params, (const unsigned char*)params + psize);
  return g_launch_rc;
}

static bool call_obj(PyObject** out, const char* fn, const char* fmt, ...) {
  (void)fn; (void)fmt;
  if (g_scenario == 0) { *out = py_object(PY_OTHER); return true; }      // a dict or string where a list belongs
  if (g_scenario == 1) { *out = py_object(PY_TUPLE); return true; }      // a tuple where a list belongs
  if (g_scenario == 2) {                                                // [(0, 4), (4, 4)]
    PyObject* l = py_object(PY_LIST);
    for (int i = 0; i < 2; ++i) {
      PyObject* t = py_object(PY_TUPLE);
      t->items.push_back(py_object(PY_LONG, i * 4));
      t->items.push_back(py_object(PY_LONG, 4));
      l->items.push_back(t);
    }
    *out = l;
    return true;
  }
  if (g_scenario == 3) { *out = py_object(PY_LIST); return true; }       // a kernel with no parameters
  *out = nullptr;                                                       // the call itself failed
  return false;
}

static void reset_state() {
  g_py_err = false;
  g_decrefs = 0;
  g_launch_calls = 0;
  g_launch_psize = 0;
  g_launch_buf.clear();
  g_launch_rc = cudaSuccess;
}

static void run_case(int scenario) {
  reset_state();
  g_scenario = scenario;
  Func fn;
  fn.kid = 7;
  unsigned char a0[4] = {1, 2, 3, 4};
  unsigned char a1[4] = {5, 6, 7, 8};
  void* args[2] = {a0, a1};
  cudaError_t rc = pack_and_launch(&fn, dim3{2, 1, 1}, dim3{4, 1, 1}, 16, args, nullptr);
  std::printf("case %d: rc=%d launch=%zu psize=%zu decref=%d buf=", scenario, (int)rc, g_launch_calls,
              g_launch_psize, g_decrefs);
  for (unsigned char c : g_launch_buf) std::printf("%02x", (unsigned)c);
  std::printf("\n");
}

int main() {
  for (int scenario = 0; scenario < 5; ++scenario) run_case(scenario);
  return 0;
}
"""

CASES = [
    dict(name="a non-list kernel_params fails the launch instead of launching with no parameters", scenario=0,
         want="rc=4 launch=0 psize=0 decref=1 buf="),
    dict(name="a tuple from kernel_params fails the launch the same way", scenario=1,
         want="rc=4 launch=0 psize=0 decref=1 buf="),
    dict(name="an ordinary parameter list still packs every argument and launches", scenario=2,
         want="rc=0 launch=1 psize=8 decref=1 buf=0102030405060708"),
    dict(name="an empty parameter list still launches with no parameters", scenario=3,
         want="rc=0 launch=1 psize=0 decref=1 buf="),
    dict(name="a kernel_params call that fails still fails before packing", scenario=4,
         want="rc=4 launch=0 psize=0 decref=0 buf="),
]


def compiler():
    for name in ("g++", "clang++", "c++"):
        path = shutil.which(name)
        if path:
            return path
    return None


def blank_comments(text):
    """Same length as text, with comments and string literals blanked so a brace in either is not counted."""
    out = list(text)
    i, n = 0, len(text)
    while i < n:
        if text.startswith("//", i):
            j = text.find("\n", i)
            j = n if j < 0 else j
            for k in range(i, j):
                out[k] = " "
            i = j
        elif text.startswith("/*", i):
            j = text.find("*/", i + 2)
            j = n if j < 0 else j + 2
            for k in range(i, j):
                if out[k] != "\n":
                    out[k] = " "
            i = j
        elif text[i] in "\"'":
            q = text[i]
            i += 1
            while i < n:
                if text[i] == "\\":
                    i += 2
                    continue
                if text[i] == q or text[i] == "\n":
                    i += 1
                    break
                out[i] = " "
                i += 1
        else:
            i += 1
    return "".join(out)


def function_source(text):
    """The real pack_and_launch definition, or None when the tree does not have it."""
    clean = blank_comments(text)
    m = re.search(r"\bcudaError_t\s+pack_and_launch\s*\(", clean)
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
                return text[m.start():i + 1]
    return None


def build(cxx, work, body):
    src = Path(work) / "pack_and_launch.cpp"
    exe = Path(work) / "pack_and_launch"
    src.write_text(PRELUDE + "\n// the real function from " + str(SHIM) + "\n" + body + "\n" + EPILOGUE,
                   encoding="utf-8")
    # -Wall never -Werror: the point is the run, not the warnings
    built = subprocess.run([cxx, "-std=c++17", "-O0", "-o", str(exe), str(src)],
                           capture_output=True, text=True, timeout=180)
    if built.returncode != 0:
        errors = [l.strip() for l in (built.stdout + built.stderr).splitlines() if " error: " in l]
        return "the real pack_and_launch does not build against the fake CPython (" + \
               (errors[0] if errors else "see the compile log") + ")"
    run = subprocess.run([str(exe)], capture_output=True, text=True, timeout=120)
    if run.returncode != 0:
        return "the built program exited %d: %s" % (run.returncode, (run.stdout + run.stderr).strip()[-200:])
    return run.stdout


def main():
    if not (ROOT / SHIM).is_file():
        print("FAIL run this test from the copy root: " + str(SHIM) + " is not in " + str(ROOT))
        return 1
    text = (ROOT / SHIM).read_text(encoding="utf-8", errors="replace")
    body = function_source(text)
    if body is None:
        print("FAIL " + str(SHIM) + " has no pack_and_launch definition to test")
        return 1
    cxx = compiler()
    if cxx is None:
        print("FAIL no C++ compiler was found (looked for g++, clang++, c++)")
        return 1
    work = tempfile.mkdtemp(prefix="kernel-params-")
    failures = 0
    try:
        out = build(cxx, work, body)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    lines = {}
    for line in out.splitlines():
        m = re.match(r"case (\d+): (.*)$", line)
        if m:
            lines[int(m.group(1))] = m.group(2)
    if not lines:
        print("FAIL the built program printed no case lines: " + out.strip()[-300:])
        return 1
    for case in CASES:
        got = lines.get(case["scenario"])
        if got is None:
            ok, detail = False, "the built program printed no line for this case"
        elif got != case["want"]:
            ok, detail = False, "got %r, expected %r" % (got, case["want"])
        else:
            ok, detail = True, ""
        print(("PASS " if ok else "FAIL ") + case["name"] + (": " + detail if detail else ""))
        failures += 0 if ok else 1
    print("%d case(s) failed" % failures if failures else "all cases passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
