#!/usr/bin/env python3
"""tests/test_fix_0095_3.py - a malformed kernel_params tuple must fail the launch, not overflow it.

sidecar-shim/tgcudart.cpp:321 reads
    size_t off = PyLong_AsSize_t(PyTuple_GetItem(t, 0)), sz = PyLong_AsSize_t(PyTuple_GetItem(t, 1));
    if (buf.size() < off + sz) buf.resize(off + sz);
    memcpy(buf.data() + off, args[i], sz);
A missing tuple item or a value PyLong_AsSize_t cannot convert leaves (size_t)-1 in off or sz; a valid but huge
offset and size overflow off + sz. Today the shim never looks at PyTuple_GetItem or PyErr_Occurred, so a malformed
kernel_params reply either resizes the packing buffer to a wrapped size or copies through a bad pointer. That can
throw or corrupt the launch instead of returning cudaErrorLaunchFailure.

No GPU, no CUDA headers and no Python headers are needed: these cases take the real pack_and_launch function out of
sidecar-shim/tgcudart.cpp, compile it against a stub Python C API and a stub do_launch, and run one child process per
case so a crash on today's code is reported and the other cases still run.
  - a tuple with no size, a non-integer offset, a non-integer size, a negative size and an offset plus size that
    overflow size_t are all refused before they reach the launch;
  - ordinary parameter tuples still pack the same bytes and launch once.
Prints one PASS or FAIL line per case and exits nonzero on any FAIL.
Run: python3 tests/test_fix_0095_3.py
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = os.getcwd()   # the tree under test: every test here runs with cwd at the copy root, never at the test file
SHIM = os.path.join("sidecar-shim", "tgcudart.cpp")
SIGNATURE = "cudaError_t pack_and_launch("

# The stub side: the Python C API and runtime calls pack_and_launch needs, then the real function below.
PRELUDE = """// built by tests/test_fix_0095_3.py: the stub Python C API and launch sink, then the real function below
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

typedef int cudaError_t;
typedef void* cudaStream_t;
typedef long Py_ssize_t;

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

struct Func { const void* fat; std::string name; int kid; int max_dyn_shared; };

struct PyObject {
  int kind;
  unsigned long long value;
  bool negated;
  std::vector<PyObject*> items;
};

enum { PY_INT = 1, PY_TUPLE = 2, PY_LIST = 3, PY_OTHER = 4 };

static int g_pyerr = 0;
static PyObject* g_result = nullptr;
static int launch_calls = 0;
static int fail_calls = 0;
static std::vector<uint8_t> g_params;

static PyObject* py_int(unsigned long long v) { return new PyObject{PY_INT, v, false, {}}; }
static PyObject* py_negative() { return new PyObject{PY_INT, 0, true, {}}; }
static PyObject* py_other() { return new PyObject{PY_OTHER, 0, false, {}}; }
static PyObject* py_tuple(std::vector<PyObject*> items) { return new PyObject{PY_TUPLE, 0, false, items}; }
static PyObject* py_list(std::vector<PyObject*> items) { return new PyObject{PY_LIST, 0, false, items}; }

static int PyErr_Occurred() { return g_pyerr; }
static void PyErr_Clear() { g_pyerr = 0; }
static void Py_DECREF(PyObject*) {}

static Py_ssize_t PyList_Size(PyObject* o) {
  if (!o || o->kind != PY_LIST) { g_pyerr = 1; return -1; }
  return (Py_ssize_t)o->items.size();
}
static PyObject* PyList_GetItem(PyObject* o, Py_ssize_t i) {
  if (!o || o->kind != PY_LIST || i < 0 || (size_t)i >= o->items.size()) { g_pyerr = 1; return nullptr; }
  return o->items[(size_t)i];
}
static PyObject* PyTuple_GetItem(PyObject* o, Py_ssize_t i) {
  if (!o || o->kind != PY_TUPLE || i < 0 || (size_t)i >= o->items.size()) { g_pyerr = 1; return nullptr; }
  return o->items[(size_t)i];
}
static size_t PyLong_AsSize_t(PyObject* o) {
  if (!o || o->kind != PY_INT || o->negated) { g_pyerr = 1; return (size_t)-1; }
  return (size_t)o->value;
}

static bool call_obj(PyObject** out, const char* fn, const char* fmt, ...) {
  (void)fn; (void)fmt;
  *out = g_result;
  return g_result != nullptr;
}

static cudaError_t fail(cudaError_t e, const char* what) { (void)what; fail_calls++; return e; }

static cudaError_t do_launch(Func* fn, dim3 g, dim3 b, size_t shmem, const void* params, size_t psize,
                             cudaStream_t st, bool coop = false) {
  (void)fn; (void)g; (void)b; (void)shmem; (void)st; (void)coop;
  launch_calls++;
  const uint8_t* p = (const uint8_t*)params;
  g_params.assign(p, p + psize);
  return cudaSuccess;
}
"""

MAIN = """
static const char* err_name(cudaError_t e) {
  switch (e) {
    case cudaSuccess: return "success";
    case cudaErrorLaunchFailure: return "launch_failure";
    case cudaErrorInvalidConfiguration: return "invalid_configuration";
    case cudaErrorInvalidDeviceFunction: return "invalid_device_function";
    default: return "other";
  }
}

static void scenario(const char* name, std::vector<void*>& args) {
  if (!std::strcmp(name, "ordinary")) {
    g_result = py_list({py_tuple({py_int(2), py_int(4)}), py_tuple({py_int(8), py_int(2)})});
    args = { (void*)"WXYZ", (void*)"hi" };
  } else if (!std::strcmp(name, "ordinary_zero")) {
    g_result = py_list({py_tuple({py_int(0), py_int(4)})});
    args = { (void*)"abcd" };
  } else if (!std::strcmp(name, "missing_item")) {
    g_result = py_list({py_tuple({py_int(0)})});
    args = { (void*)"abcd" };
  } else if (!std::strcmp(name, "non_integer_offset")) {
    g_result = py_list({py_tuple({py_other(), py_int(4)})});
    args = { (void*)"abcd" };
  } else if (!std::strcmp(name, "non_integer_size")) {
    g_result = py_list({py_tuple({py_int(0), py_other()})});
    args = { (void*)"abcd" };
  } else if (!std::strcmp(name, "negative_size")) {
    g_result = py_list({py_tuple({py_int(0), py_negative()})});
    args = { (void*)"abcd" };
  } else if (!std::strcmp(name, "overflow")) {
    // off + sz wraps to 8: both values fit size_t, but the sum does not
    g_result = py_list({py_tuple({py_int((unsigned long long)((size_t)-1 - 7)), py_int(16)})});
    args = { (void*)"0123456789abcdef" };
  }
}

int main(int argc, char** argv) {
  if (argc != 2) { std::printf("usage: %s scenario\\n", argv[0]); return 2; }
  std::vector<void*> args;
  scenario(argv[1], args);
  Func fn{nullptr, "probe_kernel", 0, 0};
  cudaError_t e = pack_and_launch(&fn, dim3(1, 1, 1), dim3(1, 1, 1), 0, args.data(), nullptr);
  std::printf("probe %s rc=%s launches=%d fails=%d psize=%zu data=", argv[1], err_name(e), launch_calls,
              fail_calls, g_params.size());
  for (uint8_t c : g_params) std::printf("%02x", c);
  std::printf("\\n");
  return 0;
}
"""

# want = (error name, launches, fail calls, packed bytes in hex)
CASES = [
    dict(name="a tuple with no size is refused before the launch", probe="missing_item",
         want=("launch_failure", 0, 1, "")),
    dict(name="a non-integer offset is refused before the launch", probe="non_integer_offset",
         want=("launch_failure", 0, 1, "")),
    dict(name="a non-integer size is refused before the launch", probe="non_integer_size",
         want=("launch_failure", 0, 1, "")),
    dict(name="a negative size is refused before the launch", probe="negative_size",
         want=("launch_failure", 0, 1, "")),
    dict(name="an offset plus size that overflows size_t is refused before the launch", probe="overflow",
         want=("launch_failure", 0, 1, "")),
    dict(name="ordinary parameters with a gap still pack the same bytes and launch", probe="ordinary",
         want=("success", 1, 0, "00005758595a00006869")),
    dict(name="ordinary parameters at offset zero still pack and launch", probe="ordinary_zero",
         want=("success", 1, 0, "61626364")),
]


def compiler():
    for name in ("g++", "clang++", "c++"):
        path = shutil.which(name)
        if path:
            return path
    return None


def function_text(src):
    """The real pack_and_launch, from its signature to its matching close brace, comments and literals skipped."""
    start = src.find(SIGNATURE)
    if start < 0:
        return None
    i = src.find("{", start)
    depth = 0
    while 0 <= i < len(src):
        if src.startswith("//", i):
            i = src.find("\n", i)
            if i < 0:
                return None
        elif src.startswith("/*", i):
            i = src.find("*/", i)
            if i < 0:
                return None
            i += 1
        elif src[i] in "\"'":
            q = src[i]
            i += 1
            while i < len(src) and src[i] != q:
                i += 2 if src[i] == "\\" else 1
        elif src[i] == "{":
            depth += 1
        elif src[i] == "}":
            depth -= 1
            if depth == 0:
                return src[start:i + 1]
        i += 1
    return None


def build(work):
    shim = os.path.join(ROOT, SHIM)
    try:
        with open(shim, encoding="utf-8") as f:
            text = f.read()
    except OSError as e:
        return None, "cannot read %s: %s" % (SHIM, e)
    body = function_text(text)
    if body is None:
        return None, "%s has no %s that the test can compile" % (SHIM, SIGNATURE.strip())
    cxx = compiler()
    if not cxx:
        return None, "no C++ compiler (g++, clang++, c++) on PATH"
    src = os.path.join(work, "pack_and_launch_probe.cpp")
    exe = os.path.join(work, "pack_and_launch_probe")
    with open(src, "w", encoding="utf-8") as f:
        f.write(PRELUDE + "\n" + body + "\n" + MAIN)
    built = subprocess.run([cxx, "-std=c++17", "-O0", "-o", exe, src], capture_output=True, text=True, timeout=180)
    if built.returncode != 0:
        errors = [l.strip() for l in (built.stdout + built.stderr).splitlines() if " error: " in l]
        return None, "the pack_and_launch probe does not build (" + (errors[0] if errors else "see the compile log") + ")"
    return exe, ""


def run_probe(exe, case):
    try:
        r = subprocess.run([exe, case["probe"]], capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired:
        return None, "the probe timed out"
    if r.returncode < 0:
        return None, "the probe was killed by signal %d" % -r.returncode
    if r.returncode != 0:
        return None, "the probe exited %d: %s" % (r.returncode, (r.stdout + r.stderr).strip()[-200:])
    m = re.search(r"^probe (\S+) rc=(\S+) launches=(\d+) fails=(\d+) psize=(\d+) data=(\S*)$", r.stdout, re.M)
    if m is None:
        return None, "unreadable probe output: " + r.stdout.strip()[-200:]
    return (m.group(1), m.group(2), int(m.group(3)), int(m.group(4)), int(m.group(5)), m.group(6)), ""


def main():
    work = tempfile.mkdtemp(prefix="fix-0095-3-")
    try:
        exe, error = build(work)
        bad = 0
        for case in CASES:
            if exe is None:
                ok, detail = False, error
            else:
                got, why = run_probe(exe, case)
                if got is None:
                    ok, detail = False, why
                else:
                    name, rc, launches, fails, psize, data = got
                    ok = (name, rc, launches, fails, data) == (case["probe"],) + case["want"] and psize == len(data) // 2
                    detail = "rc=%s launches=%d fails=%d psize=%d data=%s" % (rc, launches, fails, psize, data)
                    if not ok:
                        detail += ", expected rc=%s launches=%d fails=%d data=%s" % case["want"]
            if not ok:
                bad += 1
            print(("PASS " if ok else "FAIL ") + case["name"] + (": " + detail if detail else ""))
        return 1 if bad else 0
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
