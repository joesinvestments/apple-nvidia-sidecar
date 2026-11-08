#!/usr/bin/env python3
"""tests/test_fix_0094_6.py - a bad tgcu_api.symbol reply must not become a device address.

sidecar-shim/tgcudart.cpp:293 reads v.addr = PyLong_AsUnsignedLongLong(PyTuple_GetItem(r, 0)) after call_obj has only
checked that the Python call returned something. When tgcu_api.symbol returns an empty tuple, a list or a non-int item,
PyTuple_GetItem or PyLong_AsUnsignedLongLong fails: the cast yields (unsigned long long)-1 and the Python error stays
set, so var_addr hands its callers 0xFFFFFFFFFFFFFFFF as a device address and nobody reports the failure.
These cases cut the real var_addr out of the shim and compile it into a small library loaded with ctypes, so the
conversion runs against the real CPython C API and against a fake tgcu_api:
  - a good (address, size) reply still resolves to that address and leaves no error set;
  - a symbol at address 0 still reads as unresolved, the way production's (0, 0) reply does;
  - an empty tuple, a list, a negative int and a non-int item resolve to 0 and leave no error set;
  - a good address above 4 GiB is kept whole.
Prints one PASS or FAIL line per case and exits nonzero on any FAIL. Run: python3 tests/test_fix_0094_6.py
"""
import ctypes
import glob
import os
import shutil
import subprocess
import sys
import sysconfig
import tempfile

ROOT = os.getcwd()   # the tree under test: every test here runs with cwd at the copy root, never at the test file
SHIM = os.path.join('sidecar-shim', 'tgcudart.cpp')
SIGNATURE = 'uint64_t var_addr('

# the shim's helpers and globals that var_addr uses, then the function itself, then the ctypes entry point
HARNESS = r'''
#include <Python.h>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <map>
#include <string>

// the shapes var_addr takes from sidecar-shim/tgcudart.cpp
struct FatHandle { const void* data; int mod = -1; };
struct Var { FatHandle* fat; std::string name; size_t size; uint64_t addr = 0; };
std::map<const void*, Var> g_vars;
static FatHandle g_fat{nullptr};
static char g_host_var;
PyObject* g_api = nullptr;

void py_fatal(const char* where) {   // the shim's own reporter: prints the Python error and clears it
  fprintf(stderr, "[tgcu] python error in %s\n", where);
  PyErr_Print();
}

int module_of(FatHandle*) { return 0; }   // the module id the fake tgcu_api.symbol accepts

bool call_obj(PyObject** out, const char* fn, const char* fmt, ...) {   // the shim's call path, without the call stats
  va_list ap; va_start(ap, fmt);
  PyObject* args = Py_VaBuildValue(fmt, ap);
  va_end(ap);
  PyObject* f = args ? PyObject_GetAttrString(g_api, fn) : nullptr;
  PyObject* r = f ? PyObject_CallObject(f, args) : nullptr;
  Py_XDECREF(f);
  Py_XDECREF(args);
  *out = r;
  return r != nullptr;
}

// the real function, cut from sidecar-shim/tgcudart.cpp by this test
@VAR_ADDR@

extern "C" unsigned long long probe(PyObject* api, const char* name, int* err) {
  g_api = api;
  const void* host_var = &g_host_var;
  g_vars[host_var] = Var{&g_fat, name ? name : "", 16};
  uint64_t a = var_addr(host_var, nullptr);
  *err = PyErr_Occurred() != nullptr;   // record the pending Python error, then clear it: PyDLL raises one it still sees
  PyErr_Clear();
  g_api = nullptr;
  return a;
}
'''


class FakeApi:
    """The fake tgcu_api: symbol returns what the case plants, production returns (address, size)."""

    def __init__(self, reply):
        self.reply = reply

    def symbol(self, mod, name):
        return self.reply


CASES = [
    dict(name='a good (address, size) reply resolves to that address',
         reply=(0x7f0000123000, 4096), want=0x7f0000123000, err=0),
    dict(name='a symbol at address 0 reads as unresolved',
         reply=(0, 0), want=0, err=0),
    dict(name='an empty tuple resolves to no address and leaves no error',
         reply=(), want=0, err=0),
    dict(name='a list reply resolves to no address and leaves no error',
         reply=[0x1000, 16], want=0, err=0),
    dict(name='a negative int resolves to no address and leaves no error',
         reply=(-1,), want=0, err=0),
    dict(name='a non-int item resolves to no address and leaves no error',
         reply=('4096', 16), want=0, err=0),
]


def compiler():
    for name in ('clang++', 'g++', 'c++'):
        path = shutil.which(name)
        if path:
            return path
    return None


def python_include():
    """The include directory holding the Python.h of the interpreter running this test."""
    major, minor = sys.version_info[:2]
    ver = 'python%d.%d' % (major, minor)
    home = os.path.expanduser('~')
    cands = [sysconfig.get_paths().get('include'), sysconfig.get_config_var('INCLUDEPY'),
             sysconfig.get_config_var('CONFINCLUDEPY')]
    for base in (sysconfig.get_config_var('prefix'), sysconfig.get_config_var('exec_prefix'), sys.base_prefix):
        if base:
            cands += [os.path.join(base, 'include', ver), os.path.join(base, 'Headers')]
    for fw in ('/Library/Developer/CommandLineTools/Library/Frameworks/Python3.framework',
               '/Applications/Xcode.app/Contents/Developer/Library/Frameworks/Python3.framework',
               '/Library/Frameworks/Python.framework'):
        v = os.path.join(fw, 'Versions', '%d.%d' % (major, minor))
        cands += [os.path.join(v, 'include', ver), os.path.join(v, 'Headers')]
    cands += sorted(glob.glob(os.path.join(home, '.local', 'share', 'uv', 'python',
                                           'cpython-%d.%d*' % (major, minor), 'include', ver)))
    for c in cands:
        if c and os.path.isfile(os.path.join(c, 'Python.h')):
            return c
    return None


def cut_var_addr(text):
    """The whole var_addr function from the shim: from its signature to its matching closing brace."""
    start = text.find(SIGNATURE)
    if start < 0:
        return None
    open_brace = text.find('{', start)
    if open_brace < 0:
        return None
    depth = 0
    for i in range(open_brace, len(text)):
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def build(cxx, inc, work, source):
    lib = os.path.join(work, 'tgcu_var' + ('.dylib' if sys.platform == 'darwin' else '.so'))
    cmd = [cxx, '-std=c++17', '-O0', '-shared', '-fPIC', '-I', inc, '-o', lib, source]
    if sys.platform == 'darwin':
        cmd[-1:-1] = ['-undefined', 'dynamic_lookup']   # the Py* symbols resolve from the running interpreter
    built = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if built.returncode != 0:
        errors = [l.strip() for l in (built.stdout + built.stderr).splitlines() if ' error: ' in l]
        return None, 'does not build against Python.h (' + (errors[0] if errors else 'see the compile log').replace('"', "'") + ')'
    return lib, ''


def main():
    if not os.path.isfile(os.path.join(ROOT, SHIM)):
        print('FAIL run this test from the copy root: ' + SHIM + ' is not in ' + ROOT)
        return 1
    cxx = compiler()
    if cxx is None:
        print('FAIL no C++ compiler was found (looked for clang++, g++, c++)')
        return 1
    inc = python_include()
    if inc is None:
        print('FAIL no Python.h was found for python%d.%d' % sys.version_info[:2])
        return 1
    with open(os.path.join(ROOT, SHIM), encoding='utf-8', errors='replace') as f:
        body = cut_var_addr(f.read())
    if body is None:
        print('FAIL ' + SHIM + ' has no var_addr function to run')
        return 1
    work = tempfile.mkdtemp(prefix='tgcu-symbol-')
    failures = 0
    try:
        source = os.path.join(work, 'tgcu_var.cpp')
        with open(source, 'w', encoding='utf-8') as f:
            f.write(HARNESS.replace('@VAR_ADDR@', body))
        lib, detail = build(cxx, inc, work, source)
        if lib is None:
            print('FAIL the harness ' + detail)
            return 1
        probe = ctypes.PyDLL(lib)   # PyDLL keeps the GIL: the probe calls the Python C API
        probe.probe.restype = ctypes.c_ulonglong
        probe.probe.argtypes = [ctypes.py_object, ctypes.c_char_p, ctypes.POINTER(ctypes.c_int)]
        for case in CASES:
            err = ctypes.c_int(-1)
            got = probe.probe(FakeApi(case['reply']), b'table', ctypes.byref(err))
            ok = got == case['want'] and err.value == case['err']
            if ok:
                print('PASS ' + case['name'] + ': var_addr returned 0x%x, %d pending error' % (got, err.value))
            else:
                print('FAIL ' + case['name'] + ': var_addr returned 0x%x with %d pending error, expected 0x%x with %d'
                      % (got, err.value, case['want'], case['err']))
                failures += 1
    finally:
        shutil.rmtree(work, ignore_errors=True)
    print('%d case(s) failed' % failures if failures else 'all cases passed')
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
