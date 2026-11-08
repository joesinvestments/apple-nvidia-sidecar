#!/usr/bin/env python3
"""tests/test_fix_0094_5.py - a non-int answer from tgcu_api is an error, never a zero.

sidecar-shim/tgcudart.cpp:241 reads

    long long v = PyLong_Check(r) ? PyLong_AsLongLong(r) : 0;

so a tgcu_api function that answers None (or a tuple) is read as 0, and 0 is a valid module index: module_of stores
it in f->mod, func_of then looks the kernel up in module 0 as if the load had succeeded, and the real failure (an
exception raised inside the Python answer, or a tgcu_api that changed shape) is swallowed.  Every function call_i
calls is annotated "-> int" in sidecar-shim/tgcu_api.py, so a non-int answer can only be a bug and has to read as
the error it is.

This test takes call_i and module_of out of the shim source as they are written, compiles them against a stand-in
for the CPython object they receive and a stand-in callv that answers with a planted result, and runs the planted
answers through them:
  - module_load answering None is an error, not module 0 (today: module 0);
  - module_load answering a tuple is an error, not module 0 (today: module 0);
  - module_load answering a real index still reaches the caller unchanged, 0 (the first module) and 3;
  - module_load answering its own -1 still reaches the caller unchanged;
  - an int answer through call_i passes through unchanged;
  - a non-int answer leaves a diagnostic on stderr that names the tgcu_api function.
No GPU, no model, no weights, no Python is linked: the answer is a stand-in object and the compiler is the only
tool needed.  Prints one PASS or FAIL line per case and exits nonzero on any FAIL.
Run: python3 tests/test_fix_0094_5.py
"""
import os
import shutil
import subprocess
import sys
import tempfile

# The tree under test: the copy prove_test runs us in, else the one this file sits in.
SRC_REL = os.path.join('sidecar-shim', 'tgcudart.cpp')


def find_root():
    for base in (os.getcwd(), os.path.dirname(os.path.dirname(os.path.abspath(__file__)))):
        if os.path.isfile(os.path.join(base, SRC_REL)):
            return base
    return os.getcwd()


ROOT = find_root()

# The stand-ins: a PyObject with a kind (0 int, 1 None, 2 a tuple), the three CPython calls call_i makes on it, and a
# callv that hands back the planted answer.  FatHandle is the shim's own struct shape for what module_of receives.
STUB = r'''// built by tests/test_fix_0094_5.py
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

struct PyObject { int kind; long long v; };   // kind 0 = int, 1 = None, 2 = a tuple
static PyObject g_answer;
static const char* g_called = "";

PyObject* callv(const char* fn, const char* fmt, va_list ap) { (void) fmt; (void) ap; g_called = fn; return &g_answer; }
static int PyLong_Check(PyObject* o) { return o && o->kind == 0; }
long long PyLong_AsLongLong(PyObject* o) { return o ? o->v : 0; }
static void Py_DECREF(PyObject*) {}

struct FatHandle { const void* data; int mod = -1; };

// ---- taken from sidecar-shim/tgcudart.cpp, never retyped here ----
@@CODE@@
// ---- end of the code under test ----

int main(int argc, char** argv) {
  const char* mode = argv[1];    // "module": module_of on a fresh FatHandle; "call": call_i
  const char* fn = argv[2];      // the tgcu_api name the shim calls
  const char* kind = argv[3];    // "int", "none" or "tuple"
  g_answer.kind = kind[0] == 'i' ? 0 : kind[0] == 'n' ? 1 : 2;
  g_answer.v = argc > 4 ? std::atoll(argv[4]) : 0;
  if (std::strcmp(mode, "module") == 0) {
    FatHandle f{nullptr, -1};
    const int mod = module_of(&f);
    std::printf("result=%d called=%s\n", mod, g_called);
  } else {
    const long long v = call_i(fn, "(K)", (unsigned long long)0);
    std::printf("result=%lld called=%s\n", v, g_called);
  }
  return 0;
}
'''

# want "error" means a negative answer (the shim's own error code); a number means exactly that answer.
CASES = [
    dict(name='module_load answering None is an error, not module 0',
         mode='module', fn='module_load', kind='none', want='error'),
    dict(name='module_load answering a tuple is an error, not module 0',
         mode='module', fn='module_load', kind='tuple', want='error'),
    dict(name='module_load answering 0 still reaches the caller as the first module',
         mode='module', fn='module_load', kind='int', value='0', want='0'),
    dict(name='module_load answering 3 still reaches the caller as module 3',
         mode='module', fn='module_load', kind='int', value='3', want='3'),
    dict(name='module_load answering its own -1 still reaches the caller',
         mode='module', fn='module_load', kind='int', value='-1', want='-1'),
    dict(name='an int answer through call_i passes through unchanged',
         mode='call', fn='kernel_lookup', kind='int', value='7', want='7'),
    dict(name='a non-int answer is an error and names the tgcu_api function on stderr',
         mode='call', fn='kernel_lookup', kind='none', want='error', names='kernel_lookup'),
]

results = []


def case(name, ok, detail=''):
    results.append(bool(ok))
    print(('PASS ' if ok else 'FAIL ') + name + (':' + detail if detail else ''))


def compiler():
    for name in ('clang++', 'g++', 'c++'):
        path = shutil.which(name)
        if path:
            return path
    return None


def extract(src, signature):
    """The whole definition of `signature` (signature line and its balanced body), skipping comments and literals.

    A forward declaration of the same name is passed over: its first ; comes before its {.
    """
    at = src.find(signature)
    while at >= 0:
        brace = src.find('{', at)
        semi = src.find(';', at)
        if brace >= 0 and (semi < 0 or brace < semi):
            depth = 0
            i = brace
            n = len(src)
            while i < n:
                c = src[i]
                if c == '/' and src.startswith('//', i):
                    j = src.find('\n', i)
                    i = n if j < 0 else j
                    continue
                if c == '/' and src.startswith('/*', i):
                    j = src.find('*/', i + 2)
                    i = n if j < 0 else j + 2
                    continue
                if c in ('"', "'"):
                    quote = c
                    i += 1
                    while i < n:
                        if src[i] == '\\':
                            i += 2
                            continue
                        if src[i] == quote:
                            break
                        i += 1
                    i += 1
                    continue
                if c == '{':
                    depth += 1
                elif c == '}':
                    depth -= 1
                    if depth == 0:
                        return src[at:i + 1]
                i += 1
            raise LookupError(signature + ' has no closing brace')
        at = src.find(signature, at + 1)
    raise LookupError('no definition of ' + signature + ' in ' + SRC_REL)


def run(exe, cased):
    cmd = [exe, cased['mode'], cased['fn'], cased['kind']]
    if 'value' in cased:
        cmd.append(cased['value'])
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    return r


def main():
    src_path = os.path.join(ROOT, SRC_REL)
    try:
        with open(src_path, encoding='utf-8') as f:
            src = f.read()
    except OSError as e:
        case('the shim source is readable', False, str(e))
        return 1
    code = []
    try:
        code.append(extract(src, 'long long call_i(const char* fn'))
        code.append(extract(src, 'int module_of(FatHandle* f)'))
    except LookupError as e:
        case('the shim source still has call_i and module_of', False, str(e))
        return 1
    cxx = compiler()
    if not cxx:
        case('a C++ compiler is available to build the shim code', False, 'no clang++, g++ or c++ on PATH')
        return 1
    work = tempfile.mkdtemp(prefix='fix-0094-5-')
    try:
        src_file = os.path.join(work, 'shim_code.cpp')
        exe = os.path.join(work, 'shim_code')
        with open(src_file, 'w', encoding='utf-8') as f:
            f.write(STUB.replace('@@CODE@@', '\n\n'.join(code)))
        built = subprocess.run([cxx, '-std=c++17', '-O0', '-o', exe, src_file], capture_output=True, text=True, timeout=180)
        if built.returncode != 0:
            errs = [l.strip() for l in (built.stdout + built.stderr).splitlines() if ' error: ' in l]
            case('the shim code builds against the stand-ins', False, errs[0] if errs else 'see the compile log')
            return 1
        case('the shim code builds against the stand-ins', True)
        for cased in CASES:
            r = run(exe, cased)
            out = r.stdout.strip()
            if r.returncode != 0:
                case(cased['name'], False, 'the built program exited %d: %s' % (r.returncode, out[-200:]))
                continue
            answer = [ln for ln in out.splitlines() if ln.startswith('result=')]
            if not answer:
                case(cased['name'], False, 'the program printed %r' % out[-200:])
                continue
            got = int(answer[-1].split('=', 1)[1].split()[0])
            named = answer[-1].split('called=', 1)[-1]
            if cased['want'] == 'error':
                ok = got < 0
                detail = 'the shim answered %d, expected an error (negative)' % got
            else:
                ok = got == int(cased['want'])
                detail = 'the shim answered %d, expected %s' % (got, cased['want'])
            if not cased['fn'] in named:
                ok, detail = False, 'the shim called %r, expected %r' % (named, cased['fn'])
            if ok and cased.get('names') and cased['names'] not in r.stderr:
                ok, detail = False, 'nothing on stderr names %r, so the swallowed error stays invisible: %r' % (
                    cased['names'], r.stderr.strip()[-200:])
            case(cased['name'], ok, '' if ok else detail)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    fails = len(results) - sum(results)
    print('%d case(s), %d PASS, %d FAIL' % (len(results), sum(results), fails))
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())
