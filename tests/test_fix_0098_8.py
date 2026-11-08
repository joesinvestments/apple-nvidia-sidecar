#!/usr/bin/env python3
"""tests/test_fix_0098_8.py - a builtin kernel the driver does not know must not stay in the cache.

sidecar-shim/tgcudart.cpp tgcu_builtin_launch() keeps a per-name cache of Func pointers. On a miss it allocates the
Func, asks tgcu's builtin_kernel for the name and, when the answer is negative, returns 1 while leaving that Func
with kid<0 in the cache. The next call with the same name finds the entry, skips the creation block and hands the
invalid Func to pack_and_launch. pack_and_launch asks kernel_params(kid) and launch(kid, ...), and on tgcu's side
those are kernels[kid] (sidecar-shim/tgcu_api.py:558 and :596). A Python list read with -1 is the last kernel, so the
second call does not fail: it returns 0 after running the last kernel the process registered, with the parameters of
the call that failed. cublasGemmEx reaches this through tgcu_builtin_launch (sidecar-shim/tgcublas.cpp:40 and :43),
so a tgd MM name the builtin module does not carry makes a later GEMM launch a different kernel and report success.

These cases compile the real tgcu_builtin_launch out of sidecar-shim/tgcudart.cpp against a fake Python side (no GPU,
no CUDA headers, no Python embedding, no model) and run it:
  - the first call with a name the driver does not know reports failure and launches nothing;
  - the second call with the same name still reports failure and launches nothing. This case FAILS on today's code;
  - the second call asks the driver again instead of reusing a failed entry. This case FAILS on today's code;
  - a name the driver knows launches its own kernel on every call and is asked only once;
  - a failed name does not disturb a name the driver knows.
Prints one PASS or FAIL line per case and exits nonzero on any FAIL. Run from the copy root:
python3 tests/test_fix_0098_8.py
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = os.getcwd()   # the tree under test: every test here runs with cwd at the copy root, never at the test file
SOURCE = os.path.join('sidecar-shim', 'tgcudart.cpp')
SIGNATURE = 'tgcu_builtin_launch'

# The fake Python side is sidecar-shim/tgcu_api.py: builtin_kernel() appends the kernel it found to kernels and
# answers its index, kernel_params(kid) and launch(kid, ...) read kernels[kid], and a negative index counts from the
# end of that list. pack_and_launch below is that same path with the launch replaced by a record of what it ran.
PRELUDE = r"""
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

typedef int cudaError_t;
typedef void* cudaStream_t;
static const cudaError_t cudaSuccess = 0;
static const cudaError_t cudaErrorLaunchFailure = 719;
#define EXPORT
#define API_ENTER(name) ((void)0)

struct FatHandle { const void* data; int mod = -1; };
struct Func { FatHandle* fat; std::string name; int kid = -1; int max_dyn_shared = 48 << 10; };
struct dim3 { unsigned x, y, z; dim3(unsigned a, unsigned b, unsigned c) : x(a), y(b), z(c) {} };

static std::vector<std::string> g_kernels;         // tgcu_api.py kernels: the kernels the process has registered
static std::map<std::string, bool> g_builtin;      // tgcu_api.py builtin: the names the loaded builtin module has
static int g_asked = 0;                            // calls to builtin_kernel
static std::vector<int> g_pack_kids;               // the kid pack_and_launch was handed
static std::vector<std::string> g_launched;        // what launch(kid, ...) ran, kernels[kid] as Python reads it

static std::string joined(const std::vector<int>& v) {
  std::string s;
  for (size_t i = 0; i < v.size(); i++) { if (i) s += ','; s += std::to_string(v[i]); }
  return s.empty() ? "-" : s;
}
static std::string joined(const std::vector<std::string>& v) {
  std::string s;
  for (size_t i = 0; i < v.size(); i++) { if (i) s += ','; s += v[i]; }
  return s.empty() ? "-" : s;
}

static long long call_i(const char* fn, const char* fmt, ...) {
  if (std::strcmp(fn, "builtin_kernel")) return -1;
  va_list ap; va_start(ap, fmt);
  const char* name = va_arg(ap, const char*);
  va_end(ap);
  ++g_asked;
  if (!g_builtin[name]) return -1;
  g_kernels.push_back(name);
  return (long long) g_kernels.size() - 1;
}

static cudaError_t pack_and_launch(Func* fn, dim3, dim3, size_t, void**, cudaStream_t) {
  const long long kid = fn->kid, n = (long long) g_kernels.size();
  g_pack_kids.push_back((int) kid);
  if (kid < n && -kid <= n) {
    g_launched.push_back(g_kernels[kid < 0 ? n + kid : kid]);
    return cudaSuccess;
  }
  g_launched.push_back("<no kernel>");
  return cudaErrorLaunchFailure;
}
"""

MAIN = r"""
static int launch(const char* name, void** args) { return tgcu_builtin_launch(name, 1, 1, 256, args, nullptr); }

static void report(const char* call, int rc) {
  std::printf("%s rc=%d asked=%d pack_kids=%s launched=%s\n", call, rc, g_asked,
              joined(g_pack_kids).c_str(), joined(g_launched).c_str());
}

int main(int argc, char** argv) {
  const std::string sc = argc > 1 ? argv[1] : "";
  void* args[4] = {nullptr, nullptr, nullptr, nullptr};
  g_kernels.push_back("tgcu_memset32");   // a kernel the engine registered before (a memset ran), so the list is not empty
  g_builtin["tgcu_memcpy"] = true;        // a builtin name the loaded module does carry
  if (sc == "known_twice") {
    report("call1", launch("tgcu_memcpy", args));
    report("call2", launch("tgcu_memcpy", args));
  } else if (sc == "unknown_twice") {
    report("call1", launch("tgcu_gemm2_tn_bf16_f32", args));
    report("call2", launch("tgcu_gemm2_tn_bf16_f32", args));
  } else if (sc == "unknown_then_known") {
    report("call1", launch("tgcu_gemm2_tn_bf16_f32", args));
    report("call2", launch("tgcu_memcpy", args));
  } else {
    std::fprintf(stderr, "unknown scenario %s\n", sc.c_str());
    return 2;
  }
  return 0;
}
"""


def function_text(src):
    """The real tgcu_builtin_launch, from the start of its line to its matching close brace, comments and literals skipped."""
    m = re.search(r'^[^\n]*\b' + SIGNATURE + r'\s*\(', src, re.M)
    if m is None:
        return None
    start = m.start()
    i = src.find('{', m.end())
    if i < 0:
        return None
    depth = 0
    while 0 <= i < len(src):
        if src.startswith('//', i):
            i = src.find('\n', i)
            if i < 0:
                return None
        elif src.startswith('/*', i):
            i = src.find('*/', i + 2)
            if i < 0:
                return None
            i += 1
        elif src[i] in '"\'':
            q = src[i]
            i += 1
            while i < len(src) and src[i] != q:
                i += 2 if src[i] == '\\' else 1
        elif src[i] == '{':
            depth += 1
        elif src[i] == '}':
            depth -= 1
            if depth == 0:
                return src[start:i + 1]
        i += 1
    return None


def compiler():
    for name in ('clang++', 'c++', 'g++'):
        path = shutil.which(name)
        if path:
            return path
    return None


def build(tmp):
    """Compile the real function against the fake Python side: no GPU, no CUDA headers, no model."""
    cxx = compiler()
    if cxx is None:
        return None, 'no C++ compiler was found (looked for clang++, c++, g++)'
    try:
        with open(SOURCE, encoding='utf-8', errors='replace') as f:
            src = f.read()
    except OSError as e:
        return None, 'cannot read %s: %s' % (SOURCE, e)
    body = function_text(src)
    if body is None:
        return None, '%s has no %s() definition that the test can compile' % (SOURCE, SIGNATURE)
    prog = os.path.join(tmp, 'builtin_launch.cpp')
    with open(prog, 'w', encoding='utf-8') as f:
        f.write(PRELUDE + '\n' + body + '\n' + MAIN)
    exe = os.path.join(tmp, 'builtin_launch')
    r = subprocess.run([cxx, '-std=c++17', '-O0', '-Wno-everything', '-o', exe, prog],
                       capture_output=True, text=True, timeout=300)
    if r.returncode != 0:
        errors = [l.strip() for l in (r.stdout + r.stderr).splitlines() if ' error: ' in l]
        return None, 'compiling %s failed: %s' % (SOURCE, errors[0] if errors else (r.stderr or r.stdout).strip()[-300:])
    return exe, ''


def run(exe, scenario):
    r = subprocess.run([exe, scenario], capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        return None, 'scenario %s exited %d: %s' % (scenario, r.returncode, (r.stdout + r.stderr).strip()[-200:])
    got = {}
    for line in r.stdout.splitlines():
        parts = line.split()
        if len(parts) != 5 or not parts[0].startswith('call'):
            continue
        row = {}
        for p in parts[1:]:
            key, _, value = p.partition('=')
            row[key] = value
        got[parts[0]] = row
    return got, ''


def kids(got, call):
    """The kids pack_and_launch was handed, as ints; None when the call was not reported."""
    row = got.get(call)
    if row is None:
        return None
    v = row.get('pack_kids')
    if v in (None, '-'):
        return []
    try:
        return [int(x) for x in v.split(',')]
    except ValueError:
        return None


def launched(got, call):
    row = got.get(call)
    if row is None:
        return None
    v = row.get('launched')
    return [] if v in (None, '-') else v.split(',')


def check_first_unknown(got):
    row = got.get('call1')
    if row is None:
        return 'the first call was not reported'
    if row.get('rc') != '1':
        return 'the first call with a name the driver does not know returned %s, wanted 1' % row.get('rc')
    if kids(got, 'call1') != []:
        return 'the first call handed pack_and_launch kid(s) %s: a kernel the driver does not know must not launch' % (
            kids(got, 'call1'))
    return ''


def check_second_unknown(got):
    row = got.get('call2')
    if row is None:
        return 'the second call was not reported'
    if row.get('rc') != '1':
        return ('the second call with the same unknown name returned %s after launching %s, wanted 1: the failed entry '
                'stayed cached and the invalid kid reached launch' % (row.get('rc'), launched(got, 'call2')))
    if kids(got, 'call2') != []:
        return 'the second call handed pack_and_launch kid(s) %s: the failed cache entry was reused' % (kids(got, 'call2'))
    return ''


def check_asked_again(got):
    row = got.get('call2')
    if row is None:
        return 'the second call was not reported'
    if row.get('asked') != '2':
        return ('builtin_kernel was asked %s time(s) by the second call, wanted 2: the shim reused the failed entry '
                'instead of asking the driver again' % row.get('asked'))
    return ''


def check_known(got):
    wanted = {'call1': ['tgcu_memcpy'], 'call2': ['tgcu_memcpy', 'tgcu_memcpy']}   # the second call adds one launch
    for call in ('call1', 'call2'):
        row = got.get(call)
        if row is None:
            return 'the %s of a known name was not reported' % call
        if row.get('rc') != '0':
            return 'a name the driver knows returned %s on %s, wanted 0' % (row.get('rc'), call)
        if launched(got, call) != wanted[call]:
            return 'the %s had launched %s, wanted %s' % (call, launched(got, call), wanted[call])
        if row.get('asked') != '1':
            return ('builtin_kernel was asked %s time(s) by the %s, wanted 1: a known name is still cached'
                    % (row.get('asked'), call))
    return ''


def check_after_failure(got):
    row = got.get('call2')
    if row is None:
        return 'the call after a failed name was not reported'
    if row.get('rc') != '0':
        return 'the known name after a failed one returned %s, wanted 0' % row.get('rc')
    if launched(got, 'call2') != ['tgcu_memcpy']:
        return 'the known name after a failed one launched %s, wanted tgcu_memcpy' % (launched(got, 'call2'))
    if kids(got, 'call2') != [1]:
        return ('the known name after a failed one was handed kid(s) %s, wanted 1 (the kernel it appended after the '
                'memset the engine registered first)' % (kids(got, 'call2')))
    return ''


CASES = [
    dict(name='the first call with a name the driver does not know reports failure and launches nothing',
         scenario='unknown_twice', check=check_first_unknown),
    dict(name='the second call with the same unknown name still reports failure and launches nothing',
         scenario='unknown_twice', check=check_second_unknown),
    dict(name='the second call asks the driver again instead of reusing the failed cache entry',
         scenario='unknown_twice', check=check_asked_again),
    dict(name='a name the driver knows launches its own kernel on every call and is asked only once',
         scenario='known_twice', check=check_known),
    dict(name='a failed name does not disturb a name the driver knows',
         scenario='unknown_then_known', check=check_after_failure),
]


def main():
    if not os.path.isfile(SOURCE):
        print('FAIL run this test from the copy root: ' + SOURCE + ' is not in ' + ROOT)
        return 1
    tmp = tempfile.mkdtemp(prefix='fix-0098-8-')
    try:
        exe, why = build(tmp)
        if exe is None:
            print('FAIL the builtin launch could not be built: ' + why)
            return 1
        results, failures = {}, 0
        for case in CASES:
            if case['scenario'] not in results:
                got, why = run(exe, case['scenario'])
                results[case['scenario']] = (got, why)
            got, why = results[case['scenario']]
            detail = why or case['check'](got)
            print(('PASS ' if not detail else 'FAIL ') + case['name'] + (': ' + detail if detail else ''))
            failures += 0 if not detail else 1
        print('%d case(s) failed' % failures if failures else 'all cases passed')
        return 1 if failures else 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    sys.exit(main())
