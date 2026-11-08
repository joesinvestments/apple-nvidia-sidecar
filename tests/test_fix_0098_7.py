#!/usr/bin/env python3
"""tests/test_fix_0098_7.py - a failed cuMemAddressFree or cuMemRelease must not be reported as CUDA_SUCCESS.

sidecar-shim/tgcudart.cpp calls tgcu's vmm_free_va in vmm_AddressFree and vmm_release in vmm_Release, and both throw
the answer away: the entry point then returns CUDA_SUCCESS whatever the Python side did. tgcu_api.vmm_free_va and
tgcu_api.vmm_release return 0 only when the work was done (a Python exception there leaves callv with no result, so
call_i answers -1), so a failed free or release was in the shim's hand and went back to the caller as success. The
callers reach these entry points through cudaGetDriverEntryPoint (src/core/vmm.cpp for the elastic K/V --kv-grow,
src/core/expert_cache.cpp for the segmented arena --vram-elastic), and the reserved range or the 2 MiB physical chunk
stays behind with no error to act on.

These cases compile the real vmm_AddressFree and vmm_Release out of sidecar-shim/tgcudart.cpp against a stub tgcu (no
GPU, no CUDA headers, no Python): call_i answers what tgcu_api would (0 on success, -1 when the Python call raised, a
positive code otherwise) and records the arguments, and the shim's range registry is planted before each run.
  - a successful cuMemAddressFree hands the range and its size to tgcu and still reports CUDA_SUCCESS;
  - a failed cuMemAddressFree reports a CUDA error, not CUDA_SUCCESS;
  - a positive failure code from tgcu on cuMemAddressFree is a CUDA error too;
  - cuMemAddressFree of a range the shim never reserved is still refused and never reaches tgcu;
  - a successful cuMemRelease hands the handle to tgcu and still reports CUDA_SUCCESS;
  - a failed cuMemRelease reports a CUDA error, not CUDA_SUCCESS.
Prints one PASS or FAIL line per case and exits nonzero on any FAIL. Run: python3 tests/test_fix_0098_7.py
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = os.getcwd()   # the tree under test: every test here runs with cwd at the copy root, never at the test file
SHIM = os.path.join('sidecar-shim', 'tgcudart.cpp')

GRAN = 2 << 20                 # the shim's kVmmGran: one 2 MiB chunk
VA = 0x7f0000000000            # the address range vmm_AddressReserve registered in g_dev
HANDLE = 0x1000001             # a cuMemCreate handle (physical address plus one)
CUDA_SUCCESS = 0
CUDA_ERROR_INVALID_VALUE = 1

# the stub CUDA surface: the driver's types and error numbers (cuda.h) plus tgcu behind call_i. VMM_ENTER is the
# shim's own init / GIL / flush prologue, stubbed here so the case sees only the answer the entry point gives.
PRELUDE = r"""
#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>

typedef int CUresult;
typedef unsigned long long CUdeviceptr;
typedef unsigned long long CUmemGenericAllocationHandle;

#define EXPORT
#define VMM_ENTER

static const CUresult CUDA_SUCCESS = 0;
static const CUresult CUDA_ERROR_INVALID_VALUE = 1;
static const CUresult CUDA_ERROR_OUT_OF_MEMORY = 2;
static const CUresult CUDA_ERROR_NOT_INITIALIZED = 3;

// the shim's registry of reserved ranges (a static map in sidecar-shim/tgcudart.cpp), planted by each case
static std::map<uint64_t, size_t> g_dev;

// tgcu_api behind call_i: 0 on success, -1 when the Python call raised, a positive code otherwise.
static long long g_free_va_rc = 0, g_release_rc = 0;
static int g_free_va_calls = 0, g_release_calls = 0;
static unsigned long long g_free_va_arg0 = 0, g_free_va_arg1 = 0, g_release_arg0 = 0;

static long long call_i(const char* fn, const char* fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  long long r = 0;
  if (std::strcmp(fn, "vmm_free_va") == 0) {
    g_free_va_arg0 = va_arg(ap, unsigned long long);
    g_free_va_arg1 = va_arg(ap, unsigned long long);
    ++g_free_va_calls;
    r = g_free_va_rc;
  } else if (std::strcmp(fn, "vmm_release") == 0) {
    g_release_arg0 = va_arg(ap, unsigned long long);
    ++g_release_calls;
    r = g_release_rc;
  }
  va_end(ap);
  return r;
}
"""

MAIN = r"""
static void report(const char* mode, CUresult rc) {
  const unsigned long long a0 = g_free_va_calls ? g_free_va_arg0 : g_release_arg0;
  std::printf("mode=%s rc=%d free_va_calls=%d release_calls=%d a0=%llu a1=%llu\n", mode, (int) rc, g_free_va_calls,
              g_release_calls, a0, g_free_va_arg1);
}

int main(int argc, char** argv) {
  if (argc != 2) {
    std::printf("usage: %s mode\n", argv[0]);
    return 2;
  }
  const std::string mode = argv[1];
  const CUdeviceptr va = 0x7f0000000000ull;
  const size_t size = 2u << 20;
  g_dev[va] = size;
  CUresult rc = CUDA_SUCCESS;
  if (mode == "free-ok") {
    g_free_va_rc = 0;
    rc = vmm_AddressFree(va, size);
  } else if (mode == "free-fail") {
    g_free_va_rc = -1;   // tgcu_api.vmm_free_va raised: call_i has no result and answers -1
    rc = vmm_AddressFree(va, size);
  } else if (mode == "free-fail-positive") {
    g_free_va_rc = 1;
    rc = vmm_AddressFree(va, size);
  } else if (mode == "free-unknown") {
    g_free_va_rc = 0;
    rc = vmm_AddressFree(va + size, size);   // a range the shim never reserved
  } else if (mode == "release-ok") {
    g_release_rc = 0;
    rc = vmm_Release(0x1000001ull);
  } else if (mode == "release-fail") {
    g_release_rc = -1;
    rc = vmm_Release(0x1000001ull);
  } else {
    std::printf("unknown mode %s\n", mode.c_str());
    return 2;
  }
  report(mode.c_str(), rc);
  return 0;
}
"""

NEEDED = ['vmm_AddressFree', 'vmm_Release']

ERROR_NAMES = {
    CUDA_ERROR_INVALID_VALUE: 'CUDA_ERROR_INVALID_VALUE',
    2: 'CUDA_ERROR_OUT_OF_MEMORY',
    3: 'CUDA_ERROR_NOT_INITIALIZED',
}

# case: the answer tgcu gives, and what the entry point must return. rc 'error' means any nonzero CUresult: the call
# failed, so the shim must not answer CUDA_SUCCESS. The ordinary cases (ok, unknown range) answer the same before and
# after the fix; the two failures are the harm.
CASES = [
    dict(name='a successful cuMemAddressFree hands the range to tgcu and reports CUDA_SUCCESS',
         mode='free-ok', rc=CUDA_SUCCESS, free_va_calls='1', release_calls='0', a0=str(VA), a1=str(GRAN)),
    dict(name='a failed cuMemAddressFree reports a CUDA error, not CUDA_SUCCESS',
         mode='free-fail', rc='error', free_va_calls='1'),
    dict(name='a positive failure code from tgcu on cuMemAddressFree is a CUDA error too',
         mode='free-fail-positive', rc='error', free_va_calls='1'),
    dict(name='cuMemAddressFree of a range the shim never reserved is still refused and never reaches tgcu',
         mode='free-unknown', rc=CUDA_ERROR_INVALID_VALUE, free_va_calls='0', release_calls='0'),
    dict(name='a successful cuMemRelease hands the handle to tgcu and reports CUDA_SUCCESS',
         mode='release-ok', rc=CUDA_SUCCESS, free_va_calls='0', release_calls='1', a0=str(HANDLE)),
    dict(name='a failed cuMemRelease reports a CUDA error, not CUDA_SUCCESS',
         mode='release-fail', rc='error', release_calls='1'),
]


def read_shim():
    with open(SHIM, encoding='utf-8', errors='replace') as f:
        return f.read()


def matching_brace(text, i):
    """The index of the brace that closes the one at i, with comments and literals skipped."""
    depth, n = 0, len(text)
    while i < n:
        c = text[i]
        if text.startswith('//', i):
            j = text.find('\n', i)
            i = n if j < 0 else j
        elif text.startswith('/*', i):
            j = text.find('*/', i + 2)
            i = n if j < 0 else j + 2
        elif c in '"\'':
            quote = c
            i += 1
            while i < n and text[i] != quote:
                i += 2 if text[i] == '\\' else 1
            i += 1
        else:
            if c == '{':
                depth += 1
            elif c == '}':
                depth -= 1
                if depth == 0:
                    return i
            i += 1
    return -1


def definition(text, name):
    """The whole definition of name, signature through its closing brace, or None."""
    pat = re.compile(r'\b' + re.escape(name) + r'\s*\(')
    off = 0
    for line in text.split('\n'):
        code = line.split('//')[0]
        m = pat.search(code)
        brace = code.find('{', m.end()) if m else -1
        if brace >= 0:
            close = matching_brace(text, off + brace)
            if close > 0:
                return text[off:close + 1]
        off += len(line) + 1
    return None


def harness(text):
    parts = [PRELUDE]
    for name in NEEDED:
        body = definition(text, name)
        if body is None:
            return None, name
        parts.append(body)
    parts.append(MAIN)
    return '\n'.join(parts), None


def compiler():
    for name in ('g++', 'clang++', 'c++'):
        path = shutil.which(name)
        if path:
            return path
    return None


def build(cxx, work, source):
    src = os.path.join(work, 'vmm_free.cpp')
    exe = os.path.join(work, 'vmm_free')
    with open(src, 'w', encoding='utf-8') as f:
        f.write(source)
    built = subprocess.run([cxx, '-std=c++17', '-O0', '-o', exe, src], capture_output=True, text=True, timeout=300)
    if built.returncode != 0:
        errors = [l.strip() for l in (built.stdout + built.stderr).splitlines() if ' error: ' in l]
        return None, (errors[0] if errors else 'see the compile log').replace('"', "'")
    return exe, ''


def run(exe, mode):
    r = subprocess.run([exe, mode], capture_output=True, text=True, timeout=120)
    text = r.stdout + r.stderr
    for line in text.splitlines():
        parts = line.split()
        if parts and parts[0].startswith('mode='):
            return dict(p.split('=', 1) for p in parts if '=' in p), ''
    return None, 'the built program exited %d without a result: %s' % (r.returncode, text.strip()[-300:])


def check(case, got):
    if got is None:
        return 'no result'
    rc = got.get('rc')
    if rc is None:
        return 'the program printed no rc'
    if case['rc'] == 'error':
        if rc == '0':
            return 'rc=0 (CUDA_SUCCESS): the failed call was reported as success'
    elif rc != str(case['rc']):
        return 'rc=%s, want %d (%s)' % (rc, case['rc'], ERROR_NAMES.get(case['rc'], 'CUDA_SUCCESS'))
    for key in ('free_va_calls', 'release_calls', 'a0', 'a1'):
        if key in case and got.get(key) != case[key]:
            return '%s=%s, want %s' % (key, got.get(key), case[key])
    return ''


def main():
    if not os.path.isfile(SHIM):
        print('FAIL run this test from the copy root: ' + SHIM + ' is not in ' + ROOT)
        return 1
    source, missing = harness(read_shim())
    if source is None:
        print('FAIL sidecar-shim/tgcudart.cpp has no definition of ' + missing)
        return 1
    cxx = compiler()
    if cxx is None:
        print('FAIL no C++ compiler was found (looked for g++, clang++, c++)')
        return 1
    work = tempfile.mkdtemp(prefix='shim-vmm-free-')
    try:
        exe, detail = build(cxx, work, source)
        if exe is None:
            print('FAIL the shim\'s vmm entry points do not build against the stub tgcu (' + detail + ')')
            return 1
        results = {case['mode']: run(exe, case['mode']) for case in CASES}
    finally:
        shutil.rmtree(work, ignore_errors=True)
    failures = 0
    for case in CASES:
        got, why = results[case['mode']]
        detail = why or check(case, got)
        print(('PASS ' if not detail else 'FAIL ') + case['name'] + (': ' + detail if detail else ''))
        failures += 0 if not detail else 1
    print('%d case(s) failed' % failures if failures else 'all cases passed')
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
