#!/usr/bin/env python3
"""tests/test_fix_0098_3.py - the shim's occupancy query must not divide by a zero warp count.

sidecar-shim/tgcudart.cpp:1128 derives the warp count as warps = (block + 31) / 32 and line 1129 then divides the
register budget by warps. A caller that asks with block 0 (a runtime block size that came out zero) gets warps 0, so
65536 / (0 * regs-per-warp * 256) is a division by zero when the kernel reports numRegs > 0: on arm64 that yields 0
and the query answers "no block fits", on x86 it raises SIGFPE and kills the serving process. Line 1130 guards the
same value with std::max(1, ...) for by_threads only, so the register arm is the one left unguarded.

These cases take the real function out of sidecar-shim/tgcudart.cpp, compile it against a stub cudaFuncGetAttributes
(no GPU, no CUDA headers, no model) and run the built program on planted attributes:
  - block 0 with a kernel that reports registers must answer like the smallest legal block, not 0;
  - block 0 with a register-heavy kernel keeps the register limit (8 blocks) instead of 0;
  - block 0 with a large dynamic shared memory request keeps the shared memory limit (1 block) instead of 0;
  - block 32, block 128 and block 0 with no reported registers answer the same as today (ordinary input).
Prints one PASS or FAIL line per case and exits nonzero on any FAIL. Run: python3 tests/test_fix_0098_3.py
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = os.getcwd()   # the tree under test: every test here runs with cwd at the copy root, never at the test file
SOURCE = os.path.join('sidecar-shim', 'tgcudart.cpp')
SIGNATURE = 'EXPORT cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags('

# the harness supplies what the function needs from libcudart: the error type, the attribute record and the query the
# shim answers from tgcu. Everything about the occupancy arithmetic is the real function, compiled as it is written.
HARNESS = r"""
#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <cstdlib>

#define EXPORT
typedef int cudaError_t;
static const cudaError_t cudaSuccess = 0;

struct cudaFuncAttributes {
  int numRegs;
  size_t sharedSizeBytes;
  size_t localSizeBytes;
  size_t constSizeBytes;
  size_t maxDynamicSharedSizeBytes;
  int maxThreadsPerBlock;
};

static cudaFuncAttributes g_attrs;

cudaError_t cudaFuncGetAttributes(cudaFuncAttributes* a, const void*) {
  *a = g_attrs;
  return cudaSuccess;
}

// taken from sidecar-shim/tgcudart.cpp by tests/test_fix_0098_3.py
@@BODY@@

int main(int argc, char** argv) {
  if (argc != 5) { std::printf("usage: %s numRegs sharedSizeBytes block shmem\n", argv[0]); return 2; }
  g_attrs.numRegs = std::atoi(argv[1]);
  g_attrs.sharedSizeBytes = (size_t) std::strtoull(argv[2], nullptr, 10);
  const int block = std::atoi(argv[3]);
  const size_t shmem = (size_t) std::strtoull(argv[4], nullptr, 10);
  int n = -1;
  const cudaError_t rc = cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(&n, nullptr, block, shmem, 0);
  std::printf("rc=%d n=%d\n", (int) rc, n);
  return 0;
}
"""

# case: name, numRegs, sharedSizeBytes, block, shmem, wanted return code, wanted blocks per multiprocessor.
# The wanted counts are worked out by hand from CUDA's limits: 65536 registers per multiprocessor in 256 register
# granules, 1536 threads (48 warps) per multiprocessor, 100 KiB of shared memory in 128 B granules, at most 16 blocks.
CASES = [
    dict(name='block 0 with 32 registers answers like block 32, not 0 blocks',
         attrs=(32, 0, 0, 0), rc=0, n=16),          # warps 1, by_regs 65536/1024 = 64, by_threads 48, by_shmem 100
    dict(name='block 0 with 255 registers keeps the register limit of 8 blocks',
         attrs=(255, 0, 0, 0), rc=0, n=8),          # regs per warp 8192 B, by_regs 65536/8192 = 8
    dict(name='block 0 with 50 KiB of dynamic shared memory keeps the 1 block limit',
         attrs=(32, 0, 0, 50 * 1024), rc=0, n=1),   # sh 52224 B, by_shmem 102400/52224 = 1
    dict(name='block 32 with 32 registers still answers 16 blocks',
         attrs=(32, 0, 32, 0), rc=0, n=16),
    dict(name='block 128 with 64 registers still answers 8 blocks',
         attrs=(64, 0, 128, 0), rc=0, n=8),         # warps 4, by_regs 65536/8192 = 8, by_threads 12
    dict(name='block 0 with no reported registers still answers 16 blocks',
         attrs=(0, 0, 0, 0), rc=0, n=16),           # by_regs 16 for numRegs 0, before and after the guard
]


def function_text(src):
    """the real occupancy function, from its signature to its matching close brace, comments and literals skipped"""
    start = src.find(SIGNATURE)
    if start < 0: return None
    i = src.find('{', start)
    depth = 0
    while 0 <= i < len(src):
        if src.startswith('//', i):
            i = src.find('\n', i)
            if i < 0: return None
        elif src.startswith('/*', i):
            i = src.find('*/', i)
            if i < 0: return None
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
            if depth == 0: return src[start:i + 1]
        i += 1
    return None


def build(tmp):
    """compile the real function with a stub attribute query: no GPU, no CUDA headers, no model"""
    cxx = next((c for c in ('clang++', 'c++', 'g++') if shutil.which(c)), None)
    if cxx is None: return None, 'no C++ compiler on PATH'
    try:
        src = open(SOURCE).read()
    except OSError as e:
        return None, f'cannot read {SOURCE}: {e}'
    body = function_text(src)
    if body is None: return None, f'{SOURCE} has no {SIGNATURE.strip()} that the test can compile'
    prog = os.path.join(tmp, 'occupancy.cpp')
    with open(prog, 'w') as f:
        f.write(HARNESS.replace('@@BODY@@', body))
    exe = os.path.join(tmp, 'occupancy')
    r = subprocess.run([cxx, '-std=c++17', '-O0', '-o', exe, prog], capture_output=True, text=True)
    if r.returncode != 0: return None, f'{cxx} could not build the occupancy function: ' + (r.stdout + r.stderr).strip()[-300:]
    return exe, ''


def run(exe, attrs):
    r = subprocess.run([exe] + [str(a) for a in attrs], capture_output=True, text=True)
    if r.returncode < 0: return None, None, f'killed by signal {-r.returncode}'
    if r.returncode != 0: return None, None, f'exit {r.returncode}: ' + (r.stdout + r.stderr).strip()[-200:]
    m = re.search(r'rc=(-?\d+) n=(-?\d+)', r.stdout)
    if m is None: return None, None, f'unreadable output: {r.stdout.strip()[-200:]}'
    return int(m.group(1)), int(m.group(2)), ''


def main():
    tmp = tempfile.mkdtemp(prefix='fix-0098-3-')
    try:
        exe, why = build(tmp)
        if exe is None:
            print('FAIL the occupancy function could not be built: ' + why)
            return 1
        fails = 0
        for c in CASES:
            rc, n, why = run(exe, c['attrs'])
            regs, shared, block, shmem = c['attrs']
            got = why or f'rc={rc} n={n}'
            ok = rc == c['rc'] and n == c['n']
            fails += 0 if ok else 1
            print(('PASS ' if ok else 'FAIL ') + c['name'] +
                  f' (numRegs {regs}, shared {shared} B, block {block}, shmem {shmem} B: {got}, want rc={c["rc"]} n={c["n"]})')
        return 1 if fails else 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    sys.exit(main())
