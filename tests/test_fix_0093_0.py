#!/usr/bin/env python3
"""tests/test_fix_0093_0.py - the force-included CUDA launch header must carry the types it declares.

sidecar-shim/tgcuda_launch.h:7 declares cudaConfigureCall with cudaError_t, dim3 and cudaStream_t, but the header
includes no CUDA header. -include processes a header before the translation unit's own #includes, so a host-only CUDA
compile that force-includes this header first meets four undeclared types and fails before it ever reads the source.
These cases stub the CUDA toolkit headers in a temp folder (no GPU, no toolkit, no network) and build the header the
way a force-include does:
  - the header force-included alone declares the three launch functions and their types;
  - a force-included caller can call cudaConfigureCall with its default sharedMem and stream;
  - explicit sharedMem and stream reach the driver call;
  - the order the only build site uses today (cuda_runtime.h first, then this header) keeps working.
Ordinary input, the order the build site uses, gives the same result before and after the change.
Prints one PASS or FAIL line per case and exits nonzero on any FAIL. Run: python3 tests/test_fix_0093_0.py
"""
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.getcwd()   # the tree under test: the test runs with cwd at the copy root, never at the test file
HEADER = os.path.join('sidecar-shim', 'tgcuda_launch.h')
RUNTIME_H = 'cuda_runtime.h'

# The smallest stubs that behave like the CUDA 12 headers this build understands: vector_types.h declares dim3,
# cuda_runtime_api.h pulls it in and adds cudaError_t and cudaStream_t, cuda_runtime.h pulls in both. Nothing
# declares them from the source file, so the force-included header must include them itself.
VECTOR_TYPES = """#ifndef STUB_VECTOR_TYPES_H
#define STUB_VECTOR_TYPES_H
struct dim3 { unsigned int x, y, z; };
#endif
"""

RUNTIME_API = """#ifndef STUB_CUDA_RUNTIME_API_H
#define STUB_CUDA_RUNTIME_API_H
#include <stddef.h>
#include "vector_types.h"
typedef int cudaError_t;
#define cudaSuccess 0
struct CUstream_st { int tag; };
typedef struct CUstream_st *cudaStream_t;
#endif
"""

RUNTIME = """#ifndef STUB_CUDA_RUNTIME_H
#define STUB_CUDA_RUNTIME_H
#include "cuda_runtime_api.h"
#include "vector_types.h"
#endif
"""

PROGRAM = """// built by tests/test_fix_0093_0.py
#include <cstdio>
#include <cstddef>
%sint main() {
%s    return 0;
}
"""

LAUNCH_STUB = """static int seen_calls = 0;
extern "C" cudaError_t cudaConfigureCall(dim3 gridDim, dim3 blockDim, size_t sharedMem, cudaStream_t stream) {
    (void) blockDim; (void) sharedMem; (void) stream;
    ++seen_calls;
    return (int) (gridDim.x * 100 + gridDim.y * 10 + gridDim.z);
}
extern "C" cudaError_t cudaSetupArgument(const void *arg, size_t size, size_t offset) {
    (void) arg; (void) size; (void) offset;
    return ++seen_calls;
}
extern "C" cudaError_t cudaLaunch(const void *func) {
    (void) func;
    return ++seen_calls;
}
"""

CONFIGURE_STUB = """static int seen_grid = 0, seen_block = 0, seen_shared = 0, seen_stream_null = 0, seen_stream_tag = -1;
extern "C" cudaError_t cudaConfigureCall(dim3 gridDim, dim3 blockDim, size_t sharedMem, cudaStream_t stream) {
    seen_grid = (int) (gridDim.x * 100 + gridDim.y * 10 + gridDim.z);
    seen_block = (int) (blockDim.x + blockDim.y + blockDim.z);
    seen_shared = (int) sharedMem;
    seen_stream_null = stream == nullptr ? 1 : 0;
    seen_stream_tag = stream == nullptr ? -1 : stream->tag;
    return seen_grid + seen_block + seen_shared;
}
"""

CASES = [
    dict(name='the force-included header declares its three launch functions',
         stub=LAUNCH_STUB, front=('header',),
         want='rc=421,2,3 calls=3 grid=4,2,1', body='''
    dim3 grid = {4, 2, 1};
    dim3 block = {32, 1, 1};
    int a = cudaConfigureCall(grid, block);
    int b = cudaSetupArgument(&grid, sizeof grid, 0);
    int c = cudaLaunch(&block);
    std::printf("rc=%d,%d,%d calls=%d grid=%u,%u,%u\\n", a, b, c, seen_calls, grid.x, grid.y, grid.z);
'''),
    dict(name='a force-included caller can call cudaConfigureCall with its default arguments',
         stub=CONFIGURE_STUB, front=('header',),
         want='rc=455 grid=421 block=34 shared=0 stream_null=1', body='''
    int rc = cudaConfigureCall(dim3{4, 2, 1}, dim3{32, 1, 1});
    std::printf("rc=%d grid=%d block=%d shared=%d stream_null=%d\\n",
                rc, seen_grid, seen_block, seen_shared, seen_stream_null);
'''),
    dict(name='explicit sharedMem and stream reach the driver call',
         stub=CONFIGURE_STUB, front=('header',),
         want='rc=266 grid=123 block=15 shared=128 stream_seen=7', body='''
    CUstream_st stream = {7};
    int rc = cudaConfigureCall(dim3{1, 2, 3}, dim3{4, 5, 6}, 128, &stream);
    std::printf("rc=%d grid=%d block=%d shared=%d stream_seen=%d\\n",
                rc, seen_grid, seen_block, seen_shared, seen_stream_tag);
'''),
    dict(name='the build order used today (cuda_runtime.h first) keeps working',
         stub=CONFIGURE_STUB, front=('runtime', 'header'),
         want='rc=455 grid=421 block=34 shared=0 stream_null=1', body='''
    int rc = cudaConfigureCall(dim3{4, 2, 1}, dim3{32, 1, 1});
    std::printf("rc=%d grid=%d block=%d shared=%d stream_null=%d\\n",
                rc, seen_grid, seen_block, seen_shared, seen_stream_null);
'''),
]


def compiler():
    for name in ('clang++', 'g++', 'c++'):
        path = shutil.which(name)
        if path:
            return path
    return None


def build_and_run(cxx, work, stub_dir, case):
    header = os.path.join(ROOT, HEADER)
    stem = ''.join(c if c.isalnum() else '_' for c in case['name']).strip('_')
    src = os.path.join(work, stem + '.cpp')
    exe = os.path.join(work, stem)
    with open(src, 'w', encoding='utf-8') as f:
        f.write(PROGRAM % (case['stub'], case['body']))
    cmd = [cxx, '-std=c++17', '-O0', '-I', stub_dir]
    for which in case['front']:
        cmd += ['-include', header if which == 'header' else os.path.join(stub_dir, RUNTIME_H)]
    cmd += ['-o', exe, src]
    built = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    if built.returncode != 0:
        errors = [l.strip() for l in (built.stdout + built.stderr).splitlines() if ' error: ' in l]
        return False, 'does not build with the header force-included (' + (errors[0] if errors else 'see the compile log').replace('"', "'") + ')'
    run = subprocess.run([exe], capture_output=True, text=True, timeout=120)
    got = run.stdout.strip()
    if run.returncode != 0:
        return False, 'the built program exited %d: %s' % (run.returncode, got[-200:])
    if got != case['want']:
        return False, 'the program printed %r, expected %r' % (got, case['want'])
    return True, ''


def main():
    if not os.path.isfile(os.path.join(ROOT, HEADER)):
        print('FAIL run this test from the copy root: ' + HEADER + ' is not in ' + ROOT)
        return 1
    cxx = compiler()
    if cxx is None:
        print('FAIL no C++ compiler was found (looked for clang++, g++, c++)')
        return 1
    work = tempfile.mkdtemp(prefix='fix-0093-')
    failures = 0
    try:
        stub_dir = os.path.join(work, 'stub-cuda')
        os.makedirs(stub_dir)
        for name, text in (('vector_types.h', VECTOR_TYPES), ('cuda_runtime_api.h', RUNTIME_API), (RUNTIME_H, RUNTIME)):
            with open(os.path.join(stub_dir, name), 'w', encoding='utf-8') as f:
                f.write(text)
        for case in CASES:
            ok, detail = build_and_run(cxx, work, stub_dir, case)
            print(('PASS ' if ok else 'FAIL ') + case['name'] + (': ' + detail if detail else ''))
            failures += 0 if ok else 1
    finally:
        shutil.rmtree(work, ignore_errors=True)
    print('%d case(s) failed' % failures if failures else 'all cases passed')
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
