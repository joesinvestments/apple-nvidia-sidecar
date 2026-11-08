#!/usr/bin/env python3
"""tests/test_fix_0300_0.py - a failed host or device copy in the xport8 launcher must not read as success.

src/prefill/xport8_gpu_launch.cu:15,16,22,23,24 throw away the cudaError_t of every cudaMemcpy, and the function
still returns 0 at line 26. A failed host to device copy leaves the kernels reading stale device bytes; a failed
device to host copy leaves the caller's output buffers holding whatever they held before, while
xp8_gpu_roundtrip reports success. There is no GPU here, so these cases copy the real launcher into a temp tree,
put a stub xport8_kernels.cuh (plain host kernels that write a fixed pattern) and a stub cuda_runtime.h (fake device
arena with one injectable failed copy) beside it, build with the system C++ compiler and run the built program:
  - an ordinary run returns 0 and fills all three outputs the same way whatever the buffers held before;
  - a failure of each of the five copies (2 host to device, 3 device to host) comes back nonzero;
  - a failed device to host copy is not reported as a filled output;
  - a failed host to device copy of x does not launch the kernels on the stale device bytes.
Prints one PASS or FAIL line per case and exits nonzero on any FAIL. Run: python3 tests/test_fix_0300_0.py
"""
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.getcwd()   # the tree under test: every test here runs with cwd at the copy root, never at the test file
LAUNCHER = os.path.join('src', 'prefill', 'xport8_gpu_launch.cu')

# The fake device: one arena, five copies in the launcher's order, and the copy at g_fail_at returns an error
# without moving a byte, the way a failed cudaMemcpy does. The stub kernels count their launches.
STUB_CUDA_RUNTIME = r'''
#pragma once
#include <cstddef>
enum cudaError { cudaSuccess = 0, cudaErrorInvalidValue = 1, cudaErrorMemoryAllocation = 2 };
typedef cudaError cudaError_t;
enum cudaMemcpyKind { cudaMemcpyHostToDevice = 1, cudaMemcpyDeviceToHost = 2, cudaMemcpyDeviceToDevice = 3 };
typedef void* cudaStream_t;
cudaError_t cudaMalloc(void** p, size_t n);
template <class T> inline cudaError_t cudaMalloc(T** p, size_t n) { return cudaMalloc((void**) p, n); }
cudaError_t cudaFree(void* p);
cudaError_t cudaMemset(void* p, int v, size_t n);
cudaError_t cudaMemcpy(void* dst, const void* src, size_t n, cudaMemcpyKind kind);
cudaError_t cudaDeviceSynchronize();
const char* cudaGetErrorString(cudaError_t e);
'''

STUB_KERNELS = r'''
// stub of src/prefill/xport8_kernels.cuh built by tests/test_fix_0300_0.py: no GPU here, so the device kernels
// are plain host functions that write a fixed pattern. The launcher under test is the real file.
#pragma once
#include <cstdint>
#include "cuda_runtime.h"

extern int xp8_stub_launches;

namespace strata::prefill::xp8 {
inline void launch_quant(const uint16_t* x16, uint8_t* out, int64_t n, cudaStream_t stream) {
    (void) stream;
    ++xp8_stub_launches;
    for (int64_t i = 0; i < n; ++i) out[i] = (uint8_t) (x16[i] & 0xff);
    for (int64_t i = 0; i < n / 16; ++i) out[n + i] = (uint8_t) (0x80 | i);
}
inline void launch_dequant(const uint8_t* in, uint16_t* out16, int64_t n, cudaStream_t stream) {
    (void) stream;
    ++xp8_stub_launches;
    for (int64_t i = 0; i < n; ++i) out16[i] = (uint16_t) (0x4000 | in[i]);
}
}  // namespace strata::prefill::xp8
'''

HARNESS = r'''
// built by tests/test_fix_0300_0.py: the fake device and the caller of xp8_gpu_roundtrip
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include "cuda_runtime.h"

extern "C" int xp8_gpu_roundtrip(const uint16_t* x, int64_t n, uint8_t* packed_out, uint16_t* dq_of_gpu_packed,
                                 const uint8_t* packed_in, uint16_t* dq_of_given_packed);

int xp8_stub_launches = 0;

static int g_copies = 0;
static int g_fail_at = -1;
static unsigned char g_arena[1 << 16];
static size_t g_used = 0;

cudaError_t cudaMalloc(void** p, size_t n) {
    if (g_used + n > sizeof g_arena) return cudaErrorMemoryAllocation;
    *p = g_arena + g_used;
    g_used += (n + 15) & ~(size_t) 15;
    return cudaSuccess;
}
cudaError_t cudaFree(void* p) { (void) p; return cudaSuccess; }
cudaError_t cudaMemset(void* p, int v, size_t n) { std::memset(p, v, n); return cudaSuccess; }
cudaError_t cudaMemcpy(void* dst, const void* src, size_t n, cudaMemcpyKind kind) {
    (void) kind;
    const int i = g_copies++;
    if (i == g_fail_at) return cudaErrorInvalidValue;   // a failed copy moves nothing
    std::memmove(dst, src, n);
    return cudaSuccess;
}
cudaError_t cudaDeviceSynchronize() { return cudaSuccess; }
const char* cudaGetErrorString(cudaError_t e) { return e == cudaSuccess ? "no error" : "injected copy failure"; }

static void hex(const char* key, const void* p, size_t n) {
    const unsigned char* b = (const unsigned char*) p;
    std::printf("%s=", key);
    for (size_t i = 0; i < n; ++i) std::printf("%02x", b[i]);
    std::printf("\n");
}

int main(int argc, char** argv) {
    const int64_t n = 64;
    const size_t pb = (size_t) n + (size_t) n / 16;
    g_fail_at = argc > 1 ? std::atoi(argv[1]) : -1;
    const unsigned sentinel = argc > 2 ? (unsigned) std::strtoul(argv[2], nullptr, 0) : 0xa5u;
    std::vector<uint16_t> x((size_t) n), dq_gpu((size_t) n, (uint16_t) (sentinel * 0x0101u)),
        dq_given((size_t) n, (uint16_t) (sentinel * 0x0101u));
    std::vector<uint8_t> packed_in(pb), packed_out(pb, (uint8_t) sentinel);
    for (int64_t i = 0; i < n; ++i) {
        x[(size_t) i] = (uint16_t) (0x3c00 + i);
        packed_in[(size_t) i] = (uint8_t) (0x10 + i);
    }
    for (size_t i = (size_t) n; i < pb; ++i) packed_in[i] = (uint8_t) (0x40 + i);
    const int rc = xp8_gpu_roundtrip(x.data(), n, packed_out.data(), dq_gpu.data(), packed_in.data(), dq_given.data());
    std::printf("rc=%d\n", rc);
    std::printf("copies=%d\n", g_copies);
    std::printf("launches=%d\n", xp8_stub_launches);
    hex("packed", packed_out.data(), pb);
    hex("dq_gpu", dq_gpu.data(), (size_t) n * 2);
    hex("dq_given", dq_given.data(), (size_t) n * 2);
    return 0;
}
'''

# The five cudaMemcpy calls of src/prefill/xport8_gpu_launch.cu, in the order the function makes them.
FAILED_COPIES = [
    (0, 'the host to device copy of x'),
    (1, 'the host to device copy of the packed input'),
    (2, 'the device to host copy of the packed output'),
    (3, 'the device to host copy of the gpu packed dequant'),
    (4, 'the device to host copy of the given packed dequant'),
]
OUTPUTS = ('packed', 'dq_gpu', 'dq_given')


def compiler():
    for name in ('g++', 'clang++', 'c++'):
        path = shutil.which(name)
        if path:
            return path
    return None


def build(cxx, work):
    stub = os.path.join(work, 'stub')
    os.makedirs(stub)
    with open(os.path.join(stub, 'cuda_runtime.h'), 'w', encoding='utf-8') as f:
        f.write(STUB_CUDA_RUNTIME)
    src = os.path.join(work, 'src')
    os.makedirs(src)
    shutil.copyfile(os.path.join(ROOT, LAUNCHER), os.path.join(src, 'xport8_gpu_launch.cu'))
    with open(os.path.join(src, 'xport8_kernels.cuh'), 'w', encoding='utf-8') as f:
        f.write(STUB_KERNELS)
    harness = os.path.join(work, 'harness.cpp')
    with open(harness, 'w', encoding='utf-8') as f:
        f.write(HARNESS)
    exe = os.path.join(work, 'harness')
    cmd = [cxx, '-std=c++17', '-O0', '-I', stub, '-o', exe, '-x', 'c++', harness,
           os.path.join(src, 'xport8_gpu_launch.cu')]
    built = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if built.returncode != 0:
        errors = [l.strip() for l in (built.stdout + built.stderr).splitlines() if ' error: ' in l]
        return None, 'the launcher does not build against the stub device (' + \
            (errors[0] if errors else 'see the compile log').replace('"', "'") + ')'
    return exe, ''


def run(exe, fail_at, sentinel):
    r = subprocess.run([exe, str(fail_at), str(sentinel)], capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        return None, 'the built program exited %d: %s' % (r.returncode, (r.stdout + r.stderr).strip()[-200:])
    fields = {}
    for line in r.stdout.splitlines():
        if '=' in line:
            key, value = line.split('=', 1)
            fields[key.strip()] = value.strip()
    for key in ('rc', 'copies', 'launches') + OUTPUTS:
        if key not in fields:
            return None, 'the built program printed no %s line' % key
    return fields, ''


def case_ordinary(exe):
    """An ordinary run still returns 0 and fills every output the same way, whatever the buffers held."""
    a, detail = run(exe, -1, 0xa5)
    if a is None:
        return False, detail
    b, detail = run(exe, -1, 0x5a)
    if b is None:
        return False, detail
    if a['rc'] != '0' or b['rc'] != '0':
        return False, 'an ordinary run returned %s and %s, expected 0' % (a['rc'], b['rc'])
    if a['copies'] != '5':
        return False, 'an ordinary run made %s copies, expected the five of the launcher' % a['copies']
    differing = [k for k in OUTPUTS if a[k] != b[k]]
    if differing:
        return False, 'an ordinary run left the start of the buffers showing through: ' + ', '.join(differing)
    return True, ''


def case_failed_copy(index, what):
    def check(exe):
        a, detail = run(exe, index, 0xa5)
        if a is None:
            return False, detail
        if a['rc'] == '0':
            return False, 'a failed %s was reported as success (rc=0)' % what
        return True, ''
    return check


def case_failed_copy_back_not_claimed(exe):
    """A failed device to host copy of the gpu dequant: its buffer keeps the sentinel, so success is a lie."""
    a, detail = run(exe, 3, 0xa5)
    if a is None:
        return False, detail
    if a['rc'] == '0' and a['dq_gpu'] == 'a5a5' * 64:
        return False, 'the run reported success with dq_of_gpu_packed still holding the caller\'s old bytes'
    return True, ''


def case_no_kernels_on_stale_bytes(exe):
    """A failed host to device copy of x must not start the kernels that then read stale device bytes."""
    a, detail = run(exe, 0, 0xa5)
    if a is None:
        return False, detail
    if a['launches'] != '0':
        return False, 'a failed copy of x still launched %s kernel(s) on stale device bytes' % a['launches']
    return True, ''


def main():
    if not os.path.isfile(os.path.join(ROOT, LAUNCHER)):
        print('FAIL run this test from the copy root: ' + LAUNCHER + ' is not in ' + ROOT)
        return 1
    cxx = compiler()
    if cxx is None:
        print('FAIL no C++ compiler was found (looked for g++, clang++, c++)')
        return 1
    work = tempfile.mkdtemp(prefix='xport8-copies-')
    failures = 0
    try:
        exe, detail = build(cxx, work)
        if exe is None:
            print('FAIL ' + detail)
            return 1
        cases = [('an ordinary run returns 0 and fills every output', case_ordinary)]
        cases += [('%02d %s fails the run' % (i, what), case_failed_copy(i, what)) for i, what in FAILED_COPIES]
        cases += [
            ('a failed copy back is not reported as a filled output', case_failed_copy_back_not_claimed),
            ('a failed copy of x does not launch the kernels on stale bytes', case_no_kernels_on_stale_bytes),
        ]
        for name, check in cases:
            ok, detail = check(exe)
            print(('PASS ' if ok else 'FAIL ') + name + (': ' + detail if detail else ''))
            failures += 0 if ok else 1
    finally:
        shutil.rmtree(work, ignore_errors=True)
    print('%d case(s) failed' % failures if failures else 'all cases passed')
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
