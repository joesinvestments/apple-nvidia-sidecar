#!/usr/bin/env python3
"""tests/test_fix_0109_0.py - conversation_kv_copy_device must be complete, and have failed here, when it returns.

src/core/conversation_snapshot.cpp:311 queues each part of the device-to-device K/V copy as

    const cudaError_t e = cudaMemcpyAsync(dst[i], src[i], sizes[i], cudaMemcpyDefault, nullptr);

on the legacy (null) stream and returns true straight away (line 317), so a successful return says nothing about the
destination: a caller that reads `to`, frees either state or reuses the buffers before the transfer has run sees stale
bytes or freed memory.  An execution failure is not the launch's, so it is not reported here at all: it surfaces on the
caller's next synchronize, attributed to whatever that call was.  The null stream is also the one the rest of this
file's copies avoid on purpose (conversation_copy.hpp: a plain legacy-stream copy can be refused while another thread
captures a graph on a blocking stream).

These cases call the real function, compiled against the real headers and a stub CUDA runtime that models a device:
copies on a stream are queued and only run at a synchronize, and a copy can be told to fail at execution time.  No
GPU, no model, no weights.

  - the destination holds the source's bytes for every part when the function returns true;
  - a copy that fails while it runs comes back as false from this call, with the reason in `error`;
  - no part is queued on the legacy stream (the private copy stream is the one this file's copies use);
  - mismatched layouts return false and write nothing (the same result before and after the fix);
  - an empty extent returns true and issues no copy (the same result before and after the fix);
  - the same harm on production's storage format: the selected config's --kv int8 codes and fp16 scales.

Prints one PASS or FAIL line per case and exits nonzero on any FAIL.  Run: python3 tests/test_fix_0109_0.py
"""
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.getcwd()   # the tree under test: this test runs with cwd at the copy root
SOURCE = os.path.join('src', 'core', 'conversation_snapshot.cpp')

# The CUDA runtime the real source and the program below compile against: declarations only, no driver.
CUDA_H = """
#pragma once
#include <cstddef>
#include <cstdint>
typedef int cudaError_t;
static const cudaError_t cudaSuccess = 0;
static const cudaError_t cudaErrorMemoryAllocation = 2;
struct CUstream_st; typedef CUstream_st* cudaStream_t;
struct CUevent_st; typedef CUevent_st* cudaEvent_t;
struct CUgraphExec_st; typedef CUgraphExec_st* cudaGraphExec_t;
struct CUgraph_st; typedef CUgraph_st* cudaGraph_t;
enum cudaMemcpyKind { cudaMemcpyHostToHost, cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost,
                      cudaMemcpyDeviceToDevice, cudaMemcpyDefault };
enum cudaMemoryType { cudaMemoryTypeUnregistered, cudaMemoryTypeHost, cudaMemoryTypeDevice,
                      cudaMemoryTypeManaged };
struct cudaPointerAttributes { cudaMemoryType type; int device; void* devicePointer; void* hostPointer; };
static const unsigned cudaStreamNonBlocking = 1;
static const unsigned cudaEventDisableTiming = 2;
static const unsigned cudaEventBlockingSync = 1;
cudaError_t cudaMalloc(void** p, size_t n);
cudaError_t cudaFree(void* p);
cudaError_t cudaMallocHost(void** p, size_t n);
cudaError_t cudaFreeHost(void* p);
cudaError_t cudaMemcpy(void* d, const void* s, size_t n, cudaMemcpyKind k);
cudaError_t cudaMemcpyAsync(void* d, const void* s, size_t n, cudaMemcpyKind k, cudaStream_t st);
cudaError_t cudaStreamCreateWithFlags(cudaStream_t* s, unsigned f);
cudaError_t cudaStreamDestroy(cudaStream_t s);
cudaError_t cudaStreamSynchronize(cudaStream_t s);
cudaError_t cudaStreamWaitEvent(cudaStream_t s, cudaEvent_t e, unsigned f);
cudaError_t cudaEventCreateWithFlags(cudaEvent_t* e, unsigned f);
cudaError_t cudaEventRecord(cudaEvent_t e, cudaStream_t s);
cudaError_t cudaEventSynchronize(cudaEvent_t e);
cudaError_t cudaEventDestroy(cudaEvent_t e);
cudaError_t cudaGetDevice(int* d);
cudaError_t cudaSetDevice(int d);
cudaError_t cudaDeviceSynchronize();
cudaError_t cudaGetLastError();
const char* cudaGetErrorString(cudaError_t e);
cudaError_t cudaPointerGetAttributes(cudaPointerAttributes* a, const void* p);
cudaError_t cudaMemGetInfo(size_t* fr, size_t* tot);
"""

PROGRAM = r"""
// built by tests/test_fix_0109_0.py: the real conversation_kv_copy_device on a stub device.
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <deque>
#include <string>

#include "cuda_runtime.h"
#include "strata/core/conversation_snapshot.hpp"

namespace fake {
// one queued copy: it has not run yet, so nothing reads `dst` before a synchronize does
struct Copy { void* dst; const void* src; size_t n; cudaStream_t st; bool fails; };
std::deque<Copy>& pending() { static std::deque<Copy> q; return q; }
long& issued() { static long n = 0; return n; }   // copy requests made (queued or immediate)
long& legacy() { static long n = 0; return n; }   // copy requests made on the legacy stream
long fail_at = -1;                                // 1-based copy request that fails while it runs (-1: none)
void reset() { pending().clear(); issued() = 0; legacy() = 0; fail_at = -1; }
bool fails_now() { return fail_at > 0 && issued() == fail_at; }
cudaError_t run(const Copy& c) {
    if (c.fails) return cudaErrorMemoryAllocation;
    std::memcpy(c.dst, c.src, c.n);
    return cudaSuccess;
}
cudaError_t drain() {
    cudaError_t e = cudaSuccess;
    while (!pending().empty()) {
        const Copy c = pending().front();
        pending().pop_front();
        if (run(c) != cudaSuccess) e = cudaErrorMemoryAllocation;
    }
    return e;
}
}  // namespace fake

cudaError_t cudaMalloc(void** p, size_t n) {
    *p = std::malloc(n ? n : 1);
    if (!*p) return cudaErrorMemoryAllocation;
    std::memset(*p, 0, n);
    return cudaSuccess;
}
cudaError_t cudaFree(void* p) { std::free(p); return cudaSuccess; }
cudaError_t cudaMallocHost(void** p, size_t n) { return cudaMalloc(p, n); }
cudaError_t cudaFreeHost(void* p) { std::free(p); return cudaSuccess; }
// a synchronous copy runs before it returns, so an execution failure is its own to report
cudaError_t cudaMemcpy(void* d, const void* s, size_t n, cudaMemcpyKind) {
    const bool fails = fake::fails_now();
    fake::issued() = fake::issued() + 1;
    if (fails) return cudaErrorMemoryAllocation;
    std::memcpy(d, s, n);
    return cudaSuccess;
}
cudaError_t cudaMemcpyAsync(void* d, const void* s, size_t n, cudaMemcpyKind, cudaStream_t st) {
    const bool fails = fake::fails_now();
    fake::issued() = fake::issued() + 1;
    if (st == nullptr) fake::legacy() = fake::legacy() + 1;
    fake::pending().push_back(fake::Copy{d, s, n, st, fails});
    return cudaSuccess;
}
cudaError_t cudaStreamCreateWithFlags(cudaStream_t* s, unsigned) { *s = (cudaStream_t) 1; return cudaSuccess; }
cudaError_t cudaStreamDestroy(cudaStream_t) { return cudaSuccess; }
cudaError_t cudaStreamSynchronize(cudaStream_t) { return fake::drain(); }
cudaError_t cudaStreamWaitEvent(cudaStream_t, cudaEvent_t, unsigned) { return cudaSuccess; }
cudaError_t cudaEventCreateWithFlags(cudaEvent_t* e, unsigned) { *e = (cudaEvent_t) 1; return cudaSuccess; }
cudaError_t cudaEventRecord(cudaEvent_t, cudaStream_t) { return cudaSuccess; }
cudaError_t cudaEventSynchronize(cudaEvent_t) { return cudaSuccess; }
cudaError_t cudaEventDestroy(cudaEvent_t) { return cudaSuccess; }
cudaError_t cudaGetDevice(int* d) { *d = 0; return cudaSuccess; }
cudaError_t cudaSetDevice(int) { return cudaSuccess; }
cudaError_t cudaDeviceSynchronize() { return fake::drain(); }
cudaError_t cudaGetLastError() { return cudaSuccess; }
const char* cudaGetErrorString(cudaError_t e) {
    if (e == cudaSuccess) return "no error";
    if (e == cudaErrorMemoryAllocation) return "out of memory";
    return "cuda error";
}
cudaError_t cudaPointerGetAttributes(cudaPointerAttributes* a, const void* p) {
    a->type = cudaMemoryTypeDevice; a->device = 0; a->devicePointer = (void*) p; a->hostPointer = nullptr;
    return cudaSuccess;
}
cudaError_t cudaMemGetInfo(size_t* fr, size_t* tot) {
    if (fr) *fr = (size_t) 1 << 30;
    if (tot) *tot = (size_t) 2 << 30;
    return cudaSuccess;
}

namespace {

const int64_t kTokens = 16;   // one page granule times four: k, v and four indexer rows

void fill(void* p, size_t n, unsigned char v) { std::memset(p, v, n); }
bool all_are(const void* p, size_t n, unsigned char v) {
    const unsigned char* b = (const unsigned char*) p;
    for (size_t i = 0; i < n; ++i)
        if (b[i] != v) return false;
    return true;
}

// two mode-0 states of one shape, `from` holding the patterns below and `to` zeroed.  int8: the selected serving
// config's --kv int8 (codes plus one fp16 scale per 64 values), else fp16 pools.
struct Pair {
    strata::core::QsaState from, to;
    size_t kbytes = 0, sbytes = 0, pbytes = 0;
};
void setUp(Pair& p, int64_t upto, bool int8 = false) {
    const strata::core::ModelGeometry g;
    const int64_t cells = (upto + 3) / 4 * 4;
    p.kbytes = (size_t) cells * (size_t) g.n_head_kv * (size_t) g.head_dim * (int8 ? 1 : 2);
    p.sbytes = int8 ? (size_t) cells * (size_t) g.n_head_kv * (size_t) (g.head_dim / 64) * 2 : 0;
    p.pbytes = (size_t) (upto > 0 ? upto / 4 + 1 : 0) * (size_t) g.idx_key_dim * sizeof(float);
    for (strata::core::QsaState* s : {&p.from, &p.to}) {
        s->kv_mode = 0;
        s->max_cells = std::max<int64_t>(64, cells);
        s->n_pages = cells > 0 ? cells / 4 : 1;   // a state's slot count is never zero
        s->n_slots = s->n_pages;
        s->idx_pooled_rows = upto > 0 ? upto / 4 + 1 : 0;
        s->kv_int8 = int8;
        if (int8) {
            (void) cudaMalloc((void**) &s->k_q, p.kbytes);
            (void) cudaMalloc((void**) &s->v_q, p.kbytes);
            (void) cudaMalloc((void**) &s->k_scale, p.sbytes);
            (void) cudaMalloc((void**) &s->v_scale, p.sbytes);
        } else {
            (void) cudaMalloc((void**) &s->k_pool, p.kbytes);
            (void) cudaMalloc((void**) &s->v_pool, p.kbytes);
        }
        (void) cudaMalloc((void**) &s->idx_pooled, p.pbytes);
    }
    if (int8) {
        fill(p.from.k_q, p.kbytes, 0x11);
        fill(p.from.v_q, p.kbytes, 0x22);
        fill(p.from.k_scale, p.sbytes, 0x33);
        fill(p.from.v_scale, p.sbytes, 0x44);
        fill(p.from.idx_pooled, p.pbytes, 0x55);
    } else {
        fill(p.from.k_pool, p.kbytes, 0x11);
        fill(p.from.v_pool, p.kbytes, 0x22);
        fill(p.from.idx_pooled, p.pbytes, 0x33);
    }
}

void caseCopyComplete() {
    fake::reset();
    Pair p;
    setUp(p, kTokens);
    const strata::core::ModelGeometry g;
    std::string error;
    const bool ok = strata::core::conversation_kv_copy_device(p.from, p.to, g, kTokens, true, error);
    std::printf("CASE copy_complete ok=%d k=%d v=%d pooled=%d queued=%d\n", (int) ok,
                (int) all_are(p.to.k_pool, p.kbytes, 0x11), (int) all_are(p.to.v_pool, p.kbytes, 0x22),
                (int) all_are(p.to.idx_pooled, p.pbytes, 0x33), (int) fake::pending().size());
}

void caseExecutionError() {
    fake::reset();
    fake::fail_at = 2;   // the second part fails while it runs, not at its launch
    Pair p;
    setUp(p, kTokens);
    const strata::core::ModelGeometry g;
    std::string error;
    const bool ok = strata::core::conversation_kv_copy_device(p.from, p.to, g, kTokens, true, error);
    std::printf("CASE execution_error ok=%d reason=%d\n", (int) ok,
                (int) (error.find("out of memory") != std::string::npos));
}

void casePrivateStream() {
    fake::reset();
    Pair p;
    setUp(p, kTokens);
    const strata::core::ModelGeometry g;
    std::string error;
    (void) strata::core::conversation_kv_copy_device(p.from, p.to, g, kTokens, true, error);
    std::printf("CASE private_stream copied=%d legacy=%d\n", (int) fake::issued(), (int) fake::legacy());
}

void caseLayoutMismatch() {
    fake::reset();
    Pair p;
    setUp(p, kTokens);
    p.to.kv_int8 = true;   // a different storage format for the same tokens
    const strata::core::ModelGeometry g;
    std::string error;
    const bool ok = strata::core::conversation_kv_copy_device(p.from, p.to, g, kTokens, true, error);
    std::printf("CASE layout_mismatch ok=%d wrote=%d reason=%d\n", (int) ok,
                (int) !all_are(p.to.k_pool, p.kbytes, 0x00),
                (int) (error.find("layouts differ") != std::string::npos));
}

void caseEmptyExtent() {
    fake::reset();
    Pair p;
    setUp(p, 0);
    const strata::core::ModelGeometry g;
    std::string error;
    const bool ok = strata::core::conversation_kv_copy_device(p.from, p.to, g, 0, true, error);
    std::printf("CASE empty_extent ok=%d copied=%d quiet=%d\n", (int) ok, (int) fake::issued(),
                (int) error.empty());
}

// the selected config's storage: --kv int8, 2048 tokens of a conversation (codes and scales both move)
void caseProductionInt8() {
    fake::reset();
    Pair p;
    setUp(p, 2048, true);
    const strata::core::ModelGeometry g;
    std::string error;
    const bool ok = strata::core::conversation_kv_copy_device(p.from, p.to, g, 2048, true, error);
    std::printf("CASE production_int8 ok=%d k=%d v=%d ks=%d vs=%d pooled=%d queued=%d\n", (int) ok,
                (int) all_are(p.to.k_q, p.kbytes, 0x11), (int) all_are(p.to.v_q, p.kbytes, 0x22),
                (int) all_are(p.to.k_scale, p.sbytes, 0x33), (int) all_are(p.to.v_scale, p.sbytes, 0x44),
                (int) all_are(p.to.idx_pooled, p.pbytes, 0x55), (int) fake::pending().size());
}

}  // namespace

int main() {
    caseCopyComplete();
    caseExecutionError();
    casePrivateStream();
    caseLayoutMismatch();
    caseEmptyExtent();
    caseProductionInt8();
    return 0;
}
"""

# the printed fields each case must show once the copy is complete on return
CASES = [
    dict(name='the destination holds every part when the function returns true',
         key='copy_complete', want='ok=1 k=1 v=1 pooled=1 queued=0'),
    dict(name='a copy that fails while it runs is this call\'s error',
         key='execution_error', want='ok=0 reason=1'),
    dict(name='no part is queued on the legacy stream',
         key='private_stream', want='copied=3 legacy=0'),
    dict(name='mismatched layouts copy nothing and say so',
         key='layout_mismatch', want='ok=0 wrote=0 reason=1'),
    dict(name='an empty extent copies nothing and succeeds',
         key='empty_extent', want='ok=1 copied=0 quiet=1'),
    dict(name='the selected config\'s int8 K/V and its scales both land',
         key='production_int8', want='ok=1 k=1 v=1 ks=1 vs=1 pooled=1 queued=0'),
]


def compiler():
    for name in ('g++', 'clang++', 'c++'):
        found = shutil.which(name)
        if found:
            return found
    return None


def build(cxx, work):
    include = os.path.join(work, 'include')
    os.makedirs(include)
    with open(os.path.join(include, 'cuda_runtime.h'), 'w', encoding='utf-8') as f:
        f.write(CUDA_H)
    program = os.path.join(work, 'probe.cpp')
    with open(program, 'w', encoding='utf-8') as f:
        f.write(PROGRAM)
    exe = os.path.join(work, 'probe')
    cmd = [cxx, '-std=c++20', '-O0', '-w', '-I', include, '-I', os.path.join(ROOT, 'include')]
    if sys.platform == 'darwin':
        cmd += ['-Wl,-dead_strip']   # the parts of the source these cases do not call keep their unresolved names
    else:
        cmd += ['-ffunction-sections', '-Wl,--gc-sections']
    cmd += ['-o', exe, program, os.path.join(ROOT, SOURCE)]
    built = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if built.returncode != 0:
        errors = [l.strip() for l in (built.stdout + built.stderr).splitlines() if ' error: ' in l]
        return None, 'does not build (' + (errors[0] if errors else 'see the compile log').replace('"', "'") + ')'
    return exe, ''


def main():
    if not os.path.isfile(os.path.join(ROOT, SOURCE)):
        print('FAIL run this test from the copy root: ' + SOURCE + ' is not in ' + ROOT)
        return 1
    cxx = compiler()
    if cxx is None:
        print('FAIL no C++ compiler was found (looked for g++, clang++, c++)')
        return 1
    work = tempfile.mkdtemp(prefix='fix-0109-')
    failures = 0
    try:
        exe, detail = build(cxx, work)
        if exe is None:
            print('FAIL the real ' + SOURCE + ' ' + detail)
            return 1
        run = subprocess.run([exe], capture_output=True, text=True, timeout=300)
        if run.returncode != 0:
            print('FAIL the built probe exited %d: %s' % (run.returncode, run.stdout.strip()[-300:]))
            return 1
        got = {}
        for line in run.stdout.splitlines():
            parts = line.split()
            if parts[:1] == ['CASE'] and len(parts) > 1:
                got[parts[1]] = ' '.join(parts[2:])
        for case in CASES:
            seen = got.get(case['key'])
            ok = seen == case['want']
            print(('PASS ' if ok else 'FAIL ') + case['name'] +
                  ('' if ok else ': the built code printed %r, expected %r' % (seen, case['want'])))
            failures += 0 if ok else 1
    finally:
        shutil.rmtree(work, ignore_errors=True)
    print('%d case(s) failed' % failures if failures else 'all cases passed')
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
