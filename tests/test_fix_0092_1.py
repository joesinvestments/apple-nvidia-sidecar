#!/usr/bin/env python3
"""tests/test_fix_0092_1.py - cublasGemmEx must refuse a negative dimension instead of reporting success.

sidecar-shim/tgcublas.cpp:31 reads `if (m <= 0 || n <= 0) return CUBLAS_STATUS_SUCCESS;`, so a caller that passes a
negative m, n or k is told the GEMM succeeded where cuBLAS returns CUBLAS_STATUS_INVALID_VALUE (7) and the caller's
dimension bug is masked. These cases build the real shim against stub CUDA headers and a host stand-in for the shim's
WMMA kernels (tgcu_builtin_launch computes C = alpha * A^T B + beta * C in host memory, as tgcu_builtin.cu's gemm_tn
and gemm2_tn do, a zero K loop included), then call cublasGemmEx with no GPU, no model and no weights:
  - a negative m, a negative n and a negative k come back as CUBLAS_STATUS_INVALID_VALUE with no kernel launched;
  - m == 0 stays a successful no-op that launches nothing;
  - an ordinary bf16 GEMM still computes C = alpha A^T B + beta C, so valid work is unchanged;
  - k == 0 still computes C = beta C, which the kernels already do (the K loop runs zero times and the epilogue
    writes C), so that half of the finding does not reproduce here and the guard keeps it that way.
Prints one PASS or FAIL line per case and exits nonzero on any FAIL. Run: python3 tests/test_fix_0092_1.py
"""
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.getcwd()   # the tree under test: every test here runs with cwd at the copy root, never at the test file
SHIM = os.path.join('sidecar-shim', 'tgcublas.cpp')
CUBLAS_INCLUDE = os.path.join('sidecar-shim', 'include')

# The shim's cublas_v2.h includes the two CUDA headers below; only the stream type and the data type enum are used.
STUBS = {
    'cuda_runtime_api.h': '''// stub for tests/test_fix_0092_1.py: cublas_v2.h names the stream type and size_t, nothing else
#pragma once
#include <stddef.h>
typedef struct CUstream_st* cudaStream_t;
''',
    'library_types.h': '''// stub for tests/test_fix_0092_1.py: the data types cublasGemmEx takes
#pragma once
typedef enum cudaDataType_t {
  CUDA_R_32F = 0, CUDA_R_64F = 1, CUDA_R_16F = 2, CUDA_R_8I = 3, CUDA_R_8U = 4, CUDA_R_32I = 5, CUDA_R_32U = 6,
  CUDA_R_64U = 12, CUDA_R_64I = 13, CUDA_R_16BF = 14
} cudaDataType;
''',
}

LAUNCHER = '''// built by tests/test_fix_0092_1.py
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include "cublas_v2.h"

static int launches = 0;

static float bf16_to_float(uint16_t v) {
  const uint32_t u = (uint32_t) v << 16;
  float f;
  std::memcpy(&f, &u, sizeof f);
  return f;
}
static uint16_t float_to_bf16(float f) {
  uint32_t u;
  std::memcpy(&u, &f, sizeof u);
  return (uint16_t) (u >> 16);
}

// Host stand-in for tgcu_builtin.cu's gemm_tn / gemm2_tn: C (m x n, column-major, ldc) = alpha * A^T B + beta * C with
// A (m rows of k, lda) and B (n rows of k, ldb) bf16, C f32. A zero K loop adds nothing and the epilogue writes beta * C.
extern "C" int tgcu_builtin_launch(const char* name, unsigned gx, unsigned gy, unsigned bx, void** args, cudaStream_t) {
  (void) name; (void) gx; (void) gy; (void) bx;
  ++launches;
  const int m = *(int*) args[0], n = *(int*) args[1], k = *(int*) args[2];
  const float alpha = *(float*) args[3], beta = *(float*) args[8];
  const uint16_t* A = (const uint16_t*) *(void**) args[4]; const int lda = *(int*) args[5];
  const uint16_t* B = (const uint16_t*) *(void**) args[6]; const int ldb = *(int*) args[7];
  float* C = (float*) *(void**) args[9]; const int ldc = *(int*) args[10];
  for (int j = 0; j < n; j++)
    for (int i = 0; i < m; i++) {
      float acc = 0.0f;
      for (int l = 0; l < k; l++) acc += bf16_to_float(A[(size_t) i * lda + l]) * bf16_to_float(B[(size_t) j * ldb + l]);
      float v = alpha * acc;
      if (beta != 0.0f) v += beta * C[(size_t) j * ldc + i];
      C[(size_t) j * ldc + i] = v;
    }
  return 0;
}

int main() {
%s    return 0;
}
'''

FILL = '''
  alignas(16) uint16_t A[4 * 8];
  alignas(16) uint16_t B[3 * 8];
  float C[3 * 4];
  for (int i = 0; i < 4; i++)
    for (int l = 0; l < 8; l++) A[i * 8 + l] = float_to_bf16((float) ((i + 1) * (l % 3 + 1)));
  for (int j = 0; j < 3; j++)
    for (int l = 0; l < 8; l++) B[j * 8 + l] = float_to_bf16((float) ((j + 1) + (l % 2)));
'''

CASES = [
    dict(name='an ordinary bf16 GEMM still computes C = alpha A^T B + beta C',
         body=FILL + '''
  for (int e = 0; e < 12; e++) C[e] = 1.5f;
  const float alpha = 2.0f, beta = 0.5f;
  cublasHandle_t h;
  cublasCreate(&h);
  const cublasStatus_t st = cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, 4, 3, 8, &alpha, A, CUDA_R_16BF, 8, B, CUDA_R_16BF,
                                         8, &beta, C, CUDA_R_32F, 4, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
  cublasDestroy(h);
  double worst = 0.0;
  for (int j = 0; j < 3; j++)
    for (int i = 0; i < 4; i++) {
      double acc = 0.0;
      for (int l = 0; l < 8; l++) acc += (double) bf16_to_float(A[i * 8 + l]) * (double) bf16_to_float(B[j * 8 + l]);
      const double want = 2.0 * acc + 0.5 * 1.5;
      const double err = std::fabs((double) C[j * 4 + i] - want);
      if (err > worst) worst = err;
    }
  std::printf("status=%d diff=%.6f launches=%d\\n", (int) st, worst, launches);
''',
         want='status=0 diff=0.000000 launches=1'),
    dict(name='a negative m comes back as CUBLAS_STATUS_INVALID_VALUE with no kernel',
         body='''
  alignas(16) uint16_t A[8], B[8];
  float C[4];
  for (int e = 0; e < 8; e++) { A[e] = float_to_bf16(1.0f); B[e] = float_to_bf16(1.0f); }
  for (int e = 0; e < 4; e++) C[e] = 7.0f;
  const float alpha = 1.0f, beta = 0.0f;
  cublasHandle_t h;
  cublasCreate(&h);
  const cublasStatus_t st = cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, -2, 2, 8, &alpha, A, CUDA_R_16BF, 8, B, CUDA_R_16BF,
                                         8, &beta, C, CUDA_R_32F, 4, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
  cublasDestroy(h);
  std::printf("status=%d launches=%d c0=%.6f\\n", (int) st, launches, (double) C[0]);
''',
         want='status=7 launches=0 c0=7.000000'),
    dict(name='a negative n comes back as CUBLAS_STATUS_INVALID_VALUE with no kernel',
         body='''
  alignas(16) uint16_t A[8], B[8];
  float C[4];
  for (int e = 0; e < 8; e++) { A[e] = float_to_bf16(1.0f); B[e] = float_to_bf16(1.0f); }
  for (int e = 0; e < 4; e++) C[e] = 7.0f;
  const float alpha = 1.0f, beta = 0.0f;
  cublasHandle_t h;
  cublasCreate(&h);
  const cublasStatus_t st = cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, 2, -2, 8, &alpha, A, CUDA_R_16BF, 8, B, CUDA_R_16BF,
                                         8, &beta, C, CUDA_R_32F, 4, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
  cublasDestroy(h);
  std::printf("status=%d launches=%d c0=%.6f\\n", (int) st, launches, (double) C[0]);
''',
         want='status=7 launches=0 c0=7.000000'),
    dict(name='a negative k comes back as CUBLAS_STATUS_INVALID_VALUE with no kernel',
         body='''
  alignas(16) uint16_t A[8], B[8];
  float C[4];
  for (int e = 0; e < 8; e++) { A[e] = float_to_bf16(1.0f); B[e] = float_to_bf16(1.0f); }
  for (int e = 0; e < 4; e++) C[e] = 7.0f;
  const float alpha = 1.0f, beta = 0.0f;
  cublasHandle_t h;
  cublasCreate(&h);
  const cublasStatus_t st = cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, 2, 2, -4, &alpha, A, CUDA_R_16BF, 8, B, CUDA_R_16BF,
                                         8, &beta, C, CUDA_R_32F, 2, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
  cublasDestroy(h);
  std::printf("status=%d launches=%d c0=%.6f\\n", (int) st, launches, (double) C[0]);
''',
         want='status=7 launches=0 c0=7.000000'),
    dict(name='m == 0 stays a successful no-op that launches nothing',
         body='''
  alignas(16) uint16_t A[8], B[8];
  float C[4];
  for (int e = 0; e < 8; e++) { A[e] = float_to_bf16(1.0f); B[e] = float_to_bf16(1.0f); }
  for (int e = 0; e < 4; e++) C[e] = 7.0f;
  const float alpha = 1.0f, beta = 0.0f;
  cublasHandle_t h;
  cublasCreate(&h);
  const cublasStatus_t st = cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, 0, 2, 8, &alpha, A, CUDA_R_16BF, 8, B, CUDA_R_16BF,
                                         8, &beta, C, CUDA_R_32F, 4, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
  cublasDestroy(h);
  std::printf("status=%d launches=%d c0=%.6f\\n", (int) st, launches, (double) C[0]);
''',
         want='status=0 launches=0 c0=7.000000'),
    dict(name='k == 0 still computes C = beta C through the kernel',
         body=FILL + '''
  for (int e = 0; e < 12; e++) C[e] = 1.5f;
  const float alpha = 1.0f, beta = 0.5f;
  cublasHandle_t h;
  cublasCreate(&h);
  const cublasStatus_t st = cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, 4, 3, 0, &alpha, A, CUDA_R_16BF, 8, B, CUDA_R_16BF,
                                         8, &beta, C, CUDA_R_32F, 4, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
  cublasDestroy(h);
  std::printf("status=%d launches=%d c0=%.6f\\n", (int) st, launches, (double) C[0]);
''',
         want='status=0 launches=1 c0=0.750000'),
]


def compiler():
    for name in ('g++', 'clang++', 'c++'):
        path = shutil.which(name)
        if path:
            return path
    return None


def build_and_run(cxx, work, case, index):
    stem = 'case%d' % index
    driver = os.path.join(work, stem + '.cpp')
    exe = os.path.join(work, stem)
    with open(driver, 'w', encoding='utf-8') as f:
        f.write(LAUNCHER % case['body'])
    cmd = [cxx, '-std=c++17', '-O0', '-I', os.path.join(work, 'stub'), '-I', os.path.join(ROOT, CUBLAS_INCLUDE),
           '-o', exe, driver, os.path.join(ROOT, SHIM)]
    built = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    if built.returncode != 0:
        errors = [l.strip() for l in (built.stdout + built.stderr).splitlines() if ' error: ' in l]
        return False, 'does not build against the shim (' + (errors[0] if errors else 'see the compile log').replace('"', "'") + ')'
    run = subprocess.run([exe], capture_output=True, text=True, timeout=120)
    got = run.stdout.strip()
    if run.returncode != 0:
        return False, 'the built program exited %d: %s' % (run.returncode, got[-200:])
    if got != case['want']:
        return False, 'the program printed %r, expected %r' % (got, case['want'])
    return True, ''


def main():
    if not os.path.isfile(os.path.join(ROOT, SHIM)):
        print('FAIL run this test from the copy root: ' + SHIM + ' is not in ' + ROOT)
        return 1
    if not os.path.isfile(os.path.join(ROOT, CUBLAS_INCLUDE, 'cublas_v2.h')):
        print('FAIL run this test from the copy root: ' + os.path.join(CUBLAS_INCLUDE, 'cublas_v2.h') + ' is missing')
        return 1
    cxx = compiler()
    if cxx is None:
        print('FAIL no C++ compiler was found (looked for g++, clang++, c++)')
        return 1
    work = tempfile.mkdtemp(prefix='cublas-negative-dims-')
    failures = 0
    try:
        stub = os.path.join(work, 'stub')
        os.makedirs(stub)
        for name, text in STUBS.items():
            with open(os.path.join(stub, name), 'w', encoding='utf-8') as f:
                f.write(text)
        for index, case in enumerate(CASES):
            ok, detail = build_and_run(cxx, work, case, index)
            print(('PASS ' if ok else 'FAIL ') + case['name'] + (': ' + detail if detail else ''))
            failures += 0 if ok else 1
    finally:
        shutil.rmtree(work, ignore_errors=True)
    print('%d case(s) failed' % failures if failures else 'all cases passed')
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
