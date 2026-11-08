// xport8_gpu_launch.cu - standalone launcher of the device kernels (xport8_kernels.cuh), for xport8_gpu_test.cpp (a different translation unit: nvcc cannot parse <arm_neon.h>).
//   nvcc -O3 -std=c++17 -arch=native -c xport8_gpu_launch.cu -o xport8_gpu_launch.o
#include "xport8_kernels.cuh"

#include <cstdio>

extern "C" int xp8_gpu_roundtrip(const uint16_t* x, int64_t n, uint8_t* packed_out, uint16_t* dq_of_gpu_packed, const uint8_t* packed_in, uint16_t* dq_of_given_packed) {
    const size_t pb = (size_t) n + (size_t) n / 16;
    uint16_t *dx = nullptr, *dy = nullptr, *dz = nullptr;
    uint8_t *dp = nullptr, *dq = nullptr;
    auto ck = [](cudaError_t e, const char* w) { if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", w, cudaGetErrorString(e)); return 1; } return 0; };
    if (ck(cudaMalloc(&dx, (size_t) n * 2), "malloc x") || ck(cudaMalloc(&dy, (size_t) n * 2), "malloc y") || ck(cudaMalloc(&dz, (size_t) n * 2), "malloc z") ||
        ck(cudaMalloc(&dp, pb), "malloc p") || ck(cudaMalloc(&dq, pb), "malloc q"))
        return 1;
    if (ck(cudaMemcpy(dx, x, (size_t) n * 2, cudaMemcpyHostToDevice), "copy x") || ck(cudaMemcpy(dq, packed_in, pb, cudaMemcpyHostToDevice), "copy packed in"))
        return 1;
    cudaMemset(dp, 0xcd, pb);
    strata::prefill::xp8::launch_quant(dx, dp, n, 0);                 // x -> packed (the card's quantiser)
    strata::prefill::xp8::launch_dequant(dp, dy, n, 0);               // that packed -> fp16
    strata::prefill::xp8::launch_dequant(dq, dz, n, 0);               // a host-made packed (the NEON quantiser's) -> fp16: the y path
    if (ck(cudaDeviceSynchronize(), "sync")) return 1;
    if (ck(cudaMemcpy(packed_out, dp, pb, cudaMemcpyDeviceToHost), "copy packed out") ||
        ck(cudaMemcpy(dq_of_gpu_packed, dy, (size_t) n * 2, cudaMemcpyDeviceToHost), "copy gpu packed dequant") ||
        ck(cudaMemcpy(dq_of_given_packed, dz, (size_t) n * 2, cudaMemcpyDeviceToHost), "copy given packed dequant"))
        return 1;
    cudaFree(dx); cudaFree(dy); cudaFree(dz); cudaFree(dp); cudaFree(dq);
    return 0;
}
