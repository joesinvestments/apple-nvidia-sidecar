// xport8_kernels.cuh - the device half of the 8-bit block-32 lane transport (STRATA_PF_XPORT8; format + semantics: include/strata/prefill/xport8.hpp).
// Included by kernels.cu (public wrappers xport8_quant / xport8_dequant in kernels.hpp) and by the standalone byte-exactness test (xport8_gpu_test.cu).
// quant: one warp per 32-value block; amax by warp shuffle; IEEE fp32 divisions (__fdiv_rn), round-to-nearest-even conversions - bit for bit the host code's arithmetic.
// dequant: one thread per value.
#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace strata::prefill::xp8 {

// x[n] (fp16 bits) -> q[n] int8 | scale[n / 32] fp16 bits; nb = n / 32 blocks
__global__ void quant_kernel(const uint16_t* __restrict__ x, int8_t* __restrict__ q, uint16_t* __restrict__ sc, int64_t nb) {
    const int64_t b = ((int64_t) blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (b >= nb) return;   // a whole warp leaves together (blockDim.x % 32 == 0)
    const int lane = threadIdx.x & 31;
    const int64_t i = b * 32 + lane;
    const float v = __half2float(__ushort_as_half(x[i]));
    float a = fabsf(v);
    a = (a == a) ? a : 0.0f;   // NaNs are ignored (the host's vmaxnm / a > amax)
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
    const uint16_t sh = __half_as_ushort(__float2half_rn(__fdiv_rn(a, 127.0f)));
    const float sf = __half2float(__ushort_as_half(sh));
    int qi = 0;
    if (sf != 0.0f) {
        qi = __float2int_rn(__fdiv_rn(v, sf));   // saturating, NaN -> 0
        qi = min(max(qi, -127), 127);
    }
    q[i] = (int8_t) qi;
    if (lane == 0) sc[b] = sh;
}

// q[n] | scale[n / 32] -> out[n] fp16 bits
__global__ void dequant_kernel(const int8_t* __restrict__ q, const uint16_t* __restrict__ sc, uint16_t* __restrict__ out, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = (float) q[i] * __half2float(__ushort_as_half(sc[i >> 5]));
    v = fminf(fmaxf(v, -65504.0f), 65504.0f);
    out[i] = __half_as_ushort(__float2half_rn(v));
}

// launchers (n % 32 == 0; `in` / `out` of the packed form: n bytes of q, then n / 16 bytes of scales, 2-byte aligned)
inline void launch_quant(const uint16_t* x16, uint8_t* out, int64_t n, cudaStream_t stream) {
    const int64_t nb = n / 32;
    const unsigned blocks = (unsigned) ((nb * 32 + 255) / 256);
    quant_kernel<<<blocks, 256, 0, stream>>>(x16, reinterpret_cast<int8_t*>(out), reinterpret_cast<uint16_t*>(out + n), nb);
}
inline void launch_dequant(const uint8_t* in, uint16_t* out16, int64_t n, cudaStream_t stream) {
    dequant_kernel<<<(unsigned) ((n + 255) / 256), 256, 0, stream>>>(reinterpret_cast<const int8_t*>(in), reinterpret_cast<const uint16_t*>(in + n), out16, n);
}

}  // namespace strata::prefill::xp8
