// xport8.hpp - 8-bit block-32 transport of the Mac lane's x (card -> Mac) and y (Mac -> card) rows (STRATA_PF_XPORT8, Prefill600 follow-up "Xport8").
//
// Why: in steady state the lane's USB host thread is the binding cost (per chunk-layer: x download 31.5 ms + y push ~36 ms + ~21 ms other API calls = 88 ms);
// the link moves fp16 rows (T x 2560 x 2 B = 10.2 MB each way).  An int8 block-32 row (one signed byte per value + one fp16 scale per 32 values) is 17/16 B per
// value = 53% of fp16's bytes.
//
// Wire format of n values (n a multiple of 32; the rows of a T x 2560 matrix are consecutive, a block never crosses a row):
//     [ q: n x int8 ][ scale: n/32 x fp16 (little endian) ]          = n + n/16 bytes
// Quantisation of a block (the CUDA kernels in prefill/kernels.cu and the NEON code below do exactly this, bit for bit):
//     amax  = max |x_i|  over the 32 fp16 values read as fp32 (NaNs ignored, like fmaxf / vmaxnm)
//     sh    = fp16(amax / 127)  (round to nearest even; IEEE fp32 division)           sf = fp32(sh)
//     q_i   = sf == 0 ? 0 : clamp(rne(x_i / sf), -127, 127)   (IEEE fp32 division; a NaN quotient gives 0)
// Dequantisation: out_i = fp16(clamp(fp32(q_i) * sf, -65504, 65504))  (the product is exact in fp32; the clamp keeps 127 * sf, which can be a hair above the largest
// fp16, from rounding to infinity).
//
// This header: the scalar reference (ref_*), the NEON versions (quant / dequant; scalar elsewhere), and a dispatch_apply split over blocks (the result does not depend
// on the split).  Used by prefill.cpp (host side), xport8_test.cpp (byte-exact checks) and mirrored by the device kernels.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif
#if defined(__APPLE__)
#include <dispatch/dispatch.h>
#endif

namespace strata::prefill::xport8 {

inline constexpr int64_t kBlock = 32;
inline constexpr float kQMax = 127.0f;
inline constexpr float kHalfMax = 65504.0f;

// bytes of the packed form of n values (n % 32 == 0): n int8 + n / 32 fp16 scales
inline constexpr size_t packed_bytes(int64_t n) { return (size_t) n + (size_t) (n / kBlock) * 2; }
inline constexpr bool packable(int64_t n) { return n > 0 && n % kBlock == 0; }

inline float h2f(uint16_t h) { _Float16 v; std::memcpy(&v, &h, 2); return (float) v; }
inline uint16_t f2h(float f) { const _Float16 v = (_Float16) f; uint16_t h; std::memcpy(&h, &v, 2); return h; }

// ---- scalar reference
inline void ref_quant(const uint16_t* x, uint8_t* out, int64_t n) {
    int8_t* q = reinterpret_cast<int8_t*>(out);
    uint8_t* sc = out + n;
    for (int64_t b = 0; b < n / kBlock; ++b) {
        const uint16_t* xb = x + b * kBlock;
        float amax = 0.0f;
        for (int i = 0; i < kBlock; ++i) { const float a = std::fabs(h2f(xb[i])); if (a > amax) amax = a; }
        const uint16_t sh = f2h(amax / kQMax);
        const float sf = h2f(sh);
        std::memcpy(sc + b * 2, &sh, 2);
        for (int i = 0; i < kBlock; ++i) {
            int qi = 0;
            if (sf != 0.0f) {
                float r = h2f(xb[i]) / sf;
                if (r == r) {
                    r = std::nearbyint(r);
                    r = r > kQMax ? kQMax : (r < -kQMax ? -kQMax : r);
                    qi = (int) r;
                }
            }
            q[b * kBlock + i] = (int8_t) qi;
        }
    }
}
inline void ref_dequant(const uint8_t* in, uint16_t* out, int64_t n) {
    const int8_t* q = reinterpret_cast<const int8_t*>(in);
    const uint8_t* sc = in + n;
    for (int64_t b = 0; b < n / kBlock; ++b) {
        uint16_t sh;
        std::memcpy(&sh, sc + b * 2, 2);
        const float sf = h2f(sh);
        for (int i = 0; i < kBlock; ++i) {
            float v = (float) q[b * kBlock + i] * sf;
            v = v > kHalfMax ? kHalfMax : (v < -kHalfMax ? -kHalfMax : v);
            out[b * kBlock + i] = f2h(v);
        }
    }
}

// ---- the same, blocks [b0, b1) of a packed array of n values (NEON when available)
inline void quant_blocks(const uint16_t* x, uint8_t* out, int64_t n, int64_t b0, int64_t b1) {
#if defined(__ARM_NEON)
    int8_t* q = reinterpret_cast<int8_t*>(out);
    uint8_t* sc = out + n;
    for (int64_t b = b0; b < b1; ++b) {
        const uint16_t* xb = x + b * kBlock;
        float32x4_t v[8];
        for (int j = 0; j < 4; ++j) {
            const float16x8_t h = vreinterpretq_f16_u16(vld1q_u16(xb + 8 * j));
            v[2 * j] = vcvt_f32_f16(vget_low_f16(h));
            v[2 * j + 1] = vcvt_high_f32_f16(h);
        }
        float32x4_t m = vabsq_f32(v[0]);
        for (int j = 1; j < 8; ++j) m = vmaxnmq_f32(m, vabsq_f32(v[j]));
        const float amax = vmaxnmvq_f32(m);
        const uint16_t sh = f2h(amax / kQMax);
        const float sf = h2f(sh);
        std::memcpy(sc + b * 2, &sh, 2);
        if (sf == 0.0f) { std::memset(q + b * kBlock, 0, kBlock); continue; }
        const float32x4_t vs = vdupq_n_f32(sf);
        for (int j = 0; j < 4; ++j) {
            int32x4_t a = vcvtnq_s32_f32(vdivq_f32(v[2 * j], vs)), c = vcvtnq_s32_f32(vdivq_f32(v[2 * j + 1], vs));
            // clamp to +-127 in the integer domain (vcvtn saturates and maps NaN to 0, as the device's cvt.rni does)
            a = vminq_s32(vmaxq_s32(a, vdupq_n_s32(-127)), vdupq_n_s32(127));
            c = vminq_s32(vmaxq_s32(c, vdupq_n_s32(-127)), vdupq_n_s32(127));
            vst1_s8(q + b * kBlock + 8 * j, vmovn_s16(vcombine_s16(vmovn_s32(a), vmovn_s32(c))));
        }
    }
#else
    // scalar fallback = the reference on these blocks
    for (int64_t b = b0; b < b1; ++b) {
        uint8_t tmp[kBlock + 2];
        ref_quant(x + b * kBlock, tmp, kBlock);
        std::memcpy(out + b * kBlock, tmp, kBlock);
        std::memcpy(out + n + b * 2, tmp + kBlock, 2);
    }
#endif
}
inline void dequant_blocks(const uint8_t* in, uint16_t* out, int64_t n, int64_t b0, int64_t b1) {
#if defined(__ARM_NEON)
    const int8_t* q = reinterpret_cast<const int8_t*>(in);
    const uint8_t* sc = in + n;
    const float32x4_t hi = vdupq_n_f32(kHalfMax), lo = vdupq_n_f32(-kHalfMax);
    for (int64_t b = b0; b < b1; ++b) {
        uint16_t sh;
        std::memcpy(&sh, sc + b * 2, 2);
        const float32x4_t vs = vdupq_n_f32(h2f(sh));
        for (int j = 0; j < 2; ++j) {   // 16 values per step
            const int8x16_t w = vld1q_s8(q + b * kBlock + 16 * j);
            const int16x8_t w0 = vmovl_s8(vget_low_s8(w)), w1 = vmovl_s8(vget_high_s8(w));
            const int32x4_t i4[4] = {vmovl_s16(vget_low_s16(w0)), vmovl_s16(vget_high_s16(w0)), vmovl_s16(vget_low_s16(w1)), vmovl_s16(vget_high_s16(w1))};
            uint16x4_t r[4];
            for (int k = 0; k < 4; ++k) {
                float32x4_t f = vmulq_f32(vcvtq_f32_s32(i4[k]), vs);
                f = vminq_f32(vmaxq_f32(f, lo), hi);
                r[k] = vreinterpret_u16_f16(vcvt_f16_f32(f));
            }
            vst1q_u16(out + b * kBlock + 16 * j, vcombine_u16(r[0], r[1]));
            vst1q_u16(out + b * kBlock + 16 * j + 8, vcombine_u16(r[2], r[3]));
        }
    }
#else
    for (int64_t b = b0; b < b1; ++b) {
        uint8_t tmp[kBlock + 2];
        std::memcpy(tmp, in + b * kBlock, kBlock);
        std::memcpy(tmp + kBlock, in + n + b * 2, 2);
        ref_dequant(tmp, out + b * kBlock, kBlock);
    }
#endif
}

// ---- over all blocks, split over `threads` (dispatch_apply on macOS; serial elsewhere or for small arrays)
namespace detail {
struct Job {
    const void* in;
    void* out;
    int64_t n, nb, per;
    bool is_quant;
};
inline void run_part(const Job& j, int64_t part) {
    const int64_t b0 = part * j.per, b1 = std::min(j.nb, b0 + j.per);
    if (b0 >= b1) return;
    if (j.is_quant) quant_blocks((const uint16_t*) j.in, (uint8_t*) j.out, j.n, b0, b1);
    else dequant_blocks((const uint8_t*) j.in, (uint16_t*) j.out, j.n, b0, b1);
}
#if defined(__APPLE__)
inline void apply_fn(void* ctx, size_t i) { run_part(*static_cast<const Job*>(ctx), (int64_t) i); }
#endif
inline void run(const Job& j, int threads) {
    const int64_t parts = std::max<int64_t>(1, std::min<int64_t>(threads, j.nb / 1024));   // >= 1024 blocks (32K values) per part
    Job jj = j;
    jj.per = (j.nb + parts - 1) / parts;
#if defined(__APPLE__)
    if (parts > 1) {
        dispatch_apply_f((size_t) parts, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), &jj, apply_fn);
        return;
    }
#endif
    for (int64_t p = 0; p < parts; ++p) run_part(jj, p);
}
}  // namespace detail

// x16[n] -> out[packed_bytes(n)]
inline void quant(const uint16_t* x16, uint8_t* out, int64_t n, int threads = 1) {
    detail::run(detail::Job{x16, out, n, n / kBlock, 0, true}, threads);
}
// in[packed_bytes(n)] -> out16[n]
inline void dequant(const uint8_t* in, uint16_t* out16, int64_t n, int threads = 1) {
    detail::run(detail::Job{in, out16, n, n / kBlock, 0, false}, threads);
}

}  // namespace strata::prefill::xport8
