// xport8_test.cpp - card-free checks of the 8-bit block-32 lane transport (include/strata/prefill/xport8.hpp).
//   clang++ -std=c++20 -O2 -mcpu=apple-m4 -I include src/prefill/xport8_test.cpp -o xport8_test && ./xport8_test [real.u16 [rows]]
// 1. golden blocks (hand-computed expectations of the format)
// 2. scalar reference vs the NEON path (quant and dequant), byte for byte, over many distributions, array sizes and thread splits
// 3. round-trip error (rel RMS / max rel to the block max) per distribution
// 4. optional: a raw little-endian fp16 file of [rows x 2560] real activations: rel RMS of the round trip (the numbers quoted in EXL3_PLAN.md)
#include "strata/prefill/xport8.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

using namespace strata::prefill::xport8;

static int g_fail = 0;
#define CHECK(c, ...) do { if (!(c)) { ++g_fail; std::printf("FAIL %s:%d: ", __FILE__, __LINE__); std::printf(__VA_ARGS__); std::printf("\n"); } } while (0)

static int8_t qat(const std::vector<uint8_t>& p, int64_t i) { return (int8_t) p[(size_t) i]; }
static uint16_t sat(const std::vector<uint8_t>& p, int64_t n, int64_t b) { uint16_t h; std::memcpy(&h, p.data() + n + b * 2, 2); return h; }

static void golden() {
    std::vector<uint16_t> x(64, 0);
    for (int i = 0; i < 32; ++i) x[(size_t) i] = f2h(1.0f);                  // block 0: all ones
    for (int i = 0; i < 32; ++i) x[(size_t) (32 + i)] = f2h(i % 2 ? -2.0f : 0.0f);   // block 1: 0, -2, 0, -2 ...
    std::vector<uint8_t> p(packed_bytes(64));
    ref_quant(x.data(), p.data(), 64);
    CHECK(packed_bytes(64) == 64 + 4, "packed_bytes");
    // block 0: amax 1 -> scale fp16(1/127), q = rne(1 / scale) = 127 (scale is rounded: 1/127 = 0.007874..., fp16 0.0078735 -> 1/0.0078735 = 127.0024 -> 127)
    CHECK(sat(p, 64, 0) == f2h(1.0f / 127.0f), "scale0 %04x", sat(p, 64, 0));
    for (int i = 0; i < 32; ++i) CHECK(qat(p, i) == 127, "q0[%d]=%d", i, qat(p, i));
    // block 1: amax 2 -> q(-2) = -127, q(0) = 0
    CHECK(sat(p, 64, 1) == f2h(2.0f / 127.0f), "scale1");
    for (int i = 0; i < 32; ++i) CHECK(qat(p, 32 + i) == (i % 2 ? -127 : 0), "q1[%d]=%d", i, qat(p, 32 + i));
    std::vector<uint16_t> y(64);
    ref_dequant(p.data(), y.data(), 64);
    CHECK(std::fabs(h2f(y[0]) - 1.0f) < 1e-3f, "y0 %g", h2f(y[0]));
    CHECK(std::fabs(h2f(y[33]) + 2.0f) < 2e-3f, "y33 %g", h2f(y[33]));
    CHECK(y[32] == f2h(0.0f) || y[32] == 0x8000, "y32 zero");
    // zeros -> scale 0, q 0, back to 0 (no NaN)
    std::vector<uint16_t> z(32, f2h(0.0f)), zy(32, 0xffff);
    std::vector<uint8_t> zp(packed_bytes(32), 0x55);
    quant(z.data(), zp.data(), 32);
    dequant(zp.data(), zy.data(), 32);
    for (int i = 0; i < 32; ++i) CHECK(h2f(zy[(size_t) i]) == 0.0f, "zero roundtrip %g", h2f(zy[(size_t) i]));
    // the largest fp16 survives (127 * rounded scale can exceed 65504: the dequant clamps instead of overflowing to inf)
    std::vector<uint16_t> big(32, f2h(65504.0f)), by(32);
    std::vector<uint8_t> bp(packed_bytes(32));
    quant(big.data(), bp.data(), 32);
    dequant(bp.data(), by.data(), 32);
    for (int i = 0; i < 32; ++i) CHECK(std::isfinite(h2f(by[(size_t) i])) && h2f(by[(size_t) i]) > 65000.0f, "big %g", h2f(by[(size_t) i]));
}

static std::vector<uint16_t> gen(int kind, int64_t n, std::mt19937_64& rng) {
    std::vector<uint16_t> x((size_t) n);
    std::normal_distribution<float> g(0.0f, 1.0f);
    std::uniform_real_distribution<float> u(0.0f, 1.0f);
    for (int64_t i = 0; i < n; ++i) {
        float v = 0;
        switch (kind) {
            case 0: v = g(rng); break;                                   // gaussian
            case 1: v = g(rng) * (u(rng) < 0.01f ? 60.0f : 1.0f); break;   // outlier channels
            case 2: v = g(rng) * std::exp(8.0f * (u(rng) - 0.5f)); break;  // log-uniform scale
            case 3: v = g(rng) * 1e-6f; break;                           // tiny (scale underflows to subnormal / zero)
            case 4: v = (u(rng) < 0.5f ? 0.0f : g(rng) * 30000.0f); v = std::fmax(std::fmin(v, 65504.0f), -65504.0f); break;   // huge + zeros
            case 5: v = (float) ((int) (g(rng) * 8)) * 0.25f; break;       // exactly representable grid
            default: v = 0;
        }
        x[(size_t) i] = f2h(v);
    }
    return x;
}

static void exact(int kind, int64_t n, int threads, std::mt19937_64& rng) {
    const auto x = gen(kind, n, rng);
    std::vector<uint8_t> a(packed_bytes(n)), b(packed_bytes(n), 0xcd);
    ref_quant(x.data(), a.data(), n);
    quant(x.data(), b.data(), n, threads);
    CHECK(std::memcmp(a.data(), b.data(), a.size()) == 0, "quant kind %d n %lld threads %d: first diff at %zu", kind, (long long) n, threads,
          (size_t) (std::mismatch(a.begin(), a.end(), b.begin()).first - a.begin()));
    std::vector<uint16_t> ya((size_t) n), yb((size_t) n, 0xdead);
    ref_dequant(a.data(), ya.data(), n);
    dequant(a.data(), yb.data(), n, threads);
    CHECK(std::memcmp(ya.data(), yb.data(), (size_t) n * 2) == 0, "dequant kind %d n %lld threads %d", kind, (long long) n, threads);
}

struct Err { double rel_rms, max_rel_blk; };
static Err roundtrip(const std::vector<uint16_t>& x, int64_t n) {
    std::vector<uint8_t> p(packed_bytes(n));
    std::vector<uint16_t> y((size_t) n);
    quant(x.data(), p.data(), n, 4);
    dequant(p.data(), y.data(), n, 4);
    double se = 0, sx = 0, mx = 0;
    for (int64_t b = 0; b < n / kBlock; ++b) {
        float amax = 0;
        for (int i = 0; i < kBlock; ++i) amax = std::fmax(amax, std::fabs(h2f(x[(size_t) (b * kBlock + i)])));
        for (int i = 0; i < kBlock; ++i) {
            const double a = h2f(x[(size_t) (b * kBlock + i)]), c = h2f(y[(size_t) (b * kBlock + i)]);
            se += (a - c) * (a - c);
            sx += a * a;
            if (amax > 0) mx = std::fmax(mx, std::fabs(a - c) / amax);
        }
    }
    return {std::sqrt(se / std::fmax(sx, 1e-300)), mx};
}

int main(int argc, char** argv) {
    golden();
    std::mt19937_64 rng(12345);
    int cases = 0;
    for (int kind = 0; kind <= 5; ++kind)
        for (int64_t n : {32LL, 64LL, 2560LL, 32768LL, 32LL * 1024 * 3 + 32, 2560LL * 2048, 2560LL * 1648})
            for (int th : {1, 2, 4, 8}) { exact(kind, n, th, rng); ++cases; }
    std::printf("byte-exact scalar-vs-NEON cases: %d (fail %d)\n", cases, g_fail);
    for (int kind = 0; kind <= 5; ++kind) {
        const int64_t n = 2560LL * 512;
        const auto x = gen(kind, n, rng);
        const Err e = roundtrip(x, n);
        std::printf("round trip kind %d: rel RMS %.4f%%  max |err|/blockmax %.4f%% (bound 0.5/127 = %.4f%% + fp16 rounding; kind 3 = amax/127 below fp16's range: flushed to 0)\n", kind, 100 * e.rel_rms, 100 * e.max_rel_blk, 100 * 0.5 / 127);
        // bound: half a step of the (fp16-rounded) scale, plus the output's own fp16 rounding (2^-11 of the value)
        if (kind != 3) CHECK(e.max_rel_blk < 0.5 / 127 + 1.0 / 1024, "max err kind %d %g", kind, e.max_rel_blk);
        else CHECK(e.rel_rms > 0.9, "kind 3 (all blocks below 3.8e-6) flushes to zero by design");
    }
    // speed of the NEON path on a 2048 x 2560 chunk
    {
        const int64_t n = 2560LL * 2048;
        const auto x = gen(1, n, rng);
        std::vector<uint8_t> p(packed_bytes(n));
        std::vector<uint16_t> y((size_t) n);
        for (int th : {1, 4, 8}) {
            double tq = 1e9, td = 1e9;
            for (int r = 0; r < 6; ++r) {
                auto t0 = std::chrono::steady_clock::now();
                quant(x.data(), p.data(), n, th);
                auto t1 = std::chrono::steady_clock::now();
                dequant(p.data(), y.data(), n, th);
                auto t2 = std::chrono::steady_clock::now();
                tq = std::min(tq, std::chrono::duration<double, std::milli>(t1 - t0).count());
                td = std::min(td, std::chrono::duration<double, std::milli>(t2 - t1).count());
            }
            std::printf("2048x2560: threads %d: quant %.3f ms, dequant %.3f ms\n", th, tq, td);
        }
    }
    if (argc > 1) {
        FILE* f = std::fopen(argv[1], "rb");
        if (!f) { std::printf("cannot open %s\n", argv[1]); return 2; }
        std::fseek(f, 0, SEEK_END);
        const long sz = std::ftell(f);
        std::fseek(f, 0, SEEK_SET);
        int64_t n = sz / 2 / 2560 * 2560;
        if (argc > 2) n = std::min<int64_t>(n, (int64_t) std::atoll(argv[2]) * 2560);
        std::vector<uint16_t> x((size_t) n);
        if (std::fread(x.data(), 2, (size_t) n, f) != (size_t) n) { std::printf("short read\n"); return 2; }
        std::fclose(f);
        const Err e = roundtrip(x, n);
        std::printf("REAL %s: %lld rows x 2560: rel RMS %.4f%%  max |err|/blockmax %.4f%%\n", argv[1], (long long) (n / 2560), 100 * e.rel_rms, 100 * e.max_rel_blk);
    }
    std::printf(g_fail ? "FAILED %d\n" : "ALL OK\n", g_fail);
    return g_fail != 0;
}
