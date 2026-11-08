// xport8_gpu_test.cpp - the CUDA kernels against the host code (scalar reference and NEON), byte for byte, on a real GPU.  Run on an aarch64 box with nvcc and a GPU
// (a DGX Spark; the RTX 3090 stays with the server):
//   nvcc -O3 -std=c++17 -arch=native -c xport8_gpu_launch.cu -o xport8_gpu_launch.o
//   g++ -O2 -std=c++20 -I ../../include -I /usr/local/cuda/include xport8_gpu_test.cpp xport8_gpu_launch.o -L /usr/local/cuda/lib64 -lcudart -o xport8_gpu_test && ./xport8_gpu_test [real.u16]
#include "strata/prefill/xport8.hpp"

#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

using namespace strata::prefill::xport8;

extern "C" int xp8_gpu_roundtrip(const uint16_t* x, int64_t n, uint8_t* packed_out, uint16_t* dq_of_gpu_packed, const uint8_t* packed_in, uint16_t* dq_of_given_packed);

static int g_fail = 0;

static std::vector<uint16_t> gen(int kind, int64_t n, std::mt19937_64& rng) {
    std::vector<uint16_t> x((size_t) n);
    std::normal_distribution<float> g(0.0f, 1.0f);
    std::uniform_real_distribution<float> u(0.0f, 1.0f);
    for (int64_t i = 0; i < n; ++i) {
        float v = 0;
        switch (kind) {
            case 0: v = g(rng); break;
            case 1: v = g(rng) * (u(rng) < 0.01f ? 60.0f : 1.0f); break;
            case 2: v = g(rng) * std::exp(8.0f * (u(rng) - 0.5f)); break;
            case 3: v = g(rng) * 1e-6f; break;
            case 4: v = (u(rng) < 0.5f ? 0.0f : g(rng) * 30000.0f); v = std::fmax(std::fmin(v, 65504.0f), -65504.0f); break;
            case 5: v = (float) ((int) (g(rng) * 8)) * 0.25f; break;
            case 6: v = g(rng) * 0.3f + 0.003f * (float) (i % 7); break;
            default: v = 0;
        }
        x[(size_t) i] = f2h(v);
    }
    return x;
}

static void run(const char* what, const std::vector<uint16_t>& x) {
    const int64_t n = (int64_t) x.size();
    const size_t pb = packed_bytes(n);
    std::vector<uint8_t> ref(pb), neon(pb), gpu(pb);
    std::vector<uint16_t> ref_dq((size_t) n), neon_dq((size_t) n), gpu_dq((size_t) n), gpu_dq_of_host((size_t) n);
    ref_quant(x.data(), ref.data(), n);
    quant(x.data(), neon.data(), n, 1);
    ref_dequant(ref.data(), ref_dq.data(), n);
    dequant(neon.data(), neon_dq.data(), n, 1);
    if (xp8_gpu_roundtrip(x.data(), n, gpu.data(), gpu_dq.data(), neon.data(), gpu_dq_of_host.data()) != 0) { ++g_fail; std::printf("FAIL %s: gpu run\n", what); return; }
    const bool q_ref = std::memcmp(ref.data(), gpu.data(), pb) == 0, q_neon = std::memcmp(neon.data(), gpu.data(), pb) == 0;
    const bool d_gpu = std::memcmp(ref_dq.data(), gpu_dq.data(), (size_t) n * 2) == 0;       // GPU dequant of GPU-quantised == ref dequant of ref-quantised
    const bool d_host = std::memcmp(neon_dq.data(), gpu_dq_of_host.data(), (size_t) n * 2) == 0;   // GPU dequant of NEON-quantised == NEON dequant of it
    std::printf("%-28s n=%-9lld quant GPU==ref %s  GPU==NEON %s | dequant GPU==ref %s  (host packed) GPU==NEON %s\n", what, (long long) n, q_ref ? "yes" : "NO", q_neon ? "yes" : "NO", d_gpu ? "yes" : "NO", d_host ? "yes" : "NO");
    if (!q_ref || !q_neon || !d_gpu || !d_host) {
        ++g_fail;
        for (size_t i = 0; i < pb; ++i) if (ref[i] != gpu[i]) { std::printf("  first quant byte diff at %zu: ref %02x gpu %02x\n", i, ref[i], gpu[i]); break; }
        for (size_t i = 0; i < (size_t) n; ++i) if (ref_dq[i] != gpu_dq[i]) { std::printf("  first dequant diff at %zu: ref %04x gpu %04x\n", i, ref_dq[i], gpu_dq[i]); break; }
    }
}

int main(int argc, char** argv) {
    std::mt19937_64 rng(777);
    static const char* names[] = {"gaussian", "outlier channels", "log-uniform scale", "tiny (flush)", "huge + zeros", "grid", "offset ramp"};
    for (int kind = 0; kind <= 6; ++kind)
        for (int64_t n : {32LL, 2560LL, 32LL * 4099, 2560LL * 2048}) {
            char nm[64];
            std::snprintf(nm, sizeof nm, "%s", names[kind]);
            run(nm, gen(kind, n, rng));
        }
    if (argc > 1) {
        FILE* f = std::fopen(argv[1], "rb");
        if (f) {
            std::fseek(f, 0, SEEK_END);
            const long sz = std::ftell(f);
            std::fseek(f, 0, SEEK_SET);
            std::vector<uint16_t> x((size_t) (sz / 2 / 2560 * 2560));
            if (std::fread(x.data(), 2, x.size(), f) == x.size()) run("real activations", x);
            std::fclose(f);
        }
    }
    std::printf(g_fail ? "FAILED %d\n" : "ALL OK\n", g_fail);
    return g_fail != 0;
}
