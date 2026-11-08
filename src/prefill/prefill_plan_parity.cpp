// strata/src/prefill/prefill_plan_parity.cpp - the card's blocked expert grouping (prefill_plan_dev.cu) against the
// host algorithm it mirrors (prefill_plan_blocked, prefill_plan.hpp): per-block histograms, the exclusive scan of
// those across blocks per expert, the start of every expert in expert-id order with the Mac lane's block last when
// mac_cold_last, and the stable in-block rank that gives slot / src.
//
// Layers as the prompt path plans them: T 1..8192 tokens, K 8 or 10, n_expert 64 or 512, residency 0 / 0.3 / 1,
// routed ids with and without duplicates inside a token, mac_cold_last both ways, with and without a residency
// table, and block 1 / 7 / 64 / 256 / 1024 (the grouping's block of entries, not the card's). cnt, off, slot and src
// are compared word for word with the host's. Then the argument bounds. GPU, synthetic, no model.
#include "prefill/prefill_plan.hpp"
#include "prefill/prefill_plan_dev.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

using namespace strata::prefill;

namespace {

void ck(cudaError_t e, const char* w) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", w, cudaGetErrorString(e));
        std::exit(2);
    }
}

// the first word two vectors differ at, or -1 when they are equal word for word
long long first_diff(const std::vector<int32_t>& got, const std::vector<int32_t>& want) {
    const size_t m = got.size() < want.size() ? got.size() : want.size();
    for (size_t i = 0; i < m; ++i)
        if (got[i] != want[i]) return (long long) i;
    return got.size() == want.size() ? -1 : (long long) m;
}

}  // namespace

int main() {
    const int64_t kMaxN = 8192 * 10;   // the largest chunk x top-k the prompt path plans
    const int32_t kMaxE = 1024;        // prefill_plan_device's own bound on n_expert
    const int32_t kPoolE = 512;        // what the random layers draw their ids from
    cudaStream_t stream = nullptr;
    int32_t *d_ids = nullptr, *d_res = nullptr, *d_hist = nullptr, *d_cnt = nullptr;
    int32_t *d_off = nullptr, *d_slot = nullptr, *d_src = nullptr;
    ck(cudaMalloc((void**) &d_ids, (size_t) kMaxN * 4), "ids");
    ck(cudaMalloc((void**) &d_res, (size_t) kMaxE * 4), "res");
    ck(cudaMalloc((void**) &d_hist, (size_t) kMaxN * (size_t) kPoolE * 4), "hist");
    ck(cudaMalloc((void**) &d_cnt, (size_t) kMaxE * 4), "cnt");
    ck(cudaMalloc((void**) &d_off, (size_t) (kMaxE + 1) * 4), "off");
    ck(cudaMalloc((void**) &d_slot, (size_t) kMaxN * 4), "slot");
    ck(cudaMalloc((void**) &d_src, (size_t) kMaxN * 4), "src");
    ck(cudaStreamCreate(&stream), "stream");

    int cases = 0, bad = 0;
    // one layer: the ids and (when there is one) the residency table up, the four kernels on `stream`, then cnt, off,
    // slot and src back down against the host's plan for the same layer
    auto run = [&](const std::vector<int32_t>& ids, const std::vector<int32_t>* res_layer, int64_t T, int32_t K,
                   int32_t ne, bool mac, int32_t block) {
        const int64_t n = T * (int64_t) K;
        PrefillPlan want;
        if (!prefill_plan_blocked(ids.data(), T, K, ne, res_layer == nullptr ? nullptr : res_layer->data(), mac, block,
                                  want)) {
            std::printf("    *** the host refuses a valid layer (T %lld K %d ne %d mac %d block %d) ***\n",
                        (long long) T, K, ne, (int) mac, block);
            ++bad;
            return;
        }
        if (res_layer != nullptr)
            ck(cudaMemcpyAsync(d_res, res_layer->data(), (size_t) ne * 4, cudaMemcpyHostToDevice, stream), "res");
        ck(cudaMemcpyAsync(d_ids, ids.data(), (size_t) n * 4, cudaMemcpyHostToDevice, stream), "ids");
        if (prefill_plan_device(d_ids, T, K, ne, res_layer == nullptr ? nullptr : d_res, mac, block, d_hist, d_cnt,
                                d_off, d_slot, d_src, stream) != 0) {
            std::printf("    *** prefill_plan_device refuses a valid layer (T %lld K %d ne %d mac %d block %d) ***\n",
                        (long long) T, K, ne, (int) mac, block);
            ++bad;
            return;
        }
        ck(cudaStreamSynchronize(stream), "sync");

        std::vector<int32_t> cnt((size_t) ne), off((size_t) ne + 1), slot((size_t) n), src((size_t) n);
        ck(cudaMemcpyAsync(cnt.data(), d_cnt, (size_t) ne * 4, cudaMemcpyDeviceToHost, stream), "cnt");
        ck(cudaMemcpyAsync(off.data(), d_off, (size_t) (ne + 1) * 4, cudaMemcpyDeviceToHost, stream), "off");
        ck(cudaMemcpyAsync(slot.data(), d_slot, (size_t) n * 4, cudaMemcpyDeviceToHost, stream), "slot");
        ck(cudaMemcpyAsync(src.data(), d_src, (size_t) n * 4, cudaMemcpyDeviceToHost, stream), "src");
        ck(cudaStreamSynchronize(stream), "sync");

        ++cases;
        const char* names[] = {"cnt", "off", "slot", "src"};
        const std::vector<int32_t>* got[] = {&cnt, &off, &slot, &src};
        const std::vector<int32_t>* exp[] = {&want.cnt, &want.off, &want.slot, &want.src};
        for (int f = 0; f < 4; ++f) {
            const long long q = first_diff(*got[f], *exp[f]);
            if (q < 0) continue;
            std::printf("    *** %s[%lld] = %d, host %d (T %lld K %d ne %d mac %d block %d) ***\n", names[f], q,
                        (size_t) q < got[f]->size() ? (*got[f])[(size_t) q] : -1,
                        (size_t) q < exp[f]->size() ? (*exp[f])[(size_t) q] : -1, (long long) T, K, ne, (int) mac,
                        block);
            ++bad;
            break;
        }
    };

    std::mt19937 rng(47);
    const int64_t Ts[] = {1, 2, 3, 17, 100, 167, 511, 1183, 2048, 8192};
    const int32_t blocks[] = {1, 7, 64, 256, 1024};
    for (int it = 0; it < 300 && !bad; ++it) {
        const int64_t T = Ts[it % 10];
        const int32_t K = (it % 3 == 0) ? 8 : 10;
        const int32_t ne = (it % 4 == 0) ? 64 : kPoolE;
        const double frac = (it % 5 == 0) ? 0.0 : (it % 5 == 1) ? 1.0 : 0.3;
        const bool dup = it % 7 == 0, mac = it % 2 == 0;

        std::vector<int32_t> res((size_t) ne);
        for (int32_t e = 0; e < ne; ++e)
            res[(size_t) e] = (std::uniform_real_distribution<>(0.0, 1.0)(rng) < frac) ? e : -1;

        std::vector<int32_t> ids((size_t) (T * K));
        for (int64_t t = 0; t < T; ++t)
            for (int32_t k = 0; k < K; ++k) {
                int32_t e;
                bool ok;
                do {
                    e = std::uniform_int_distribution<int32_t>(0, ne - 1)(rng);
                    ok = true;
                    if (!dup)
                        for (int32_t j = 0; j < k; ++j) ok = ok && ids[(size_t) (t * K + j)] != e;
                } while (!ok);
                ids[(size_t) (t * K + k)] = e;
            }

        for (int32_t block : blocks) run(ids, &res, T, K, ne, mac, block);
        // no residency table: every expert is the card's, whatever mac_cold_last says
        if (it % 11 == 0)
            for (int32_t block : blocks) run(ids, nullptr, T, K, ne, mac, block);
    }

    // the argument bounds: n_expert 1..1024, block 1..1024, T*K >= 1
    {
        std::vector<int32_t> all_on((size_t) kMaxE, 0), one((size_t) 1, 0);
        ck(cudaMemcpy(d_res, all_on.data(), (size_t) kMaxE * 4, cudaMemcpyHostToDevice), "res_bound");
        ck(cudaMemcpy(d_ids, one.data(), 4, cudaMemcpyHostToDevice), "id0");
        struct Bound {
            const char* name;
            int64_t T;
            int32_t K, ne, block;
            bool want_zero;
        };
        const Bound bounds[] = {
            {"n_expert 0", 1, 1, 0, 1, false},
            {"n_expert 1025", 1, 1, 1025, 1, false},
            {"block 0", 1, 1, 4, 0, false},
            {"block 1025", 1, 1, 4, 1025, false},
            {"T*K 0", 0, 1, 4, 1, false},
            {"K 0", 1, 0, 4, 1, false},
            {"n_expert 1024 block 1024", 1, 1, 1024, 1024, true},
            {"block 1", 1, 1, 4, 1, true},
        };
        for (const Bound& b : bounds) {
            const int rc = prefill_plan_device(d_ids, b.T, b.K, b.ne, d_res, true, b.block, d_hist, d_cnt, d_off,
                                               d_slot, d_src, stream);
            if ((rc == 0) != b.want_zero) {
                std::printf("    *** %s: prefill_plan_device returned %d, want %s ***\n", b.name, rc,
                            b.want_zero ? "0" : "nonzero");
                ++bad;
            }
        }
        ck(cudaStreamSynchronize(stream), "sync");
    }

    ck(cudaStreamDestroy(stream), "stream_destroy");
    std::printf("prefill_plan: %d cases, %d failures\n", cases, bad);
    if (bad) return 1;
    std::printf("PREFILL PLAN PARITY OK %d\n", cases);
    return 0;
}
