// The prompt path's per-layer expert grouping in the blocked form the card runs: a histogram per block of `block`
// entries, an exclusive scan of those histograms across blocks per expert, and a stable rank within the block. Host
// C++ only (no CUDA here): each of the three loops below maps to one kernel of the next job, and the result is
// prefill.cpp's host grouping for one MoE layer on one card with no peer share.
#pragma once

#include <cstdint>
#include <vector>

namespace strata::prefill {

// What prefill.cpp's host loop publishes for one layer: cnt, off in expert-id order with the Mac lane's block last
// when mac_cold_last, slot/src token-major, order (the card's experts by id) and mac_cold (the lane's experts, whose
// rows are the plan's last block) with their row count.
struct PrefillPlan {
    std::vector<int32_t> cnt, off, slot, src, order, mac_cold;
    int64_t rows_cpu = 0;
};

// res[e] >= 0 means expert e is resident on the card; mac_cold_last stands for "mac_x_issued" (the Mac lane takes
// this layer's routed experts that are not resident). False (out cleared) on an id outside [0, n_expert).
inline bool prefill_plan_blocked(const int32_t* ids, int64_t T, int32_t K, int32_t n_expert, const int32_t* res,
                                 bool mac_cold_last, int32_t block, PrefillPlan& out) {
    out.cnt.clear();
    out.off.clear();
    out.slot.clear();
    out.src.clear();
    out.order.clear();
    out.mac_cold.clear();
    out.rows_cpu = 0;
    if (ids == nullptr || T <= 0 || K <= 0 || n_expert <= 0 || block <= 0) return false;
    const int64_t n = T * K;
    const size_t ne = (size_t) n_expert;
    for (int64_t i = 0; i < n; ++i)
        if (ids[i] < 0 || ids[i] >= n_expert) return false;
    const int64_t nblk = (n + (int64_t) block - 1) / (int64_t) block;

    // Step 1, one kernel: a thread block per `block` entries builds its block's histogram (a shared-memory array on
    // the card, one row of `hist` here). No step below reads another block's entries except through these histograms.
    std::vector<int32_t> hist((size_t) nblk * ne, 0);
    for (int64_t b = 0; b < nblk; ++b) {
        int32_t* h = hist.data() + (size_t) b * ne;
        int64_t i1 = (b + 1) * (int64_t) block;
        if (i1 > n) i1 = n;
        for (int64_t i = b * (int64_t) block; i < i1; ++i) ++h[(size_t) ids[i]];
    }

    // Step 2, one kernel: one thread per expert scans the blocks, so cnt[e] is its total and hist[b][e] becomes the
    // exclusive prefix, the base of expert e in block b.
    out.cnt.assign(ne, 0);
    for (int32_t e = 0; e < n_expert; ++e) {
        int32_t run = 0;
        for (int64_t b = 0; b < nblk; ++b) {
            int32_t& v = hist[(size_t) b * ne + (size_t) e];
            const int32_t c = v;
            v = run;
            run += c;
        }
        out.cnt[(size_t) e] = run;
    }

    // The Mac lane's experts of this layer: their rows are the plan's last block (prefill.cpp's on_cpu block).
    std::vector<char> on_cpu;
    if (mac_cold_last && res != nullptr) {
        for (int32_t e = 0; e < n_expert; ++e)
            if (out.cnt[(size_t) e] > 0 && res[e] < 0) out.mac_cold.push_back(e);
        if (!out.mac_cold.empty()) {
            on_cpu.assign(ne, 0);
            for (int32_t e : out.mac_cold) {
                on_cpu[(size_t) e] = 1;
                out.rows_cpu += out.cnt[(size_t) e];
            }
        }
    }
    out.off.assign(ne + 1, 0);
    int32_t r = 0;
    for (int32_t e = 0; e < n_expert; ++e)
        if (on_cpu.empty() || !on_cpu[(size_t) e]) {
            out.off[(size_t) e] = r;
            r += out.cnt[(size_t) e];
        }
    for (int32_t e = 0; e < n_expert && !on_cpu.empty(); ++e)   // the Mac lane's block, last
        if (on_cpu[(size_t) e]) {
            out.off[(size_t) e] = r;
            r += out.cnt[(size_t) e];
        }
    out.off[ne] = r;

    // Step 3, one kernel: a thread block per `block` entries ranks each entry among the equal ids before it in its
    // block (a shared-memory counter array on the card), slot = the expert's start + its block's base + the rank.
    out.slot.assign((size_t) n, -1);
    out.src.assign((size_t) n, -1);
    std::vector<int32_t> rank(ne, 0);
    for (int64_t b = 0; b < nblk; ++b) {
        const int32_t* base = hist.data() + (size_t) b * ne;
        int64_t i1 = (b + 1) * (int64_t) block;
        if (i1 > n) i1 = n;
        for (int64_t i = b * (int64_t) block; i < i1; ++i) {
            const int32_t e = ids[i];
            const int32_t p = out.off[(size_t) e] + base[(size_t) e] + rank[(size_t) e];
            ++rank[(size_t) e];
            out.slot[(size_t) i] = p;
            out.src[(size_t) p] = (int32_t) (i / K);
        }
        for (int64_t i = b * (int64_t) block; i < i1; ++i) rank[(size_t) ids[i]] = 0;
    }

    // The experts, in id order: the card's resident ones. The Mac lane's block is not run here.
    for (int32_t e = 0; e < n_expert; ++e)
        if (out.cnt[(size_t) e] > 0 && (on_cpu.empty() || !on_cpu[(size_t) e])) out.order.push_back(e);
    return true;
}

}  // namespace strata::prefill
