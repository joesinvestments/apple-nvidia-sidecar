// strata/src/prefill/prefill_plan_dev.cu - the card's blocked expert grouping: the four kernels that mirror
// prefill_plan_blocked (prefill_plan.hpp) step for step, launched in order by prefill_plan_device on the caller's
// stream. No allocation, no host round trip, no engine wiring yet (docs/prefill-device-grouping.md section 1).
#include "prefill/prefill_plan_dev.hpp"

#include <cuda_runtime.h>

namespace strata::prefill {
namespace {

// the argument bounds prefill_plan_device enforces, and what the shared-memory arrays below are sized for
constexpr int32_t kPlanMaxExpert = 1024;
constexpr int32_t kPlanMaxBlock = 1024;
constexpr int kPlanThreads = 256;

// Step 1, one thread block per `block` entries: the block's histogram in shared memory, written as one row of
// d_hist. Ids are the caller's contract (in [0, n_expert)), as in the host loop.
__global__ void prefill_plan_hist_kernel(const int32_t* __restrict__ ids, int64_t n, int32_t n_expert, int32_t block,
                                         int32_t* __restrict__ hist) {
    extern __shared__ int32_t s_hist[];
    for (int32_t e = (int32_t) threadIdx.x; e < n_expert; e += (int32_t) blockDim.x) s_hist[e] = 0;
    __syncthreads();
    const int64_t i0 = (int64_t) blockIdx.x * (int64_t) block;
    int64_t i1 = i0 + (int64_t) block;
    if (i1 > n) i1 = n;
    for (int64_t i = i0 + (int64_t) threadIdx.x; i < i1; i += (int64_t) blockDim.x) atomicAdd(&s_hist[ids[i]], 1);
    __syncthreads();
    int32_t* const row = hist + (int64_t) blockIdx.x * (int64_t) n_expert;
    for (int32_t e = (int32_t) threadIdx.x; e < n_expert; e += (int32_t) blockDim.x) row[e] = s_hist[e];
}

// Step 2, one thread per expert: the exclusive scan of its histogram down the blocks, in place, so
// hist[b * n_expert + e] becomes the base of expert e in block b, and its total in d_cnt.
__global__ void prefill_plan_scan_kernel(int32_t* __restrict__ hist, int64_t nblk, int32_t n_expert,
                                         int32_t* __restrict__ cnt) {
    const int32_t e = (int32_t) (blockIdx.x * blockDim.x + threadIdx.x);
    if (e >= n_expert) return;
    int32_t run = 0;
    for (int64_t b = 0; b < nblk; ++b) {
        int32_t& v = hist[b * (int64_t) n_expert + e];
        const int32_t c = v;
        v = run;
        run += c;
    }
    cnt[e] = run;
}

// Step 3, one thread block of n_expert threads: d_off in expert-id order, the Mac lane's block last when
// mac_cold_last (an expert with rows and res[e] < 0), and d_off[n_expert] = the row count, as the host loop.
__global__ void prefill_plan_off_kernel(const int32_t* __restrict__ cnt, const int32_t* __restrict__ res,
                                        int32_t n_expert, bool mac_cold_last, int32_t* __restrict__ off) {
    __shared__ uint8_t s_cold[kPlanMaxExpert];
    const int32_t e = (int32_t) threadIdx.x;   // launched with exactly n_expert threads
    s_cold[e] = (mac_cold_last && res != nullptr && cnt[e] > 0 && res[e] < 0) ? 1 : 0;
    __syncthreads();
    if (e != 0) return;
    int32_t r = 0;
    for (int32_t q = 0; q < n_expert; ++q)
        if (!s_cold[q]) {
            off[q] = r;
            r += cnt[q];
        }
    for (int32_t q = 0; q < n_expert; ++q)
        if (s_cold[q]) {
            off[q] = r;
            r += cnt[q];
        }
    off[n_expert] = r;
}

// Step 4, one thread block per `block` entries: each entry's stable rank among the equal ids before it in its block
// (d_hist now holds the block's bases), then slot = the expert's start + its block's base + the rank, and
// src[slot] = the entry's token.
__global__ void prefill_plan_place_kernel(const int32_t* __restrict__ ids, int64_t n, int32_t K, int32_t n_expert,
                                          int32_t block, const int32_t* __restrict__ hist,
                                          const int32_t* __restrict__ off, int32_t* __restrict__ slot,
                                          int32_t* __restrict__ src) {
    __shared__ int32_t s_ids[kPlanMaxBlock];
    const int64_t i0 = (int64_t) blockIdx.x * (int64_t) block;
    int64_t i1 = i0 + (int64_t) block;
    if (i1 > n) i1 = n;
    const int32_t n_here = (int32_t) (i1 - i0);
    for (int32_t j = (int32_t) threadIdx.x; j < n_here; j += (int32_t) blockDim.x) s_ids[j] = ids[i0 + j];
    __syncthreads();
    const int32_t* const base = hist + (int64_t) blockIdx.x * (int64_t) n_expert;
    for (int32_t j = (int32_t) threadIdx.x; j < n_here; j += (int32_t) blockDim.x) {
        const int32_t e = s_ids[j];
        int32_t rank = 0;
        for (int32_t q = 0; q < j; ++q) rank += (s_ids[q] == e) ? 1 : 0;
        const int32_t p = off[e] + base[e] + rank;
        slot[i0 + j] = p;
        src[p] = (int32_t) ((i0 + j) / (int64_t) K);
    }
}

}  // namespace

int prefill_plan_device(const int32_t* d_ids, int64_t T, int32_t K, int32_t n_expert, const int32_t* d_res,
                        bool mac_cold_last, int32_t block, int32_t* d_hist, int32_t* d_cnt, int32_t* d_off,
                        int32_t* d_slot, int32_t* d_src, cudaStream_t s) {
    if (n_expert < 1 || n_expert > kPlanMaxExpert) return 1;
    if (block < 1 || block > kPlanMaxBlock) return 1;
    if (T < 1 || K < 1 || d_ids == nullptr) return 1;

    const int64_t n = T * (int64_t) K;
    const int64_t nblk = (n + (int64_t) block - 1) / (int64_t) block;
    const size_t shmem = (size_t) n_expert * sizeof(int32_t);
    const unsigned grid = (unsigned) nblk;

    prefill_plan_hist_kernel<<<grid, kPlanThreads, shmem, s>>>(d_ids, n, n_expert, block, d_hist);
    prefill_plan_scan_kernel<<<(unsigned) ((n_expert + kPlanThreads - 1) / kPlanThreads), kPlanThreads, 0, s>>>(
        d_hist, nblk, n_expert, d_cnt);
    prefill_plan_off_kernel<<<1, (unsigned) n_expert, 0, s>>>(d_cnt, d_res, n_expert, mac_cold_last, d_off);
    prefill_plan_place_kernel<<<grid, kPlanThreads, 0, s>>>(d_ids, n, K, n_expert, block, d_hist, d_off, d_slot,
                                                            d_src);
    return 0;
}

}  // namespace strata::prefill
