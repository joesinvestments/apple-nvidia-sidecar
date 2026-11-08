// strata/src/prefill/prefill_plan_dev.hpp - the prompt path's per-layer expert grouping on the card: the device twin
// of prefill_plan_blocked (prefill_plan.hpp), same arguments, same result. No engine wiring here: the caller owns
// every buffer and the stream. See docs/prefill-device-grouping.md section 1.
#pragma once

#include <cstdint>

#include <cuda_runtime.h>

namespace strata::prefill {

// d_hist holds ceil(T*K/block) * n_expert int32, d_cnt n_expert, d_off n_expert + 1, d_slot and d_src T*K; all of
// them are written (d_hist and d_slot/d_src as the host loop writes them). Returns 0, or nonzero for a bad argument:
// n_expert outside 1..1024, block outside 1..1024, T*K < 1, ids null. Ids outside [0, n_expert) are the caller's
// contract, as in the host loop. No allocation, every launch on `s`.
int prefill_plan_device(const int32_t* d_ids, int64_t T, int32_t K, int32_t n_expert, const int32_t* d_res,
                        bool mac_cold_last, int32_t block, int32_t* d_hist, int32_t* d_cnt, int32_t* d_off,
                        int32_t* d_slot, int32_t* d_src, cudaStream_t s);

}  // namespace strata::prefill
