/* xch_logic.h - the cold-pair compaction of the GPU <-> Metal exchange (docs/strata-exchange-map.md 3.1 step 1, 3.4,
 * 3.5): the 4080 walks a layer's routed ids in i = t*K + j order and keeps every pair whose expert is not resident on
 * the card.  Plain C/C++: included by the CUDA kernels (device) and by the host (its reference and the parity test).
 */
#ifndef STRATA_XCH_LOGIC_H
#define STRATA_XCH_LOGIC_H

#include <stdint.h>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define XCH_HD __host__ __device__
#else
#define XCH_HD
#endif

/* One layer's cold pairs, in i = t*K + j order (3.5: the order fixes the fp32 sum).  An id is cold when it is valid
 * (0 <= id < n_expert) and not resident on the card (d_res == 0, or d_res[id] < 0); every other id is skipped.  Writes
 * ex[p] = id, row[p] = i / k, wo[p] = w[i] for p = 0 .. count-1 and returns the count (at most n*k). */
XCH_HD inline int xch_compact(const int32_t* ids, const float* w, const int32_t* d_res, int n_expert, int n, int k,
                              int32_t* ex, int32_t* row, float* wo) {
    int c = 0;
    for (int i = 0; i < n * k; ++i) {
        const int32_t id = ids[i];
        if (id < 0 || id >= n_expert) continue;
        if (d_res && d_res[id] >= 0) continue;
        ex[c] = id;
        row[c] = i / k;
        wo[c] = w[i];
        ++c;
    }
    return c;
}

#endif
