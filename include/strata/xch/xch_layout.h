/* xch_layout.h - the GPU <-> Metal exchange region (docs/strata-exchange-map.md 3.1-3.5): one Mac DMA region from tgcuSysmemAlloc,
 * 16 KiB aligned, read and written in place by the 4080 (over PCIe) and by Metal (NoCopy wrap, coherent(system) accesses).
 * Plain C: included by the CUDA kernels, the Metal kernels and the host. Byte offsets; every flag word sits on its own 128 B line.
 *
 * Per window (one verify window = one Metal command buffer + one CUDA graph launch), per layer l, slot S = XCH_SLOT(l):
 *   4080  writes S.x (T rows, fp16, row-major 2560), S.pairs (count entries, cold pairs in i = t*K + j order), S.count, S.nrows,
 *         S.ids (T*K routed expert ids, -1 padding), fences (system), then S.seq = epoch.
 *   Metal waits S.seq == epoch (bounded), computes the cold experts, writes S.y (T rows, fp32), fences (system), S.landed = epoch.
 *   4080  waits S.landed == epoch (bounded), adds S.y into its residual.
 *   Timing (in each slot's free bytes 264..300; 0 = that side never got there): Metal writes S.mtl_spins when it
 *         sees seq, the 4080 writes S.nv_pub_ns just before S.seq, S.nv_wait0_ns on its wait entry, and S.nv_wait1_ns
 *         and S.nv_spins when it sees landed.
 * Header words: XCH_H_EPOCH (host writes before the commit/launch), XCH_H_ABORT (nonzero = both sides stop spinning and drain),
 * status words (0 = ok; else XCH_ST_* code, layer, spins), miss words (Metal found a table entry 0; host serves it).
 * Compares are wrap-safe: (int32_t)(v - want) >= 0.
 */
#ifndef STRATA_XCH_LAYOUT_H
#define STRATA_XCH_LAYOUT_H

#define XCH_H              2560u                 /* hidden size */
#define XCH_K              10u                   /* routed experts per token */
#define XCH_MAX_ROWS       8u                    /* T <= 8 tokens per window */
#define XCH_MAX_PAIRS      80u                   /* XCH_MAX_ROWS * XCH_K */
#define XCH_N_LAYERS       48u

/* header page */
#define XCH_HDR_BYTES      16384u
#define XCH_H_EPOCH        0u                    /* u32 */
#define XCH_H_ABORT        128u                  /* u32 */
#define XCH_H_MTL_STATUS   256u                  /* u32[4]: code, layer, spins, aux */
#define XCH_H_NV_STATUS    384u                  /* u32[4]: code, layer, spins, aux */
#define XCH_H_MISS_SEQ     512u                  /* u32: Metal raises (epoch << 8 | layer) when it needs table entries */
#define XCH_H_MISS_FIXED   640u                  /* u32: host answers with the same value once the entries are written */
#define XCH_H_MISS_COUNT   768u                  /* u32 */
#define XCH_H_MISS_LIST    1024u                 /* u32[2 * XCH_MAX_PAIRS]: (layer, expert) */

/* status codes */
#define XCH_ST_OK          0u
#define XCH_ST_TIMEOUT     1u                    /* a bounded spin ran out */
#define XCH_ST_ABORT       2u                    /* the abort word was seen */
#define XCH_ST_SHAPE       3u                    /* count/rows out of range */

/* per-layer slot */
#define XCH_SLOT_BYTES     131072u
#define XCH_SLOT(l)        (XCH_HDR_BYTES + (l) * XCH_SLOT_BYTES)
#define XCH_S_SEQ          0u                    /* u32 */
#define XCH_S_LANDED       128u                  /* u32 */
#define XCH_S_COUNT        256u                  /* u32: cold pairs in this layer (0 = nothing to do, Metal still lands) */
#define XCH_S_NROWS        260u                  /* u32: T */
#define XCH_S_MTL_SPINS    264u                  /* u32: Metal writes (xw_wait_build): its spin count when it saw seq == epoch */
#define XCH_S_NV_PUB_NS    272u                  /* u64: 4080 writes (xch_publish_cold): globaltimer just before slot.seq */
#define XCH_S_NV_WAIT0_NS  280u                  /* u64: 4080 writes (xch_wait_flag_sys): globaltimer on entry */
#define XCH_S_NV_WAIT1_NS  288u                  /* u64: 4080 writes (xch_wait_flag_sys): globaltimer when it saw landed */
#define XCH_S_NV_SPINS     296u                  /* u32: 4080 writes (xch_wait_flag_sys): its poll count when it saw landed */
#define XCH_S_PAIRS        512u                  /* XCH_MAX_PAIRS x { int32 expert; int32 row; float weight; } = 960 B */
#define XCH_S_IDS          2048u                 /* int32[XCH_MAX_PAIRS]: all routed ids, t*K + j, -1 = none (logs/counters) */
#define XCH_S_X            4096u                 /* half[XCH_MAX_ROWS * XCH_H] = 40 KiB */
#define XCH_S_Y            49152u                /* float[XCH_MAX_ROWS * XCH_H] = 80 KiB, ends at XCH_SLOT_BYTES */

#define XCH_REGION_BYTES   (XCH_HDR_BYTES + XCH_N_LAYERS * XCH_SLOT_BYTES)   /* 6 MiB + 16 KiB */

#endif
