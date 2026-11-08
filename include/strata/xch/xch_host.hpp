// xch_host.hpp - host side of the GPU <-> Metal exchange (docs/strata-exchange-map.md 3.1-3.7, include/strata/xch/xch_layout.h).
// STRATA_XCH=1 turns it on (static residency, v1): the Metal LRU holds every expert that can be cold, the expert table is built once,
// d7/d18 refills and LRU drops are off, and each verify window is one Metal command buffer + one CUDA graph with no per-layer host work.
#pragma once
#include <cstddef>
#include <cstdint>

extern "C" {
// expert_source.cpp (the Metal lane's owner). 0 = ok, < 0 = error (a message on stderr).
// setup: attach the exchange region to the Metal lane's context, build the expert table (layer * n_expert + e -> the blob in the
// locked tier or in an LRU slot; NULL when neither holds it), and switch the lane to static residency (strata_lru_drop and every
// LRU eviction become no-ops). host_res[layer * n_expert + e] >= 0 = the expert is in VRAM (the card computes it).
int strata_xch_setup(void* src, void* region_host, size_t len, int n_layers, int n_expert, const int32_t* host_res);
int strata_xch_submit(int n_layers, int n_rows, uint32_t epoch, uint32_t spin_budget);   // one window: exl3m_xch_window_submit
int strata_xch_wait(uint32_t status[4]);                                                 // exl3m_xch_window_wait
int strata_xch_on(void);                                                                 // 1 after a successful setup
#if defined(STRATA_TGCU)
// shim/tgcudart.cpp (J1): real Mac DMA memory the GPU reads and writes in place. Returns cudaError_t (0 = cudaSuccess).
int tgcuSysmemAlloc(size_t size, void** host, void** dev);
#endif
}
