// libtgd: in-process access to our TinyGPU dext variant .
// One connection per process. Not thread-safe except tgd_spin_ge / tgd_land / tgd_mmio_copy, which touch no state.
#ifndef LIBTGD_H
#define LIBTGD_H
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// 0 on success; -1 = no IOService named "tinygpu" (dext not loaded or no GPU); else the IOServiceOpen kern_return_t.
// Registers a termination notification: if the device goes away the process exits (fail closed).
int   tgd_open(void);
// Maps BAR `bar` (0..5) into this process. wc = write-combined (dext map type 8 + bar). NULL on failure.
void *tgd_map_bar(int bar, int wc, uint64_t *size);
// Anonymous page-aligned memory (size rounded up to the 16 KiB page), prepared for DMA by the dext (selector 3).
// iova_pairs receives [iova, len] pairs ended by [0, 0]; it must hold 2 * max_pairs uint64_t. NULL on failure.
void *tgd_dma_alloc(uint64_t size, uint64_t *iova_pairs, uint32_t max_pairs, uint64_t *handle);
// CompleteDMA in the dext (selector 4), then munmap. 0 on success.
int   tgd_dma_free(uint64_t handle, void *ptr, uint64_t size);
// Registers EXISTING page-aligned memory for DMA (selector 3, no mmap): -1 on bad arguments, -2 if iova_pairs is too small, else 0 or the kern_return_t.
int   tgd_dma_register(void *ptr, uint64_t size, uint64_t *iova_pairs, uint32_t max_pairs, uint64_t *handle);
// CompleteDMA in the dext (selector 4) for memory registered by tgd_dma_register; the memory stays the caller's. 0 on success.
int   tgd_dma_unregister(uint64_t handle);
// MMIO-safe copy: aligned 32-bit volatile accesses, byte accesses at unaligned edges (server.c:62-67 semantics).
void  tgd_mmio_copy(volatile void *dst, const volatile void *src, uint64_t n);
int   tgd_cfg_read(uint32_t off, uint32_t sz, uint32_t *val);
int   tgd_cfg_write(uint32_t off, uint32_t sz, uint32_t val);
int   tgd_reset(void);
// Spin on a local word until (int32_t)(*word - want) >= 0 (wrap-safe). Returns the value seen, 0 on timeout.
// Sequence numbers must therefore start at 1.
uint32_t tgd_spin_ge(volatile uint32_t *word, uint32_t want, uint64_t timeout_ns);
// dsb st, then one 32-bit store of seq (the "landed" posted write into BAR1).
void     tgd_land(volatile uint32_t *bar_word, uint32_t seq);

#ifdef __cplusplus
}
#endif
#endif
