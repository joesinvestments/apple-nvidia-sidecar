// libtgd: Build: see Makefile (clang -O2 -dynamiclib -framework IOKit -framework CoreFoundation).
#include "libtgd.h"
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include <sys/mman.h>
#include <dispatch/dispatch.h>
#include <IOKit/IOKitLib.h>
#include <IOKit/IOMessage.h>
#include <mach/mach.h>

// dext user-client selectors and map types
enum { SEL_READ_CFG = 0, SEL_WRITE_CFG = 1, SEL_RESET = 2, SEL_PREPARE_DMA = 3, SEL_COMPLETE_DMA = 4 };
enum { MAP_WC_BASE = 8 };
#define IOVA_OUT_BYTES 8192 // > 4096 so it arrives in the dext as a descriptor (UserClient.cpp:133), as server.c:147

static io_connect_t g_conn = IO_OBJECT_NULL;

static void on_terminate(void *refcon, io_service_t svc, uint32_t msg, void *arg) {
  (void)refcon; (void)svc; (void)arg;
  if (msg == kIOMessageServiceIsTerminated) {
    // fail closed: BAR pointers are about to become invalid; same policy as server.c:92-94
    static const char m[] = "libtgd: tinygpu service terminated, exiting\n";
    write(2, m, sizeof(m) - 1);
    _exit(70);
  }
}

int tgd_open(void) {
  if (g_conn != IO_OBJECT_NULL) return 0;
  io_service_t svc = IOServiceGetMatchingService(kIOMainPortDefault, IOServiceNameMatching("tinygpu"));
  if (!svc) return -1;

  static io_object_t notif;
  if (!notif) {
    IONotificationPortRef port = IONotificationPortCreate(kIOMainPortDefault);
    IONotificationPortSetDispatchQueue(port, dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0));
    IOServiceAddInterestNotification(port, svc, kIOGeneralInterest, on_terminate, NULL, &notif);
  }

  kern_return_t kr = IOServiceOpen(svc, mach_task_self(), 0, &g_conn);
  IOObjectRelease(svc);
  if (kr != KERN_SUCCESS) { g_conn = IO_OBJECT_NULL; return kr; }
  return 0;
}

void *tgd_map_bar(int bar, int wc, uint64_t *size) {
  if (g_conn == IO_OBJECT_NULL || bar < 0 || bar > 5) return NULL;
  mach_vm_address_t addr = 0;
  mach_vm_size_t sz = 0;
  // cache mode comes from the dext's descriptor (WC for type 8 + bar); no client-side cache flag
  if (IOConnectMapMemory64(g_conn, (uint32_t)(wc ? MAP_WC_BASE + bar : bar), mach_task_self(), &addr, &sz, kIOMapAnywhere)) return NULL;
  if (size) *size = sz;
  return (void *)addr;
}

void *tgd_dma_alloc(uint64_t size, uint64_t *iova_pairs, uint32_t max_pairs, uint64_t *handle) {
  if (g_conn == IO_OBJECT_NULL || !iova_pairs || max_pairs < 1 || !handle) return NULL;
  uint64_t pg = (uint64_t)getpagesize(); // 16 KiB on this Mac; also keeps the struct input > 4096 bytes
  size = size ? (size + pg - 1) & ~(pg - 1) : pg;

  void *ptr = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_ANON | MAP_PRIVATE, -1, 0);
  if (ptr == MAP_FAILED) return NULL;

  uint64_t out[IOVA_OUT_BYTES / 8] = {0};
  size_t out_sz = sizeof(out);
  uint64_t h = 0;
  uint32_t h_cnt = 1;
  kern_return_t kr = IOConnectCallMethod(g_conn, SEL_PREPARE_DMA, NULL, 0, ptr, size, &h, &h_cnt, out, &out_sz);
  if (kr != KERN_SUCCESS || h_cnt != 1) {
    if (kr == KERN_SUCCESS) { // old dext: prepared but no handle, cannot free; keep the memory alive, report failure
      fprintf(stderr, "libtgd: PrepareDMA returned no handle (stock TinyGPU dext?)\n");
      return NULL;
    }
    munmap(ptr, size);
    return NULL;
  }

  uint32_t n = 0;
  for (; n + 1 < max_pairs && n < IOVA_OUT_BYTES / 16 && out[2 * n + 1]; n++) {
    iova_pairs[2 * n] = out[2 * n];
    iova_pairs[2 * n + 1] = out[2 * n + 1];
  }
  if (n < IOVA_OUT_BYTES / 16 && out[2 * n + 1]) { // caller's array too small: undo
    tgd_dma_free(h, ptr, size);
    return NULL;
  }
  iova_pairs[2 * n] = 0;
  iova_pairs[2 * n + 1] = 0;
  *handle = h;
  return ptr;
}

int tgd_dma_free(uint64_t handle, void *ptr, uint64_t size) {
  if (g_conn == IO_OBJECT_NULL) return -1;
  kern_return_t kr = IOConnectCallScalarMethod(g_conn, SEL_COMPLETE_DMA, &handle, 1, NULL, NULL);
  if (kr != KERN_SUCCESS) return kr; // keep the pages: the device may still own them
  uint64_t pg = (uint64_t)getpagesize();
  return ptr ? munmap(ptr, (size + pg - 1) & ~(pg - 1)) : 0;
}

// Register memory the caller already owns: same PrepareDMA round trip as tgd_dma_alloc, but no mmap and no munmap.
int tgd_dma_register(void *ptr, uint64_t size, uint64_t *iova_pairs, uint32_t max_pairs, uint64_t *handle) {
  uint64_t pg = (uint64_t)getpagesize();
  if (g_conn == IO_OBJECT_NULL || !ptr || !size || (size % pg) || ((uintptr_t)ptr % pg) || !iova_pairs || !handle) return -1;

  uint64_t out[IOVA_OUT_BYTES / 8] = {0};
  size_t out_sz = sizeof(out);
  uint64_t h = 0;
  uint32_t h_cnt = 1;
  kern_return_t kr = IOConnectCallMethod(g_conn, SEL_PREPARE_DMA, NULL, 0, ptr, size, &h, &h_cnt, out, &out_sz);
  if (kr != KERN_SUCCESS || h_cnt != 1) {
    if (kr == KERN_SUCCESS) { // old dext: prepared but no handle, nothing to undo
      fprintf(stderr, "libtgd: PrepareDMA returned no handle (stock TinyGPU dext?)\n");
      return -1;
    }
    return kr;
  }

  uint32_t n = 0;
  for (; n + 1 < max_pairs && n < IOVA_OUT_BYTES / 16 && out[2 * n + 1]; n++) {
    iova_pairs[2 * n] = out[2 * n];
    iova_pairs[2 * n + 1] = out[2 * n + 1];
  }
  if (n < IOVA_OUT_BYTES / 16 && out[2 * n + 1]) { // caller's array too small: undo and report
    tgd_dma_unregister(h);
    return -2;
  }
  iova_pairs[2 * n] = 0;
  iova_pairs[2 * n + 1] = 0;
  *handle = h;
  return 0;
}

// Undo tgd_dma_register: CompleteDMA only, the memory was never ours to unmap.
int tgd_dma_unregister(uint64_t handle) {
  if (g_conn == IO_OBJECT_NULL) return -1;
  return IOConnectCallScalarMethod(g_conn, SEL_COMPLETE_DMA, &handle, 1, NULL, NULL);
}

void tgd_mmio_copy(volatile void *dst, const volatile void *src, uint64_t n) {
  volatile uint8_t *d = (volatile uint8_t *)dst;
  const volatile uint8_t *s = (const volatile uint8_t *)src;
  while (n && ((uintptr_t)d & 3)) { *d++ = *s++; n--; }
  if (((uintptr_t)s & 3) == 0) {
    for (; n >= 4; n -= 4, d += 4, s += 4) *(volatile uint32_t *)d = *(const volatile uint32_t *)s;
  } else { // unaligned source: byte loads (Device memory allows them), aligned 32-bit store
    for (; n >= 4; n -= 4, d += 4, s += 4)
      *(volatile uint32_t *)d = (uint32_t)s[0] | (uint32_t)s[1] << 8 | (uint32_t)s[2] << 16 | (uint32_t)s[3] << 24;
  }
  while (n--) *d++ = *s++;
}

int tgd_cfg_read(uint32_t off, uint32_t sz, uint32_t *val) {
  if (g_conn == IO_OBJECT_NULL || !val) return -1;
  uint64_t in[2] = {off, sz}, out = 0;
  uint32_t out_cnt = 1;
  kern_return_t kr = IOConnectCallScalarMethod(g_conn, SEL_READ_CFG, in, 2, &out, &out_cnt);
  if (kr == KERN_SUCCESS) *val = (uint32_t)out;
  return kr;
}

int tgd_cfg_write(uint32_t off, uint32_t sz, uint32_t val) {
  if (g_conn == IO_OBJECT_NULL) return -1;
  uint64_t in[3] = {off, sz, val};
  return IOConnectCallScalarMethod(g_conn, SEL_WRITE_CFG, in, 3, NULL, NULL);
}

int tgd_reset(void) {
  if (g_conn == IO_OBJECT_NULL) return -1;
  return IOConnectCallScalarMethod(g_conn, SEL_RESET, NULL, 0, NULL, NULL);
}

uint32_t tgd_spin_ge(volatile uint32_t *word, uint32_t want, uint64_t timeout_ns) {
  uint64_t deadline = clock_gettime_nsec_np(CLOCK_UPTIME_RAW) + timeout_ns;
  for (;;) {
    for (int i = 0; i < 256; i++) {
      uint32_t v = __atomic_load_n(word, __ATOMIC_ACQUIRE);
      if ((int32_t)(v - want) >= 0) return v;
    }
    if (clock_gettime_nsec_np(CLOCK_UPTIME_RAW) >= deadline) return 0;
  }
}

void tgd_land(volatile uint32_t *bar_word, uint32_t seq) {
#if defined(__aarch64__)
  __asm__ volatile("dsb st" ::: "memory"); // y and any WC data out before the flag
#else
  __atomic_thread_fence(__ATOMIC_SEQ_CST);
#endif
  *bar_word = seq;
}
