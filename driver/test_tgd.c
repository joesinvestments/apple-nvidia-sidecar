// Unit test of the parts of libtgd that need no driver: tgd_mmio_copy, tgd_spin_ge, tgd_land against ordinary memory,
// plus "every driver call fails cleanly with no connection". Exit 0 = all pass.
#include "libtgd.h"
#include <assert.h>
#include <pthread.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

static volatile uint32_t g_word;
static void *bump_later(void *arg) {
  usleep(2000);
  __atomic_store_n(&g_word, (uint32_t)(uintptr_t)arg, __ATOMIC_RELEASE);
  return NULL;
}
static uint64_t now(void) { return clock_gettime_nsec_np(CLOCK_UPTIME_RAW); }

int main(void) {
  // tgd_mmio_copy: every dst/src alignment 0..7, lengths 0..67, against memcpy; guard bytes untouched
  uint8_t src[96], dst[96], ref[96];
  int cases = 0;
  for (int so = 0; so < 8; so++)
    for (int dof = 0; dof < 8; dof++)
      for (int n = 0; n < 68; n++) {
        for (int i = 0; i < 96; i++) { src[i] = (uint8_t)(i * 7 + 3); dst[i] = ref[i] = 0xa5; }
        memcpy(ref + 8 + dof, src + so, n);
        tgd_mmio_copy(dst + 8 + dof, src + so, n);
        assert(!memcmp(dst, ref, sizeof dst));
        cases++;
      }
  printf("tgd_mmio_copy: %d alignment/length cases match memcpy, guards intact\n", cases);

  // tgd_spin_ge: already satisfied, wrap-safe, timeout, and woken by another thread
  g_word = 5;
  assert(tgd_spin_ge(&g_word, 5, 1000) == 5);
  assert(tgd_spin_ge(&g_word, 3, 1000) == 5);
  g_word = 2; // wrap: 2 is "after" 0xfffffff0
  assert(tgd_spin_ge(&g_word, 0xfffffff0u, 1000) == 2);
  g_word = 1;
  uint64_t t0 = now();
  assert(tgd_spin_ge(&g_word, 2, 5000000) == 0);
  uint64_t waited = now() - t0;
  assert(waited >= 5000000 && waited < 50000000);
  printf("tgd_spin_ge: timeout after %.2f ms (asked 5 ms)\n", waited / 1e6);

  pthread_t th;
  g_word = 0;
  t0 = now();
  pthread_create(&th, NULL, bump_later, (void *)(uintptr_t)42);
  uint32_t got = tgd_spin_ge(&g_word, 42, 1000000000);
  uint64_t woke = now() - t0;
  pthread_join(th, NULL);
  assert(got == 42);
  printf("tgd_spin_ge: saw 42 from another thread after %.2f ms (writer slept 2 ms)\n", woke / 1e6);

  // tgd_land: one store of seq
  volatile uint32_t landed = 0;
  tgd_land(&landed, 7);
  assert(landed == 7);
  printf("tgd_land: stored 7\n");

  // no connection: every driver call fails, nothing crashes
  uint64_t pairs[8], h = 0, sz = 0;
  uint32_t v = 0;
  assert(tgd_map_bar(0, 0, &sz) == NULL);
  assert(tgd_dma_alloc(16384, pairs, 4, &h) == NULL);
  assert(tgd_dma_free(0, NULL, 0) != 0);
  assert(tgd_cfg_read(0, 2, &v) != 0 && tgd_cfg_write(0, 2, 0) != 0 && tgd_reset() != 0);
  printf("no-connection calls: all fail cleanly\n");

  printf("PASS\n");
  return 0;
}
