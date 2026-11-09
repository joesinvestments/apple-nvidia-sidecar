// tgdprobe: open our dext in-process and print the card and its memory regions.
// Exit 0 = opened and printed; 2 = driver service not found; 3 = open refused (user-client entitlement).
#include "libtgd.h"
#include <stdio.h>
#include <string.h>

// usage: tgdprobe [--wc]   (--wc maps BAR1 write-combined instead of uncached; never both, no mixed-attribute aliases)
int main(int argc, char **argv) {
  int wc1 = argc > 1 && !strcmp(argv[1], "--wc");
  int rc = tgd_open();
  if (rc == -1) {
    fprintf(stderr, "tgdprobe: no IOService named \"tinygpu\": our dext is not loaded or no GPU is attached.\n"
                    "          Check: systemextensionsctl list; ioreg -l -n tinygpu; see driver/LOAD-TOMORROW.md\n");
    return 2;
  }
  if (rc != 0) {
    fprintf(stderr, "tgdprobe: IOServiceOpen on \"tinygpu\" failed: 0x%08x (client not allowed to open the dext?)\n", rc);
    return 3;
  }

  uint32_t ven = 0, dev = 0;
  tgd_cfg_read(0x00, 2, &ven);
  tgd_cfg_read(0x02, 2, &dev);
  printf("pci %04x:%04x\n", ven, dev);

  for (int bar = 0; bar < 6; bar++) {
    uint64_t sz = 0;
    int wc = bar == 1 && wc1;
    void *p = tgd_map_bar(bar, wc, &sz);
    if (p) printf("bar%d %s size 0x%llx (%llu MiB)\n", bar, wc ? "wc" : "uc", (unsigned long long)sz, (unsigned long long)(sz >> 20));
    else printf("bar%d %s not mapped%s\n", bar, wc ? "wc" : "uc", wc ? " (WC mapping refused)" : "");
    if (bar == 0 && p) printf("bar0[0] (NV_PMC_BOOT_0) = 0x%08x\n", *(volatile uint32_t *)p);
  }
  return 0;
}
