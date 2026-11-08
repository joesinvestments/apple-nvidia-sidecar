// include/strata/kernels/cpu/mac_mem_guard.hpp - does a slot count fit the Mac's RAM before the lane wires it.
//
// The Mac lane locks every slot it maps (mmap + mlock + register, ml_init): a slot count whose stride sum pushes the machine's wired
// memory past a share of its physical RAM is refused there, before the slots are mapped.  On our test machine (64 GB) an FP4 config with
// 21,500 slots of 2.77 MB NVFP4 blobs on top of a 40 GiB locked RAM tier wired about 95 GiB and the Mac hung until the watchdog
// panicked it, twice.  The share is STRATA_MAC_LOCK_FRAC (default 0.90): the EXL3 exchange that runs today wires about 55.6 GiB of
// 64 and must pass.
#pragma once

#include <cstdint>
#include <cstdio>
#include <string>

namespace strata::kernels::cpu {

/// True when wiring `want` more bytes on top of the `wired_now` the system already holds stays within `frac` of `phys`;
/// `frac` outside (0, 1] means 0.90.  On false `why` names the want, the wired memory and the limit, all in GiB.
inline bool mac_lock_fits(uint64_t want, uint64_t wired_now, uint64_t phys, double frac, std::string& why) {
    why.clear();
    if (!(frac > 0.0 && frac <= 1.0)) frac = 0.90;
    const double limit = frac * (double) phys;
    if ((double) wired_now + (double) want <= limit) return true;
    const double gib = 1024.0 * 1024.0 * 1024.0;
    char line[256];
    std::snprintf(line, sizeof line,
                  "the slots want %.1f GiB, the system has %.1f GiB wired now, and %.0f%% of %.1f GiB is %.1f GiB",
                  (double) want / gib, (double) wired_now / gib, frac * 100.0, (double) phys / gib, limit / gib);
    why = line;
    return false;
}

}  // namespace strata::kernels::cpu
