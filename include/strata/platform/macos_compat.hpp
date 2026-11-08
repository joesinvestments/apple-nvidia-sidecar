// include/strata/platform/macos_compat.hpp - macOS (Apple silicon) shims for the Linux-only calls of the host code.
// Force-included into Strata's own sources by the tgcu macOS build; empty everywhere else.
#pragma once
#if defined(__APPLE__)
#include <fcntl.h>
#include <unistd.h>
#include <sys/types.h>
#include <mach/mach.h>

// hugetlb / THP: macOS has neither; the mappings fall back to normal pages
#include <sys/mman.h>
#ifndef MAP_HUGETLB
#define MAP_HUGETLB 0
#endif
#ifndef MADV_HUGEPAGE
#define MADV_HUGEPAGE MADV_NORMAL
#endif
#ifndef MADV_NOHUGEPAGE
#define MADV_NOHUGEPAGE MADV_NORMAL
#endif

// O_DIRECT: macOS has no open flag; DirectFile sets F_NOCACHE on the descriptor instead
#ifndef O_DIRECT
#define O_DIRECT 0
#define STRATA_MACOS_NOCACHE 1
#endif

// posix_fadvise: WILLNEED becomes F_RDADVISE (read-ahead hint), the rest are advisory no-ops
#ifndef POSIX_FADV_NORMAL
#define POSIX_FADV_NORMAL 0
#define POSIX_FADV_RANDOM 1
#define POSIX_FADV_SEQUENTIAL 2
#define POSIX_FADV_WILLNEED 3
#define POSIX_FADV_DONTNEED 4
#define POSIX_FADV_NOREUSE 5
static inline int posix_fadvise(int fd, off_t offset, off_t len, int advice) {
    if (advice == POSIX_FADV_WILLNEED && len > 0) {
        struct radvisory ra;
        ra.ra_offset = offset;
        ra.ra_count = len > (off_t) 0x7fffffff ? 0x7fffffff : (int) len;
        (void) fcntl(fd, F_RDADVISE, &ra);
    }
    return 0;
}
#endif

// _SC_AVPHYS_PAGES: free + inactive + purgeable + speculative pages from the Mach VM statistics
#ifndef _SC_AVPHYS_PAGES
#define _SC_AVPHYS_PAGES 0x7a11
static inline long strata_macos_avphys_pages() {
    vm_statistics64_data_t vm;
    mach_msg_type_number_t n = HOST_VM_INFO64_COUNT;
    if (host_statistics64(mach_host_self(), HOST_VM_INFO64, (host_info64_t) &vm, &n) != KERN_SUCCESS) return -1;
    return (long) (vm.free_count + vm.inactive_count + vm.purgeable_count + vm.speculative_count);
}
static inline long strata_macos_sysconf(int name) { return name == _SC_AVPHYS_PAGES ? strata_macos_avphys_pages() : sysconf(name); }
#define sysconf(n) strata_macos_sysconf(n)
#endif
#endif
