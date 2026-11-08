// src/kernels/cpu/pool_affinity_mac.hpp - macOS has no thread-to-core pinning API; the pool runs unpinned there.
#pragma once
#if defined(__APPLE__)
#include <thread>
#include <vector>

namespace strata::kernels::cpu::detail {
inline int get_thread_affinity(std::vector<unsigned long>& mask, std::vector<int>* cpus = nullptr) {
    const unsigned n = std::thread::hardware_concurrency();
    mask.assign((n + 63) / 64, 0);
    for (unsigned i = 0; i < n; ++i) mask[i / 64] |= 1ul << (i % 64);
    if (cpus) { cpus->clear(); for (unsigned i = 0; i < n; ++i) cpus->push_back((int) i); }
    return 0;
}
inline int set_thread_affinity(const std::vector<unsigned long>&) { return 0; }
inline int pin_thread_to_cpu(int) { return 0; }
}  // namespace strata::kernels::cpu::detail
#endif
