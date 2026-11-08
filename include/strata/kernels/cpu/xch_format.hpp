// include/strata/kernels/cpu/xch_format.hpp - which blob format a pack has, for the exchange window's host side.
//
// The Mac library's lane reads one kind of blob at a time.  An EXL3 pack (exl3_experts.txt) is the trellis blob the library has always
// decoded; an all-NVFP4 native pack (tools/nvfp4_local_pack.py, tools/iq_pack.py: every layer's gate/up and down a ggml block_nvfp4
// tensor, 2,764,816 B per expert with the 16-byte scale tail, no Hadamard rotation) runs the library's NVFP4 kernels instead.  The lane
// asks this table before it starts and tells the library the answer (exl3m_xch_set_format); a pack that is neither is refused,
// because the EXL3 kernels decode such blobs as trellises and report no error.
#pragma once

#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/native_expert.hpp"

#include <string>

namespace strata::kernels::cpu {

/// The NVFP4 blob the Mac library reads (exl3-metal/exl3_metal.mm's NVFP4_BLOB): gate | up | down of block_nvfp4 rows, then the scale tail.
inline constexpr size_t kXchNvfp4Blob = 2764816;

/// 0 for an EXL3 pack, 1 for a native pack whose every layer is an NVFP4 blob the Mac lane reads, -1 for anything else with the reason
/// (naming the layer that does not fit) in `why`.
inline int xch_blob_format(const ExpertLayout& L, std::string& why) {
    why.clear();
    if (L.exl3) return 0;
    if (!L.native) {
        why = "the pack is neither an EXL3 pack nor a native pack";
        return -1;
    }
    if (L.n_layers <= 0 || L.fmt.size() != (size_t) L.n_layers) {
        why = "the pack has " + std::to_string(L.fmt.size()) + " layer formats for " + std::to_string(L.n_layers) + " layers";
        return -1;
    }
    for (int64_t l = 0; l < L.n_layers; ++l) {
        const NativeFmt& f = L.fmt[(size_t) l];
        const std::string at = "layer " + std::to_string(l) + ": ";
        if (f.gu_type != kNvfp4Type || f.d_type != kNvfp4Type) {
            why = at + "gate/up type " + std::to_string(f.gu_type) + " and down type " + std::to_string(f.d_type) +
                  " are not NVFP4 (" + std::to_string(kNvfp4Type) + ")";
            return -1;
        }
        if (f.had_block != 0) {
            why = at + "the experts are Hadamard-rotated (had_block " + std::to_string(f.had_block) + ")";
            return -1;
        }
        if (f.bytes != kXchNvfp4Blob) {
            why = at + "a blob of " + std::to_string((unsigned long long) f.bytes) + " B, the lane reads " +
                  std::to_string((unsigned long long) kXchNvfp4Blob) + " B";
            return -1;
        }
        if (f.tail_off == 0) {
            why = at + "the blob has no per-expert NVFP4 scale tail";
            return -1;
        }
    }
    return 1;
}

}  // namespace strata::kernels::cpu
