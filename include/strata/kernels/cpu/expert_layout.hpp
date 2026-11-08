// include/strata/kernels/cpu/expert_layout.hpp - plan v0.3 P6: where each routed expert lives in experts.bin.
//
// A Q2_0 pack (tools/strata_pack.py) has one blob size for every layer, `BLOB`, in the Strata expert form.  A
// native pack (tools/iq_pack.py, the IQ2_XS / IQ3_XXS files) keeps each expert's raw GGUF slices, so the blob
// size and the formats change from layer to layer; `native_experts.txt` says how.  Everything that touches an
// expert blob - the arena, the VRAM tier, the prompt path, the CPU pool, the GPU window - asks this table.
#pragma once

#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/cpu/native_expert.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace strata::kernels::cpu {

/// M4: the nine tensors of an EXL3 expert blob, in exllamav3 exl3_moe_coop's MoeCoopParams order (g_trellis, g_suh,
/// g_svh, u_trellis, ...), which is also Exl3PlanParams::field_off's order.  The bytes inside the blob are in the
/// packer's order (three trellises, then the six fp16 vectors); `ExpertLayout::exl3_field_off` maps one to the other.
enum Exl3Field {
    EXL3_G_TRELLIS, EXL3_G_SUH, EXL3_G_SVH, EXL3_U_TRELLIS, EXL3_U_SUH, EXL3_U_SVH, EXL3_D_TRELLIS, EXL3_D_SUH,
    EXL3_D_SVH, EXL3_FIELDS
};

struct ExpertLayout {
    bool native = false;
    /// M4: an EXL3 3 bpw pack (~/exl3/pack_exl3.py, exl3_experts.txt): one fixed blob for every layer, K3 MUL1
    /// trellises plus fp16 sign/scale vectors, expert ids 0..n_expert-1 of the unpruned model.
    bool exl3 = false;
    /// EXL3 packs: blob-relative byte offsets of the nine tensors (256-byte aligned), indexed by Exl3Field.
    uint32_t exl3_field_off[EXL3_FIELDS] = {};
    /// EXL3 packs with `keep` lines (a pruned model's view of a full-model experts.bin): per layer, the router's
    /// expert index -> the blob index in the file, whose layers hold `file_n_expert` blobs each.  Empty: identity.
    std::vector<std::vector<uint16_t>> remap;
    int64_t file_n_expert = 0;
    /// EXL3 mixed precision: the header's trellis bit width (3, 4 or 5), so a layer at the header's K keeps the
    /// header's offsets and a smaller-K layer in the same file stride is rebased.
    uint8_t exl3_k_header = 3;
    /// Per layer the trellis bit width (the header's K for its blob, 2 for a smaller blob in the same file stride,
    /// e.g. stored in a view's unused positions); the file's blob stride stays the header's blob.
    std::vector<uint8_t> exl3_k;
    uint64_t exl3_stride = 0;
    /// Byte offset of field f inside layer `layer`'s blob (trellises are K-sized; the fp16 vectors follow them).
    uint32_t exl3_off(int64_t layer, int f) const {
        const int k = exl3_k.empty() ? exl3_k_header : exl3_k[(size_t) layer];
        if (k == exl3_k_header) return exl3_field_off[f];
        const uint32_t t_h = 2560u * 640u * (uint32_t) exl3_k_header / 8u, tk = 2560u * 640u * (uint32_t) k / 8u;
        return f % 3 == 0 ? (uint32_t) (f / 3) * tk : exl3_field_off[f] - 3u * t_h + 3u * tk;
    }
    int exl3_bits(int64_t layer) const { return exl3_k.empty() ? exl3_k_header : exl3_k[(size_t) layer]; }
    int64_t n_layers = 0, n_expert = NE;
    std::vector<NativeFmt> fmt;           ///< per layer (native packs)
    std::vector<uint64_t> offset, bytes;  ///< per layer: where its 512 blobs start, bytes per blob
    /// Plan v0.3 P6: per layer, the absolute offsets of the gate / up / down tensors in their GGUF files, so the
    /// arena can be filled from the GGUF itself when the pack has no experts.bin (3 x n_layers, 0 = unknown).
    std::vector<uint64_t> gguf_off;
    /// Per layer PER ROLE (3 x n_layers, `3 * layer + role`, matching `gguf_off`), the GGUF file - a name beside
    /// the --native shard - that holds that role's tensor.  Empty = the --native shard itself.  Per role because a
    /// shard boundary can fall inside a layer: Unsloth's UD-Q4_K_XL has layer 11's down in shard 2 and its gate
    /// and up in shard 3 (native_experts.txt v4, the `gate,up,down` shard column).
    std::vector<std::string> gguf_file;
    int version = 0;                      ///< native_experts.txt's / exl3_experts.txt's header version (0 = none)
    uint64_t max_blob = BLOB;
    uint64_t total = 0;                   ///< experts.bin size

    uint64_t blob_bytes(int64_t layer) const { return native || exl3 ? bytes[(size_t) layer] : (uint64_t) BLOB; }
    uint64_t layer_offset(int64_t layer) const {
        return native || exl3 ? offset[(size_t) layer] : (uint64_t) layer * (uint64_t) n_expert * (uint64_t) BLOB;
    }
    uint64_t blob_offset(int64_t layer, int64_t expert) const {
        const uint64_t e = remap.empty() ? (uint64_t) expert : (uint64_t) remap[(size_t) layer][(size_t) expert];
        return layer_offset(layer) + e * (exl3 && exl3_stride ? exl3_stride : blob_bytes(layer));
    }
};

/// Plan v0.3 P6: whether this CPU (and its OS) runs the AVX-512 kernels (F, BW, VL, VNNI, VBMI).  Probed in a
/// file compiled without AVX-512, so asking is safe everywhere; STRATA_FORCE_AVX2=1 answers no (for tests).
bool cpu_avx512_ok();
/// Whether this CPU runs AVX-512F + AVX-512BW + AVX-512VL + AVX-512DQ (e.g. Skylake-X i9-9920X) for iq_avx512.cpp.
bool cpu_avx512bw_ok();
/// Whether this CPU (and its OS) runs the AVX2 kernels (AVX, AVX2, FMA, F16C): the floor of every expert kernel
/// (q2_avx2.cpp, iq_avx2.cpp, and ggml-cpu in the portable build).  STRATA_FORCE_AVX2 does not change it.
bool cpu_avx2_ok();
/// Whether this CPU (and its OS) runs AVX (Sandy Bridge and newer): the AVX1 router dot (kq_avx1.cpp) and ggml-cpu in
/// a STRATA_ISA_FLOOR=avx build.  The other expert rows then run on ggml-cpu (experimental, slow).
bool cpu_avx1_ok();
/// SSE4.2 + POPCNT: ggml-cpu in a STRATA_ISA_FLOOR=none build.
bool cpu_sse42_ok();
/// STRATA_FORCE_ISA=avx2|avx|sse caps the probes above (3 = no cap, 2 = AVX2, 1 = AVX, 0 = SSE4.2), so the older-CPU
/// paths run on any PC: tests and the experimental builds' checks.  ggml-cpu keeps what it was compiled for.
int cpu_isa_cap();
/// The experimental older-CPU build this engine is ("avx", "sse4.2"), or "" for the normal build (AVX2 floor).
const char* isa_floor_build();
/// STRATA_IQ256_GATHER, the AVX-2 i-quant kernels' gathered grid decode (iq_avx2.cpp): -1 unset (auto, see
/// cpu_gather_fast_here), 0 the scalar decode on every core, 1 the gathered one on every core.
int iq256_gather_setting();
/// Whether this CPU's performance cores gather the IQ grid entries faster than they assemble them from scalar loads:
/// an Intel CPU with AVX-VNNI, i.e. Alder Lake / Sapphire Rapids or newer, and not one of the E-core-only parts.
/// The older Intel cores gather slowly (Haswell, Broadwell) or under the Downfall (GDS) microcode (Skylake to Tiger
/// Lake), AMD Zen 2/3 gather slowly, and Zen 4/5 run the AVX-512 kernels.  STRATA_FORCE_ISA answers no.
bool cpu_gather_fast();
/// cpu_gather_fast() and the CALLING THREAD runs on a performance core: CPUID 1Ah core type 40h on a hybrid CPU (its
/// E-cores, 20h, gather slower than they assemble).  Probed once per thread: the pool pins each worker to one core.
bool cpu_gather_fast_here();
/// Whether the AVX-2 expert kernels take their AVX-VNNI forms (vpdpwssd / vpdpbusd: Alder Lake, Sapphire Rapids and
/// later, P- and E-cores alike; the same integer sums): cpu_avx2_ok() and CPUID 7.1:EAX[4], not capped by
/// STRATA_FORCE_ISA, and not an older-CPU build (STRATA_ISA_FLOOR, which stays on its floor and AVX2).
/// STRATA_NO_AVXVNNI=1 answers no.
bool cpu_avxvnni_ok();
/// Whether a native pack's layer of this ggml type runs on Strata's own Q2_0 kernels (AVX2 / AVX-512).  On a
/// CPU without AVX2 (a native, non-portable build: ggml-cpu compiled for this CPU's SSE) a Q2_0 layer takes
/// ggml-cpu's own Q2_0 vec_dot like every other native type: slower, but it runs.
inline bool q2_native_kernels(int type) { return type == 42 && cpu_avx2_ok(); }
/// The CPU's brand string (CPUID 0x80000002..4), for messages; "unknown" when it has none.
std::string cpu_name();
/// Q2_0 GGUF rows / activation quantizer on the kernels this CPU has.
void q2_rows_any(const uint8_t* w, size_t row_bytes, int nblocks, const ActQ* const* a, int nt, float* const* out,
                 int r0, int r1);
void act_quant_any(const float* x, int n, ActQ& a);
/// The same dispatch, always the pre-bit-plane kernel (tests and A/B; the bit-plane switch off by construction).
void q2_rows_any_legacy(const uint8_t* w, size_t row_bytes, int nblocks, const ActQ* const* a, int nt, float* const* out,
                        int r0, int r1);

/// The process-wide layout (canonical Q2_0 until `expert_layout_load` finds a native or EXL3 pack).
const ExpertLayout& expert_layout();
/// Reads `<pack_dir>/native_experts.txt` when it exists (a native pack), `<pack_dir>/exl3_experts.txt` when it exists
/// (an EXL3 pack), else sets the canonical layout.  A pack with both is refused, and so is one with exl3_pack.json
/// (pack_exl3.py's working manifest) but no exl3_experts.txt: the packer writes that marker last, once every layer
/// is in experts.bin, so its absence means the pack is still being converted.
/// Versions up to kExpertLayoutVersion / kExl3LayoutVersion are read; a newer one is refused (a newer packer wrote it).
bool expert_layout_load(const std::string& pack_dir, int64_t n_layers, int64_t n_expert, std::string& err);
/// The newest native_experts.txt this engine reads.  v4 = v3 plus the per-role shard column `gate,up,down`,
/// written only when some layer's roles are in different shards (every other pack stays v3, byte for byte).
/// v5 = v4 plus `hadamard.txt` beside it: Hadamard-folded experts (hadamard.hpp), written only for such models,
/// so an engine that cannot rotate the activations refuses the pack instead of running it wrong.
inline constexpr int kExpertLayoutVersion = 5;
/// The newest exl3_experts.txt this engine reads.  v1 header:
///   # strata exl3 experts v1: exl3-k3-mul1-v1 (n_expert 512, n_embd 2560, n_ff 640, K 3, cb 2, blob 1862400)
/// then one `field <name> <offset> <bytes>` line per tensor (names gate_trellis .. down_svh) and one
/// `<layer> <offset> <blob> <sha256 of the layer's section>` line per layer, contiguous from offset 0.
inline constexpr int kExl3LayoutVersion = 1;

}  // namespace strata::kernels::cpu
