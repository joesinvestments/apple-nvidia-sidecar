#pragma once
// M4: the EXL3 GPU arms (libstrata_exl3.dylib, ~/exl3/strata_m4): decode MoE for the verify window (device plan +
// exllamav3 exl3_moe_coop, graph-capture safe) and the prompt path's FP16 dequant.  Loaded with dlopen on first use
// (STRATA_EXL3_GPU_LIB, default $HOME/exl3/strata_m4/libstrata_exl3.dylib); one context per process, sized from the
// EXL3 expert layout.  The library links the same tgcu libcudart as the engine, so its kernels share the device.
#include <dlfcn.h>
#include <cstdio>
#include <vector>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdlib>
#include <string>
#include "strata/kernels/cpu/expert_layout.hpp"

namespace strata::core {

struct Exl3Gpu {
    static constexpr int kMaxRows = 8;   // verify-window rows per group the context's scratch holds
    void* ctx = nullptr;
    int (*pre)(void*, int, const float*, int, const int32_t*, const float*, int, const int32_t*, const unsigned long long*,
               unsigned long long, long long, int, const unsigned*, void*) = nullptr;
    int (*post)(void*, int, float*, int, int, void*) = nullptr;
    int (*dequant)(void*, const uint8_t*, uint16_t*, uint16_t*, int, const unsigned*, void*) = nullptr;
    int (*mask)(void*, float*, int, int, const int32_t*, int, void*) = nullptr;        // device residency row
    int (*mask_host)(void*, float*, int, int, const int32_t*, int, void*) = nullptr;   // host residency row
    // prompt path: exllamav3 exl3_moe (grouped, token-sorted) over a layer's experts in <= 5 launches
    void* pf_ctx = nullptr;
    // STRATA_EXL3_KEEP_MASK=<file>: int32 [n_layers][n_expert], >= 0 = routable (a quality probe: route among a fixed
    // per-layer expert subset, misses still go to the CPU); host copy for the prompt path, device copy for the window
    std::vector<int32_t> keep_host;
    int32_t* keep_dev = nullptr;
    int (*prefill_moe)(void*, const uint16_t*, float*, int, const int32_t*, const int32_t*, const unsigned long long*,
                       int, const unsigned*, void*) = nullptr;
    /// mixed precision: layer l's trellis bits and blob field offsets (ExpertLayout::exl3_off)
    struct LayerFmt { int bits; unsigned off[9]; };
    LayerFmt fmt(int64_t l) const {
        const auto& lay = strata::kernels::cpu::expert_layout();
        LayerFmt f{lay.exl3_bits(l), {}};
        for (int i = 0; i < 9; ++i) f.off[i] = lay.exl3_off(l, i);
        return f;
    }
};

/// STRATA_EXL3_PF=0: the prompt path dequantizes each EXL3 expert to FP16 (exl3s_dequant + GEMMs) instead of the
/// grouped exl3_moe kernel (default on).
inline bool exl3_pf_grouped() {
    static const bool on = [] { const char* v = std::getenv("STRATA_EXL3_PF"); return !(v && v[0] == '0'); }();
    return on;
}

/// STRATA_EXL3_RESIDENT_ONLY=1: route among the VRAM-resident experts only (the router's logits of the others -> -inf
/// before the model's own top-k) in the verify window and the prompt path - a prune to the resident set with the router
/// renormalised, so no expert runs on the CPU and nothing streams.  Changes the output (a smaller expert set).
inline bool exl3_resident_only() {
    static const bool on = [] { const char* v = std::getenv("STRATA_EXL3_RESIDENT_ONLY"); return v && v[0] == '1'; }();
    return on;
}

/// The process's EXL3 arms, or nullptr with `err` set (not an EXL3 layout, library missing, context failed).
inline Exl3Gpu* exl3_gpu(std::string& err) {
    static std::string load_err;
    static Exl3Gpu* g = [] () -> Exl3Gpu* {
        const auto& lay = strata::kernels::cpu::expert_layout();
        if (!lay.exl3) { load_err = "the expert layout is not EXL3"; return nullptr; }
        const char* e = std::getenv("STRATA_EXL3_GPU_LIB");
        const char* home = std::getenv("HOME");
        const std::string path = e ? e : std::string(home ? home : "") + "/exl3/strata_m4/libstrata_exl3.dylib";
        void* h = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (!h) { load_err = "dlopen " + path + ": " + dlerror(); return nullptr; }
        using Create = void* (*)(int, int, int, int, int, const unsigned*, unsigned long long, char*, int);
        auto create = (Create) dlsym(h, "exl3s_create");
        auto* r = new Exl3Gpu;
        r->pre = (decltype(r->pre)) dlsym(h, "exl3s_dec_pre");
        r->post = (decltype(r->post)) dlsym(h, "exl3s_dec_post");
        r->dequant = (decltype(r->dequant)) dlsym(h, "exl3s_dequant");
        r->mask = (decltype(r->mask)) dlsym(h, "exl3s_mask");
        r->mask_host = (decltype(r->mask_host)) dlsym(h, "exl3s_mask_host");
        using PfCreate = void* (*)(int, int, const unsigned*, char*, int);
        auto pf_create = (PfCreate) dlsym(h, "exl3s_pf_create");
        r->prefill_moe = (decltype(r->prefill_moe)) dlsym(h, "exl3s_prefill_moe");
        if (!create || !r->pre || !r->post || !r->dequant || !r->mask || !r->mask_host || !pf_create || !r->prefill_moe) { load_err = path + ": missing exl3s_* symbols"; return nullptr; }
        unsigned off[strata::kernels::cpu::EXL3_FIELDS];
        for (int i = 0; i < strata::kernels::cpu::EXL3_FIELDS; ++i) off[i] = lay.exl3_field_off[i];
        char buf[256] = {};
        const auto& geo = strata::core::kCompiledGeometry;   // the layout the EXL3 arms and the pack header are built for
        r->ctx = create((int) lay.n_expert, (int) geo.n_embd, (int) geo.n_ff, (int) geo.n_expert_used,
                        Exl3Gpu::kMaxRows, off, lay.max_blob, buf, (int) sizeof buf);
        if (!r->ctx) { load_err = buf; return nullptr; }
        // rows: up to an 8192-token chunk x top-10 (the scratch is mostly fixed-size: ~37 MB)
        r->pf_ctx = pf_create(8192 * 10, (int) lay.n_expert, off, buf, (int) sizeof buf);
        if (!r->pf_ctx) { load_err = buf; return nullptr; }
        if (const char* km = std::getenv("STRATA_EXL3_KEEP_MASK")) {
            const size_t n = (size_t) lay.n_layers * (size_t) lay.n_expert;
            r->keep_host.resize(n);
            FILE* f = std::fopen(km, "rb");
            if (!f || std::fread(r->keep_host.data(), 4, n, f) != n) { load_err = std::string("cannot read ") + km; if (f) std::fclose(f); return nullptr; }
            std::fclose(f);
            if (cudaMalloc((void**) &r->keep_dev, n * 4) != cudaSuccess ||
                cudaMemcpy(r->keep_dev, r->keep_host.data(), n * 4, cudaMemcpyHostToDevice) != cudaSuccess) {
                load_err = "keep mask upload"; return nullptr;
            }
            std::fprintf(stderr, "strata exl3: keep mask %s loaded (quality probe)\n", km);
        }
        return r;
    }();
    if (!g) err = "EXL3 GPU arms: " + load_err;
    return g;
}

}  // namespace strata::core
