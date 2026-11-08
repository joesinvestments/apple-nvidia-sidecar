// exl3_metal.mm - host side of libexl3metal (see exl3_metal.h). Objective-C++ internally, plain C API.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <mach/mach_time.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <thread>
#include <vector>
#include "exl3_metal.h"
#include <exl3_metallib.h>   // exl3_metallib[] / exl3_metallib_len (xxd -i of the offline-compiled metallib); angle brackets: the
                             // generated header lives in the build dir (-I"$OUT" in build.sh) and must win over a stale copy here
#include "../include/strata/xch/xch_layout.h"   // the GPU <-> Metal exchange region (docs/strata-exchange-map.md 3.3)

namespace {
constexpr size_t PAGE = 16384;
constexpr int NU_MAX = 96, PAIR_MAX = 4096;
// pf_short_tok accepts up to PF_SHORT_NV4, where the NVFP4 short path meets the batched lane (measured on our test machine, K=8:
// 256 7.8 vs 29.9 ms, 512 9.9 vs 30.1, 1024 17.6 vs 32.2, 1536 24.6 vs 31.9, 2048 31.9 vs 31.7); EXL3 keeps its own 256
constexpr int PF_SHORT_EXL3 = 256, PF_SHORT_NV4 = 2048;
constexpr size_t PIECE_MAX = (size_t)1 << 30;
constexpr size_t NVFP4_BLOB = 2764816;   // one NVFP4 expert blob, 2764816 B at every trellis width (the register overlap covers it)
constexpr uint32_t H = EXL3M_H, FF = EXL3M_FF;
// decode table buffer layout
constexpr size_t T_UNITS = 0, T_PAIRS = 4096, T_ROWPTR = 4096 + (size_t)PAIR_MAX * 16, T_END = T_ROWPTR + 64;
// exchange window table buffer layout (one set: the window's encoder is serial)
constexpr size_t XT_UNITS = 0, XT_PAIRS = 96 * 40, XT_ROWPTR = XT_PAIRS + (size_t)XCH_MAX_PAIRS * 16, XT_END = XT_ROWPTR + 64;

thread_local std::string g_err;
int fail(const std::string &m, int code = -1) { g_err = m; return code; }
double now_ms() {
    static mach_timebase_info_data_t tb;
    if (!tb.denom) mach_timebase_info(&tb);
    return (double)mach_absolute_time() * tb.numer / tb.denom / 1e6;
}

struct UnitDH { uint64_t blob; int32_t rows[8]; };            // = MSL UnitD (40 B)
struct PairDH { uint32_t u, slot; float w; uint32_t pad; };
struct DPH { uint32_t mrx, nu, bits, p0; };
struct GPH { uint32_t kind, p[3]; };
struct XWH { uint32_t layer, epoch, budget, n_experts, n_rows, mrx, bits, p1; };   // = MSL XW (32 B)
struct UnitPH { uint64_t blob; uint32_t row0, nrows; };       // = MSL UnitP (16 B)
struct PPH { uint32_t nrows, cap, bits, p0; };
struct DQH { uint32_t which, bits, p0, p1; };                  // = MSL DQ (16 B)
struct GTH { uint32_t u, r0, nv, pad; };
struct CPH { uint32_t pr; float w; };

struct Piece { uintptr_t base, end; id<MTLBuffer> buf; };
struct Region { uintptr_t key; std::vector<Piece> pieces; };

struct Slot {
    bool busy = false;
    id<MTLBuffer> d_x, d_tbl, d_xq, d_yg, d_zq, d_yd, d_out;   // decode scratch
    id<MTLBuffer> nv_z, nv_stage; size_t nv_stage_n = 0;         // NVFP4 scratch: the fp32 mid rows, and staging for unregistered blobs
    id<MTLBuffer> stage; size_t stage_blobs = 0;                 // staging for unregistered blobs
    id<MTLBuffer> p_x, p_tbl, p_A, p_C, p_Z, p_Yd, p_Wg, p_Wu, p_Wd, p_y, p_yh;   // prefill scratch (grow only); p_yh = the fp16 result rows (half-I/O jobs)
    size_t p_x_n = 0, p_tbl_n = 0, p_cap = 0, p_wg_n = 0, p_y_n = 0, p_yh_n = 0;
    id<MTLBuffer> p_nvst; size_t p_nvst_n = 0;   // staging for NVFP4 prompt blobs no registered region holds (blobs, not bytes)
    id<MTLBuffer> s_x, s_tbl, s_out, s_yh, s_yd;   // short-prompt scratch: the decode kernels over the prompt's pairs (grow only); s_yd = every unit's rows
    size_t s_x_n = 0, s_tbl_n = 0, s_out_n = 0, s_yh_n = 0, s_yd_n = 0;
};

struct Pipes { id<MTLComputePipelineState> rot_dec, mid_dec, comb_dec, gemv[6][9], rot_pf, rot_pf_h, y_f2h, decode_w, gemm_gu, gemm_d, mid_pf, comb_pf;
               id<MTLComputePipelineState> decode_w_plain, gemm_gu_t, gemm_d_t, decode_w_nv4_plain;   // pf_tensor_ops = 1 (nil if the device cannot build them)
               id<MTLComputePipelineState> xw_wait, rot_dec_sys, comb_dec_sys, xw_land;
               // NVFP4 cold experts ([SG 4 or 2][mrx]); adds the window's D1 (x fp16 -> fp32) and D5 (comb -> slot)
               id<MTLComputePipelineState> nv4_gemv[2][9], nv4_mid, nv4_comb, nv4_x, nv4_comb_sys;
               // the NVFP4 prompt lane's stages (weight decode, chunk rows, mid with the tail scales, combine)
               id<MTLComputePipelineState> decode_w_nv4, x_pf_nv4, x_pf_nv4_h, mid_pf_nv4, comb_pf_nv4; };
}  // namespace

struct exl3m_ctx {
    id<MTLDevice> dev;
    id<MTLCommandQueue> q;
    Pipes P;
    std::shared_mutex rmu;
    std::vector<Region> regions;
    std::mutex smu;
    std::condition_variable scv;
    std::vector<Slot> slots;
    std::mutex stmu;
    exl3m_stats st{};
    std::string dname;
    int pf_chunk_rows = 4096, pf_chunk_experts = 16;
    int pf_short_tok = 0;   // EXL3M_PF_SHORT_TOK: a prompt of at most this many tokens runs through the decode kernels (0 = off; 0..2048:
                            // the NVFP4 lane's measured crossover; an EXL3 prompt uses it up to 256 only)
    int tensor_ops = 0;   // pf_tensor_ops: 0 = the simdgroup_matrix GEMMs (default), 1 = the matmul2d ones (needs the pipelines below)
    int bits = 3;         // the trellis width K of the blobs this context reads ("bits", 3, 4 or 5; default 3)
    size_t blob = EXL3M_BLOB_BYTES;   // one expert's blob at bits: 614400*bits + 19200
    int wait_spin = 0, unretained = 0;   // decode latency knobs: poll the command buffer status instead of waitUntilCompleted; unretained references
    int64_t skip_mask = 0;   // profiling only: bit i skips prefill kernel i (rot, decode_w x3, gemm_gu, mid, gemm_d, comb)
    // exchange window (exl3m_xch_*): the region, the expert table, one set of scratch buffers and the window in flight
    id<MTLBuffer> xch_buf, xch_tbl, xch_dp, xch_args, xch_tab, xch_dead, xch_xq, xch_yg, xch_zq, xch_yd;
    id<MTLBuffer> xch_nx, xch_nz;   // the NVFP4 window's scratch (the fp32 x rows and the fp32 mid rows), fmt 1 only
    id<MTLCommandBuffer> xch_cb;
    void *xch_region = nullptr;
    size_t xch_len = 0;
    int xch_nl = 0, xch_ne = 0;
    int xch_fmt = 0;   // 0 = EXL3 trellis blobs (the window as it has been), 1 = NVFP4 blobs (2,764,816 bytes)
    std::vector<id<MTLBuffer>> xch_used;   // the table's pieces plus every registered piece (the fallback path's useResource list)
    // macOS 15: one residency set on the queue holds every allocation a window touches, so the window marks no resource itself
    id xch_rs = nil;              // the queue's id<MTLResidencySet> (macOS 15); kept as id so the field carries no availability
    NSUInteger xch_rs_n = 0;      // its allocation count; 0 = not in step with the regions any more, window_submit builds it again
    bool xch_rs_scratch = false;  // the set already holds the window's scratch buffers
    bool xch_rs_logged = false, xch_rs_off = false;   // the once-only size line; no residency sets here (macOS < 15, or refused)
    // EXL3M_KEEPWARM_MS: a background thread keeps the queue busy while idle (see exl3m_create)
    std::atomic<double> last_commit_ms{0};   // when a command buffer was last committed on c->q
    std::atomic<bool> kw_stop{false};
    std::thread kw;
    id<MTLBuffer> kw_buf = nil;              // the keep-warm thread's own tiny buffer (no slot, no job, no caller buffer)
};

struct exl3m_job {
    Slot *slot;
    id<MTLCommandBuffer> cb;
    void *y_out;
    size_t y_bytes;
    id<MTLBuffer> src;       // shared buffer holding the result
    double t_submit, t_encode, t_copy;
    bool dec = false;         // a decode-shape job (the only kind the spin wait applies to)
};

namespace {
id<MTLComputePipelineState> mkpipe(exl3m_ctx *c, id<MTLLibrary> lib, NSString *name) {
    NSError *err = nil;
    id<MTLFunction> f = [lib newFunctionWithName:name];
    if (!f) { g_err = std::string("missing kernel ") + name.UTF8String; return nil; }
    id<MTLComputePipelineState> p = [c->dev newComputePipelineStateWithFunction:f error:&err];
    if (!p) g_err = std::string("pipeline ") + name.UTF8String + ": " + (err ? err.localizedDescription.UTF8String : "?");
    return p;
}

id<MTLBuffer> mkbuf(exl3m_ctx *c, size_t n, MTLResourceOptions o = MTLResourceStorageModeShared) {
    return [c->dev newBufferWithLength:(n + 255) & ~(size_t)255 options:o];
}
const MTLResourceOptions PRIV = MTLResourceStorageModePrivate;

Slot *acquire(exl3m_ctx *c) {
    std::unique_lock<std::mutex> l(c->smu);
    for (;;) {
        for (auto &s : c->slots) if (!s.busy) { s.busy = true; return &s; }
        c->scv.wait(l);
    }
}
void release(exl3m_ctx *c, Slot *s) {
    { std::lock_guard<std::mutex> l(c->smu); s->busy = false; }
    c->scv.notify_one();
}

// Resolve a blob pointer to (GPU address, buffer). Returns false if not covered by a registered piece.
bool resolve(exl3m_ctx *c, const void *p, uint64_t &addr, id<MTLBuffer> __strong &buf, size_t n = 0) {
    uintptr_t a = (uintptr_t)p;
    if (!n) n = c->blob;   // an NVFP4 blob is 2764816 B at every trellis width, so the caller passes its length
    std::shared_lock<std::shared_mutex> l(c->rmu);
    for (auto &r : c->regions)
        for (auto &pc : r.pieces)
            if (a >= pc.base && a + n <= pc.end) { addr = pc.buf.gpuAddress + (a - pc.base); buf = pc.buf; return true; }
    return false;
}

void use_all(id<MTLComputeCommandEncoder> en, const std::vector<id<MTLBuffer>> &bufs) {
    std::vector<id<MTLBuffer>> u;
    for (auto &b : bufs) { bool dup = false; for (auto &x : u) if (x == b) { dup = true; break; } if (!dup) u.push_back(b); }
    for (auto &b : u) [en useResource:b usage:MTLResourceUsageRead];
}

// macOS 15: one residency set on the queue holds every allocation the exchange window touches (the region, the expert table, the
// table's pieces and the window's scratch), so the window does not mark the registered pieces resident itself. Older macOS, or a
// device that refuses the set, leaves c->xch_rs nil and the window keeps the useResource path.
void xch_rs_stale(exl3m_ctx *c) { if (c->xch_rs) c->xch_rs_n = 0; }   // a region changed: window_submit rebuilds the set

void xch_rs_sync(exl3m_ctx *c) {
    if (c->xch_rs_off) return;
    if (@available(macOS 15.0, *)) {
        std::vector<id<MTLBuffer>> bufs;
        auto add = [&bufs](id<MTLBuffer> b) {
            if (!b) return;
            for (auto &x : bufs) if (x == b) return;   // a table blob is usually one of the registered pieces too
            bufs.push_back(b);
        };
        add(c->xch_buf); add(c->xch_tbl);
        { std::shared_lock<std::shared_mutex> l(c->rmu); for (auto &r : c->regions) for (auto &pc : r.pieces) add(pc.buf); }
        add(c->xch_dp); add(c->xch_args); add(c->xch_tab); add(c->xch_dead);
        add(c->xch_xq); add(c->xch_yg); add(c->xch_zq); add(c->xch_yd); add(c->xch_nx); add(c->xch_nz);
        const bool scratch = c->xch_dp != nil;
        if (c->xch_rs && c->xch_rs_n == (NSUInteger)bufs.size() && (c->xch_rs_scratch || !scratch)) return;   // unchanged
        if (c->xch_rs) { [c->q removeResidencySet:c->xch_rs]; c->xch_rs = nil; }                              // drop the old one first
        MTLResidencySetDescriptor *d = [MTLResidencySetDescriptor new];
        d.label = @"exl3m exchange";
        d.initialCapacity = bufs.size();
        NSError *err = nil;
        id<MTLResidencySet> s = [c->dev newResidencySetWithDescriptor:d error:&err];
        if (!s) { c->xch_rs_off = true; return; }   // no residency sets on this device: keep the per-window useResource path
        for (auto &b : bufs) [s addAllocation:b];
        [s commit];
        [s requestResidency];
        [c->q addResidencySet:s];
        c->xch_rs = s; c->xch_rs_n = (NSUInteger)bufs.size(); c->xch_rs_scratch = scratch;
        if (scratch && !c->xch_rs_logged) {
            uint64_t bytes = 0;
            for (auto &b : bufs) bytes += (uint64_t)b.length;
            fprintf(stderr, "exl3m: exchange residency set: %zu allocations, %.2f GB\n", bufs.size(), (double)bytes / 1e9);
            c->xch_rs_logged = true;
        }
        return;
    }
    c->xch_rs_off = true;   // macOS < 15: no residency sets, the window keeps the useResource path
}

// the fallback, as a decode job does it: the table's pieces, the region and the expert table become resident for this window
void xch_use_resources(exl3m_ctx *c, id<MTLComputeCommandEncoder> en) {
    use_all(en, c->xch_used);
    [en useResource:c->xch_buf usage:MTLResourceUsageRead | MTLResourceUsageWrite];
    [en useResource:c->xch_tbl usage:MTLResourceUsageRead];
}

void ensure_stage(exl3m_ctx *c, Slot *s, size_t nblobs) {
    if (s->stage_blobs >= nblobs) return;
    s->stage = mkbuf(c, nblobs * c->blob);
    s->stage_blobs = nblobs;
}
void grow(exl3m_ctx *c, id<MTLBuffer> __strong &b, size_t &cur, size_t need, MTLResourceOptions o) {
    if (cur >= need) return;
    size_t n = std::max(need, cur + cur / 2);
    b = mkbuf(c, n, o); cur = n;
}
}  // namespace

// ---------------------------------------------------------------------------------------------------------------------
extern "C" {

const char *exl3m_last_error(void) { return g_err.c_str(); }
const char *exl3m_device_name(exl3m_ctx *c) { return c->dname.c_str(); }

exl3m_ctx *exl3m_create(int n_threads, int n_slots) {
    (void)n_threads;
    @autoreleasepool {
        exl3m_ctx *c = new exl3m_ctx();
        c->dev = MTLCreateSystemDefaultDevice();
        if (!c->dev) { g_err = "no Metal device"; delete c; return nullptr; }
        c->dname = c->dev.name.UTF8String;
        c->q = [c->dev newCommandQueue];
        NSError *err = nil;
        dispatch_data_t dd = dispatch_data_create(exl3_metallib, exl3_metallib_len, nil, DISPATCH_DATA_DESTRUCTOR_DEFAULT);
        id<MTLLibrary> lib = [c->dev newLibraryWithData:dd error:&err];
        if (!lib) { g_err = std::string("metallib: ") + (err ? err.localizedDescription.UTF8String : "?"); delete c; return nullptr; }
        Pipes &P = c->P;
        P.rot_dec = mkpipe(c, lib, @"rot_in_dec"); P.mid_dec = mkpipe(c, lib, @"mid_dec"); P.comb_dec = mkpipe(c, lib, @"comb_dec");
        for (int kb = 3; kb <= 5; kb++)   // gemv_dec is specialised per trellis width: P.gemv[bits][mrx]
            for (int m = 1; m <= 8; m *= 2) P.gemv[kb][m] = mkpipe(c, lib, [NSString stringWithFormat:@"gemv_dec_%d_k%d", m, kb]);
        P.rot_pf = mkpipe(c, lib, @"rot_in_pf"); P.decode_w = mkpipe(c, lib, @"decode_w"); P.gemm_gu = mkpipe(c, lib, @"gemm_gu");
        P.gemm_d = mkpipe(c, lib, @"gemm_d"); P.mid_pf = mkpipe(c, lib, @"mid_pf"); P.comb_pf = mkpipe(c, lib, @"comb_pf");
        P.rot_pf_h = mkpipe(c, lib, @"rot_in_pf_h"); P.y_f2h = mkpipe(c, lib, @"y_f2h");
        P.decode_w_plain = mkpipe(c, lib, @"decode_w_plain"); P.gemm_gu_t = mkpipe(c, lib, @"gemm_gu_t"); P.gemm_d_t = mkpipe(c, lib, @"gemm_d_t");
        P.decode_w_nv4_plain = mkpipe(c, lib, @"decode_w_nv4_plain");   // the NVFP4 prompt lane on the tensor-op GEMMs
        P.xw_wait = mkpipe(c, lib, @"xw_wait_build"); P.rot_dec_sys = mkpipe(c, lib, @"rot_in_dec_sys");
        P.comb_dec_sys = mkpipe(c, lib, @"comb_dec_sys"); P.xw_land = mkpipe(c, lib, @"xw_land");
        // NVFP4 cold experts. Their absence only fails exl3m_moe_rows_nvfp4, not the context (the EXL3 paths never use them)
        P.nv4_mid = mkpipe(c, lib, @"mid_nv4"); P.nv4_comb = mkpipe(c, lib, @"comb_nv4");
        P.nv4_x = mkpipe(c, lib, @"x_f32_sys"); P.nv4_comb_sys = mkpipe(c, lib, @"comb_nv4_sys");   // the NVFP4 window's D1 and D5
        for (int m = 1; m <= 8; m *= 2) {
            P.nv4_gemv[0][m] = mkpipe(c, lib, [NSString stringWithFormat:@"gemv_nv4_%d", m]);
            P.nv4_gemv[1][m] = mkpipe(c, lib, [NSString stringWithFormat:@"gemv_nv4_dn%d", m]);
        }
        // the NVFP4 prompt lane's own stages (exl3m_moe_prefill_nvfp4 fails without them, as the decode kernels above)
        P.decode_w_nv4 = mkpipe(c, lib, @"decode_w_nv4"); P.mid_pf_nv4 = mkpipe(c, lib, @"mid_pf_nv4");
        P.comb_pf_nv4 = mkpipe(c, lib, @"comb_pf_nv4"); P.x_pf_nv4 = mkpipe(c, lib, @"x_pf_nv4"); P.x_pf_nv4_h = mkpipe(c, lib, @"x_pf_nv4_h");
        bool gemvs = true;   // every (K, MR) pair the host can pick: P.gemv[3][1] .. P.gemv[5][8]
        for (int kb = 3; kb <= 5; kb++) for (int m = 1; m <= 8; m *= 2) gemvs = gemvs && P.gemv[kb][m] != nil;
        if (!P.rot_dec || !P.mid_dec || !P.comb_dec || !gemvs || !P.rot_pf || !P.rot_pf_h || !P.y_f2h ||
            !P.decode_w || !P.gemm_gu || !P.gemm_d || !P.mid_pf || !P.comb_pf || !P.xw_wait || !P.rot_dec_sys || !P.comb_dec_sys || !P.xw_land) { delete c; return nullptr; }
        if (!P.decode_w_plain || !P.gemm_gu_t || !P.gemm_d_t || !P.decode_w_nv4_plain) {   // no tensor ops here (device or OS): creation still works, pf_tensor_ops 1 is refused
            P.decode_w_plain = P.gemm_gu_t = P.gemm_d_t = P.decode_w_nv4_plain = nil;
            g_err.clear();
        }
        if (n_slots <= 0) n_slots = 2;
        if (n_slots > 8) n_slots = 8;
        if (const char *e = getenv("EXL3M_WAIT_SPIN")) c->wait_spin = atoi(e);        // decode jobs: poll the command buffer instead of waitUntilCompleted
        if (const char *e = getenv("EXL3M_UNRETAINED")) c->unretained = atoi(e);
        if (const char *e = getenv("EXL3M_PF_CHUNK_ROWS")) { if (atoi(e) >= 64) c->pf_chunk_rows = atoi(e); }          // prefill scratch size (rows per chunk) ...
        if (const char *e = getenv("EXL3M_PF_CHUNK_EXPERTS")) { if (atoi(e) >= 1) c->pf_chunk_experts = atoi(e); }     // ... and decoded-weight experts per chunk: smaller = less Metal memory
        if (const char *e = getenv("EXL3M_PF_SHORT_TOK")) { if (atoi(e) >= 0 && atoi(e) <= PF_SHORT_NV4) c->pf_short_tok = atoi(e); }   // short prompts through the decode kernels
        c->slots.resize((size_t)n_slots);
        for (auto &s : c->slots) {
            s.d_x = mkbuf(c, 8 * (size_t)H * 4);
            s.d_tbl = mkbuf(c, T_END);
            s.d_xq = mkbuf(c, 2 * (size_t)NU_MAX * 8 * H * 2, PRIV);
            s.d_yg = mkbuf(c, 2 * (size_t)NU_MAX * 8 * FF * 4, PRIV);
            s.d_zq = mkbuf(c, (size_t)NU_MAX * 8 * FF * 2, PRIV);
            s.d_yd = mkbuf(c, (size_t)NU_MAX * 8 * H * 4, PRIV);
            s.d_out = mkbuf(c, 8 * (size_t)H * 4);
            s.nv_z = mkbuf(c, (size_t)NU_MAX * 8 * FF * 4, PRIV);   // NVFP4 mid rows, fp32
        }
        // keep-warm (EXL3M_KEEPWARM_MS=N, N > 0): the queue's residency set lapses when it is idle and the first job after that
        // waited ~240 ms, so while no command buffer has been committed for N ms the thread commits a tiny one of its own
        double kw_period = 0;
        if (const char *e = getenv("EXL3M_KEEPWARM_MS")) kw_period = atof(e);
        if (kw_period > 0) {
            c->kw_buf = mkbuf(c, 256, PRIV);
            c->last_commit_ms = now_ms();
            c->kw = std::thread([c, kw_period] {
                const double step = std::max(10.0, kw_period / 4);
                while (!c->kw_stop) {
                    std::this_thread::sleep_for(std::chrono::milliseconds((long long)step));
                    if (now_ms() - c->last_commit_ms < kw_period) continue;
                    @autoreleasepool {
                        id<MTLCommandBuffer> kb = [c->q commandBuffer];
                        id<MTLBlitCommandEncoder> bl = [kb blitCommandEncoder];
                        [bl fillBuffer:c->kw_buf range:NSMakeRange(0, 256) value:0];
                        [bl endEncoding];
                        [kb commit];   // not waited on: the commit alone keeps the residency set warm
                        c->last_commit_ms = now_ms();
                    }
                }
            });
        }
        return c;
    }
}

void exl3m_destroy(exl3m_ctx *c) {
    if (c) {
        @autoreleasepool {
            c->kw_stop = true;                        // stop the keep-warm thread before the queue and the context go away
            if (c->kw.joinable()) c->kw.join();
            delete c;
        }
    }
}

int64_t exl3m_register(exl3m_ctx *c, const void *base, size_t len) {
    @autoreleasepool {
        uintptr_t b = ((uintptr_t)base + PAGE - 1) & ~(uintptr_t)(PAGE - 1), e = ((uintptr_t)base + len) & ~(uintptr_t)(PAGE - 1);
        if (e <= b) return fail("region smaller than one page");
        // the pieces are VM pages, so a region that ends inside its last page wraps that page to the end: a blob ending at base + len
        // then still lies inside one piece (NVFP4 blobs are 2,764,816 B, so 512 of them end half a page past the core's end)
        uintptr_t we = ((uintptr_t)base + len + PAGE - 1) & ~(uintptr_t)(PAGE - 1);
        Region r; r.key = (uintptr_t)base;
        // the overlap covers the largest blob the context may resolve (EXL3 at any K, and NVFP4), so a blob that straddles a
        // piece boundary lies wholly inside the next piece (an EXL3 sized overlap left NVFP4 blobs unresolvable)
        size_t overlap = (std::max(c->blob, NVFP4_BLOB) + PAGE - 1) & ~(PAGE - 1);
        uintptr_t pos = b;
        size_t maxlen = std::min<size_t>(PIECE_MAX, c->dev.maxBufferLength & ~(PAGE - 1));
        if (const char *pm = getenv("EXL3M_PIECE_MB")) maxlen = std::min<size_t>(maxlen, ((size_t)atoll(pm) << 20) & ~(PAGE - 1));   // test hook
        while (pos < we) {
            size_t n = std::min<size_t>(we - pos, maxlen);
            id<MTLBuffer> buf = [c->dev newBufferWithBytesNoCopy:(void *)pos length:n
                                  options:MTLResourceStorageModeShared | MTLResourceHazardTrackingModeUntracked deallocator:nil];
            if (!buf) return fail("newBufferWithBytesNoCopy failed (len " + std::to_string(n) + ")");
            r.pieces.push_back({pos, pos + n, buf});
            if (pos + n >= we) break;
            pos += n - overlap;
        }
        std::unique_lock<std::shared_mutex> l(c->rmu);
        c->regions.push_back(std::move(r));
        xch_rs_stale(c);
        if (getenv("EXL3M_LOG_ALLOC")) fprintf(stderr, "exl3m: registered %.2f GB; Metal allocated %.2f GB of recommended %.2f GB\n", (double)(e - b) / 1e9, (double)c->dev.currentAllocatedSize / 1e9, (double)c->dev.recommendedMaxWorkingSetSize / 1e9);
        return (int64_t)(e - b);
    }
}

int exl3m_unregister(exl3m_ctx *c, const void *base) {
    std::unique_lock<std::shared_mutex> l(c->rmu);
    for (size_t i = 0; i < c->regions.size(); i++)
        if (c->regions[i].key == (uintptr_t)base) { c->regions.erase(c->regions.begin() + (long)i); xch_rs_stale(c); return 0; }
    return fail("region not registered");
}

int exl3m_set_param(exl3m_ctx *c, const char *key, int64_t v) {
    if (!strcmp(key, "bits")) {   // the trellis width K: the blob size and every field offset follow it (docs/exl3-kbits-map.md)
        if (v != 3 && v != 4 && v != 5) return fail("bits must be 3, 4 or 5");
        c->bits = (int)v;
        c->blob = EXL3M_BLOB_BYTES_K(v);
        return 0;
    }
    if (!strcmp(key, "pf_tensor_ops")) {
        if (v != 0 && v != 1) return fail("pf_tensor_ops must be 0 or 1");
        if (v == 1 && (!c->P.gemm_gu_t || !c->P.gemm_d_t || !c->P.decode_w_plain)) return fail("the tensor-op pipelines did not build on this device");
        c->tensor_ops = (int)v; return 0;
    }
    if (!strcmp(key, "pf_chunk_rows") && v >= 64) { c->pf_chunk_rows = (int)v; return 0; }
    if (!strcmp(key, "pf_chunk_experts") && v >= 1) { c->pf_chunk_experts = (int)v; return 0; }
    if (!strcmp(key, "pf_short_tok")) {   // a prompt of at most this many tokens runs through the decode kernels (0 = off)
        if (v < 0 || v > PF_SHORT_NV4) return fail("pf_short_tok must be 0..2048");
        c->pf_short_tok = (int)v; return 0;
    }
    if (!strcmp(key, "wait_spin")) { c->wait_spin = (int)v; return 0; }
    if (!strcmp(key, "unretained")) { c->unretained = (int)v; return 0; }
    if (!strcmp(key, "debug_skip_mask")) { c->skip_mask = v; return 0; }
    return -1;
}
void exl3m_get_stats(exl3m_ctx *c, exl3m_stats *out) { std::lock_guard<std::mutex> l(c->stmu); *out = c->st; }
int64_t exl3m_blob_bytes(exl3m_ctx *c) { return (int64_t)c->blob; }
// Metal memory accounting: bytes this process holds as Metal resources vs the device's recommended working set (HostTailAndPrefill 2026-10-12)
int64_t exl3m_alloc_bytes(exl3m_ctx *c) { return (int64_t)c->dev.currentAllocatedSize; }
int64_t exl3m_recommended_bytes(exl3m_ctx *c) { return (int64_t)c->dev.recommendedMaxWorkingSetSize; }

// ---------------------------------------------------------------------------------------------------------------------
int exl3m_moe_rows_submit(exl3m_ctx *c, const float *x, int m, int n_pairs, const int32_t *eidx, const int32_t *ridx, const float *w,
                          const void *const *blobs, float *y_out, exl3m_job **jobp) {
    @autoreleasepool {
        if (m < 1 || m > 8) return fail("m must be 1..8");
        if (n_pairs < 1 || n_pairs > PAIR_MAX) return fail("n_pairs must be 1..4096");
        const double t0 = now_ms();
        int32_t emax = -1;
        for (int p = 0; p < n_pairs; p++) {
            if (eidx[p] < 0 || ridx[p] < 0 || ridx[p] >= m) return fail("bad expert/row index");
            emax = std::max(emax, eidx[p]);
        }
        std::vector<int> unit_of((size_t)emax + 1, -1);
        std::vector<UnitDH> units; std::vector<int> cnt; std::vector<int32_t> uexp;
        std::vector<PairDH> pairs((size_t)n_pairs);
        for (int p = 0; p < n_pairs; p++) {
            int e = eidx[p], u = unit_of[(size_t)e];
            if (u < 0) {
                if ((int)units.size() >= NU_MAX) return fail("more than 96 distinct experts");
                u = (int)units.size(); unit_of[(size_t)e] = u;
                UnitDH ud{}; for (int r = 0; r < 8; r++) ud.rows[r] = -1;
                units.push_back(ud); cnt.push_back(0); uexp.push_back(e);
            }
            int slot = cnt[(size_t)u]++;
            if (slot >= 8) return fail("more than 8 pairs for one expert");
            units[(size_t)u].rows[slot] = ridx[p];
            pairs[(size_t)p] = { (uint32_t)u, (uint32_t)slot, w[p], 0 };
        }
        const int nu = (int)units.size();
        int maxslot = *std::max_element(cnt.begin(), cnt.end()), mrx = maxslot <= 1 ? 1 : maxslot <= 2 ? 2 : maxslot <= 4 ? 4 : 8;

        Slot *s = acquire(c);
        double tc = 0;
        // resolve blobs
        std::vector<id<MTLBuffer>> used;
        size_t nstaged = 0;
        for (int u = 0; u < nu; u++) {
            const void *bp = blobs[uexp[(size_t)u]];
            if (!bp) { release(c, s); return fail("null blob pointer"); }
            uint64_t addr; id<MTLBuffer> buf;
            if (resolve(c, bp, addr, buf)) { units[(size_t)u].blob = addr; used.push_back(buf); }
            else {
                if (!nstaged) ensure_stage(c, s, NU_MAX);
                double a = now_ms();
                memcpy((char *)s->stage.contents + nstaged * c->blob, bp, c->blob);
                tc += now_ms() - a;
                units[(size_t)u].blob = s->stage.gpuAddress + nstaged * c->blob; nstaged++;
            }
        }
        if (nstaged) used.push_back(s->stage);
        // CSR per output row in pair order
        std::vector<uint32_t> rowptr((size_t)m + 1, 0);
        for (int p = 0; p < n_pairs; p++) rowptr[(size_t)ridx[p] + 1]++;
        for (int r = 0; r < m; r++) rowptr[(size_t)r + 1] += rowptr[(size_t)r];
        std::vector<PairDH> sorted((size_t)n_pairs);
        { std::vector<uint32_t> fill(rowptr.begin(), rowptr.end() - 1); for (int p = 0; p < n_pairs; p++) sorted[fill[(size_t)ridx[p]]++] = pairs[(size_t)p]; }
        {
            double a = now_ms();
            memcpy(s->d_x.contents, x, (size_t)m * H * 4);
            char *t = (char *)s->d_tbl.contents;
            memcpy(t + T_UNITS, units.data(), units.size() * sizeof(UnitDH));
            memcpy(t + T_PAIRS, sorted.data(), sorted.size() * sizeof(PairDH));
            memcpy(t + T_ROWPTR, rowptr.data(), rowptr.size() * 4);
            tc += now_ms() - a;
        }

        id<MTLCommandBuffer> cb = c->unretained ? [c->q commandBufferWithUnretainedReferences] : [c->q commandBuffer];
        id<MTLComputeCommandEncoder> en = [cb computeCommandEncoder];
        use_all(en, used);
        DPH dp = { (uint32_t)mrx, (uint32_t)nu, (uint32_t)c->bits, 0 };
        GPH g0 = { 0, {0, 0, 0} }, g1 = { 1, {0, 0, 0} };
        const MTLSize tg128 = MTLSizeMake(128, 1, 1), tgG = MTLSizeMake(128, 1, 1);   // tgG: gemv_dec is SG = 4 simdgroups wide (split-K)
        [en setComputePipelineState:c->P.rot_dec];
        [en setBuffer:s->d_x offset:0 atIndex:0]; [en setBuffer:s->d_tbl offset:T_UNITS atIndex:1]; [en setBuffer:s->d_xq offset:0 atIndex:2];
        [en setBytes:&dp length:sizeof dp atIndex:3];
        if (!(c->skip_mask & 0x100)) [en dispatchThreadgroups:MTLSizeMake(20, (NSUInteger)nu * mrx, 2) threadsPerThreadgroup:tg128];
        [en setComputePipelineState:c->P.gemv[c->bits][mrx]];
        [en setBuffer:s->d_tbl offset:T_UNITS atIndex:0]; [en setBuffer:s->d_xq offset:0 atIndex:1]; [en setBuffer:s->d_yg offset:0 atIndex:2];
        [en setBytes:&dp length:sizeof dp atIndex:3]; [en setBytes:&g0 length:sizeof g0 atIndex:4];
        if (!(c->skip_mask & 0x200)) [en dispatchThreadgroups:MTLSizeMake(40, (NSUInteger)nu, 2) threadsPerThreadgroup:tgG];
        [en setComputePipelineState:c->P.mid_dec];
        [en setBuffer:s->d_tbl offset:T_UNITS atIndex:0]; [en setBuffer:s->d_yg offset:0 atIndex:1]; [en setBuffer:s->d_zq offset:0 atIndex:2];
        [en setBytes:&dp length:sizeof dp atIndex:3];
        if (!(c->skip_mask & 0x400)) [en dispatchThreadgroups:MTLSizeMake(5, (NSUInteger)nu * mrx, 1) threadsPerThreadgroup:tg128];
        [en setComputePipelineState:c->P.gemv[c->bits][mrx]];
        [en setBuffer:s->d_tbl offset:T_UNITS atIndex:0]; [en setBuffer:s->d_zq offset:0 atIndex:1]; [en setBuffer:s->d_yd offset:0 atIndex:2];
        [en setBytes:&dp length:sizeof dp atIndex:3]; [en setBytes:&g1 length:sizeof g1 atIndex:4];
        if (!(c->skip_mask & 0x800)) [en dispatchThreadgroups:MTLSizeMake(160, (NSUInteger)nu, 1) threadsPerThreadgroup:tgG];
        [en setComputePipelineState:c->P.comb_dec];
        [en setBuffer:s->d_tbl offset:T_UNITS atIndex:0]; [en setBuffer:s->d_yd offset:0 atIndex:1]; [en setBuffer:s->d_tbl offset:T_PAIRS atIndex:2];
        [en setBuffer:s->d_tbl offset:T_ROWPTR atIndex:3]; [en setBuffer:s->d_out offset:0 atIndex:4]; [en setBytes:&dp length:sizeof dp atIndex:5];
        if (!(c->skip_mask & 0x1000)) [en dispatchThreadgroups:MTLSizeMake(20, (NSUInteger)m, 1) threadsPerThreadgroup:tg128];
        [en endEncoding];
        exl3m_job *j = new exl3m_job{ s, cb, y_out, (size_t)m * H * 4, s->d_out, now_ms(), 0, tc };
        j->dec = true;
        j->t_encode = j->t_submit - t0;
        [cb commit];
        c->last_commit_ms = now_ms();   // activity: the keep-warm thread stays quiet while work flows
        *jobp = j;
        { std::lock_guard<std::mutex> l(c->stmu); c->st.decode_calls++; c->st.blobs_staged += (uint64_t)nstaged; c->st.blobs_mapped += (uint64_t)(nu - (int)nstaged); }
        return 0;
    }
}

int exl3m_wait(exl3m_ctx *c, exl3m_job *j) {
    @autoreleasepool {
        if (c->wait_spin && j->dec) { while (j->cb.status < MTLCommandBufferStatusCompleted) {} } else [j->cb waitUntilCompleted];
        if (!j->dec && getenv("EXL3M_LOG_ALLOC")) {
            static int n = 0;
            if (++n % 50 == 1) fprintf(stderr, "exl3m: prefill job %d: Metal allocated %.2f GB of recommended %.2f GB\n", n, (double)c->dev.currentAllocatedSize / 1e9, (double)c->dev.recommendedMaxWorkingSetSize / 1e9);
        }
        const double t1 = now_ms();
        int rc = 0;
        if (j->cb.status != MTLCommandBufferStatusCompleted) {
            NSError *e = j->cb.error;
            rc = fail(std::string("command buffer failed: ") + (e ? e.localizedDescription.UTF8String : "?"));
        } else {
            memcpy(j->y_out, j->src.contents, j->y_bytes);
        }
        const double t2 = now_ms();
        {
            std::lock_guard<std::mutex> l(c->stmu);
            c->st.gpu_ms_last = (j->cb.GPUEndTime - j->cb.GPUStartTime) * 1e3;
            c->st.wall_ms_last = t1 - j->t_submit;
            c->st.encode_ms_last = j->t_encode;
            c->st.copy_ms_last = j->t_copy + (t2 - t1);
        }
        release(c, j->slot);
        delete j;
        return rc;
    }
}

int exl3m_moe_rows(exl3m_ctx *c, const float *x, int m, int n_pairs, const int32_t *eidx, const int32_t *ridx, const float *w,
                   const void *const *blobs, float *y_out) {
    exl3m_job *j = nullptr;
    int rc = exl3m_moe_rows_submit(c, x, m, n_pairs, eidx, ridx, w, blobs, y_out, &j);
    return rc ? rc : exl3m_wait(c, j);
}

// ---------------------------------------------------------------------------------------------------------------------
// NVFP4 cold experts: the same units, pairs and stages as exl3m_moe_rows_submit, over upstream Strata's QAD pack
// blobs (NVFP4_BLOB bytes each: gate | up | down of ggml block_nvfp4 rows plus the fp32 tail, see the .metal's NV4_GT).
// No trellis and no input rotation: gemv_nv4 x2 -> mid_nv4 (tail scales, SiLU, product, fp32) -> gemv_nv4 (down) -> comb_nv4.
// Not declared in exl3_metal.h on purpose (Strata mirrors that header; the entry point is only this extern "C" symbol).
static constexpr int NV4_SG_GU = 4, NV4_NC_GU = 2;   // gemv_nv4 for gate/up: simdgroups that split k, and columns a lane decodes
static constexpr int NV4_SG_DN = 2, NV4_NC_DN = 2;   // the same for the down GEMV (k = 640)
extern "C" int exl3m_moe_rows_nvfp4(exl3m_ctx *c, const float *x, int m, int n_pairs, const int32_t *eidx, const int32_t *ridx,
                                    const float *w, const void *const *blobs, float *y_out) {
    @autoreleasepool {
        if (m < 1 || m > 8) return fail("m must be 1..8");
        if (n_pairs < 1 || n_pairs > PAIR_MAX) return fail("n_pairs must be 1..4096");
        if (!c->P.nv4_mid || !c->P.nv4_comb || !c->P.nv4_gemv[0][1] || !c->P.nv4_gemv[1][1]) return fail("the NVFP4 pipelines did not build on this device");
        const double t0 = now_ms();
        int32_t emax = -1;
        for (int p = 0; p < n_pairs; p++) {
            if (eidx[p] < 0 || ridx[p] < 0 || ridx[p] >= m) return fail("bad expert/row index");
            emax = std::max(emax, eidx[p]);
        }
        std::vector<int> unit_of((size_t)emax + 1, -1);
        std::vector<UnitDH> units; std::vector<int> cnt; std::vector<int32_t> uexp;
        std::vector<PairDH> pairs((size_t)n_pairs);
        for (int p = 0; p < n_pairs; p++) {
            int e = eidx[p], u = unit_of[(size_t)e];
            if (u < 0) {
                if ((int)units.size() >= NU_MAX) return fail("more than 96 distinct experts");
                u = (int)units.size(); unit_of[(size_t)e] = u;
                UnitDH ud{}; for (int r = 0; r < 8; r++) ud.rows[r] = -1;
                units.push_back(ud); cnt.push_back(0); uexp.push_back(e);
            }
            int slot = cnt[(size_t)u]++;
            if (slot >= 8) return fail("more than 8 pairs for one expert");
            units[(size_t)u].rows[slot] = ridx[p];
            pairs[(size_t)p] = { (uint32_t)u, (uint32_t)slot, w[p], 0 };
        }
        const int nu = (int)units.size();
        int maxslot = *std::max_element(cnt.begin(), cnt.end()), mrx = maxslot <= 1 ? 1 : maxslot <= 2 ? 2 : maxslot <= 4 ? 4 : 8;

        Slot *s = acquire(c);
        double tc = 0;
        std::vector<id<MTLBuffer>> used;
        size_t nstaged = 0;
        for (int u = 0; u < nu; u++) {
            const void *bp = blobs[uexp[(size_t)u]];
            if (!bp) { release(c, s); return fail("null blob pointer"); }
            uint64_t addr; id<MTLBuffer> buf;
            if (resolve(c, bp, addr, buf, NVFP4_BLOB)) { units[(size_t)u].blob = addr; used.push_back(buf); }
            else {
                if (!s->nv_stage_n) { s->nv_stage = mkbuf(c, (size_t)NU_MAX * NVFP4_BLOB); s->nv_stage_n = NU_MAX; }
                double a = now_ms();
                memcpy((char *)s->nv_stage.contents + nstaged * NVFP4_BLOB, bp, NVFP4_BLOB);
                tc += now_ms() - a;
                units[(size_t)u].blob = s->nv_stage.gpuAddress + nstaged * NVFP4_BLOB; nstaged++;
            }
        }
        if (nstaged) used.push_back(s->nv_stage);
        std::vector<uint32_t> rowptr((size_t)m + 1, 0);
        for (int p = 0; p < n_pairs; p++) rowptr[(size_t)ridx[p] + 1]++;
        for (int r = 0; r < m; r++) rowptr[(size_t)r + 1] += rowptr[(size_t)r];
        std::vector<PairDH> sorted((size_t)n_pairs);
        { std::vector<uint32_t> fill(rowptr.begin(), rowptr.end() - 1); for (int p = 0; p < n_pairs; p++) sorted[fill[(size_t)ridx[p]]++] = pairs[(size_t)p]; }
        {
            double a = now_ms();
            memcpy(s->d_x.contents, x, (size_t)m * H * 4);
            char *t = (char *)s->d_tbl.contents;
            memcpy(t + T_UNITS, units.data(), units.size() * sizeof(UnitDH));
            memcpy(t + T_PAIRS, sorted.data(), sorted.size() * sizeof(PairDH));
            memcpy(t + T_ROWPTR, rowptr.data(), rowptr.size() * 4);
            tc += now_ms() - a;
        }

        id<MTLCommandBuffer> cb = c->unretained ? [c->q commandBufferWithUnretainedReferences] : [c->q commandBuffer];
        id<MTLComputeCommandEncoder> en = [cb computeCommandEncoder];
        use_all(en, used);
        DPH dp = { (uint32_t)mrx, (uint32_t)nu, 0, 0 };
        GPH g0 = { 0, {0, 0, 0} }, g1 = { 1, {0, 0, 0} };
        const MTLSize tg128 = MTLSizeMake(128, 1, 1);
        const MTLSize tg_gu = MTLSizeMake(32 * NV4_SG_GU, 1, 1), tg_dn = MTLSizeMake(32 * NV4_SG_DN, 1, 1);   // gemv_nv4's SG simdgroups
        const NSUInteger nv4_tiles = FF / (8 * NV4_NC_GU);   // gemv_nv4's tile is 8 * NC columns of the 640-row side
        [en setComputePipelineState:c->P.nv4_gemv[0][mrx]];
        [en setBuffer:s->d_tbl offset:T_UNITS atIndex:0]; [en setBuffer:s->d_x offset:0 atIndex:1]; [en setBuffer:s->d_yg offset:0 atIndex:2];
        [en setBytes:&dp length:sizeof dp atIndex:3]; [en setBytes:&g0 length:sizeof g0 atIndex:4];
        if (!(c->skip_mask & 0x100)) [en dispatchThreadgroups:MTLSizeMake(nv4_tiles, (NSUInteger)nu, 2) threadsPerThreadgroup:tg_gu];
        [en setComputePipelineState:c->P.nv4_mid];
        [en setBuffer:s->d_tbl offset:T_UNITS atIndex:0]; [en setBuffer:s->d_yg offset:0 atIndex:1]; [en setBuffer:s->nv_z offset:0 atIndex:2];
        [en setBytes:&dp length:sizeof dp atIndex:3];
        if (!(c->skip_mask & 0x200)) [en dispatchThreadgroups:MTLSizeMake(5, (NSUInteger)nu * mrx, 1) threadsPerThreadgroup:tg128];
        [en setComputePipelineState:c->P.nv4_gemv[1][mrx]];
        [en setBuffer:s->d_tbl offset:T_UNITS atIndex:0]; [en setBuffer:s->nv_z offset:0 atIndex:1]; [en setBuffer:s->d_yd offset:0 atIndex:2];
        [en setBytes:&dp length:sizeof dp atIndex:3]; [en setBytes:&g1 length:sizeof g1 atIndex:4];
        if (!(c->skip_mask & 0x400)) [en dispatchThreadgroups:MTLSizeMake(H / (8 * NV4_NC_DN), (NSUInteger)nu, 1) threadsPerThreadgroup:tg_dn];
        [en setComputePipelineState:c->P.nv4_comb];
        [en setBuffer:s->d_tbl offset:T_UNITS atIndex:0]; [en setBuffer:s->d_yd offset:0 atIndex:1]; [en setBuffer:s->d_tbl offset:T_PAIRS atIndex:2];
        [en setBuffer:s->d_tbl offset:T_ROWPTR atIndex:3]; [en setBuffer:s->d_out offset:0 atIndex:4]; [en setBytes:&dp length:sizeof dp atIndex:5];
        if (!(c->skip_mask & 0x800)) [en dispatchThreadgroups:MTLSizeMake(20, (NSUInteger)m, 1) threadsPerThreadgroup:tg128];
        [en endEncoding];
        exl3m_job *j = new exl3m_job{ s, cb, y_out, (size_t)m * H * 4, s->d_out, now_ms(), 0, tc };
        j->dec = true;
        j->t_encode = j->t_submit - t0;
        [cb commit];
        c->last_commit_ms = now_ms();
        { std::lock_guard<std::mutex> l(c->stmu); c->st.decode_calls++; c->st.blobs_staged += (uint64_t)nstaged; c->st.blobs_mapped += (uint64_t)(nu - (int)nstaged); }
        return exl3m_wait(c, j);
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// A short prompt (n_tok <= pf_short_tok) is one chunk of a few rows per expert: the prompt kernels would dequantize every
// expert to fp16 for a GEMM that costs the same for 6 rows as for 600, so the same pairs go through the decode kernels
// instead, each expert's rows in units of up to 8 (the trellis is read once per unit). One command buffer, one serial compute
// encoder: the batches reuse the slot's xq/yg/zq buffers, which the encoder orders. An NVFP4 prompt runs the same
// plan over gemv_nv4 / mid_nv4 / comb_nv4: no rotation (the GEMV reads the fp32 rows), and the down GEMV writes fp32 into the
// slot's yd rows. Returns 1 when it does not apply, a blob that resolve() cannot place, so prefill_submit_impl stages it and
// runs the prompt path as before.
static int prefill_short(exl3m_ctx *c, const void *x, bool half_io, bool nv4, int n_tok, int64_t n_pairs, const int32_t *eidx,
                         const int32_t *ridx, const float *w, const void *const *blobs, void *y_out, exl3m_job **jobp) {
    @autoreleasepool {
        const double t0 = now_ms();
        int32_t emax = -1;
        for (int64_t p = 0; p < n_pairs; p++) emax = std::max(emax, eidx[p]);
        std::vector<int> unit_of((size_t)emax + 1, -1), cnt;
        std::vector<UnitDH> units; std::vector<int32_t> uexp;
        std::vector<PairDH> pairs((size_t)n_pairs);
        for (int64_t p = 0; p < n_pairs; p++) {
            int e = eidx[p], u = unit_of[(size_t)e];
            if (u < 0 || cnt[(size_t)u] >= 8) {
                u = (int)units.size(); unit_of[(size_t)e] = u;
                UnitDH ud{}; for (int r = 0; r < 8; r++) ud.rows[r] = -1;
                units.push_back(ud); cnt.push_back(0); uexp.push_back(e);
            }
            int slot = cnt[(size_t)u]++;
            units[(size_t)u].rows[slot] = ridx[p];
            pairs[(size_t)p] = { (uint32_t)u, (uint32_t)slot, w[p], 0 };
        }
        const int nu = (int)units.size();
        int maxslot = *std::max_element(cnt.begin(), cnt.end()), mrx = maxslot <= 1 ? 1 : maxslot <= 2 ? 2 : maxslot <= 4 ? 4 : 8;
        if (nv4 && (!c->P.nv4_gemv[0][mrx] || !c->P.nv4_gemv[1][mrx] || !c->P.nv4_mid || !c->P.nv4_comb)) return 1;
        std::vector<id<MTLBuffer>> used;
        for (int u = 0; u < nu; u++) {
            const void *bp = blobs[uexp[(size_t)u]];
            uint64_t addr; id<MTLBuffer> buf;
            if (!bp || !resolve(c, bp, addr, buf, nv4 ? NVFP4_BLOB : 0)) return 1;
            units[(size_t)u].blob = addr; used.push_back(buf);
        }
        Slot *s = acquire(c);
        double tc = 0;
        // CSR over the output rows in pair order, as the decode path
        std::vector<uint32_t> rowptr((size_t)n_tok + 1, 0);
        for (int64_t p = 0; p < n_pairs; p++) rowptr[(size_t)ridx[p] + 1]++;
        for (int r = 0; r < n_tok; r++) rowptr[(size_t)r + 1] += rowptr[(size_t)r];
        std::vector<PairDH> sorted((size_t)n_pairs);
        { std::vector<uint32_t> fill(rowptr.begin(), rowptr.end() - 1); for (int64_t p = 0; p < n_pairs; p++) sorted[fill[(size_t)ridx[p]]++] = pairs[(size_t)p]; }
        size_t toff = (units.size() * sizeof(UnitDH) + 255) & ~(size_t)255;
        const size_t o_pairs = toff;
        toff = (toff + sorted.size() * sizeof(PairDH) + 255) & ~(size_t)255;
        const size_t o_rowptr = toff, tbl_bytes = toff + rowptr.size() * 4;
        grow(c, s->s_x, s->s_x_n, (size_t)n_tok * H * 4, MTLResourceStorageModeShared);
        grow(c, s->s_tbl, s->s_tbl_n, tbl_bytes, MTLResourceStorageModeShared);
        grow(c, s->s_out, s->s_out_n, (size_t)n_tok * H * 4, MTLResourceStorageModeShared);
        if (half_io) grow(c, s->s_yh, s->s_yh_n, (size_t)n_tok * H * 2, MTLResourceStorageModeShared);
        grow(c, s->s_yd, s->s_yd_n, (size_t)nu * (size_t)mrx * H * 4, PRIV);
        {
            double a = now_ms();
            if (half_io) {   // the kernel rounds x to fp16 first, so the fp32 rows of an fp16 call are its exact values
                const uint16_t *h = (const uint16_t *)x; float *f = (float *)s->s_x.contents;
                for (size_t i = 0, n = (size_t)n_tok * H; i < n; i++) { __fp16 v; memcpy(&v, h + i, sizeof v); f[i] = (float)v; }
            } else memcpy(s->s_x.contents, x, (size_t)n_tok * H * 4);
            char *t = (char *)s->s_tbl.contents;
            memcpy(t, units.data(), units.size() * sizeof(UnitDH));
            memcpy(t + o_pairs, sorted.data(), sorted.size() * sizeof(PairDH));
            memcpy(t + o_rowptr, rowptr.data(), rowptr.size() * 4);
            tc += now_ms() - a;
        }

        id<MTLCommandBuffer> cb = [c->q commandBuffer];
        id<MTLComputeCommandEncoder> en = [cb computeCommandEncoder];
        use_all(en, used);
        const MTLSize tg128 = MTLSizeMake(128, 1, 1), tgG = MTLSizeMake(128, 1, 1);   // tgG: gemv_dec is SG = 4 simdgroups wide (split-K)
        const GPH g0 = { 0, {0, 0, 0} }, g1 = { 1, {0, 0, 0} };
        // gemv_nv4's own shapes, as in exl3m_moe_rows_nvfp4: SG simdgroups, 8 * NC columns a tile
        const MTLSize tg_gu = MTLSizeMake(32 * NV4_SG_GU, 1, 1), tg_dn = MTLSizeMake(32 * NV4_SG_DN, 1, 1);
        const NSUInteger nv4_gu_tiles = FF / (8 * NV4_NC_GU), nv4_dn_tiles = H / (8 * NV4_NC_DN);
        for (int b0 = 0; b0 < nu; b0 += NU_MAX) {
            const int nb = std::min(NU_MAX, nu - b0);
            const NSUInteger uoff = (NSUInteger)b0 * sizeof(UnitDH), yoff = (NSUInteger)b0 * (NSUInteger)mrx * H * 4;
            DPH dp = { (uint32_t)mrx, (uint32_t)nb, (uint32_t)c->bits, 0 };
            if (nv4) {   // gate/up, tail scales + SiLU product, down; x is read as the fp32 rows, no rotation
                DPH dpn = { (uint32_t)mrx, (uint32_t)nb, 0, 0 };
                [en setComputePipelineState:c->P.nv4_gemv[0][mrx]];
                [en setBuffer:s->s_tbl offset:uoff atIndex:0]; [en setBuffer:s->s_x offset:0 atIndex:1]; [en setBuffer:s->d_yg offset:0 atIndex:2];
                [en setBytes:&dpn length:sizeof dpn atIndex:3]; [en setBytes:&g0 length:sizeof g0 atIndex:4];
                [en dispatchThreadgroups:MTLSizeMake(nv4_gu_tiles, (NSUInteger)nb, 2) threadsPerThreadgroup:tg_gu];
                [en setComputePipelineState:c->P.nv4_mid];
                [en setBuffer:s->s_tbl offset:uoff atIndex:0]; [en setBuffer:s->d_yg offset:0 atIndex:1]; [en setBuffer:s->nv_z offset:0 atIndex:2];
                [en setBytes:&dpn length:sizeof dpn atIndex:3];
                [en dispatchThreadgroups:MTLSizeMake(5, (NSUInteger)nb * mrx, 1) threadsPerThreadgroup:tg128];
                [en setComputePipelineState:c->P.nv4_gemv[1][mrx]];
                [en setBuffer:s->s_tbl offset:uoff atIndex:0]; [en setBuffer:s->nv_z offset:0 atIndex:1]; [en setBuffer:s->s_yd offset:yoff atIndex:2];
                [en setBytes:&dpn length:sizeof dpn atIndex:3]; [en setBytes:&g1 length:sizeof g1 atIndex:4];
                [en dispatchThreadgroups:MTLSizeMake(nv4_dn_tiles, (NSUInteger)nb, 1) threadsPerThreadgroup:tg_dn];
                continue;
            }
            [en setComputePipelineState:c->P.rot_dec];
            [en setBuffer:s->s_x offset:0 atIndex:0]; [en setBuffer:s->s_tbl offset:uoff atIndex:1]; [en setBuffer:s->d_xq offset:0 atIndex:2];
            [en setBytes:&dp length:sizeof dp atIndex:3];
            [en dispatchThreadgroups:MTLSizeMake(20, (NSUInteger)nb * mrx, 2) threadsPerThreadgroup:tg128];
            [en setComputePipelineState:c->P.gemv[c->bits][mrx]];
            [en setBuffer:s->s_tbl offset:uoff atIndex:0]; [en setBuffer:s->d_xq offset:0 atIndex:1]; [en setBuffer:s->d_yg offset:0 atIndex:2];
            [en setBytes:&dp length:sizeof dp atIndex:3]; [en setBytes:&g0 length:sizeof g0 atIndex:4];
            [en dispatchThreadgroups:MTLSizeMake(40, (NSUInteger)nb, 2) threadsPerThreadgroup:tgG];
            [en setComputePipelineState:c->P.mid_dec];
            [en setBuffer:s->s_tbl offset:uoff atIndex:0]; [en setBuffer:s->d_yg offset:0 atIndex:1]; [en setBuffer:s->d_zq offset:0 atIndex:2];
            [en setBytes:&dp length:sizeof dp atIndex:3];
            [en dispatchThreadgroups:MTLSizeMake(5, (NSUInteger)nb * mrx, 1) threadsPerThreadgroup:tg128];
            [en setComputePipelineState:c->P.gemv[c->bits][mrx]];
            [en setBuffer:s->s_tbl offset:uoff atIndex:0]; [en setBuffer:s->d_zq offset:0 atIndex:1]; [en setBuffer:s->s_yd offset:yoff atIndex:2];
            [en setBytes:&dp length:sizeof dp atIndex:3]; [en setBytes:&g1 length:sizeof g1 atIndex:4];
            [en dispatchThreadgroups:MTLSizeMake(160, (NSUInteger)nb, 1) threadsPerThreadgroup:tgG];
        }
        DPH dpa = { (uint32_t)mrx, (uint32_t)nu, (uint32_t)c->bits, 0 };
        [en setComputePipelineState:nv4 ? c->P.nv4_comb : c->P.comb_dec];   // the same buffer indices on both
        [en setBuffer:s->s_tbl offset:0 atIndex:0]; [en setBuffer:s->s_yd offset:0 atIndex:1];
        [en setBuffer:s->s_tbl offset:o_pairs atIndex:2]; [en setBuffer:s->s_tbl offset:o_rowptr atIndex:3];
        [en setBuffer:s->s_out offset:0 atIndex:4]; [en setBytes:&dpa length:sizeof dpa atIndex:5];
        [en dispatchThreadgroups:MTLSizeMake(20, (NSUInteger)n_tok, 1) threadsPerThreadgroup:tg128];
        if (half_io) {   // the fp32 rows -> fp16, as the prompt path's y_f2h behind the last chunk
            const uint32_t nel = (uint32_t)n_tok * H;
            [en setComputePipelineState:c->P.y_f2h];
            [en setBuffer:s->s_out offset:0 atIndex:0]; [en setBuffer:s->s_yh offset:0 atIndex:1]; [en setBytes:&nel length:4 atIndex:2];
            [en dispatchThreads:MTLSizeMake(nel, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        }
        [en endEncoding];
        exl3m_job *j = new exl3m_job{ s, cb, y_out, half_io ? (size_t)n_tok * H * 2 : (size_t)n_tok * H * 4, half_io ? s->s_yh : s->s_out,
                                      now_ms(), 0, tc };
        j->t_encode = j->t_submit - t0;
        [cb commit];
        c->last_commit_ms = now_ms();   // activity: the keep-warm thread stays quiet while work flows
        *jobp = j;
        { std::lock_guard<std::mutex> l(c->stmu); c->st.prefill_calls++; c->st.blobs_mapped += (uint64_t)nu; }
        return 0;
    }
}

// nv4: the same pipeline over NVFP4 blobs (gate | up | down block_nvfp4 rows + the fp32 tail, NVFP4_BLOB bytes): the
// weight decode, the chunk rows, the mid stage and the combine are the nv4 kernels, which carry the tail scales in fp32 as
// gemv_nv4 / mid_nv4 / comb_nv4 do, and the blobs resolve and stage at NVFP4_BLOB instead of the context's trellis blob.
static int prefill_submit_impl(exl3m_ctx *c, const void *x, bool half_io, bool nv4, int n_tok, int64_t n_pairs, const int32_t *eidx,
                               const int32_t *ridx, const float *w, const void *const *blobs, void *y_out, exl3m_job **jobp) {
    @autoreleasepool {
        if (n_tok < 1 || n_tok > 16384) return fail("n_tok must be 1..16384");
        if (n_pairs < 1 || n_pairs > (1 << 20)) return fail("n_pairs must be 1..2^20");
        if (nv4 && (!c->P.decode_w_nv4 || !c->P.x_pf_nv4 || !c->P.x_pf_nv4_h || !c->P.mid_pf_nv4 || !c->P.comb_pf_nv4))
            return fail("the NVFP4 prefill pipelines did not build on this device");
        const double t0 = now_ms();
        int32_t emax = -1;
        for (int64_t p = 0; p < n_pairs; p++) {
            if (eidx[p] < 0 || ridx[p] < 0 || ridx[p] >= n_tok) return fail("bad expert/row index");
            emax = std::max(emax, eidx[p]);
        }
        const int pf_short = nv4 ? c->pf_short_tok : std::min(c->pf_short_tok, PF_SHORT_EXL3);   // EXL3 stops at its own 256
        if (pf_short > 0 && n_tok <= pf_short) {   // a short prompt: the decode kernels, one command buffer (see prefill_short)
            const int rc = prefill_short(c, x, half_io, nv4, n_tok, n_pairs, eidx, ridx, w, blobs, y_out, jobp);
            if (rc <= 0) return rc;
        }
        // counting sort by expert (stable: pair order kept within an expert)
        std::vector<uint32_t> ecount((size_t)emax + 2, 0);
        for (int64_t p = 0; p < n_pairs; p++) ecount[(size_t)eidx[p] + 1]++;
        for (size_t e = 0; e <= (size_t)emax; e++) ecount[e + 1] += ecount[e];
        std::vector<uint32_t> order((size_t)n_pairs);
        { std::vector<uint32_t> fill(ecount.begin(), ecount.end() - 1); for (int64_t p = 0; p < n_pairs; p++) order[fill[(size_t)eidx[p]]++] = (uint32_t)p; }
        std::vector<int32_t> uexp;                       // used experts ascending
        for (int32_t e = 0; e <= emax; e++) if (ecount[(size_t)e + 1] > ecount[(size_t)e]) uexp.push_back(e);
        const int nu = (int)uexp.size();
        for (int u = 0; u < nu; u++)
            if (ecount[(size_t)uexp[(size_t)u] + 1] - ecount[(size_t)uexp[(size_t)u]] > 16384) return fail("more than 16384 rows for one expert");

        // chunk plan
        struct Chunk { int u0, u1; uint32_t rows; };
        std::vector<Chunk> chunks;
        {
            int u0 = 0; uint32_t rows = 0;
            for (int u = 0; u < nu; u++) {
                uint32_t nr = ecount[(size_t)uexp[(size_t)u] + 1] - ecount[(size_t)uexp[(size_t)u]];
                if (u > u0 && (u - u0 >= c->pf_chunk_experts || rows + nr > (uint32_t)c->pf_chunk_rows)) { chunks.push_back({u0, u, rows}); u0 = u; rows = 0; }
                rows += nr;
            }
            chunks.push_back({u0, nu, rows});
        }
        uint32_t cap_rows = 0; int maxg = 0; size_t tbl_bytes = 0;
        for (auto &ch : chunks) { cap_rows = std::max(cap_rows, ch.rows); maxg = std::max(maxg, ch.u1 - ch.u0);
            tbl_bytes += 256 + (size_t)(ch.u1 - ch.u0) * 16 + 256 + (size_t)ch.rows * 8 + 256 + ((size_t)ch.rows / 32 + (size_t)(ch.u1 - ch.u0) + 1) * 16 + 256 + (size_t)ch.rows * 16 + 1024; }
        cap_rows += 64;

        Slot *s = acquire(c);
        double tc = 0;
        const size_t blen = nv4 ? NVFP4_BLOB : c->blob;   // an NVFP4 blob is 2,764,816 B at every trellis width
        std::vector<id<MTLBuffer>> used;
        std::vector<uint64_t> addr((size_t)nu);
        size_t nstaged = 0;
        for (int u = 0; u < nu; u++) {
            const void *bp = blobs[uexp[(size_t)u]];
            if (!bp) { release(c, s); return fail("null blob pointer"); }
            id<MTLBuffer> buf;
            if (resolve(c, bp, addr[(size_t)u], buf, blen)) used.push_back(buf);
            else nstaged++;
        }
        if (nstaged) {
            id<MTLBuffer> sbuf = s->stage;
            if (nv4) {
                if (s->p_nvst_n < nstaged) { s->p_nvst = mkbuf(c, nstaged * NVFP4_BLOB); s->p_nvst_n = nstaged; }
                sbuf = s->p_nvst;
            } else if (s->stage_blobs < nstaged) { sbuf = s->stage = mkbuf(c, nstaged * c->blob); s->stage_blobs = nstaged; }
            size_t k = 0;
            for (int u = 0; u < nu; u++) {
                id<MTLBuffer> buf; uint64_t a;
                if (resolve(c, blobs[uexp[(size_t)u]], a, buf, blen)) continue;
                double a0 = now_ms();
                memcpy((char *)sbuf.contents + k * blen, blobs[uexp[(size_t)u]], blen);
                tc += now_ms() - a0;
                addr[(size_t)u] = sbuf.gpuAddress + k * blen; k++;
            }
            used.push_back(sbuf);
        }
        // scratch
        const size_t xbytes = (size_t)n_tok * H * 4;                    // the fp32 accumulation rows (always fp32: a row can be touched by several expert chunks)
        const size_t xbytes_in = half_io ? xbytes / 2 : xbytes;        // the input rows as the host holds them
        grow(c, s->p_x, s->p_x_n, xbytes_in, MTLResourceStorageModeShared);
        grow(c, s->p_y, s->p_y_n, xbytes, MTLResourceStorageModeShared);
        if (half_io) grow(c, s->p_yh, s->p_yh_n, xbytes / 2, MTLResourceStorageModeShared);
        grow(c, s->p_tbl, s->p_tbl_n, tbl_bytes + 4096, MTLResourceStorageModeShared);
        if (s->p_cap < cap_rows) {
            s->p_A = mkbuf(c, 2 * (size_t)cap_rows * H * 2, PRIV);
            s->p_C = mkbuf(c, 2 * (size_t)cap_rows * FF * 4, PRIV);
            s->p_Z = mkbuf(c, (size_t)cap_rows * FF * 2, PRIV);
            s->p_Yd = mkbuf(c, (size_t)cap_rows * H * 4, PRIV);
            s->p_cap = cap_rows;
        }
        const size_t wbytes = (size_t)H * FF * 2;
        grow(c, s->p_Wg, s->p_wg_n, (size_t)maxg * wbytes, PRIV);
        if (!s->p_Wu || s->p_Wu.length < s->p_Wg.length) { s->p_Wu = mkbuf(c, s->p_Wg.length, PRIV); s->p_Wd = mkbuf(c, s->p_Wg.length, PRIV); }
        {
            double a = now_ms();
            memcpy(s->p_x.contents, x, xbytes_in);
            tc += now_ms() - a;
        }

        id<MTLCommandBuffer> cb = [c->q commandBuffer];
        id<MTLBlitCommandEncoder> bl = [cb blitCommandEncoder];
        [bl fillBuffer:s->p_y range:NSMakeRange(0, xbytes) value:0];
        [bl endEncoding];
        // pf_tensor_ops: the tensor-op GEMMs read the plain row-major weights, so the weight decode switches with them (an NVFP4
        // blob decodes into the same plain layout, its 36 B blocks straight into W[u][k*N + n])
        const bool tops = c->tensor_ops;
        id<MTLComputePipelineState> pi_dec_w = tops ? (nv4 ? c->P.decode_w_nv4_plain : c->P.decode_w_plain) : nv4 ? c->P.decode_w_nv4 : c->P.decode_w;
        id<MTLComputePipelineState> pi_gu = tops ? c->P.gemm_gu_t : c->P.gemm_gu;
        id<MTLComputePipelineState> pi_d = tops ? c->P.gemm_d_t : c->P.gemm_d;
        id<MTLComputePipelineState> pi_x = nv4 ? (half_io ? c->P.x_pf_nv4_h : c->P.x_pf_nv4) : (half_io ? c->P.rot_pf_h : c->P.rot_pf);
        id<MTLComputeCommandEncoder> en = [cb computeCommandEncoder];
        use_all(en, used);
        char *tbl = (char *)s->p_tbl.contents; size_t toff = 0;
        auto take = [&](size_t n) { toff = (toff + 255) & ~(size_t)255; size_t o = toff; toff += n; return o; };
        std::vector<int32_t> row_slot((size_t)n_tok, -1);
        const uint32_t tr = tops ? 64u : 32u;   // rows per GEMM tile: the tensor-op GEMM is 64 tall, the simdgroup GEMM 32
        const NSUInteger decw = nv4 && tops ? 25600 : 6400 * 32;   // the plain NVFP4 decode is one thread a 36 B block (K * N / 64)
        for (auto &ch : chunks) {
            const int G = ch.u1 - ch.u0; const uint32_t R = ch.rows;
            size_t o_units = take((size_t)G * 16), o_uor = take((size_t)R * 4), o_src = take((size_t)R * 4);
            size_t ntiles_max = (size_t)R / 32 + (size_t)G + 1;
            size_t o_tiles = take(ntiles_max * 16);
            size_t o_touched = take((size_t)R * 4), o_tptr = take(((size_t)R + 1) * 4), o_cps = take((size_t)R * 8);
            UnitPH *U = (UnitPH *)(tbl + o_units); uint32_t *uor = (uint32_t *)(tbl + o_uor); int32_t *src = (int32_t *)(tbl + o_src);
            GTH *tiles = (GTH *)(tbl + o_tiles); uint32_t ntiles = 0;
            std::vector<float> pw(R); std::vector<uint32_t> prow(R);
            uint32_t pr = 0;
            for (int g = 0; g < G; g++) {
                int u = ch.u0 + g; int32_t e = uexp[(size_t)u];
                uint32_t nr = ecount[(size_t)e + 1] - ecount[(size_t)e];
                U[g] = { addr[(size_t)u], pr, nr };
                for (uint32_t i = 0; i < nr; i++, pr++) {
                    uint32_t p = order[ecount[(size_t)e] + i];
                    uor[pr] = (uint32_t)g; src[pr] = ridx[p]; pw[pr] = w[p]; prow[pr] = (uint32_t)ridx[p];
                }
                for (uint32_t r = 0; r < nr; r += tr) tiles[ntiles++] = { (uint32_t)g, U[g].row0 + r, std::min<uint32_t>(tr, nr - r), 0 };
            }
            // combine CSR: touched rows, pairs in chunk order
            uint32_t *touched = (uint32_t *)(tbl + o_touched), *tptr = (uint32_t *)(tbl + o_tptr); CPH *cps = (CPH *)(tbl + o_cps);
            uint32_t nt = 0;
            for (uint32_t i = 0; i < R; i++) { int32_t &rs = row_slot[prow[i]]; if (rs < 0) { rs = (int32_t)nt; touched[nt++] = prow[i]; } }
            std::vector<uint32_t> cnt(nt + 1, 0);
            for (uint32_t i = 0; i < R; i++) cnt[(size_t)row_slot[prow[i]] + 1]++;
            for (uint32_t i = 0; i < nt; i++) cnt[i + 1] += cnt[i];
            memcpy(tptr, cnt.data(), ((size_t)nt + 1) * 4);
            { std::vector<uint32_t> fill(cnt.begin(), cnt.end() - 1); for (uint32_t i = 0; i < R; i++) cps[fill[(size_t)row_slot[prow[i]]]++] = { i, pw[i] }; }
            for (uint32_t i = 0; i < nt; i++) row_slot[touched[i]] = -1;
            // encode
            PPH pp = { R, (uint32_t)s->p_cap, (uint32_t)c->bits, 0 };
            const MTLSize tg128 = MTLSizeMake(128, 1, 1);
            [en setComputePipelineState:pi_x];
            [en setBuffer:s->p_x offset:0 atIndex:0]; [en setBuffer:s->p_tbl offset:o_units atIndex:1]; [en setBuffer:s->p_tbl offset:o_uor atIndex:2];
            [en setBuffer:s->p_tbl offset:o_src atIndex:3]; [en setBuffer:s->p_A offset:0 atIndex:4]; [en setBytes:&pp length:sizeof pp atIndex:5];
            if (!(c->skip_mask & 1)) [en dispatchThreadgroups:MTLSizeMake(20, R, 2) threadsPerThreadgroup:tg128];
            [en setComputePipelineState:pi_dec_w];
            [en setBuffer:s->p_tbl offset:o_units atIndex:0];
            for (uint32_t which = 0; which < 3; which++) {
                id<MTLBuffer> wb = which == 0 ? s->p_Wg : which == 1 ? s->p_Wu : s->p_Wd;
                DQH dq = { which, (uint32_t)c->bits, 0, 0 };
                [en setBuffer:wb offset:0 atIndex:1]; [en setBytes:&dq length:sizeof dq atIndex:2];
                if (!(c->skip_mask & 2)) [en dispatchThreads:MTLSizeMake(decw, (NSUInteger)G, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            }
            [en setComputePipelineState:pi_gu];
            [en setBuffer:s->p_A offset:0 atIndex:0]; [en setBuffer:s->p_Wg offset:0 atIndex:1]; [en setBuffer:s->p_Wu offset:0 atIndex:2];
            [en setBuffer:s->p_C offset:0 atIndex:3]; [en setBuffer:s->p_tbl offset:o_tiles atIndex:4]; [en setBytes:&pp length:sizeof pp atIndex:5];
            if (!(c->skip_mask & 4)) [en dispatchThreadgroups:MTLSizeMake(10, ntiles, 2) threadsPerThreadgroup:tg128];
            [en setComputePipelineState:nv4 ? c->P.mid_pf_nv4 : c->P.mid_pf];
            [en setBuffer:s->p_tbl offset:o_units atIndex:0]; [en setBuffer:s->p_tbl offset:o_uor atIndex:1]; [en setBuffer:s->p_C offset:0 atIndex:2];
            [en setBuffer:s->p_Z offset:0 atIndex:3]; [en setBytes:&pp length:sizeof pp atIndex:4];
            if (!(c->skip_mask & 8)) [en dispatchThreadgroups:MTLSizeMake(5, R, 1) threadsPerThreadgroup:tg128];
            [en setComputePipelineState:pi_d];
            [en setBuffer:s->p_Z offset:0 atIndex:0]; [en setBuffer:s->p_Wd offset:0 atIndex:1]; [en setBuffer:s->p_Yd offset:0 atIndex:2];
            [en setBuffer:s->p_tbl offset:o_tiles atIndex:3];
            if (!(c->skip_mask & 16)) [en dispatchThreadgroups:MTLSizeMake(40, ntiles, 1) threadsPerThreadgroup:tg128];
            [en setComputePipelineState:nv4 ? c->P.comb_pf_nv4 : c->P.comb_pf];
            [en setBuffer:s->p_tbl offset:o_units atIndex:0]; [en setBuffer:s->p_tbl offset:o_uor atIndex:1]; [en setBuffer:s->p_Yd offset:0 atIndex:2];
            const uint32_t bits = (uint32_t)c->bits;
            [en setBuffer:s->p_tbl offset:o_touched atIndex:3]; [en setBuffer:s->p_tbl offset:o_tptr atIndex:4]; [en setBuffer:s->p_tbl offset:o_cps atIndex:5];
            [en setBuffer:s->p_y offset:0 atIndex:6]; [en setBytes:&bits length:4 atIndex:7];
            if (!(c->skip_mask & 32)) [en dispatchThreadgroups:MTLSizeMake(20, nt, 1) threadsPerThreadgroup:tg128];
        }
        if (half_io) {   // fp32 rows -> fp16 on the GPU (the host's clamp + cast, bit for bit), behind the last chunk's combine
            const uint32_t nel = (uint32_t)n_tok * H;
            [en setComputePipelineState:c->P.y_f2h];
            [en setBuffer:s->p_y offset:0 atIndex:0]; [en setBuffer:s->p_yh offset:0 atIndex:1]; [en setBytes:&nel length:4 atIndex:2];
            [en dispatchThreads:MTLSizeMake(nel, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        }
        [en endEncoding];
        exl3m_job *j = new exl3m_job{ s, cb, y_out, half_io ? xbytes / 2 : xbytes, half_io ? s->p_yh : s->p_y, now_ms(), 0, tc };
        j->t_encode = j->t_submit - t0;
        [cb commit];
        c->last_commit_ms = now_ms();   // activity: the keep-warm thread stays quiet while work flows
        *jobp = j;
        { std::lock_guard<std::mutex> l(c->stmu); c->st.prefill_calls++; c->st.blobs_staged += (uint64_t)nstaged; c->st.blobs_mapped += (uint64_t)((size_t)nu - nstaged); }
        return 0;
    }
}

int exl3m_moe_prefill_submit(exl3m_ctx *c, const float *x, int n_tok, int64_t n_pairs, const int32_t *eidx, const int32_t *ridx,
                             const float *w, const void *const *blobs, float *y_out, exl3m_job **jobp) {
    return prefill_submit_impl(c, x, false, false, n_tok, n_pairs, eidx, ridx, w, blobs, y_out, jobp);
}

// fp16 in / fp16 out: x rows as fp16 (uint16 bits), y_out rows as fp16 = the fp32 accumulated rows clamped to +-65504 and rounded to nearest even
int exl3m_moe_prefill_h_submit(exl3m_ctx *c, const uint16_t *x16, int n_tok, int64_t n_pairs, const int32_t *eidx, const int32_t *ridx,
                               const float *w, const void *const *blobs, uint16_t *y16_out, exl3m_job **jobp) {
    return prefill_submit_impl(c, x16, true, false, n_tok, n_pairs, eidx, ridx, w, blobs, y16_out, jobp);
}

int exl3m_moe_prefill(exl3m_ctx *c, const float *x, int n_tok, int64_t n_pairs, const int32_t *eidx, const int32_t *ridx, const float *w,
                      const void *const *blobs, float *y_out) {
    exl3m_job *j = nullptr;
    int rc = exl3m_moe_prefill_submit(c, x, n_tok, n_pairs, eidx, ridx, w, blobs, y_out, &j);
    return rc ? rc : exl3m_wait(c, j);
}

// ---------------------------------------------------------------------------------------------------------------------
// Exchange window (docs/strata-exchange-map.md 3.1-3.5, option A): one command buffer per verify window, one serial compute
// encoder, 7 dispatches per layer. D0 xw_wait_build spins on the layer's slot in Mac memory at system scope and builds the
// units/pairs/CSR tables and the indirect arguments on the GPU; D1-D6 are the decode kernels as indirect dispatches, with x
// read from and y written to the slot. No host work per layer, no thread, no CPU relay.
// the same seven dispatches run the NVFP4 kernels (gemv_nv4, mid_nv4, comb_nv4) in place of the trellis ones when
// exl3m_xch_set_format(c, 1) precedes the table: the indirect arguments are the same, so D0 and D6 do not change.
int exl3m_xch_attach(exl3m_ctx *c, void *region, size_t len) {
    @autoreleasepool {
        if (!region) return fail("null exchange region");
        if ((uintptr_t)region & (PAGE - 1)) return fail("the exchange region must be 16 KiB aligned");
        if (len < XCH_REGION_BYTES) return fail("the exchange region must hold XCH_REGION_BYTES");
        id<MTLBuffer> b = [c->dev newBufferWithBytesNoCopy:region length:len options:MTLResourceStorageModeShared deallocator:nil];
        if (!b) return fail("newBufferWithBytesNoCopy failed for the exchange region");
        c->xch_buf = b; c->xch_region = region; c->xch_len = len;
        return 0;
    }
}

// the format of the exchange window's table blobs, set before exl3m_xch_set_table. 0 = the EXL3 trellis (the default: the
// window as it has been, docs/strata-exchange-map.md 3.1-3.5), 1 = NVFP4 (blobs, gate | up | down of block_nvfp4 rows
// plus the fp32 tail, 2,764,816 bytes each: the window then runs the nv4 kernels in place of the trellis ones). Anything else is
// refused and leaves the context as it was.
extern "C" int exl3m_xch_set_format(exl3m_ctx *c, int fmt) {
    if (fmt != 0 && fmt != 1) return fail("the window format must be 0 (EXL3 trellis) or 1 (NVFP4)");
    if (c->xch_tbl && fmt != c->xch_fmt) return fail("exl3m_xch_set_format must precede exl3m_xch_set_table");
    c->xch_fmt = fmt;
    return 0;
}

int exl3m_xch_set_table(exl3m_ctx *c, const void *const *blob_ptrs, int n_layers, int n_experts) {
    @autoreleasepool {
        if (!c->xch_buf) return fail("exl3m_xch_attach first");
        if (n_layers < 1 || n_layers > (int)XCH_N_LAYERS) return fail("n_layers must be 1..48");
        if (n_experts < 1) return fail("n_experts must be >= 1");
        const size_t blob = c->xch_fmt ? NVFP4_BLOB : c->blob;   // an NVFP4 blob is 2764816 B at every trellis width
        const size_t n = (size_t)n_layers * (size_t)n_experts;
        std::vector<uint64_t> t(n, 0);
        std::vector<id<MTLBuffer>> used;
        for (size_t i = 0; i < n; i++) {
            const void *bp = blob_ptrs[i];
            if (!bp) continue;
            uint64_t addr; id<MTLBuffer> buf;
            if (!resolve(c, bp, addr, buf, blob)) return fail("blob " + std::to_string(i) + " is not in a registered region (the window path does not stage)");
            t[i] = addr;
            used.push_back(buf);
        }
        c->xch_tbl = mkbuf(c, n * 8);
        memcpy(c->xch_tbl.contents, t.data(), n * 8);
        {   // every registered piece, for the residency set and for the no-set fallback's useResource list
            std::shared_lock<std::shared_mutex> l(c->rmu);
            for (auto &r : c->regions) for (auto &pc : r.pieces) used.push_back(pc.buf);
        }
        c->xch_used = used;
        c->xch_nl = n_layers; c->xch_ne = n_experts;
        xch_rs_sync(c);   // macOS 15: the table's pieces go into the queue's set, rebuilt once the window's scratch exists
        return 0;
    }
}

int exl3m_xch_window_submit(exl3m_ctx *c, int n_layers, int n_rows, uint32_t epoch, uint32_t spin_budget) {
    @autoreleasepool {
        if (!c->xch_buf) return fail("exl3m_xch_attach first");
        if (!c->xch_tbl) return fail("exl3m_xch_set_table first");
        if (n_layers < 1 || n_layers > (int)XCH_N_LAYERS) return fail("n_layers must be 1..48");
        if (n_layers > c->xch_nl) return fail("n_layers exceeds the expert table");
        if (n_rows < 1 || n_rows > (int)XCH_MAX_ROWS) return fail("n_rows must be 1..8");
        // indirect dispatch cannot switch pipelines, so gemv_dec<MR, KB> is fixed per window at MR = next power of two >= n_rows;
        // each row's accumulation is independent of MR, so the results are unchanged (docs/strata-exchange-map.md 3.2)
        const uint32_t mrx = n_rows <= 1 ? 1u : n_rows <= 2 ? 2u : n_rows <= 4 ? 4u : 8u;
        const bool nv4 = c->xch_fmt != 0;   // NVFP4 blobs, so the nv4 kernels take the trellis stages' places
        if (nv4 && (!c->P.nv4_x || !c->P.nv4_mid || !c->P.nv4_comb_sys || !c->P.nv4_gemv[0][mrx] || !c->P.nv4_gemv[1][mrx]))
            return fail("the NVFP4 pipelines did not build on this device");
        if (!c->xch_dp) {
            c->xch_dp = mkbuf(c, (size_t)XCH_N_LAYERS * 16);
            c->xch_args = mkbuf(c, (size_t)XCH_N_LAYERS * 6 * 16);
            c->xch_tab = mkbuf(c, XT_END);
            c->xch_dead = mkbuf(c, 16);
            c->xch_xq = mkbuf(c, 2 * (size_t)NU_MAX * 8 * H * 2, PRIV);
            c->xch_yg = mkbuf(c, 2 * (size_t)NU_MAX * 8 * FF * 4, PRIV);
            c->xch_zq = mkbuf(c, (size_t)NU_MAX * 8 * FF * 2, PRIV);
            c->xch_yd = mkbuf(c, (size_t)NU_MAX * 8 * H * 4, PRIV);
        }
        if (nv4 && !c->xch_nx) {   // the NVFP4 window's scratch: the fp32 x rows and the fp32 mid rows (once per context)
            c->xch_nx = mkbuf(c, (size_t)XCH_MAX_ROWS * H * 4);
            c->xch_nz = mkbuf(c, (size_t)NU_MAX * 8 * FF * 4, PRIV);
        }
        xch_rs_sync(c);   // the set has to hold the window's scratch buffers too
        memset((char *)c->xch_region + XCH_H_MTL_STATUS, 0, 16);   // status = {code, layer, spins, aux}
        memset(c->xch_dead.contents, 0, 4);                        // the window's private dead word
        id<MTLCommandBuffer> cb = [c->q commandBuffer];
        id<MTLComputeCommandEncoder> en = [cb computeCommandEncoder];
        if (!c->xch_rs) xch_use_resources(c, en);   // fallback only: with the set the queue keeps those buffers resident
        const MTLSize tg1 = MTLSizeMake(1, 1, 1), tg128 = MTLSizeMake(128, 1, 1), tgG = MTLSizeMake(128, 1, 1);   // tgG: gemv_dec is SG = 4 simdgroups wide (split-K)
        const GPH g0 = { 0, {0, 0, 0} }, g1 = { 1, {0, 0, 0} };
        // the NVFP4 stages take the same indirect arguments, because gemv_nv4's 16-column tiles cover the same 40 and 160
        // grid columns; only the pipelines, the two GEMV threadgroups (32 * SG) and the fp32 mid rows differ.
        id<MTLComputePipelineState> p_x = nv4 ? c->P.nv4_x : c->P.rot_dec_sys;
        id<MTLComputePipelineState> p_gu = nv4 ? c->P.nv4_gemv[0][mrx] : c->P.gemv[c->bits][mrx];
        id<MTLComputePipelineState> p_mid = nv4 ? c->P.nv4_mid : c->P.mid_dec;
        id<MTLComputePipelineState> p_dn = nv4 ? c->P.nv4_gemv[1][mrx] : c->P.gemv[c->bits][mrx];
        id<MTLComputePipelineState> p_comb = nv4 ? c->P.nv4_comb_sys : c->P.comb_dec_sys;
        id<MTLBuffer> bx = nv4 ? c->xch_nx : c->xch_xq, bz = nv4 ? c->xch_nz : c->xch_zq;
        const MTLSize tg_gu = nv4 ? MTLSizeMake(32 * NV4_SG_GU, 1, 1) : tgG, tg_dn = nv4 ? MTLSizeMake(32 * NV4_SG_DN, 1, 1) : tgG;
        for (int l = 0; l < n_layers; l++) {
            const NSUInteger aoff = (NSUInteger)l * 6 * 16, doff = (NSUInteger)l * 16, soff = (NSUInteger)XCH_SLOT(l);
            const XWH w = { (uint32_t)l, epoch, spin_budget, (uint32_t)c->xch_ne, (uint32_t)n_rows, mrx, (uint32_t)c->bits, 0 };
            [en setComputePipelineState:c->P.xw_wait];
            [en setBuffer:c->xch_buf offset:0 atIndex:0]; [en setBuffer:c->xch_tbl offset:0 atIndex:1];
            [en setBuffer:c->xch_dp offset:doff atIndex:2]; [en setBuffer:c->xch_args offset:aoff atIndex:3];
            [en setBuffer:c->xch_tab offset:XT_UNITS atIndex:4]; [en setBuffer:c->xch_tab offset:XT_PAIRS atIndex:5];
            [en setBuffer:c->xch_tab offset:XT_ROWPTR atIndex:6]; [en setBytes:&w length:sizeof w atIndex:7];
            [en setBuffer:c->xch_dead offset:0 atIndex:8];
            [en dispatchThreadgroups:tg1 threadsPerThreadgroup:tg1];
            [en setComputePipelineState:p_x];
            [en setBuffer:c->xch_buf offset:soff + XCH_S_X atIndex:0]; [en setBuffer:c->xch_tab offset:XT_UNITS atIndex:1];
            [en setBuffer:bx offset:0 atIndex:2]; [en setBuffer:c->xch_dp offset:doff atIndex:3];
            [en dispatchThreadgroupsWithIndirectBuffer:c->xch_args indirectBufferOffset:aoff threadsPerThreadgroup:tg128];
            [en setComputePipelineState:p_gu];
            [en setBuffer:c->xch_tab offset:XT_UNITS atIndex:0]; [en setBuffer:bx offset:0 atIndex:1];
            [en setBuffer:c->xch_yg offset:0 atIndex:2]; [en setBuffer:c->xch_dp offset:doff atIndex:3];
            [en setBytes:&g0 length:sizeof g0 atIndex:4];
            [en dispatchThreadgroupsWithIndirectBuffer:c->xch_args indirectBufferOffset:aoff + 16 threadsPerThreadgroup:tg_gu];
            [en setComputePipelineState:p_mid];
            [en setBuffer:c->xch_tab offset:XT_UNITS atIndex:0]; [en setBuffer:c->xch_yg offset:0 atIndex:1];
            [en setBuffer:bz offset:0 atIndex:2]; [en setBuffer:c->xch_dp offset:doff atIndex:3];
            [en dispatchThreadgroupsWithIndirectBuffer:c->xch_args indirectBufferOffset:aoff + 32 threadsPerThreadgroup:tg128];
            [en setComputePipelineState:p_dn];
            [en setBuffer:c->xch_tab offset:XT_UNITS atIndex:0]; [en setBuffer:bz offset:0 atIndex:1];
            [en setBuffer:c->xch_yd offset:0 atIndex:2]; [en setBuffer:c->xch_dp offset:doff atIndex:3];
            [en setBytes:&g1 length:sizeof g1 atIndex:4];
            [en dispatchThreadgroupsWithIndirectBuffer:c->xch_args indirectBufferOffset:aoff + 48 threadsPerThreadgroup:tg_dn];
            [en setComputePipelineState:p_comb];
            [en setBuffer:c->xch_tab offset:XT_UNITS atIndex:0]; [en setBuffer:c->xch_yd offset:0 atIndex:1];
            [en setBuffer:c->xch_tab offset:XT_PAIRS atIndex:2]; [en setBuffer:c->xch_tab offset:XT_ROWPTR atIndex:3];
            [en setBuffer:c->xch_buf offset:soff + XCH_S_Y atIndex:4]; [en setBuffer:c->xch_dp offset:doff atIndex:5];
            [en dispatchThreadgroupsWithIndirectBuffer:c->xch_args indirectBufferOffset:aoff + 64 threadsPerThreadgroup:tg128];
            [en setComputePipelineState:c->P.xw_land];
            [en setBuffer:c->xch_buf offset:soff + XCH_S_LANDED atIndex:0]; [en setBuffer:c->xch_dead offset:0 atIndex:1];
            [en setBytes:&epoch length:4 atIndex:2];
            [en dispatchThreadgroupsWithIndirectBuffer:c->xch_args indirectBufferOffset:aoff + 80 threadsPerThreadgroup:tg1];
        }
        [en endEncoding];
        [cb commit];
        c->last_commit_ms = now_ms();   // activity: the keep-warm thread stays quiet while work flows
        c->xch_cb = cb;
        return 0;
    }
}

int exl3m_xch_window_wait(exl3m_ctx *c, uint32_t status[4]) {
    @autoreleasepool {
        if (!c->xch_cb) return fail("no exchange window in flight");
        [c->xch_cb waitUntilCompleted];
        id<MTLCommandBuffer> cb = c->xch_cb; c->xch_cb = nil;
        const uint32_t *st = (const uint32_t *)((const char *)c->xch_region + XCH_H_MTL_STATUS);
        for (int i = 0; i < 4; i++) status[i] = st[i];
        if (cb.status != MTLCommandBufferStatusCompleted) {
            NSError *e = cb.error;
            return fail(std::string("window command buffer failed: ") + (e ? e.localizedDescription.UTF8String : "?"));
        }
        if (status[0] != XCH_ST_OK) return fail("window ended with status " + std::to_string(status[0]) + " at layer " + std::to_string(status[1]));
        return 0;
    }
}

}  // extern "C"

// the prompt lane over NVFP4 blobs, exl3m_moe_prefill's contract through the prompt pipeline (the expert-sorted chunks,
// the decoded fp16 weights, the GEMMs and the combine of prefill_submit_impl), not 8 rows per exl3m_moe_rows_nvfp4 call (10-28
// our test machine: that first version read 512 tokens in 39.6 ms against the EXL3 prompt kernels' 31.1 ms). Not declared in exl3_metal.h
// on purpose, as exl3m_moe_rows_nvfp4: Strata mirrors that header, the entry points are these extern "C" symbols.
extern "C" int exl3m_moe_prefill_nvfp4_submit(exl3m_ctx *c, const float *x, int n_tok, int64_t n_pairs, const int32_t *eidx,
                                              const int32_t *ridx, const float *w, const void *const *blobs, float *y_out, exl3m_job **jobp) {
    return prefill_submit_impl(c, x, false, true, n_tok, n_pairs, eidx, ridx, w, blobs, y_out, jobp);
}
// the fp16 in / fp16 out contract of exl3m_moe_prefill_h_submit
extern "C" int exl3m_moe_prefill_nvfp4_h_submit(exl3m_ctx *c, const uint16_t *x16, int n_tok, int64_t n_pairs, const int32_t *eidx,
                                                const int32_t *ridx, const float *w, const void *const *blobs, uint16_t *y16_out, exl3m_job **jobp) {
    return prefill_submit_impl(c, x16, true, true, n_tok, n_pairs, eidx, ridx, w, blobs, y16_out, jobp);
}
extern "C" int exl3m_moe_prefill_nvfp4(exl3m_ctx *c, const float *x, int n_tok, int64_t n_pairs, const int32_t *eidx, const int32_t *ridx,
                                       const float *w, const void *const *blobs, float *y_out) {
    exl3m_job *j = nullptr;
    int rc = exl3m_moe_prefill_nvfp4_submit(c, x, n_tok, n_pairs, eidx, ridx, w, blobs, y_out, &j);
    return rc ? rc : exl3m_wait(c, j);
}
