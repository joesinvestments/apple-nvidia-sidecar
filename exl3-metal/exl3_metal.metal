// exl3_metal.metal - EXL3 MUL1 expert MoE compute for the Apple GPU (2560x640 experts: gate/up [2560,640], down [640,2560]).
// The trellis width K (3, 4 or 5; docs/exl3-kbits-map.md) comes in with the job's parameters: a tile is 8K u32 words, a trellis
// 204800*K bytes, and the nine blob field offsets follow K. decode8<K> is the one recipe; its K=3 instance is the shipped K=3 code.
// Numerics mirror exl3/cpu/exl3_cpu.c + Strata pool.cpp exl3_task: x fp32 -> fp16 -> * suh (fp16 round) -> FWHT128 fp32 (stages 1..64)
// -> * R_SCALE -> fp16 ; GEMV/GEMM with fp16 operands and fp32 accumulation ; had_out = FWHT128 fp32, * INV * svh ; SwiGLU in fp32 ->
// fp16 ; down likewise ; y = sum_pairs weight * expert(x[row]).  Compiled offline without fast-math / fp-contract.
#pragma METAL internals : enable   // metal::thread_scope (the system-scope fence of the exchange window kernels, see xw_wait_build)
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>   // mpp::tensor_ops::matmul2d (the pf_tensor_ops GEMMs, Metal 4)
#pragma METAL internals : enable   // the header above ends with "internals : disable"; keep metal::thread_scope visible below
#include "../include/strata/xch/xch_layout.h"   // the GPU <-> Metal exchange region (docs/strata-exchange-map.md 3.3)
using namespace metal;
#pragma clang fp contract(off)

#define H 2560u
#define FF 640u
// blob field offsets for the trellis width k (exl3_pack.json, docs/exl3-kbits-map.md): trellis 204800*k each, then the six fp16 vectors.
inline ulong off_gt(uint k) { return 0ul; }
inline ulong off_ut(uint k) { return 204800ul * k; }
inline ulong off_dt(uint k) { return 409600ul * k; }
inline ulong off_gsuh(uint k) { return 614400ul * k; }
inline ulong off_gsvh(uint k) { return 614400ul * k + 5120ul; }
inline ulong off_usuh(uint k) { return 614400ul * k + 6400ul; }
inline ulong off_usvh(uint k) { return 614400ul * k + 11520ul; }
inline ulong off_dsuh(uint k) { return 614400ul * k + 12800ul; }
inline ulong off_dsvh(uint k) { return 614400ul * k + 14080ul; }

constant float R_SCALE = 0.088388347648f;
constant float INV = 0.08838834764831845f;

// ---------------------------------------------------------------------------------------------------------------------
inline half cb_mul1(uint s) {
  uint x = s * 0x83DCD12Du;
  uint p = (x & 0x00FF00FFu) + ((x >> 8) & 0x00FF00FFu);      // two 16-bit lanes of 2-byte sums (<= 510)
  // low lane + high lane, truncated: the sum of the 4 bytes (<= 1020, so nothing carries between the lanes). The half's
  // pattern is 0x6400 + b, and b < 0x400 shares no bit with 0x6400, so | 0x6400 is the same add.
  half h = as_type<half>((ushort)((p + (p >> 16)) | 0x6400u));
  return fma(h, as_type<half>((ushort)0x1EEE), as_type<half>((ushort)0xC931));
}

// The 8 consecutive states of lane t of a tile (8K uint32 words, stream MSB first, circular; element e's 16-bit window ends at
// bit (e+1)*K), decoded to halves in jj order. Lane t's 8 windows start K bits into the K+2-byte span that begins at stream byte
// K*t-2 (mod 32K), the same layout for every lane, so every shift below is a compile-time constant for a given K.
// K=3 is the shipped arithmetic: sb = (3t+94) % 96 and the window shifts 48-p for p = 3, 6, ... 24 in the 64-bit span word.
inline half cb_state(uint s) { return cb_mul1(s); }
template <uint K>
inline void decode8(const device uint* w, uint t, thread half* v) {
  constexpr uint TW = 8u * K, TB = 32u * K;                   // u32 words and bytes per tile: 24 / 96 at K=3
  const uint sb = (K * t + TB - 2u) % TB;
  const uint wi = sb >> 2, sh = (sb & 3u) * 8u;
  const uint i1 = (wi + 1u) % TW, i2 = (wi + 2u) % TW;
  const uint w0 = w[wi], w1 = w[i1], w2 = w[i2];
  // span bytes 0..7 as one 64-bit word: the three tile words rotated left by sh. The low sh bits of the low word come from
  // w2, and the 64-bit shift of that term needs no sh == 0 branch.
  const ulong d = ((((ulong)w0 << 32) | (ulong)w1) << sh) | (uint)(((ulong)w2 << sh) >> 32);
  for (uint jj = 0; jj < 8u; jj++) {
    const uint p = (jj + 1u) * K;                                  // window jj starts at span bit p, 16 bits MSB first
    v[jj] = cb_state((uint)(d >> (48u - p)) & 0xFFFFu);
  }
}
// the decode for the job's width: only 3, 4 and 5 exist (exl3m_set_param refuses others), and K=3 stays the shipped code
inline void decode_bits(uint k, const device uint* w, uint t, thread half* v) {
  if (k == 4u) decode8<4>(w, t, v);
  else if (k == 5u) decode8<5>(w, t, v);
  else decode8<3>(w, t, v);
}

// FWHT128 over a 128-thread threadgroup (thread i holds element i): stages 1..16 by simd shuffles, 32 and 64 through TG memory.
// Same operations in the same order as the sequential butterfly -> bitwise equal.
inline float fwht128(float v, uint i, threadgroup float* buf) {
  for (uint st = 1u; st < 32u; st <<= 1) {
    float o = simd_shuffle_xor(v, (ushort)st);
    v = (i & st) ? (o - v) : (v + o);
  }
  for (uint st = 32u; st <= 64u; st <<= 1) {
    buf[i] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float o = buf[i ^ st];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    v = (i & st) ? (o - v) : (v + o);
  }
  return v;
}

// permuted-input layout of the GEMV: element e of a row -> ((e/16)*4 + q)*4 + slot with q = ((e%16)%8)/2, slot = (e%2) | (e%16/8)*2
inline uint xq_index(uint e) {
  uint i = e >> 4, r = e & 15u;
  return ((i * 4u + ((r & 7u) >> 1)) << 2) + (r & 1u) + ((r >> 3) << 1);
}

// ---------------------------------------------------------------------------------------------------------------------
// DECODE shape: unit = one distinct expert with up to mrx rows (slots).
struct UnitD { device const uchar* blob; int rows[8]; };       // rows[slot]: row of x, -1 = unused
struct PairD { uint u; uint slot; float w; uint pad; };         // pair: unit, slot, weight
struct DP { uint mrx; uint nu; uint bits; uint pad; };

// A: rotate inputs for gate (proj 0) and up (proj 1). grid (20 blocks, nu*mrx, 2), TG 128. xq [proj][u][slot][H] permuted.
kernel void rot_in_dec(device const float* x [[buffer(0)]], device const UnitD* units [[buffer(1)]], device half* xq [[buffer(2)]],
                       constant DP& p [[buffer(3)]], uint3 tg [[threadgroup_position_in_grid]], uint i [[thread_index_in_threadgroup]]) {
  threadgroup float buf[128];
  const uint blk = tg.x, u = tg.y / p.mrx, slot = tg.y % p.mrx, proj = tg.z;
  const int row = units[u].rows[slot];
  float v = 0.f;
  if (row >= 0) {
    half a = (half)x[(ulong)row * H + blk * 128u + i];
    half s = ((device const half*)(units[u].blob + (proj ? off_usuh(p.bits) : off_gsuh(p.bits))))[blk * 128u + i];
    v = (float)(half)((float)a * (float)s);
  }
  v = fwht128(v, i, buf);
  xq[(((ulong)proj * p.nu + u) * p.mrx + slot) * H + xq_index(blk * 128u + i)] = (half)(v * R_SCALE);
}

// B: fused trellis decode + GEMV. SG simdgroups (32 lanes = the 32 lane-groups of a tile) per (16-column tile, unit, proj),
// each simdgroup walking one contiguous slice of K; the partial sums of the slices meet in threadgroup memory (one
// 32-lane simdgroup per tile left too little work in flight for the 40 cores of the M5 Max).
// kind 0: gate/up (k=2560, n=640) grid (40, nu, 2); kind 1: down (k=640, n=2560) grid (160, nu, 1). TG = 32 * SG threads.
struct GP { uint kind; uint pad[3]; };
// KB is the trellis width K as a compile-time constant: the tile stride and the field offsets fold into the inner loop
template <uint MR, uint KB, uint SG>
kernel void gemv_dec(device const UnitD* units [[buffer(0)]], device const half* xin [[buffer(1)]], device float* yout [[buffer(2)]],
                     constant DP& p [[buffer(3)]], constant GP& g [[buffer(4)]],
                     uint3 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]]) {
  const uint k = g.kind ? FF : H, n = g.kind ? H : FF, kt = k / 16u, nt = n / 16u;
  const uint j = tg.x, u = tg.y, proj = tg.z;
  device const uchar* blob = units[u].blob;
  const device uint* tr = (const device uint*)(blob + (g.kind ? off_dt(KB) : (proj ? off_ut(KB) : off_gt(KB))));
  const ulong ub = g.kind ? (ulong)u : ((ulong)proj * p.nu + u);
  device const half* x = xin + ub * MR * k;
  device float* y = yout + ub * MR * n;
  const uint sg = lane >> 5, il = lane & 31u, c = il >> 2, q = il & 3u;   // this simdgroup's K slice, the tile's lane
  threadgroup float red[SG * 16u * MR];                       // one partial sum a (slice, row, lane): the tile's columns c and c + 8
  float accA[MR], accB[MR];
  for (uint r = 0; r < MR; r++) { accA[r] = 0.f; accB[r] = 0.f; }
  const uint chunk = (kt + SG - 1u) / SG, i0 = sg * chunk, i1 = min(kt, i0 + chunk);   // contiguous K slices, kt % SG = 0 at both kinds
  device const uint* tp = tr + (ulong)j * (8u * KB) + (ulong)i0 * (nt * (8u * KB));
  device const half* xp = x + q * 4u + i0 * 16u;              // walk the tile and x with pointer adds: no per-i index math
  const uint tstep = nt * (8u * KB);
  #pragma clang loop unroll_count(2)                          // two tiles per iteration: the second tile's loads issue while the first hashes
  for (uint i = i0; i < i1; i++, tp += tstep, xp += 16u) {
    half w[8];
    decode8<KB>(tp, lane, w);
    for (uint r = 0; r < MR; r++) {
      half4 xv = *(device const half4*)(xp + (ulong)r * k);
      float a = accA[r], b = accB[r];
      // a half x half product is exact in fp32, so fma rounds exactly like mul then add, one op less
      a = fma((float)w[0], (float)xv[0], a); a = fma((float)w[1], (float)xv[1], a);
      a = fma((float)w[2], (float)xv[2], a); a = fma((float)w[3], (float)xv[3], a);
      b = fma((float)w[4], (float)xv[0], b); b = fma((float)w[5], (float)xv[1], b);
      b = fma((float)w[6], (float)xv[2], b); b = fma((float)w[7], (float)xv[3], b);
      accA[r] = a; accB[r] = b;
    }
  }
  for (uint r = 0; r < MR; r++) {                             // the 4 q lanes of a slice add up first, in registers
    float a = accA[r], b = accB[r];
    a += simd_shuffle_xor(a, (ushort)1); a += simd_shuffle_xor(a, (ushort)2);
    b += simd_shuffle_xor(b, (ushort)1); b += simd_shuffle_xor(b, (ushort)2);
    accA[r] = a; accB[r] = b;
  }
  if (SG > 1u) {                                              // then the SG slices, through red: the order changes, the sum does not
    if (q == 0) for (uint r = 0; r < MR; r++) { red[(sg * MR + r) * 16u + c] = accA[r]; red[(sg * MR + r) * 16u + 8u + c] = accB[r]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (sg == 0u && q == 0) {
    for (uint r = 0; r < MR; r++) {
      float a = accA[r], b = accB[r];
      for (uint s = 1; s < SG; s++) { a += red[(s * MR + r) * 16u + c]; b += red[(s * MR + r) * 16u + 8u + c]; }
      y[(ulong)r * n + 16u * j + c] = a; y[(ulong)r * n + 16u * j + c + 8u] = b;
    }
  }
}
// twelve instances (SG = 4 simdgroups = 128-thread groups): the host picks gemv_dec_<MR>_k<K> from the context's bits (K) and the job's slots (MR)
template [[host_name("gemv_dec_1_k3")]] kernel void gemv_dec<1, 3, 4>(device const UnitD*, device const half*, device float*, constant DP&, constant GP&, uint3, uint);
template [[host_name("gemv_dec_2_k3")]] kernel void gemv_dec<2, 3, 4>(device const UnitD*, device const half*, device float*, constant DP&, constant GP&, uint3, uint);
template [[host_name("gemv_dec_4_k3")]] kernel void gemv_dec<4, 3, 4>(device const UnitD*, device const half*, device float*, constant DP&, constant GP&, uint3, uint);
template [[host_name("gemv_dec_8_k3")]] kernel void gemv_dec<8, 3, 4>(device const UnitD*, device const half*, device float*, constant DP&, constant GP&, uint3, uint);
template [[host_name("gemv_dec_1_k4")]] kernel void gemv_dec<1, 4, 4>(device const UnitD*, device const half*, device float*, constant DP&, constant GP&, uint3, uint);
template [[host_name("gemv_dec_2_k4")]] kernel void gemv_dec<2, 4, 4>(device const UnitD*, device const half*, device float*, constant DP&, constant GP&, uint3, uint);
template [[host_name("gemv_dec_4_k4")]] kernel void gemv_dec<4, 4, 4>(device const UnitD*, device const half*, device float*, constant DP&, constant GP&, uint3, uint);
template [[host_name("gemv_dec_8_k4")]] kernel void gemv_dec<8, 4, 4>(device const UnitD*, device const half*, device float*, constant DP&, constant GP&, uint3, uint);
template [[host_name("gemv_dec_1_k5")]] kernel void gemv_dec<1, 5, 4>(device const UnitD*, device const half*, device float*, constant DP&, constant GP&, uint3, uint);
template [[host_name("gemv_dec_2_k5")]] kernel void gemv_dec<2, 5, 4>(device const UnitD*, device const half*, device float*, constant DP&, constant GP&, uint3, uint);
template [[host_name("gemv_dec_4_k5")]] kernel void gemv_dec<4, 5, 4>(device const UnitD*, device const half*, device float*, constant DP&, constant GP&, uint3, uint);
template [[host_name("gemv_dec_8_k5")]] kernel void gemv_dec<8, 5, 4>(device const UnitD*, device const half*, device float*, constant DP&, constant GP&, uint3, uint);

// C: had_out(gate), had_out(up), SwiGLU -> fp16, rotate for down. grid (5, nu*mrx), TG 128. zq [u][slot][FF] permuted.
kernel void mid_dec(device const UnitD* units [[buffer(0)]], device const float* yg [[buffer(1)]], device half* zq [[buffer(2)]],
                    constant DP& p [[buffer(3)]], uint2 tg [[threadgroup_position_in_grid]], uint i [[thread_index_in_threadgroup]]) {
  threadgroup float buf[128];
  const uint blk = tg.x, u = tg.y / p.mrx, slot = tg.y % p.mrx;
  device const uchar* blob = units[u].blob;
  const ulong rowi = (ulong)u * p.mrx + slot, plane = (ulong)p.nu * p.mrx * FF;
  const uint col = blk * 128u + i;
  float vg = fwht128(yg[rowi * FF + col], i, buf);
  float vu = fwht128(yg[plane + rowi * FF + col], i, buf);
  float gg = vg * INV * (float)((device const half*)(blob + off_gsvh(p.bits)))[col];
  float uu = vu * INV * (float)((device const half*)(blob + off_usvh(p.bits)))[col];
  half z = (half)(gg / (1.f + precise::exp(-gg)) * uu);
  float v = (float)(half)((float)z * (float)((device const half*)(blob + off_dsuh(p.bits)))[col]);
  v = fwht128(v, i, buf);
  zq[rowi * FF + xq_index(col)] = (half)(v * R_SCALE);
}

// FWHT128 held by one warp, 4 elements a lane (element i = 4 * lane + e): stages 1 and 2 pair a lane's own elements, 4 to 64 pair
// lanes, so the same butterflies in the same order as fwht128 come out bitwise equal with no threadgroup memory and no barrier.
inline float4 fwht128_warp(float4 v, uint lane) {
  float a = v.x + v.y, b = v.x - v.y, c = v.z + v.w, d = v.z - v.w;
  v = float4(a + c, b + d, a - c, b - d);
  for (uint st = 4u; st <= 64u; st <<= 1) {
    ushort sl = (ushort)(st >> 2);
    float4 o = float4(simd_shuffle_xor(v.x, sl), simd_shuffle_xor(v.y, sl), simd_shuffle_xor(v.z, sl), simd_shuffle_xor(v.w, sl));
    v = (lane & (st >> 2)) ? (o - v) : (v + o);
  }
  return v;
}

// E: had_out(down) and the weighted sum over pairs. grid (20, nrows), TG 128. out [row][H].
// Four warps take a row's pairs four at a time (warp w has pi0 + w, pi0 + w + 4, ...) instead of walking them in series, and
// their partial sums meet in red.
kernel void comb_dec(device const UnitD* units [[buffer(0)]], device const float* yd [[buffer(1)]], device const PairD* pairs [[buffer(2)]],
                     device const uint* rowptr [[buffer(3)]], device float* out [[buffer(4)]], constant DP& p [[buffer(5)]],
                     uint2 tg [[threadgroup_position_in_grid]], uint i [[thread_index_in_threadgroup]]) {
  threadgroup float red[4][128];
  const uint blk = tg.x, row = tg.y, w = i >> 5, lane = i & 31u, col = blk * 128u + 4u * lane;
  float4 acc = float4(0.f);
  for (uint pi = rowptr[row] + w; pi < rowptr[row + 1]; pi += 4u) {
    PairD pr = pairs[pi];
    float4 v = fwht128_warp(*(const device float4*)(yd + ((ulong)pr.u * p.mrx + pr.slot) * H + col), lane);
    half4 sh = *(const device half4*)((device const half*)(units[pr.u].blob + off_dsvh(p.bits)) + col);
    float4 o = (v * INV) * float4(sh);
    acc += pr.w * o;
  }
  red[w][4u * lane + 0u] = acc.x; red[w][4u * lane + 1u] = acc.y;
  red[w][4u * lane + 2u] = acc.z; red[w][4u * lane + 3u] = acc.w;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  out[(ulong)row * H + blk * 128u + i] = (red[0][i] + red[1][i]) + (red[2][i] + red[3][i]);
}

// ---------------------------------------------------------------------------------------------------------------------
// NVFP4: NVIDIA's QAD routed experts. One blob = gate 640x2560 | up 640x2560 | down 2560x640, every row k/64
// blocks of 36 B (4 UE4M3 scale bytes, one per 16 values, then 32 code bytes: byte s*8 + j of a block holds value j (low
// nibble) and value j + 8 (high nibble) of sub-block s), then a 16-byte fp32 tail {s_gate, s_up, s_down, 0}. No input
// rotation and no trellis: expert(x) = s_down * D(silu(s_gate * G x) * (s_up * U x)), the tail scales applied to the fp32
// GEMV results (folding them into fp16 weights loses 2-12% to subnormals).
#define NV4_GT 0ul
#define NV4_UP 921600ul      // 640 rows x 40 blocks x 36 B
#define NV4_DN 1843200ul     // 2560 rows x 10 blocks x 36 B
#define NV4_TAIL 2764800ul   // 16 B: s_gate, s_up, s_down, 0
#define NV4_SG_GU 4u         // gate/up: 4 simdgroups (128 threads) split the 40 blocks, 4 per row
#define NV4_NC_GU 2u         // and a lane decodes 2 columns, so a threadgroup covers 8 * NC = 16 rows
#define NV4_SG_DN 2u         // down: 10 blocks over 2 simdgroups (64 threads)
#define NV4_NC_DN 2u

// kvalues_fp4[c] = 0,1,2,3,4,6,8,12, 0,-1,-2,-3,-4,-6,-8,-12: the magnitudes in 4-bit fields of MAG
#define NV4_MAG 0xC8643210u
inline float nv4_val(uint c) {
  const float m = (float)((NV4_MAG >> ((c & 7u) << 2)) & 0xFu);
  return (c & 8u) ? -m : m;
}
// the block scale ue4m3(x) * 0.5: 0 for 0 and 0x7F, else (1 + m/8) * 2^(e-8), and m * 2^-10 for e = 0
inline float nv4_scale(uint x) {
  if (x == 0u || x == 0x7Fu) return 0.f;
  const uint e = (x >> 3) & 0xFu, m = x & 7u;
  return (e == 0u) ? (float)m * 0x1p-10f : as_type<float>(((e + 119u) << 23) | (m << 20));
}

// A: fused block decode + GEMV, fp32 accumulate. One threadgroup per (8*NC-column tile, unit, proj); SG simdgroups split
// the row into contiguous block ranges and lane il takes sub-block q = il & 3 of columns c = il >> 2, c + 8, ... c + 8(NC-1).
// kind 0: gate/up (k = 2560, n = 640), x [m][H] fp32, yout [2][nu][MR][FF], grid (640/(8*NC), nu, 2), TG 128 (SG = 4);
// kind 1: down (k = 640, n = 2560), x = the mid rows [(u * MR + slot)][FF] fp32, yout [nu][MR][H], grid (2560/(8*NC), nu, 1), TG 64.
template <uint MR, uint SG, uint NC>
kernel void gemv_nv4(device const UnitD* units [[buffer(0)]], device const float* xin [[buffer(1)]], device float* yout [[buffer(2)]],
                     constant DP& p [[buffer(3)]], constant GP& g [[buffer(4)]],
                     uint3 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]]) {
  const uint k = g.kind ? FF : H, n = g.kind ? H : FF, kb = k >> 6;   // 64 values in a 36-byte block
  const uint j = tg.x, u = tg.y, proj = tg.z;
  const device uchar* w0 = units[u].blob + (g.kind ? NV4_DN : (proj ? NV4_UP : NV4_GT));
  const ulong ub = g.kind ? (ulong)u : ((ulong)proj * p.nu + u);
  device float* y = yout + ub * MR * n;
  const uint sg = lane >> 5, il = lane & 31u, c = il >> 2, q = il & 3u;   // this simdgroup's block range, the tile's lane
  const uint cs = 8u * NC;                                   // the tile's columns: c, c + 8, ... c + 8(NC - 1)
  threadgroup float red[SG * MR * (8u * NC)];                // one partial sum a (slice, row, column)
  threadgroup half2 kv2[256];   // a code byte's two values (.x = the low nibble, .y = the high), and the 128 scale bytes
  threadgroup float kvs[128];
  const uint nt = SG * 32u;
  for (uint t = lane; t < 256u; t += nt) kv2[t] = half2((half)nv4_val(t & 15u), (half)nv4_val(t >> 4));
  for (uint t = lane; t < 128u; t += nt) kvs[t] = nv4_scale(t);
  ulong xoff[MR];   // the unit's input rows: gate/up x[units[u].rows[slot]], down its mid rows; an unused slot reads row 0
  for (uint r = 0; r < MR; r++) xoff[r] = (ulong)(g.kind ? (ub * MR + r) : (ulong)max(units[u].rows[r], 0)) * (ulong)k;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float acc[NC * MR];
  for (uint t = 0; t < NC * MR; t++) acc[t] = 0.f;
  const uint chunk = (kb + SG - 1u) / SG, i0 = sg * chunk, i1 = min(kb, i0 + chunk);   // contiguous block ranges
  device const uchar* wA = w0 + (ulong)(j * cs + c) * (kb * 36u) + (ulong)i0 * 36u;
  const ulong rb8 = (ulong)(kb * 36u) * 8u;                  // row step from column c to column c + 8
  #pragma clang loop unroll_count(2)   // two blocks per iteration: the next block's weight bytes issue while this one decodes
  for (uint i = i0; i < i1; i++, wA += 36u) {
    const uint xb = (i << 6) + (q << 4);
    // every column of the tile decodes once into registers, then each row's 16 x values load once for all of them: at
    // m >= 2 the x loads outnumber the weight bytes, and NC columns per lane are what keeps them off the GEMV's back
    float2 tv[NC][8]; float dss[NC];
    #pragma clang loop unroll(full)
    for (uint cc = 0; cc < NC; cc++) {
      device const uchar* wb = wA + cc * rb8;
      dss[cc] = kvs[wb[q]];                                                        // this sub-block's scale byte
      const uchar4 b0 = as_type<uchar4>(*(device const uint*)(wb + 4u + (q << 3)));   // its 8 code bytes: the
      const uchar4 b1 = as_type<uchar4>(*(device const uint*)(wb + 8u + (q << 3)));   // values v and v + 8 a byte
      const half2 h0 = kv2[b0.x], h1 = kv2[b0.y], h2 = kv2[b0.z], h3 = kv2[b0.w];
      const half2 h4 = kv2[b1.x], h5 = kv2[b1.y], h6 = kv2[b1.z], h7 = kv2[b1.w];
      tv[cc][0] = float2((float)h0.x, (float)h0.y); tv[cc][1] = float2((float)h1.x, (float)h1.y);
      tv[cc][2] = float2((float)h2.x, (float)h2.y); tv[cc][3] = float2((float)h3.x, (float)h3.y);
      tv[cc][4] = float2((float)h4.x, (float)h4.y); tv[cc][5] = float2((float)h5.x, (float)h5.y);
      tv[cc][6] = float2((float)h6.x, (float)h6.y); tv[cc][7] = float2((float)h7.x, (float)h7.y);
    }
    #pragma clang loop unroll(full)
    for (uint r = 0; r < MR; r++) {   // the sub-blocks' dot products, then one scaled add into each accumulator
      device const float* xp = xin + xoff[r] + xb;
      const float4 x0 = *(device const float4*)(xp), x1 = *(device const float4*)(xp + 4);
      const float4 x2 = *(device const float4*)(xp + 8), x3 = *(device const float4*)(xp + 12);
      #pragma clang loop unroll(full)
      for (uint cc = 0; cc < NC; cc++) {
        const float2 t0 = tv[cc][0], t1 = tv[cc][1], t2 = tv[cc][2], t3 = tv[cc][3];
        const float2 t4 = tv[cc][4], t5 = tv[cc][5], t6 = tv[cc][6], t7 = tv[cc][7];
        float s = t0.x * x0.x;
        s = fma(t1.x, x0.y, s); s = fma(t2.x, x0.z, s); s = fma(t3.x, x0.w, s);
        s = fma(t4.x, x1.x, s); s = fma(t5.x, x1.y, s); s = fma(t6.x, x1.z, s); s = fma(t7.x, x1.w, s);
        s = fma(t0.y, x2.x, s); s = fma(t1.y, x2.y, s); s = fma(t2.y, x2.z, s); s = fma(t3.y, x2.w, s);
        s = fma(t4.y, x3.x, s); s = fma(t5.y, x3.y, s); s = fma(t6.y, x3.z, s); s = fma(t7.y, x3.w, s);
        acc[cc * MR + r] = fma(dss[cc], s, acc[cc * MR + r]);
      }
    }
  }
  #pragma clang loop unroll(full)
  for (uint t = 0; t < NC * MR; t++) {                        // the 4 q lanes of a slice add up first, in registers
    float a = acc[t];
    a += simd_shuffle_xor(a, (ushort)1); a += simd_shuffle_xor(a, (ushort)2);
    acc[t] = a;
  }
  if (SG > 1u) {                                              // then the SG slices, through red: the order changes, the sum does not
    if (q == 0) for (uint cc = 0; cc < NC; cc++) for (uint r = 0; r < MR; r++) red[(sg * MR + r) * cs + 8u * cc + c] = acc[cc * MR + r];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (sg == 0u && q == 0) {
    for (uint cc = 0; cc < NC; cc++) for (uint r = 0; r < MR; r++) {
      float a = acc[cc * MR + r];
      for (uint s = 1; s < SG; s++) a += red[(s * MR + r) * cs + 8u * cc + c];
      y[(ulong)r * n + cs * j + 8u * cc + c] = a;
    }
  }
}
// eight instances (MR = the job's slots, SG = the simdgroups that split the row's k blocks and NC = the columns a lane
// decodes; gate/up and down differ because 40 and 10 blocks split differently over 4 and 2 simdgroups)
template [[host_name("gemv_nv4_1")]] kernel void gemv_nv4<1, NV4_SG_GU, NV4_NC_GU>(device const UnitD*, device const float*, device float*, constant DP&, constant GP&, uint3, uint);
template [[host_name("gemv_nv4_2")]] kernel void gemv_nv4<2, NV4_SG_GU, NV4_NC_GU>(device const UnitD*, device const float*, device float*, constant DP&, constant GP&, uint3, uint);
template [[host_name("gemv_nv4_4")]] kernel void gemv_nv4<4, NV4_SG_GU, NV4_NC_GU>(device const UnitD*, device const float*, device float*, constant DP&, constant GP&, uint3, uint);
template [[host_name("gemv_nv4_8")]] kernel void gemv_nv4<8, NV4_SG_GU, NV4_NC_GU>(device const UnitD*, device const float*, device float*, constant DP&, constant GP&, uint3, uint);
template [[host_name("gemv_nv4_dn1")]] kernel void gemv_nv4<1, NV4_SG_DN, NV4_NC_DN>(device const UnitD*, device const float*, device float*, constant DP&, constant GP&, uint3, uint);
template [[host_name("gemv_nv4_dn2")]] kernel void gemv_nv4<2, NV4_SG_DN, NV4_NC_DN>(device const UnitD*, device const float*, device float*, constant DP&, constant GP&, uint3, uint);
template [[host_name("gemv_nv4_dn4")]] kernel void gemv_nv4<4, NV4_SG_DN, NV4_NC_DN>(device const UnitD*, device const float*, device float*, constant DP&, constant GP&, uint3, uint);
template [[host_name("gemv_nv4_dn8")]] kernel void gemv_nv4<8, NV4_SG_DN, NV4_NC_DN>(device const UnitD*, device const float*, device float*, constant DP&, constant GP&, uint3, uint);

// B: the tail scales, SiLU and the product, all in fp32: z [(u * mrx + slot)][FF] = silu(s_gate * G x) * (s_up * U x).
// grid (5, nu*mrx), TG 128. ygu [2][nu][mrx][FF] fp32 (the gate and up GEMV results side by side).
kernel void mid_nv4(device const UnitD* units [[buffer(0)]], device const float* ygu [[buffer(1)]], device float* z [[buffer(2)]],
                    constant DP& p [[buffer(3)]], uint2 tg [[threadgroup_position_in_grid]], uint i [[thread_index_in_threadgroup]]) {
  const uint u = tg.y / p.mrx, slot = tg.y % p.mrx;
  const ulong rowi = (ulong)u * p.mrx + slot, plane = (ulong)p.nu * p.mrx * FF;
  const uint col = tg.x * 128u + i;
  device const float* tail = (device const float*)(units[u].blob + NV4_TAIL);
  const float g = ygu[rowi * FF + col] * tail[0];
  const float uu = ygu[plane + rowi * FF + col] * tail[1];
  z[rowi * FF + col] = g / (1.f + precise::exp(-g)) * uu;
}

// C: s_down and the weighted sum over the pairs, as comb_dec does it (four warps take a row's pairs four at a time).
// s_down multiplies here, on the down GEMV's fp32 output: the rule is s_down * D(silu(s_gate * G x) * (s_up * U x)), so the
// tail scales are never folded into the weights (fp16 subnormals cost 2-12%).
// grid (20, m), TG 128. out [row][H] fp32, overwritten: rows without a pair are zero.
// The body is shared with the exchange window's comb_nv4_sys, so both add the pairs up the same way.
template <typename Out>
inline void comb_nv4_body(device const UnitD* units, device const float* yd, device const PairD* pairs, device const uint* rowptr,
                          threadgroup float red[4][128], Out out, constant DP& p, uint2 tg, uint i) {
  const uint blk = tg.x, row = tg.y, w = i >> 5, lane = i & 31u, col = blk * 128u + 4u * lane;
  float4 acc = float4(0.f);
  for (uint pi = rowptr[row] + w; pi < rowptr[row + 1]; pi += 4u) {
    PairD pr = pairs[pi];
    float4 v = *(const device float4*)(yd + ((ulong)pr.u * p.mrx + pr.slot) * H + col);
    acc += (pr.w * ((device const float*)(units[pr.u].blob + NV4_TAIL))[2]) * v;
  }
  red[w][4u * lane + 0u] = acc.x; red[w][4u * lane + 1u] = acc.y;
  red[w][4u * lane + 2u] = acc.z; red[w][4u * lane + 3u] = acc.w;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  out[(ulong)row * H + blk * 128u + i] = (red[0][i] + red[1][i]) + (red[2][i] + red[3][i]);
}
kernel void comb_nv4(device const UnitD* units [[buffer(0)]], device const float* yd [[buffer(1)]], device const PairD* pairs [[buffer(2)]],
                     device const uint* rowptr [[buffer(3)]], device float* out [[buffer(4)]], constant DP& p [[buffer(5)]],
                     uint2 tg [[threadgroup_position_in_grid]], uint i [[thread_index_in_threadgroup]]) {
  threadgroup float red[4][128];
  comb_nv4_body(units, yd, pairs, rowptr, red, out, p, tg, i);
}

// ---------------------------------------------------------------------------------------------------------------------
// PREFILL shape: experts with many rows. Rows of the chunk are laid out expert after expert (row0 per unit).
struct UnitP { device const uchar* blob; uint row0; uint nrows; };
struct PP { uint nrows; uint cap; uint bits; uint pad; };      // rows in chunk; row capacity (plane stride of the rotated inputs)

// rotate inputs per chunk row. grid (20, nrows, 2), TG 128. A [proj][cap][H] plain fp16.
kernel void rot_in_pf(device const float* x [[buffer(0)]], device const UnitP* units [[buffer(1)]], device const uint* unit_of_row [[buffer(2)]],
                      device const int* src_row [[buffer(3)]], device half* A [[buffer(4)]], constant PP& p [[buffer(5)]],
                      uint3 tg [[threadgroup_position_in_grid]], uint i [[thread_index_in_threadgroup]]) {
  threadgroup float buf[128];
  const uint blk = tg.x, pr = tg.y, proj = tg.z, col = blk * 128u + i;
  half a = (half)x[(ulong)src_row[pr] * H + col];
  half s = ((device const half*)(units[unit_of_row[pr]].blob + (proj ? off_usuh(p.bits) : off_gsuh(p.bits))))[col];
  float v = fwht128((float)(half)((float)a * (float)s), i, buf);
  A[((ulong)proj * p.cap + pr) * H + col] = (half)(v * R_SCALE);
}

// decode a projection of every chunk unit into the tiled fp16 layout W[u][tile(i*nt+j)][row*16+col] (512 B per tile).
// grid x = ntiles*32 (tile = x>>5, lane = x&31), y = unit; q.which: 0 gate, 1 up, 2 down; q.bits = the trellis width.
struct DQ { uint which; uint bits; uint pad0; uint pad1; };
kernel void decode_w(device const UnitP* units [[buffer(0)]], device half* W [[buffer(1)]], constant DQ& q [[buffer(2)]],
                     uint2 gid [[thread_position_in_grid]]) {
  const uint ntiles = (H / 16u) * (FF / 16u), tile = gid.x >> 5, t = gid.x & 31u;
  if (tile >= ntiles) return;
  const uint u = gid.y;
  const device uint* w = (const device uint*)(units[u].blob + (q.which == 2 ? off_dt(q.bits) : (q.which ? off_ut(q.bits) : off_gt(q.bits))))
                       + (ulong)tile * (8u * q.bits);
  half v[8];
  decode_bits(q.bits, w, t, v);
  const uint c = t >> 2, qq = t & 3u;
  device half* o = W + ((ulong)u * ntiles + tile) * 256u;
  for (uint jj = 0; jj < 8u; jj++) {
    uint row = 2u * qq + (jj & 1u) + ((jj & 2u) ? 8u : 0u), col = c + ((jj & 4u) ? 8u : 0u);
    o[row * 16u + col] = v[jj];
  }
}

// pf_tensor_ops = 1: the same projection as a plain row-major fp16 matrix W[u][k*N + n] (K x N), the layout matmul2d reads. Same
// grid, same tile decode as decode_w; q.which: 0 gate [H, FF], 1 up [H, FF], 2 down [FF, H].
kernel void decode_w_plain(device const UnitP* units [[buffer(0)]], device half* W [[buffer(1)]], constant DQ& q [[buffer(2)]],
                           uint2 gid [[thread_position_in_grid]]) {
  const uint K = q.which == 2 ? FF : H, N = q.which == 2 ? H : FF;
  const uint ntiles = (K / 16u) * (N / 16u), tile = gid.x >> 5, t = gid.x & 31u;
  if (tile >= ntiles) return;
  const uint u = gid.y;
  const device uint* w = (const device uint*)(units[u].blob + (q.which == 2 ? off_dt(q.bits) : (q.which ? off_ut(q.bits) : off_gt(q.bits))))
                       + (ulong)tile * (8u * q.bits);
  half v[8];
  decode_bits(q.bits, w, t, v);
  const uint c = t >> 2, qq = t & 3u, nn = N / 16u;   // tile = (k >> 4) * nn + (n >> 4), as the tiled layout above
  device half* o = W + (ulong)u * K * N + ((ulong)(tile / nn) * 16u) * N + (tile % nn) * 16u;
  for (uint jj = 0; jj < 8u; jj++) {
    uint row = 2u * qq + (jj & 1u) + ((jj & 2u) ? 8u : 0u), col = c + ((jj & 4u) ? 8u : 0u);
    o[(ulong)row * N + col] = v[jj];
  }
}

// C[nv rows, N] fp32 = A[rows, K] fp16 x W[K, N] (tiled fp16, per unit) ; tile 32 x 64, K step 32, 4 simdgroups.
struct GT { uint u; uint r0; uint nv; uint pad; };
template <uint K, uint N, bool FROMB>
inline void gemm_body(device const half* A, device const half* Wu, device float* C, uint r0, uint nv, uint n0, uint t, uint sg,
                      threadgroup half* As, threadgroup half* Bs, threadgroup float* Cs) {
  constexpr uint NT = N / 16u;
  const uint sgm = sg >> 1, sgn = sg & 1u;
  simdgroup_float8x8 acc[2][4];
  for (uint a = 0; a < 2; a++) for (uint b = 0; b < 4; b++) acc[a][b] = simdgroup_float8x8(0.f);
  const uint arow = t >> 2, apart = t & 3u;                // A tile: 32 rows x 32 k: 4 x half8 per row
  const uint bt = t >> 4, br = t & 15u;                    // B tile: 8 16x16 tiles, one 16-col row (32 B) per thread
  const uint bki = bt >> 2, bnj = bt & 3u;
  for (uint k0 = 0; k0 < K; k0 += 32u) {
    *(threadgroup half4*)(As + arow * 32u + apart * 8u) = *(device const half4*)(A + (ulong)(r0 + arow) * K + k0 + apart * 8u);
    *(threadgroup half4*)(As + arow * 32u + apart * 8u + 4u) = *(device const half4*)(A + (ulong)(r0 + arow) * K + k0 + apart * 8u + 4u);
    device const half* bsrc = Wu + ((ulong)((k0 >> 4) + bki) * NT + ((n0 >> 4) + bnj)) * 256u + br * 16u;
    *(threadgroup half4*)(Bs + (bki * 16u + br) * 64u + bnj * 16u) = *(device const half4*)bsrc;
    *(threadgroup half4*)(Bs + (bki * 16u + br) * 64u + bnj * 16u + 4u) = *(device const half4*)(bsrc + 4);
    *(threadgroup half4*)(Bs + (bki * 16u + br) * 64u + bnj * 16u + 8u) = *(device const half4*)(bsrc + 8);
    *(threadgroup half4*)(Bs + (bki * 16u + br) * 64u + bnj * 16u + 12u) = *(device const half4*)(bsrc + 12);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint kk = 0; kk < 32u; kk += 8u) {
      simdgroup_half8x8 a0, a1, b0, b1, b2, b3;
      simdgroup_load(a0, As + (sgm * 16u) * 32u + kk, 32);
      simdgroup_load(a1, As + (sgm * 16u + 8u) * 32u + kk, 32);
      simdgroup_load(b0, Bs + kk * 64u + sgn * 32u, 64);
      simdgroup_load(b1, Bs + kk * 64u + sgn * 32u + 8u, 64);
      simdgroup_load(b2, Bs + kk * 64u + sgn * 32u + 16u, 64);
      simdgroup_load(b3, Bs + kk * 64u + sgn * 32u + 24u, 64);
      simdgroup_multiply_accumulate(acc[0][0], a0, b0, acc[0][0]); simdgroup_multiply_accumulate(acc[0][1], a0, b1, acc[0][1]);
      simdgroup_multiply_accumulate(acc[0][2], a0, b2, acc[0][2]); simdgroup_multiply_accumulate(acc[0][3], a0, b3, acc[0][3]);
      simdgroup_multiply_accumulate(acc[1][0], a1, b0, acc[1][0]); simdgroup_multiply_accumulate(acc[1][1], a1, b1, acc[1][1]);
      simdgroup_multiply_accumulate(acc[1][2], a1, b2, acc[1][2]); simdgroup_multiply_accumulate(acc[1][3], a1, b3, acc[1][3]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  for (uint a = 0; a < 2; a++) for (uint b = 0; b < 4; b++)
    simdgroup_store(acc[a][b], Cs + (sgm * 16u + a * 8u) * 64u + sgn * 32u + b * 8u, 64);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint e = t; e < 32u * 16u; e += 128u) {              // 32 rows x 16 float4
    uint r = e >> 4, c4 = e & 15u;
    if (r < nv) *(device float4*)(C + (ulong)(r0 + r) * N + n0 + c4 * 4u) = *(threadgroup float4*)(Cs + r * 64u + c4 * 4u);
  }
}

// gate+up: grid (10 n tiles, ntiles, 2 proj); A [proj][cap][H], W [proj? Wg : Wu][u], C [proj][cap][FF]
kernel void gemm_gu(device const half* A [[buffer(0)]], device const half* Wg [[buffer(1)]], device const half* Wu [[buffer(2)]],
                    device float* C [[buffer(3)]], device const GT* tiles [[buffer(4)]], constant PP& p [[buffer(5)]],
                    uint3 tg [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup half As[32 * 32]; threadgroup half Bs[32 * 64]; threadgroup float Cs[32 * 64];
  const GT T = tiles[tg.y];
  const uint proj = tg.z;
  device const half* W = (proj ? Wu : Wg) + (ulong)T.u * (H * FF);
  gemm_body<H, FF, true>(A + (ulong)proj * p.cap * H, W, C + (ulong)proj * p.cap * FF, T.r0, T.nv, tg.x * 64u, t, sg, As, Bs, Cs);
}
// down: grid (40 n tiles, ntiles, 1); A = Z [cap][FF], W = Wd[u], C = Yd [cap][H]
kernel void gemm_d(device const half* A [[buffer(0)]], device const half* Wd [[buffer(1)]], device float* C [[buffer(2)]],
                   device const GT* tiles [[buffer(3)]],
                   uint3 tg [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup half As[32 * 32]; threadgroup half Bs[32 * 64]; threadgroup float Cs[32 * 64];
  const GT T = tiles[tg.y];
  gemm_body<FF, H, true>(A, Wd + (ulong)T.u * (H * FF), C, T.r0, T.nv, tg.x * 64u, t, sg, As, Bs, Cs);
}

// pf_tensor_ops = 1: gemm_gu / gemm_d through Metal 4 tensor ops (mpp::tensor_ops::matmul2d), which reach the matrix units of M5
// and later. Same buffers; W is the plain row-major copy decode_w_plain writes. One 64 x 64 tile of C per threadgroup (the
// 64-row tile reaches ~63 TFLOPS on this M5 Max against ~31 for 32 x 64, docs/m5-prefill-gemm.md), K from the operand extents,
// fp16 x fp16 -> fp32 accumulator, and the host's tile table steps by 64 rows for these kernels (the simdgroup GEMM shares it and
// keeps 32). The operands are described with their storage's fastest dimension first (matmul2d's convention), so A is (K, rows)
// with strides {1, K}, W is (N, K) with {1, N} and C is (N, rows) with {1, N} (a tensor<device half, ...> wants a plain, not a
// const, device pointer, hence the parameters above). A tile is nv rows tall (nv < 64 for a unit's last tile): the A and C extents
// say so and the op masks the rows past them, so no tile can write another
// unit's rows (threadgroups of one dispatch run in no particular order).
using TH2 = tensor<device half, dextents<int32_t, 2>, tensor_inline>;
using TF2 = tensor<device float, dextents<int32_t, 2>, tensor_inline>;
inline void gemm_tensor_body(device half* A, device half* W, device float* C, int K, int N, uint r0, uint n0, uint nv) {
  constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(64, 64, static_cast<int>(dynamic_extent));
  mpp::tensor_ops::matmul2d<desc, execution_simdgroups<4>> op;
  TH2 At(A + (ulong)r0 * (uint)K, dextents<int32_t, 2>(K, (int)nv), array<int32_t, 2>{1, K});
  TH2 Bt(W + (ulong)n0, dextents<int32_t, 2>(64, K), array<int32_t, 2>{1, N});
  TF2 Ct(C + (ulong)r0 * (uint)N + n0, dextents<int32_t, 2>(64, (int)nv), array<int32_t, 2>{1, N});
  op.run(At, Bt, Ct);
}
// gate+up: grid (10 n tiles, ntiles, 2 proj), same buffers as gemm_gu
kernel void gemm_gu_t(device half* A [[buffer(0)]], device half* Wg [[buffer(1)]], device half* Wu [[buffer(2)]],
                      device float* C [[buffer(3)]], device const GT* tiles [[buffer(4)]], constant PP& p [[buffer(5)]],
                      uint3 tg [[threadgroup_position_in_grid]]) {
  const GT T = tiles[tg.y];
  const uint proj = tg.z;
  gemm_tensor_body(A + (ulong)proj * p.cap * H, (proj ? Wu : Wg) + (ulong)T.u * (H * FF), C + (ulong)proj * p.cap * FF,
                   (int)H, (int)FF, T.r0, tg.x * 64u, T.nv);
}
// down: grid (40 n tiles, ntiles, 1), same buffers as gemm_d
kernel void gemm_d_t(device half* A [[buffer(0)]], device half* Wd [[buffer(1)]], device float* C [[buffer(2)]],
                     device const GT* tiles [[buffer(3)]], uint3 tg [[threadgroup_position_in_grid]]) {
  const GT T = tiles[tg.y];
  gemm_tensor_body(A, Wd + (ulong)T.u * (H * FF), C, (int)FF, (int)H, T.r0, tg.x * 64u, T.nv);
}

// had_out(gate/up) + SwiGLU + rotate for down. grid (5, nrows). Cg [proj][cap][FF] fp32 -> Z [cap][FF] fp16 (plain).
kernel void mid_pf(device const UnitP* units [[buffer(0)]], device const uint* unit_of_row [[buffer(1)]], device const float* Cgu [[buffer(2)]],
                   device half* Z [[buffer(3)]], constant PP& p [[buffer(4)]],
                   uint2 tg [[threadgroup_position_in_grid]], uint i [[thread_index_in_threadgroup]]) {
  threadgroup float buf[128];
  const uint blk = tg.x, pr = tg.y, col = blk * 128u + i;
  device const uchar* blob = units[unit_of_row[pr]].blob;
  float vg = fwht128(Cgu[(ulong)pr * FF + col], i, buf);
  float vu = fwht128(Cgu[((ulong)p.cap + pr) * FF + col], i, buf);
  float gg = vg * INV * (float)((device const half*)(blob + off_gsvh(p.bits)))[col];
  float uu = vu * INV * (float)((device const half*)(blob + off_usvh(p.bits)))[col];
  half z = (half)(gg / (1.f + precise::exp(-gg)) * uu);
  float v = fwht128((float)(half)((float)z * (float)((device const half*)(blob + off_dsuh(p.bits)))[col]), i, buf);
  Z[(ulong)pr * FF + col] = (half)(v * R_SCALE);
}

// had_out(down) + weighted accumulation into y (+=) per touched output row, pairs of the chunk in order. grid (20, ntouched).
struct CP { uint pr; float w; };
kernel void comb_pf(device const UnitP* units [[buffer(0)]], device const uint* unit_of_row [[buffer(1)]], device const float* Yd [[buffer(2)]],
                    device const uint* touched [[buffer(3)]], device const uint* tptr [[buffer(4)]], device const CP* cps [[buffer(5)]],
                    device float* y [[buffer(6)]], constant uint& bits [[buffer(7)]], uint2 tg [[threadgroup_position_in_grid]],
                    uint i [[thread_index_in_threadgroup]]) {
  threadgroup float buf[128];
  const uint blk = tg.x, ti = tg.y, col = blk * 128u + i;
  float acc = 0.f;
  for (uint c = tptr[ti]; c < tptr[ti + 1]; c++) {
    CP cp = cps[c];
    float v = fwht128(Yd[(ulong)cp.pr * H + col], i, buf);
    float o = v * INV * (float)((device const half*)(units[unit_of_row[cp.pr]].blob + off_dsvh(bits)))[col];
    acc += cp.w * o;
  }
  const ulong yi = (ulong)touched[ti] * H + col;
  y[yi] = y[yi] + acc;
}

// ---------------------------------------------------------------------------------------------------------------------
// NVFP4 prompt stages: the trellis prompt pipeline over the block_nvfp4 blobs gemv_nv4 reads. No trellis and no
// input rotation: the weight decode writes decode_w's tiled fp16 layout straight from the 36-byte blocks, the input stage is
// a plain fp16 copy of the chunk's rows, and the tail scales land in fp32 where mid_nv4 and comb_nv4 apply them (folding
// s_gate / s_up / s_down into the fp16 weights loses 2-12% to subnormals).

// a projection of every chunk unit into decode_w's tiled fp16 layout W[u][tile(i*nt+j)][row*16+col] (row = k, col = n).
// grid x = ntiles*32 (tile = x>>5, lane = x&31), y = unit; q.which: 0 gate [H, FF], 1 up [H, FF], 2 down [FF, H]. A blob's rows
// are its n side (FF for gate/up, H for down), each k/64 36-byte blocks; a k tile of 16 values lies inside one block and one of
// its sub-blocks of 16, so lane t = 2 * nl + hb (nl = the tile's n row, hb its 4 code bytes: values 4hb..4hb+3 and 4hb+8..4hb+11).
kernel void decode_w_nv4(device const UnitP* units [[buffer(0)]], device half* W [[buffer(1)]], constant DQ& q [[buffer(2)]],
                         uint2 gid [[thread_position_in_grid]]) {
  const uint ntiles = (H / 16u) * (FF / 16u), tile = gid.x >> 5, t = gid.x & 31u;
  if (tile >= ntiles) return;
  const uint u = gid.y;
  const bool dn = q.which == 2;
  const uint nn = dn ? H / 16u : FF / 16u;                     // n side: H for down, FF for gate/up
  const uint ki = tile / nn, nj = tile % nn, nb = dn ? FF / 64u : H / 64u;   // k tiles, n tiles, blocks in a blob row
  const uint bi = ki >> 2, s = ki & 3u;                        // the k tile's block of 64 values and its sub-block of 16
  const uint nl = t >> 1, hb = t & 1u;                         // this thread's n row of the tile and its 4 code bytes
  const device uchar* blk = units[u].blob + (dn ? NV4_DN : (q.which ? NV4_UP : NV4_GT)) + ((ulong)(nj * 16u + nl) * nb + bi) * 36u;
  const uint code = *(const device uint*)(blk + 4u + s * 8u + hb * 4u);
  const float sc = nv4_scale(blk[s]);
  device half* o = W + ((ulong)u * ntiles + tile) * 256u + nl;
  for (uint i = 0; i < 4u; i++) {                              // byte j = hb * 4 + i: value j in the low nibble, j + 8 in the high
    o[(hb * 4u + i) * 16u] = (half)(nv4_val((code >> (8u * i)) & 0xFu) * sc);
    o[(hb * 4u + i + 8u) * 16u] = (half)(nv4_val((code >> (8u * i + 4u)) & 0xFu) * sc);
  }
}

// pf_tensor_ops = 1: the same projection as a plain row-major fp16 W[u][k*N + n] (decode_w_plain's layout, which matmul2d
// reads), decoded straight from the 36-byte blocks. One thread a block: x = n + N * kb (N is a multiple of 32, so a warp's 32
// threads write 32 consecutive n of one k row). grid x = N * K / 64, y = unit; q.which: 0 gate [H, FF], 1 up [H, FF], 2 down [FF, H].
kernel void decode_w_nv4_plain(device const UnitP* units [[buffer(0)]], device half* W [[buffer(1)]], constant DQ& q [[buffer(2)]],
                               uint2 gid [[thread_position_in_grid]]) {
  const uint K = q.which == 2 ? FF : H, N = q.which == 2 ? H : FF, nb = K / 64u;
  const uint n = gid.x % N, bi = gid.x / N;
  if (bi >= nb) return;
  const uint u = gid.y;
  const device uchar* blk = units[u].blob + (q.which == 2 ? NV4_DN : (q.which ? NV4_UP : NV4_GT)) + ((ulong)n * nb + bi) * 36u;
  device half* o = W + (ulong)u * K * N + (ulong)bi * 64u * N + n;   // the block's 64 k values of this n row
  for (uint s = 0; s < 4u; s++) {                                    // four sub-blocks of 16 values, one scale byte each
    const float sc = nv4_scale(blk[s]);
    const uint c0 = *(const device uint*)(blk + 4u + s * 8u), c1 = *(const device uint*)(blk + 8u + s * 8u);
    #pragma clang loop unroll(full)
    for (uint j = 0; j < 4u; j++) {                                  // byte j: value j in the low nibble, value j + 8 in the high
      const uint b0 = (c0 >> (8u * j)) & 0xFFu, b1 = (c1 >> (8u * j)) & 0xFFu;
      device half* ok = o + (s * 16u + j) * N;
      ok[0] = (half)(nv4_val(b0 & 0xFu) * sc);
      ok[8u * N] = (half)(nv4_val(b0 >> 4) * sc);
      ok[4u * N] = (half)(nv4_val(b1 & 0xFu) * sc);
      ok[12u * N] = (half)(nv4_val(b1 >> 4) * sc);
    }
  }
}

// the chunk rows as fp16 (an NVFP4 row multiplies x directly: no suh, no rotation), both gate/up planes. grid (20, nrows, 2), TG 128.
// Same argument list as rot_in_pf so the host encodes the stage the same way; units and unit_of_row go unread.
kernel void x_pf_nv4(device const float* x [[buffer(0)]], device const UnitP* units [[buffer(1)]], device const uint* unit_of_row [[buffer(2)]],
                     device const int* src_row [[buffer(3)]], device half* A [[buffer(4)]], constant PP& p [[buffer(5)]],
                     uint3 tg [[threadgroup_position_in_grid]], uint i [[thread_index_in_threadgroup]]) {
  const uint pr = tg.y, proj = tg.z, col = tg.x * 128u + i;
  A[((ulong)proj * p.cap + pr) * H + col] = (half)x[(ulong)src_row[pr] * H + col];
}
// the same with the caller's rows already fp16 (the fp16-I/O job; x_pf_nv4 rounds every x to fp16 too)
kernel void x_pf_nv4_h(device const half* x [[buffer(0)]], device const UnitP* units [[buffer(1)]], device const uint* unit_of_row [[buffer(2)]],
                       device const int* src_row [[buffer(3)]], device half* A [[buffer(4)]], constant PP& p [[buffer(5)]],
                       uint3 tg [[threadgroup_position_in_grid]], uint i [[thread_index_in_threadgroup]]) {
  const uint pr = tg.y, proj = tg.z, col = tg.x * 128u + i;
  A[((ulong)proj * p.cap + pr) * H + col] = x[(ulong)src_row[pr] * H + col];
}

// mid_nv4's tail scales and SiLU product on the chunk's rows: Z [nrows][FF] fp16, the plain layout gemm_d reads (an NVFP4 row
// has no input rotation). grid (5, nrows), TG 128. Cgu [proj][cap][FF] fp32.
kernel void mid_pf_nv4(device const UnitP* units [[buffer(0)]], device const uint* unit_of_row [[buffer(1)]], device const float* Cgu [[buffer(2)]],
                       device half* Z [[buffer(3)]], constant PP& p [[buffer(4)]],
                       uint2 tg [[threadgroup_position_in_grid]], uint i [[thread_index_in_threadgroup]]) {
  const uint blk = tg.x, pr = tg.y, col = blk * 128u + i;
  const device float* tail = (device const float*)(units[unit_of_row[pr]].blob + NV4_TAIL);
  const float g = Cgu[(ulong)pr * FF + col] * tail[0];
  const float uu = Cgu[((ulong)p.cap + pr) * FF + col] * tail[1];
  Z[(ulong)pr * FF + col] = (half)(g / (1.f + precise::exp(-g)) * uu);
}

// comb_nv4's s_down and the weighted accumulation into y (+=) per touched output row, pairs of the chunk in order. grid (20, ntouched).
kernel void comb_pf_nv4(device const UnitP* units [[buffer(0)]], device const uint* unit_of_row [[buffer(1)]], device const float* Yd [[buffer(2)]],
                        device const uint* touched [[buffer(3)]], device const uint* tptr [[buffer(4)]], device const CP* cps [[buffer(5)]],
                        device float* y [[buffer(6)]], uint2 tg [[threadgroup_position_in_grid]], uint i [[thread_index_in_threadgroup]]) {
  const uint blk = tg.x, ti = tg.y, col = blk * 128u + i;
  float acc = 0.f;
  for (uint c = tptr[ti]; c < tptr[ti + 1]; c++) {
    CP cp = cps[c];
    const float w = cp.w * ((device const float*)(units[unit_of_row[cp.pr]].blob + NV4_TAIL))[2];
    acc += w * Yd[(ulong)cp.pr * H + col];
  }
  const ulong yi = (ulong)touched[ti] * H + col;
  y[yi] = y[yi] + acc;
}

// fp16-I/O prefill (pf500): the same rotation with the input rows already fp16 (the kernel rounds x to fp16 first anyway: bit-identical to rot_in_pf on fp16-representable x)
kernel void rot_in_pf_h(device const half* x [[buffer(0)]], device const UnitP* units [[buffer(1)]], device const uint* unit_of_row [[buffer(2)]],
                        device const int* src_row [[buffer(3)]], device half* A [[buffer(4)]], constant PP& p [[buffer(5)]],
                        uint3 tg [[threadgroup_position_in_grid]], uint i [[thread_index_in_threadgroup]]) {
  threadgroup float buf[128];
  const uint blk = tg.x, pr = tg.y, proj = tg.z, col = blk * 128u + i;
  half a = x[(ulong)src_row[pr] * H + col];
  half s = ((device const half*)(units[unit_of_row[pr]].blob + (proj ? off_usuh(p.bits) : off_gsuh(p.bits))))[col];
  float v = fwht128((float)(half)((float)a * (float)s), i, buf);
  A[((ulong)proj * p.cap + pr) * H + col] = (half)(v * R_SCALE);
}

// the fp32 accumulated rows -> fp16 (what the host's clamp + _Float16 cast did): clamp to +-65504 (NaN passes), round to nearest even
kernel void y_f2h(device const float* y [[buffer(0)]], device half* yh [[buffer(1)]], constant uint& n [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  if (i < n) {
    float v = y[i];
    v = v > 65504.0f ? 65504.0f : (v < -65504.0f ? -65504.0f : v);
    yh[i] = (half)v;
  }
}

// ---------------------------------------------------------------------------------------------------------------------
// EXCHANGE window (docs/strata-exchange-map.md 3.1-3.5, option A): one command buffer per verify window, one serial encoder,
// 7 dispatches per layer. D0 xw_wait_build spins on the layer's slot in Mac memory at system scope, builds units/pairs/CSR
// exactly as exl3m_moe_rows_submit does on the CPU and writes the indirect arguments; D1-D6 are today's decode kernels as
// indirect dispatches, with x read from and y written to the slot. The arithmetic of D1/D5 is the arithmetic of rot_in_dec /
// comb_dec, so the results are bit-identical.
#ifndef __METAL_MEMORY_SCOPE_SYSTEM__
#define __METAL_MEMORY_SCOPE_SYSTEM__ 3
#endif
namespace metal { constexpr constant metal::thread_scope thread_scope_system = static_cast<thread_scope>(__METAL_MEMORY_SCOPE_SYSTEM__); }

struct XW { uint layer, epoch, budget, n_experts, n_rows, mrx, bits, pad; };

// status = {code, layer, spins, aux}, then the window is dead: every later layer zeroes its arguments and drains.
inline void xw_fail(volatile coherent(system) device uint* hdr, device uint* args, device uint* dead,
                    uint code, uint layer, uint spins, uint aux) {
  volatile coherent(system) device uint* st = hdr + XCH_H_MTL_STATUS / 4u;
  st[0] = code; st[1] = layer; st[2] = spins; st[3] = aux;
  atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_system);
  dead[0] = 1u;
  for (uint i = 0; i < 24u; i++) args[i] = 0u;
}

// D0: bounded system-scope spin on slot.seq (the lane_resident_sys pattern, exchange/xch_sys.metal), then the table build.
// grid (1, 1, 1), TG 1: thread 0 does the whole build (<= 80 pairs).
kernel void xw_wait_build(volatile coherent(system) device uchar* region [[buffer(0)]], device const ulong* table [[buffer(1)]],
                          device DP* dp [[buffer(2)]], device uint* args [[buffer(3)]], device UnitD* units [[buffer(4)]],
                          device PairD* pairs [[buffer(5)]], device uint* rowptr [[buffer(6)]], constant XW& w [[buffer(7)]],
                          device uint* dead [[buffer(8)]], uint tid [[thread_index_in_threadgroup]]) {
  if (tid != 0) return;
  if (dead[0]) { for (uint i = 0; i < 24u; i++) args[i] = 0u; return; }
  volatile coherent(system) device uint* hdr = (volatile coherent(system) device uint*)region;
  volatile coherent(system) device uint* slot = (volatile coherent(system) device uint*)(region + XCH_SLOT(w.layer));
  const uint want = w.epoch;
  uint spins = 0;
  for (;;) {
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_system);
    if (int(slot[XCH_S_SEQ / 4u] - want) >= 0) break;
    if (hdr[XCH_H_ABORT / 4u] != 0u) { xw_fail(hdr, args, dead, XCH_ST_ABORT, w.layer, spins, 0u); return; }
    if (++spins >= w.budget) { xw_fail(hdr, args, dead, XCH_ST_TIMEOUT, w.layer, spins, 0u); return; }
  }
  slot[XCH_S_MTL_SPINS / 4u] = spins;   // the card reads it from the dump with the publish/wait timestamps
  const uint count = slot[XCH_S_COUNT / 4u], nrows = slot[XCH_S_NROWS / 4u];
  if (count > XCH_MAX_PAIRS || nrows > XCH_MAX_ROWS) { xw_fail(hdr, args, dead, XCH_ST_SHAPE, w.layer, spins, 0u); return; }
  volatile coherent(system) device int* pi = (volatile coherent(system) device int*)(region + XCH_SLOT(w.layer) + XCH_S_PAIRS);
  int uexp[96];
  uint cnt[96];
  PairD tmp[XCH_MAX_PAIRS];
  uint nu = 0;
  for (uint p = 0; p < count; p++) {
    const int e = pi[3u * p], r = pi[3u * p + 1u];
    const float wt = as_type<float>(pi[3u * p + 2u]);
    if (r < 0 || (uint)r >= nrows) { xw_fail(hdr, args, dead, XCH_ST_SHAPE, w.layer, spins, (uint)e); return; }
    if (e < 0 || (uint)e >= w.n_experts) { xw_fail(hdr, args, dead, XCH_ST_SHAPE, w.layer, spins, (uint)e); return; }
    uint u = 0u;
    while (u < nu && uexp[u] != e) u++;
    if (u == nu) {
      if (nu >= 96u) { xw_fail(hdr, args, dead, XCH_ST_SHAPE, w.layer, spins, (uint)e); return; }
      uexp[nu] = e;
      for (uint s = 0; s < 8u; s++) units[nu].rows[s] = -1;
      cnt[nu] = 0u; nu++;
    }
    const uint s = cnt[u]++;
    if (s >= 8u) { xw_fail(hdr, args, dead, XCH_ST_SHAPE, w.layer, spins, (uint)e); return; }
    units[u].rows[s] = r;
    tmp[p] = { u, s, wt, 0u };
  }
  uint maxslot = 0u;
  for (uint u = 0; u < nu; u++) maxslot = max(maxslot, cnt[u]);
  if (maxslot > w.mrx) { xw_fail(hdr, args, dead, XCH_ST_SHAPE, w.layer, spins, 0u); return; }   // the window's fixed gemv_dec<MR, KB> must cover every slot
  for (uint u = 0; u < nu; u++) {
    const ulong a = table[(ulong)w.layer * w.n_experts + (ulong)uexp[u]];
    if (!a) { xw_fail(hdr, args, dead, XCH_ST_SHAPE, w.layer, spins, (uint)uexp[u]); return; }
    units[u].blob = (device const uchar*)a;
  }
  // CSR per output row, pairs in pair order inside a row (stable counting sort, as exl3m_moe_rows_submit)
  uint rp[XCH_MAX_ROWS + 1u], fill[XCH_MAX_ROWS];
  for (uint r = 0; r <= nrows; r++) rp[r] = 0u;
  for (uint p = 0; p < count; p++) rp[(uint)units[tmp[p].u].rows[tmp[p].slot] + 1u]++;
  for (uint r = 0; r < nrows; r++) rp[r + 1u] += rp[r];
  for (uint r = 0; r <= nrows; r++) { rowptr[r] = rp[r]; if (r < nrows) fill[r] = rp[r]; }
  for (uint p = 0; p < count; p++) pairs[fill[(uint)units[tmp[p].u].rows[tmp[p].slot]]++] = tmp[p];
  dp->mrx = w.mrx; dp->nu = nu; dp->bits = w.bits; dp->pad = 0u;
  // one 16 B slot per dispatch (the host's indirectBufferOffset is layer * 96 + 16 * dispatch)
  args[0] = 20u; args[1] = nu * w.mrx; args[2] = 2u;      // D1 rot_in_dec_sys
  args[4] = 40u; args[5] = nu; args[6] = 2u;              // D2 gemv_dec gate/up
  args[8] = 5u; args[9] = nu * w.mrx; args[10] = 1u;      // D3 mid_dec
  args[12] = 160u; args[13] = nu; args[14] = 1u;          // D4 gemv_dec down
  args[16] = 20u; args[17] = nrows; args[18] = 1u;        // D5 comb_dec_sys (empty CSR rows land y = 0)
  args[20] = 1u; args[21] = 1u; args[22] = 1u;            // D6 xw_land
  if (count == 0u) for (uint i = 0; i < 16u; i++) args[i] = 0u;
}

// D1: rot_in_dec with x read from the slot (fp16, the 4080 already rounded it) at system scope. grid (20, nu*mrx, 2), TG 128.
kernel void rot_in_dec_sys(volatile coherent(system) device const half* x [[buffer(0)]], device const UnitD* units [[buffer(1)]],
                           device half* xq [[buffer(2)]], constant DP& p [[buffer(3)]],
                           uint3 tg [[threadgroup_position_in_grid]], uint i [[thread_index_in_threadgroup]]) {
  threadgroup float buf[128];
  const uint blk = tg.x, u = tg.y / p.mrx, slot = tg.y % p.mrx, proj = tg.z;
  const int row = units[u].rows[slot];
  float v = 0.f;
  if (row >= 0) {
    half a = x[(ulong)row * H + blk * 128u + i];
    half s = ((device const half*)(units[u].blob + (proj ? off_usuh(p.bits) : off_gsuh(p.bits))))[blk * 128u + i];
    v = (float)(half)((float)a * (float)s);
  }
  v = fwht128(v, i, buf);
  xq[(((ulong)proj * p.nu + u) * p.mrx + slot) * H + xq_index(blk * 128u + i)] = (half)(v * R_SCALE);
}

// D1 (NVFP4): the slot's fp16 x -> the fp32 rows gemv_nv4 multiplies, one row per (unit, slot). There is no input rotation
// (an NVFP4 row multiplies x directly), so a unit's slots only need their own rows; the grid's third dimension is the trellis path's
// gate/up planes and this layer's second plane returns at once. grid (20, nu*mrx, 2), TG 128.
kernel void x_f32_sys(volatile coherent(system) device const half* x [[buffer(0)]], device const UnitD* units [[buffer(1)]],
                      device float* x32 [[buffer(2)]], constant DP& p [[buffer(3)]],
                      uint3 tg [[threadgroup_position_in_grid]], uint i [[thread_index_in_threadgroup]]) {
  if (tg.z) return;
  const uint blk = tg.x, u = tg.y / p.mrx, slot = tg.y % p.mrx;
  const int row = units[u].rows[slot];
  if (row < 0) return;
  const ulong e = (ulong)row * H + blk * 128u + i;
  x32[e] = (float)x[e];
}

// D5: comb_dec with y written into the slot at system scope. grid (20, nrows), TG 128.
kernel void comb_dec_sys(device const UnitD* units [[buffer(0)]], device const float* yd [[buffer(1)]], device const PairD* pairs [[buffer(2)]],
                         device const uint* rowptr [[buffer(3)]], volatile coherent(system) device float* out [[buffer(4)]], constant DP& p [[buffer(5)]],
                         uint2 tg [[threadgroup_position_in_grid]], uint i [[thread_index_in_threadgroup]]) {
  threadgroup float buf[128];
  const uint blk = tg.x, row = tg.y, col = blk * 128u + i;
  float acc = 0.f;
  for (uint pi = rowptr[row]; pi < rowptr[row + 1]; pi++) {
    PairD pr = pairs[pi];
    float v = fwht128(yd[((ulong)pr.u * p.mrx + pr.slot) * H + col], i, buf);
    float o = v * INV * (float)((device const half*)(units[pr.u].blob + off_dsvh(p.bits)))[col];
    acc += pr.w * o;
  }
  out[(ulong)row * H + col] = acc;
}

// D5 (NVFP4): comb_nv4 with y written into the slot at system scope, the arithmetic and the pair order of kernel.
// grid (20, nrows), TG 128.
kernel void comb_nv4_sys(device const UnitD* units [[buffer(0)]], device const float* yd [[buffer(1)]], device const PairD* pairs [[buffer(2)]],
                         device const uint* rowptr [[buffer(3)]], volatile coherent(system) device float* out [[buffer(4)]], constant DP& p [[buffer(5)]],
                         uint2 tg [[threadgroup_position_in_grid]], uint i [[thread_index_in_threadgroup]]) {
  threadgroup float red[4][128];
  comb_nv4_body(units, yd, pairs, rowptr, red, out, p, tg, i);
}

// D6: publish the layer (one system fence, then the landed word: a trailing fence only delayed the kernel's end; 10-19 M5 Max:
// 0.163 -> 0.077 ms per hand-off). grid (1, 1, 1), TG 1. A dead window does not land.
kernel void xw_land(volatile coherent(system) device uint* landed [[buffer(0)]], device const uint* dead [[buffer(1)]],
                    constant uint& epoch [[buffer(2)]], uint tid [[thread_index_in_threadgroup]]) {
  if (tid != 0 || dead[0]) return;
  atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_system);
  landed[0] = epoch;
}
