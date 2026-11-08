// include/strata/platform/tgcu.hpp - explicit mapped-memory hand-offs for the tgcu backend (an RTX 3090 behind a USB3
// bridge on macOS): the GPU cannot read host RAM there, so a mapped buffer is a VRAM mirror and the host pushes what it
// wrote and pulls what the GPU wrote at the hand-off points.  No-ops in every other build.
#pragma once
#include <cstddef>
#if defined(STRATA_TGCU)
extern "C" int tgcuMappedPush(const void* p, size_t n);
extern "C" int tgcuMappedPull(void* p, size_t n);
extern "C" int tgcuMappedManage(const void* p);
extern "C" int tgcuHsArm(void* const* ps, const size_t* ns, int cnt, const void* gate_p, unsigned want);
extern "C" int tgcuHsWait(int timeout_ms);
extern "C" int tgcuHsPushPre(void* const* ps, const size_t* ns, int cnt);
extern "C" int tgcuHsPushGo();
extern "C" int tgcuHsPushPreNext(void* const* ps, const size_t* ns, int cnt, void* const* nps, const size_t* nns, int ncnt, const void* gate_p, unsigned want);
extern "C" int tgcuHsRelease();
namespace strata::tgcu {
// pre-armed handshake (STRATA_HS_PRE): arm the copy engine to pull the segments once *gate >= want; wait for them; queue the
// gated push of segments before the payload exists, then send it
inline int hs_arm(void* const* ps, const size_t* ns, int cnt, const void* gate_p, unsigned want) { return tgcuHsArm(ps, ns, cnt, gate_p, want); }
inline int hs_wait(int timeout_ms) { return tgcuHsWait(timeout_ms); }
inline int hs_push_pre(void* const* ps, const size_t* ns, int cnt) { return tgcuHsPushPre(ps, ns, cnt); }
inline int hs_push_go() { return tgcuHsPushGo(); }
inline int hs_push_pre_next(void* const* ps, const size_t* ns, int cnt, void* const* nps, const size_t* nns, int ncnt, const void* gate_p, unsigned want) { return tgcuHsPushPreNext(ps, ns, cnt, nps, nns, ncnt, gate_p, want); }
inline int hs_release() { return tgcuHsRelease(); }
// the driver's status comes back to the caller: 0 when the hand-off happened, its own nonzero code when it did not
inline int push(const void* p, size_t n) { if (p == nullptr || n == 0) return 0; return tgcuMappedPush(p, n); }
inline int pull(void* p, size_t n) { if (p == nullptr || n == 0) return 0; return tgcuMappedPull(p, n); }
inline int manage(const void* p) { if (p == nullptr) return 0; return tgcuMappedManage(p); }
}  // namespace strata::tgcu
#else
namespace strata::tgcu {
inline int hs_arm(void* const*, const size_t*, int, const void*, unsigned) { return -1; }
inline int hs_wait(int) { return -1; }
inline int hs_push_pre(void* const*, const size_t*, int) { return -1; }
inline int hs_push_go() { return -1; }
inline int hs_push_pre_next(void* const*, const size_t*, int, void* const*, const size_t*, int, const void*, unsigned) { return -1; }
inline int hs_release() { return -1; }
inline int push(const void*, size_t) { return 0; }
inline int pull(void*, size_t) { return 0; }
inline int manage(const void*) { return 0; }
}  // namespace strata::tgcu
#endif
