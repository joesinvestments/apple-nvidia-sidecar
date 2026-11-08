// arm64: no cpuid; every x86 feature query reads as absent
#pragma once
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
// MSVC has no <cpuid.h> and no #include_next; its CPUID intrinsics live in <intrin.h>.
#include <intrin.h>
static inline void __sce_cpuidex(int out[4], int leaf, int sub) { __cpuidex(out, leaf, sub); }
static inline void __sce_cpuid(int out[4], int leaf) { __cpuid(out, leaf); }
#define __cpuid_count(leaf, sub, a, b, c, d) do { int __sce_regs[4]; __sce_cpuidex(__sce_regs, (int)(leaf), (int)(sub)); (a) = (unsigned)__sce_regs[0]; (b) = (unsigned)__sce_regs[1]; (c) = (unsigned)__sce_regs[2]; (d) = (unsigned)__sce_regs[3]; } while (0)
#define __cpuid(leaf, a, b, c, d) do { int __sce_regs[4]; __sce_cpuid(__sce_regs, (int)(leaf)); (a) = (unsigned)__sce_regs[0]; (b) = (unsigned)__sce_regs[1]; (c) = (unsigned)__sce_regs[2]; (d) = (unsigned)__sce_regs[3]; } while (0)
static inline unsigned __get_cpuid_max(unsigned ext, unsigned *sig) { int __sce_regs[4]; __sce_cpuid(__sce_regs, 0); if (sig) *sig = (unsigned)__sce_regs[1]; if (ext == 0) return (unsigned)__sce_regs[0]; __sce_cpuid(__sce_regs, (int)ext); return (unsigned)__sce_regs[0]; }
#elif defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include_next <cpuid.h>
#else
#define __cpuid_count(leaf, sub, a, b, c, d) ((a) = 0, (b) = 0, (c) = 0, (d) = 0)
#define __cpuid(leaf, a, b, c, d) ((a) = 0, (b) = 0, (c) = 0, (d) = 0)
static inline int __get_cpuid_max(unsigned ext, unsigned *sig) { if (sig) *sig = 0; return 0; }
#endif
