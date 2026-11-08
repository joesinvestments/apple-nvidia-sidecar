"""moe_combine_peer16's slot guard (src/prefill/kernels.cu, fix-0246#0), run on the host (no GPU here).

The kernel sums the routed rows of token t: for each of its K = 10 slots it reads `slot[t * 10 + k]`, and a slot at or
past `rows_local` belongs to the peer card, so the local buffer must not be read for it. Only the upper bound is checked:
a negative slot (for example -1, the "not in this group" marker the host tables use) passes `r < rows_local` and indexes
`Dm` and `row_sd` BEFORE their buffers, so the combine adds whatever lies in front of the row buffer.

What this test does: it takes the kernel's own source text out of src/prefill/kernels.cu, compiles that very function
for the host behind small shims (blockIdx / blockDim / threadIdx, __half2float, sigm, fmaf, the file's N), and runs it
on planted buffers. The rows in front of Dm and row_sd are filled with a 4096.0 sentinel, so a slot read from before the
buffer lands in the sum. Every case compares the kernel's output with the same sum over the slots it should read
(0 <= r < rows_local, the contract the kernel is called under).

One line per case, PASS or FAIL; exit non-zero on any FAIL.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

KERNEL = "moe_combine_peer16_kernel"


def root() -> Path:
    """The tree on trial: prove_test runs this file with cwd = a copy of the engine, not this file's tree."""
    d = Path.cwd().resolve()
    for p in (d, *d.parents):
        if (p / "src" / "prefill" / "kernels.cu").is_file():
            return p
    raise SystemExit(f"FAIL cannot find src/prefill/kernels.cu above {d}")


def kernel_source(cu: str) -> str:
    """The kernel's own text: from `__global__ void <name>(` to the closing brace of its body."""
    m = re.search(r"^__global__ void " + KERNEL + r"\(.*?\)\s*\{", cu, re.M | re.S)
    if not m:
        raise SystemExit(f"FAIL src/prefill/kernels.cu has no {KERNEL}")
    start, depth, i = m.start(), 0, m.end() - 1
    while i < len(cu):
        if cu[i] == "{":
            depth += 1
        elif cu[i] == "}":
            depth -= 1
            if depth == 0:
                return cu[start:i + 1]
        i += 1
    raise SystemExit(f"FAIL {KERNEL}'s body does not close")


def rows_of(cu: str) -> int:
    m = re.search(r"^constexpr int N = (\d+)", cu, re.M)
    if not m:
        raise SystemExit("FAIL src/prefill/kernels.cu has no constexpr int N")
    return int(m.group(1))


SHIM = r"""
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#define __global__
using std::fmaf;

static constexpr int N = @N@;   // src/prefill/kernels.cu's own N

struct EmuDim3 { unsigned x, y, z; };
static EmuDim3 blockIdx = {0, 0, 0}, blockDim = {0, 0, 0}, threadIdx = {0, 0, 0};

struct EmuHalf { uint16_t u; };
static EmuHalf __ushort_as_half(uint16_t u) { EmuHalf h; h.u = u; return h; }
static float __half2float(EmuHalf h) {   // the value CUDA's conversion gives (exact for every half)
    const uint32_t e = (h.u >> 10) & 0x1f, m = h.u & 0x3ff;
    if (e == 0) {   // zero or subnormal: m * 2^-24, signed
        const float v = std::ldexp((float) m, -24);
        return (h.u & 0x8000u) ? -v : v;
    }
    const uint32_t sign = (uint32_t) (h.u >> 15) << 31;
    const uint32_t bits = e == 31 ? (sign | 0x7f800000u | (m << 13)) : (sign | ((e + 112) << 23) | (m << 13));
    float f;
    std::memcpy(&f, &bits, sizeof f);
    return f;
}
static float sigm(float x) { return 1.0f / (1.0f + __builtin_expf(-x)); }

@KERNEL_TEXT@
"""

MAIN = r"""
static const float POISON = 4096.0f;   // what a slot read from before a buffer finds there
static int failed = 0;

// deterministic planted values in [-1, 1)
static float val(uint32_t seed, size_t i) {
    uint32_t h = seed * 2654435761u + (uint32_t) i * 2246822519u;
    h ^= h >> 15; h *= 2654435761u; h ^= h >> 13;
    return (float) (h % 2000) / 1000.0f - 1.0f;
}

static std::vector<float> planted(int n, uint32_t seed) {
    std::vector<float> v((size_t) n);
    for (int i = 0; i < n; ++i) v[(size_t) i] = val(seed, (size_t) i);
    return v;
}

// one case: the kernel's output against the sum over the slots it is allowed to read (0 <= r < rows_local)
static void run_case(const char* name, int T, int64_t rows_local, const std::vector<int32_t>& slot,
                     const std::vector<float>& w, const std::vector<float>& rows, const std::vector<float>& sd,
                     const std::vector<float>& shared, const std::vector<float>& sg, const std::vector<uint16_t>& peer) {
    const size_t nrow = rows.size() / (size_t) N, guard = 64;   // guard rows planted before each buffer
    std::vector<float> darena((guard + nrow + 1) * (size_t) N, POISON);
    float* Dm = darena.data() + guard * (size_t) N;
    std::memcpy(Dm, rows.data(), rows.size() * sizeof(float));
    std::vector<float> sarena(guard + nrow + 1, POISON);
    float* row_sd = sarena.data() + guard;
    std::memcpy(row_sd, sd.data(), sd.size() * sizeof(float));
    std::vector<float> bo((size_t) T * (size_t) N, 0.0f);

    blockDim = EmuDim3{256, 1, 1};
    const unsigned blocks = (unsigned) (((size_t) T * (size_t) N + 255) / 256);
    for (unsigned b = 0; b < blocks; ++b) {
        blockIdx = EmuDim3{b, 1, 1};
        for (unsigned j = 0; j < 256; ++j) {   // one thread per output element, as the launch config gives
            threadIdx = EmuDim3{j, 1, 1};
            @KERNEL@(Dm, slot.data(), w.data(), shared.data(), sg.data(), peer.data(), rows_local, bo.data(), T, row_sd);
        }
    }

    double worst = 0.0, want_at_worst = 0.0, got_at_worst = 0.0;
    size_t worst_i = 0;
    for (int t = 0; t < T; ++t) {
        for (int d = 0; d < N; ++d) {
            double s = 0.0;
            for (int k = 0; k < 10; ++k) {
                const int32_t r = slot[(size_t) t * 10 + k];
                if (r < 0 || r >= rows_local) continue;   // not this card's row: the peer's sum covers it
                s += (double) w[(size_t) t * 10 + k] * (double) rows[(size_t) r * N + d] * (double) sd[(size_t) r];
            }
            const double want = s + (double) __half2float(__ushort_as_half(peer[(size_t) t * N + d])) +
                                (double) shared[(size_t) t * N + d] * (double) sigm(sg[(size_t) t]);
            const double got = (double) bo[(size_t) t * N + d];
            const double diff = std::fabs(got - want) / (1.0 + std::fabs(want));
            if (diff > worst) { worst = diff; worst_i = (size_t) t * N + d; want_at_worst = want; got_at_worst = got; }
        }
    }
    if (worst <= 1e-4) {
        std::printf("PASS %s: %d token(s), %lld local rows, largest relative difference %.3g\n", name, T,
                    (long long) rows_local, worst);
    } else {
        ++failed;
        std::printf("FAIL %s: output[%zu] is %.9g where the slots 0 <= r < %lld sum to %.9g\n", name, worst_i,
                    got_at_worst, (long long) rows_local, want_at_worst);
    }
}

int main() {
    const int T = 2;
    // slot[t * 10 + k]: -1 is the host tables' "not in this group" marker; the values past rows_local belong to the
    // peer card; 0..19 are this card's rows, which every case must still sum
    const std::vector<int32_t> one_negative = {0, 1, -1, 2, 3, 4, 5, 6, 7, 8,   9, 10, 11, 12, 13, 14, 15, 16, 17, 18};
    const std::vector<int32_t> deep_negative = {-3, 1, 2, 3, 4, 5, 6, 7, 8, 9,   10, 11, 12, 13, 14, 15, 16, 17, 18, -2};
    const std::vector<int32_t> in_range = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9,   10, 11, 12, 13, 14, 15, 16, 17, 18, 19};
    const std::vector<int32_t> past_end = {0, 1, 2, 3, 4, 5, 6, 7, 8, 25,   10, 11, 12, 13, 14, 15, 16, 17, 18, 400};
    const std::vector<int32_t> first_k_negative = {-4, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                                                   10, 11, 12, 13, 14, 15, 16, 17, 18, -1};
    const std::vector<uint16_t> peer_zero((size_t) T * N, 0), peer_one((size_t) T * N, 0x3c00);   // 0.0 and 1.0 in FP16

    run_case("ordinary slots sum as before", T, 20, in_range, planted(T * 10, 11), planted(20 * N, 12), planted(20, 13),
             planted(T * N, 14), planted(T, 15), peer_zero);
    run_case("a slot past rows_local is still skipped", T, 20, past_end, planted(T * 10, 21), planted(20 * N, 22),
             planted(20, 23), planted(T * N, 24), planted(T, 25), peer_one);
    run_case("a negative slot is not read (row_sd on)", T, 20, one_negative, planted(T * 10, 31), planted(20 * N, 32),
             planted(20, 33), planted(T * N, 34), planted(T, 35), peer_zero);
    run_case("a negative slot is not read (row_sd off)", T, 20, deep_negative, planted(T * 10, 41), planted(20 * N, 42),
             planted(20, 43), planted(T * N, 44), planted(T, 45), peer_one);
    run_case("a negative slot at the first k", T, 20, first_k_negative, planted(T * 10, 51), planted(20 * N, 52),
             planted(20, 53), planted(T * N, 54), planted(T, 55), peer_zero);
    return failed != 0;
}
"""


def main() -> int:
    r = root()
    cu = (r / "src" / "prefill" / "kernels.cu").read_text(encoding="utf-8", errors="replace")
    src = SHIM.replace("@N@", str(rows_of(cu))).replace("@KERNEL_TEXT@", kernel_source(cu))
    src = src + MAIN.replace("@KERNEL@", KERNEL)
    tmp = tempfile.mkdtemp(prefix="fix-0246-")
    cpp, exe = os.path.join(tmp, "combine_peer16_host.cpp"), os.path.join(tmp, "combine_peer16_host")
    Path(cpp).write_text(src, encoding="utf-8")
    cc = os.environ.get("CXX") or "g++"
    build = subprocess.run([cc, "-std=c++17", "-O0", "-o", exe, cpp], capture_output=True, text=True, timeout=300)
    if build.returncode != 0:
        print("FAIL " + KERNEL + " does not build for the host: " + " | ".join(build.stderr.strip().splitlines()[:4]))
        return 1
    run = subprocess.run([exe], capture_output=True, text=True, timeout=300)
    out = run.stdout + run.stderr
    sys.stdout.write(out if out.endswith("\n") else out + "\n")
    if run.returncode != 0 or any(l.startswith("FAIL") for l in out.splitlines()):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
