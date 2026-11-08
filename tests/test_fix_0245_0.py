#!/usr/bin/env python3
"""fix-0245#0: moe_combine_peer's slot guard must bound both ends of the slot index.

src/prefill/kernels.cu, moe_combine_peer_kernel and moe_combine_peer16_kernel:

    const int32_t r = slot[t * 10 + k];
    if (r < rows_local) s = fmaf(w[t * 10 + k], row_sd ? Dm[(int64_t) r * N + d] * row_sd[r] : Dm[(int64_t) r * N + d], s);

The guard tests only the upper bound.  A slot that holds a negative sentinel (the usual encoding of an
unused slot) passes `r < rows_local`, and the body reads Dm[r * N + d] and row_sd[r] before the start of
both buffers.  With a negative slot the sum picks up memory that is not a row, so the layer's output is
wrong; the same read past the start of a device buffer is the out-of-bounds access.

The test compiles the two kernels exactly as kernels.cu holds them, against a small fake CUDA shim (no
GPU, no model), runs one thread per output element, and plants a poison band directly in front of the
Dm and row_sd rows.  On today's code a negative slot folds that poison into the output; with a guard
that also checks the lower bound the slot is skipped and the sum is the one the valid slots give.
Ordinary routing (every slot a local row) must give the same output before and after.

Run: python3 tests/test_fix_0245_0.py
Prints one PASS or FAIL line per case and exits non-zero on any FAIL.  Temp folders only.
"""
import math
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

K = 10          # top-k slots per token, as the kernel's t * 10 + k
N = 4           # the row width the harness compiles the kernels at
POISON = 1e30   # what sits in the band before Dm and row_sd


def find_root():
    """The tree under test: the copy we run in, else the one this file sits in."""
    rel = Path("src") / "prefill" / "kernels.cu"
    for base in (Path.cwd(), Path(__file__).resolve().parent.parent):
        if (base / rel).is_file():
            return base
    return Path.cwd()


ROOT = find_root()
SRC = ROOT / "src" / "prefill" / "kernels.cu"


def fail(msg):
    print("FAIL " + msg)
    return False


def ok(msg):
    print("PASS " + msg)
    return True


def kernel_text(src, name):
    """The kernel's own definition, from __global__ to its closing brace."""
    at = src.find("__global__ void " + name + "(")
    if at < 0:
        return None
    open_brace = src.find("{", at)
    if open_brace < 0:
        return None
    depth = 0
    for i in range(open_brace, len(src)):
        c = src[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return src[at:i + 1]
    return None


PROLOGUE = r'''
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
using std::fmaf;
#define __global__
struct FakeDim { int x, y, z; };
static FakeDim blockIdx = {0, 0, 0}, threadIdx = {0, 0, 0};
static const FakeDim blockDim = {256, 1, 1};
static constexpr int64_t N = %d;
static const float POISON = 1e30f;   // the band before Dm and row_sd; a read into it shows in the output
static inline float sigm(float x) { return (float) (1.0 / (1.0 + std::exp((double) -x))); }
struct FakeHalf { uint16_t bits; };
static inline FakeHalf __ushort_as_half(uint16_t b) { return FakeHalf{b}; }
static inline float __half2float(FakeHalf h) {
    const uint32_t sign = (uint32_t) (h.bits >> 15) << 31;
    uint32_t e = (h.bits >> 10) & 0x1fu, m = h.bits & 0x3ffu, f;
    if (e == 0) {
        if (m == 0) { f = sign; }
        else {
            e = 127 - 15 + 1;
            while ((m & 0x400u) == 0) { m <<= 1; --e; }
            f = sign | (e << 23) | ((m & 0x3ffu) << 13);
        }
    } else if (e == 31) {
        f = sign | 0x7f800000u | (m << 13);
    } else {
        f = sign | ((e + 127 - 15) << 23) | (m << 13);
    }
    float out;
    std::memcpy(&out, &f, sizeof out);
    return out;
}
''' % N

DRIVER = r'''
static void read_n(std::FILE* f, double* out, size_t n) {
    for (size_t i = 0; i < n; ++i)
        if (std::fscanf(f, "%lf", &out[i]) != 1) { std::fprintf(stderr, "short input\n"); std::exit(3); }
}
static void read_row(std::FILE* f, float* out, size_t n) {
    double v;
    for (size_t i = 0; i < n; ++i) { if (std::fscanf(f, "%lf", &v) != 1) std::exit(3); out[i] = (float) v; }
}
int main(int argc, char** argv) {
    if (argc < 3) return 2;
    const int which = std::atoi(argv[1]);   // 0: moe_combine_peer, 1: moe_combine_peer16
    std::FILE* f = std::fopen(argv[2], "r");
    if (f == nullptr) return 2;
    long long T = 0, rows_local = 0, n = 0;
    if (std::fscanf(f, "%lld %lld %lld", &T, &rows_local, &n) != 3) return 2;
    const size_t nslot = (size_t) (T * 10), nrow = (size_t) (T * n);
    std::vector<int32_t> slot(nslot);
    for (size_t i = 0; i < nslot; ++i) { long long v = 0; if (std::fscanf(f, "%lld", &v) != 1) return 3; slot[i] = (int32_t) v; }
    std::vector<float> w(nslot), shared(nrow), sg((size_t) T), peer(nrow), dmm((size_t) (rows_local * n));
    read_row(f, w.data(), nslot);
    read_row(f, shared.data(), nrow);
    read_row(f, sg.data(), (size_t) T);
    std::vector<uint16_t> peer16(nrow);
    if (which == 1) {
        for (size_t i = 0; i < nrow; ++i) { long long v = 0; if (std::fscanf(f, "%lld", &v) != 1) return 3; peer16[i] = (uint16_t) v; }
    } else {
        read_row(f, peer.data(), nrow);
    }
    read_row(f, dmm.data(), dmm.size());
    long long has_sd = 0;
    if (std::fscanf(f, "%lld", &has_sd) != 1) return 3;
    // the rows sit after a poison band, so a read before the buffer is visible in the output
    const size_t pad = 2 * (size_t) n;
    std::vector<float> dmb(pad + dmm.size() + pad, POISON);
    std::memcpy(dmb.data() + pad, dmm.data(), dmm.size() * sizeof(float));
    float* Dm = dmb.data() + pad;
    std::vector<float> sdb;
    const float* row_sd = nullptr;
    if (has_sd) {
        std::vector<float> rows((size_t) rows_local);
        read_row(f, rows.data(), rows.size());
        sdb.assign(2 + rows.size() + 2, POISON);
        std::memcpy(sdb.data() + 2, rows.data(), rows.size() * sizeof(float));
        row_sd = sdb.data() + 2;
    }
    std::vector<float> bo(nrow, 0.0f);
    for (int64_t i = 0; i < T * n; ++i) {
        blockIdx.x = (int) (i / 256);
        threadIdx.x = (int) (i % 256);
        if (which == 1)
            moe_combine_peer16_kernel(Dm, slot.data(), w.data(), shared.data(), sg.data(), peer16.data(), rows_local,
                                      bo.data(), T, row_sd);
        else
            moe_combine_peer_kernel(Dm, slot.data(), w.data(), shared.data(), sg.data(), peer.data(), rows_local,
                                    bo.data(), T, row_sd);
    }
    for (size_t i = 0; i < nrow; ++i) std::printf("%.9g\n", (double) bo[i]);
    std::fclose(f);
    return 0;
}
'''


def build(work):
    """Compile the two kernels from kernels.cu with the CPU shim; the executable or None."""
    src = SRC.read_text(encoding="utf-8", errors="replace")
    kernels = []
    for name in ("moe_combine_peer_kernel", "moe_combine_peer16_kernel"):
        body = kernel_text(src, name)
        if body is None:
            return None, name + " is not in " + str(SRC)
        kernels.append(body)
    cpp = work / "harness.cpp"
    cpp.write_text(PROLOGUE + "\n" + kernels[0] + "\n\n" + kernels[1] + "\n\n" + DRIVER, encoding="utf-8")
    cc = shutil.which("clang++") or shutil.which("g++") or shutil.which("c++")
    if cc is None:
        return None, "no C++ compiler on PATH to run the kernels with"
    exe = work / "harness"
    r = subprocess.run([cc, "-std=c++17", "-O1", "-o", str(exe), str(cpp)], capture_output=True, text=True)
    if r.returncode != 0:
        return None, "the kernels do not compile: " + (r.stdout + r.stderr).strip()[-300:]
    return exe, ""


def half_bits(v):
    """The IEEE half bit pattern of a value the shim's __half2float turns back into it."""
    return {0.5: 0x3800, 1.0: 0x3C00, -1.0: 0xBC00, 2.0: 0x4000, 0.25: 0x3400, -0.5: 0xB800,
            0.75: 0x3A00, -0.125: 0xB000, 0.125: 0x3000, 4.0: 0x4400}[v]


def half_value(bits):
    e = (bits >> 10) & 0x1F
    m = bits & 0x3FF
    v = (1.0 + m / 1024.0) * 2.0 ** (e - 15)
    return -v if bits & 0x8000 else v


def run_case(exe, work, case):
    """One kernel launch on the planted data: the output rows, or None and the reason."""
    T, rows = len(case["slots"]), case["rows_local"]
    lines = ["%d %d %d" % (T, rows, N)]
    lines.append(" ".join(str(v) for row in case["slots"] for v in row))
    lines.append(" ".join("%.17g" % v for v in case["w"]))
    lines.append(" ".join("%.17g" % v for v in case["shared"]))
    lines.append(" ".join("%.17g" % v for v in case["sg"]))
    if case["which"] == 1:
        lines.append(" ".join(str(v) for v in case["peer16"]))
    else:
        lines.append(" ".join("%.17g" % v for v in case["peer"]))
    lines.append(" ".join("%.17g" % v for row in case["dm"] for v in row))
    lines.append("1" if case["sd"] is not None else "0")
    if case["sd"] is not None:
        lines.append(" ".join("%.17g" % v for v in case["sd"]))
    inp = work / "input.txt"
    inp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    r = subprocess.run([str(exe), str(case["which"]), str(inp)], capture_output=True, text=True)
    if r.returncode != 0:
        return None, "the kernel run exited %d: %s" % (r.returncode, r.stderr.strip()[-200:])
    got = []
    for line in r.stdout.split():
        try:
            got.append(float(line))
        except ValueError:
            return None, "unreadable kernel output: " + line
    if len(got) != T * N:
        return None, "the kernel wrote %d of %d outputs" % (len(got), T * N)
    return got, ""


def wanted(case):
    """The sum the routed slots define: local rows only, peer rows and empty slots add nothing."""
    rows, sd = case["rows_local"], case["sd"]
    out = []
    for t in range(len(case["slots"])):
        for d in range(N):
            s = 0.0
            for k in range(K):
                r = case["slots"][t][k]
                if 0 <= r < rows:
                    v = case["dm"][r][d] * (sd[r] if sd is not None else 1.0)
                    s = case["w"][t * K + k] * v + s
            if case["which"] == 1:
                p = half_value(case["peer16"][t * N + d])
            else:
                p = case["peer"][t * N + d]
            out.append((s + p) + case["shared"][t * N + d] / (1.0 + math.exp(-case["sg"][t])))
    return out


def close(a, b):
    if not (math.isfinite(a) and math.isfinite(b)):
        return a == b
    return abs(a - b) <= 1e-4 * max(1.0, abs(a), abs(b))


def cases():
    """Planted routing: negative sentinels, peer rows, and one ordinary all-local token batch."""
    dm = [[1.0, 2.0, 3.0, 4.0],
          [0.5, 1.0, 1.5, 2.0],
          [-1.0, -2.0, -3.0, -4.0],
          [8.0, 4.0, 2.0, 1.0]]
    w = [0.5, 0.25, 0.125, 0.0625, 0.5, 0.25, 0.125, 0.0625, 0.5, 0.25,
         0.25, 0.5, 0.0625, 0.125, 0.25, 0.5, 0.0625, 0.125, 0.25, 0.5]
    shared = [0.5, 0.25, 0.125, 0.0625, 1.0, 0.5, 0.25, 0.125]
    sg = [0.5, 1.0]
    peer = [0.125, 0.25, -0.5, 1.0, 0.25, 0.5, 0.75, -0.125]
    peer16 = [half_bits(v) for v in [0.5, 1.0, -1.0, 2.0, 0.25, -0.5, 0.75, -0.125]]
    sd = [0.5, 2.0, 1.0, 0.25]
    sentinels = [[-1, 0, 1, 2, 3, 4, 5, -1, 2, 3],
                 [3, 2, 1, 0, -1, 6, 7, 0, 1, 2]]
    ordinary = [[3, 0, 1, 2, 3, 0, 1, 2, 3, 1],
                [0, 1, 2, 3, 2, 1, 0, 3, 1, 2]]
    return [
        {"name": "peer kernel: a negative slot is not a local row (no row_sd)",
         "which": 0, "rows_local": 4, "slots": sentinels, "w": w, "shared": shared, "sg": sg,
         "peer": peer, "peer16": peer16, "dm": dm, "sd": None},
        {"name": "peer kernel: a negative slot is not a local row (with row_sd)",
         "which": 0, "rows_local": 4, "slots": sentinels, "w": w, "shared": shared, "sg": sg,
         "peer": peer, "peer16": peer16, "dm": dm, "sd": sd},
        {"name": "peer16 kernel: a negative slot is not a local row",
         "which": 1, "rows_local": 4, "slots": sentinels, "w": w, "shared": shared, "sg": sg,
         "peer": peer, "peer16": peer16, "dm": dm, "sd": None},
        {"name": "ordinary routing: every slot is a local row, unchanged",
         "which": 0, "rows_local": 4, "slots": ordinary, "w": w, "shared": shared, "sg": sg,
         "peer": peer, "peer16": peer16, "dm": dm, "sd": sd},
    ]


def main():
    if not SRC.is_file():
        return 0 if fail("src/prefill/kernels.cu is missing under " + str(ROOT)) else 1
    work = Path(tempfile.mkdtemp(prefix="fix-0245-"))
    try:
        exe, why = build(work)
        if exe is None:
            fail(why)
            return 1
        good = True
        for case in cases():
            got, why = run_case(exe, work, case)
            if got is None:
                good = fail(case["name"] + ": " + why) and good
                continue
            want = wanted(case)
            bad = [i for i in range(len(want)) if not close(got[i], want[i])]
            if bad:
                i = bad[0]
                good = fail("%s: output %d is %.6g, the routed slots give %.6g"
                            % (case["name"], i, got[i], want[i])) and good
            else:
                good = ok(case["name"]) and good
        return 0 if good else 1
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
