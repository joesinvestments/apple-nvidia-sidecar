#!/usr/bin/env python3
"""fix-0116#4: the warm queue's skip-if-resident check must count pages at the host page size.

src/core/expert_source.cpp, the worker in MacWarmQ::start:

    const uintptr_t pg = 16384, p = (uintptr_t) base + j.first, a = p & ~(pg - 1);
    const size_t len = (size_t) (p + j.second - a), npg = (len + pg - 1) / pg;
    char vec[512];
    if (npg <= sizeof vec && mincore((void*) a, len, vec) == 0) {
        size_t miss = 0;
        for (size_t i = 0; i < npg; ++i) miss += (vec[i] & 1) ? 0 : 1;
        if (miss == 0) { skipped.fetch_add(1); continue; }
    }

mincore(2) reports one byte per page of the host, so the page size used to align `a` and to count
`npg` has to be the host's.  Hard-coded to the 16 KiB of an Apple Silicon Mac, on a 4 KiB-page Mac
(an Intel Mac) npg is up to four times too small: the loop inspects only the first npg bytes of
vec, which describe the first quarter of the range, and the worker drops the job as resident while
most of the blob is not in the page cache, so the look-ahead never warms it.  A range of
512 * 16384 bytes still passes the npg <= sizeof vec guard, and the kernel then writes 2048 page
bytes into the 512 byte vec: a stack overwrite on a 4 KiB-page host.

This test compiles the worker's residency block from the tree under test against a fake file, a
fake pread and a fake mincore that reports one byte per 4096 byte page, the host the finding names
(no GPU, no model, no mapping of real weights).  Ordinary input, a range whose pages are all in
the cache, gives the same answer before and after the fix.

Run from the tree root: python3 tests/test_fix_0116_4.py
Prints one PASS or FAIL line per case and exits non-zero on any FAIL.  Temp folders only.
"""
import os
import shutil
import subprocess
import sys
import tempfile

SRC_REL = os.path.join("src", "core", "expert_source.cpp")
PAGE = 4096           # the page size of the host the finding is about
VEC = 512             # the char vec[512] the block declares

HARNESS = r"""
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <vector>
#include <sys/types.h>
#include <unistd.h>

// The host this harness fakes: 4096 byte pages.  Whatever the block asks the host for its page
// size, it hears 4096 (a 4 KiB-page Mac, which is where the hard-coded 16384 is wrong).
static long fake_sysconf(int name) { return name == _SC_PAGESIZE ? 4096 : -1; }
static unsigned long vm_page_size = 4096;
#define sysconf fake_sysconf
#define getpagesize() 4096

static uint8_t* g_base = nullptr;
static std::vector<char> g_file;                 // the file behind the mapping, a byte per file byte
static std::vector<int> g_missing;               // 4 KiB page indices from g_base that are NOT in the page cache
static int g_mincore_calls = 0, g_mincore_pages = 0, g_overflow = 0;
static long long g_read_bytes = 0;

static bool not_resident(long page) {
    for (int m : g_missing) if ((long) m == page) return true;
    return false;
}

// mincore(2) on a 4 KiB-page host: one byte per 4096 byte page of [addr, addr + len).  The caller's
// vec holds 512 bytes (char vec[512]); a range of more pages than that is the overwrite the block
// must refuse before it calls us, so we cap the write and record it.
static int fake_mincore(void* addr, size_t len, char* vec) {
    ++g_mincore_calls;
    const uintptr_t a = (uintptr_t) addr;
    if (len == 0 || (a & 4095) != 0) return -1;
    const size_t npages = (len + 4095) / 4096;
    g_mincore_pages = (int) npages;
    if (npages > 512) ++g_overflow;
    for (size_t i = 0; i < npages && i < 512; ++i) {
        const long rel = (long) (a + i * 4096 - (uintptr_t) g_base);
        vec[i] = (rel >= 0 && !not_resident(rel / 4096)) ? 1 : 0;
    }
    return 0;
}
#define mincore fake_mincore

static ssize_t fake_pread(int fd, void* buf, size_t n, off_t off) {
    (void) fd;
    if ((size_t) off >= g_file.size()) return 0;
    size_t k = g_file.size() - (size_t) off;
    if (k > n) k = n;
    std::memcpy(buf, g_file.data() + off, k);
    g_read_bytes += (long long) k;
    return (ssize_t) k;
}
#define pread fake_pread

static std::atomic<int64_t> skipped{0}, reads{0};

// The block the worker runs for one predicted blob, exactly as the tree writes it.
static void warm_one(const uint8_t* base, int fd, std::pair<uint64_t, uint64_t> j) {
    std::vector<uint8_t> scratch;
    for (int once = 0; once < 1; ++once) {
@@BLOCK@@
    }
}

static void setup(const char* what) {
    const size_t span = 16u << 20;
    g_base = (uint8_t*) std::aligned_alloc(16384, span);
    if (g_base == nullptr) std::exit(4);
    std::memset(g_base, 7, span);
    g_file.assign(span, 7);
    g_missing.clear();
    if (std::strcmp(what, "part") == 0) {          // 16 pages from base + 16384: the first 4 in cache, 12 not
        for (int p = 8; p < 20; ++p) g_missing.push_back(p);
    } else if (std::strcmp(what, "unaligned") == 0) {   // 10 pages from base + 24576: the last 2 not
        g_missing.push_back(14);
        g_missing.push_back(15);
    } else if (std::strcmp(what, "big") == 0) {    // 512 * 16384 bytes from base + 16384: the tail not
        for (int p = 1000; p < 1004 + 4092; ++p) g_missing.push_back(p);
    }
}

int main(int argc, char** argv) {
    if (argc < 2) return 2;
    const char* what = argv[1];
    setup(what);
    long long off = 16384, bytes = 65536;
    if (std::strcmp(what, "unaligned") == 0) { off = 24576; bytes = 40960; }
    if (std::strcmp(what, "big") == 0) bytes = 512LL * 16384;
    warm_one(g_base, 7, std::make_pair((uint64_t) off, (uint64_t) bytes));
    std::printf("case=%s reads=%lld skipped=%lld mincore_calls=%d mincore_pages=%d overflow=%d read_bytes=%lld\n",
                what, (long long) reads.load(), (long long) skipped.load(), g_mincore_calls,
                g_mincore_pages, g_overflow, g_read_bytes);
    return 0;
}
"""

fails = 0


def check(name, ok, detail=""):
    global fails
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else ": " + detail))
    if not ok:
        fails += 1


def find_root():
    """The tree under test: the copy prove_test runs us in, else the one this file sits in."""
    for base in (os.getcwd(), os.path.dirname(os.path.dirname(os.path.abspath(__file__)))):
        if os.path.isfile(os.path.join(base, SRC_REL)):
            return base
    return os.getcwd()


def resident_block(src):
    """The warm worker's residency lines, from the page-size declaration to the read it counts."""
    at = src.find("struct MacWarmQ")
    end = src.find("MacWarmQ* mac_warm_q", at)
    if at < 0 or end < 0:
        return None, "MacWarmQ is gone from " + SRC_REL
    lines = src[at:end].splitlines()
    start = stop = None
    for i, line in enumerate(lines):
        if start is None and "uintptr_t pg" in line:
            start = i
        if start is not None and "reads.fetch_add(1);" in line:
            stop = i
            break
    if start is None:
        return None, "the warm worker's page-size declaration is gone from " + SRC_REL
    if stop is None:
        return None, "the warm worker's residency block has no end in " + SRC_REL
    return "\n".join(lines[start:stop + 1]), None


def compiler():
    for name in (os.environ.get("CXX"), "clang++", "c++", "g++"):
        if name and shutil.which(name):
            return name
    return None


def run(program, *args):
    r = subprocess.run([program, *args], capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        return None, (r.stdout + r.stderr).strip()
    return r.stdout.strip(), None


def parse(line):
    out = {}
    for part in line.split():
        k, _, v = part.partition("=")
        if k != "case":
            out[k] = int(v)
    return out


def main():
    root = find_root()
    src_path = os.path.join(root, SRC_REL)
    if not os.path.isfile(src_path):
        check("the tree holds " + SRC_REL, False, src_path + " is missing")
        return
    with open(src_path, encoding="utf-8") as f:
        src = f.read()
    block, err = resident_block(src)
    if block is None:
        check("the warm worker's residency block is readable", False, err)
        return
    check("the warm worker's residency block is readable", True)

    cxx = compiler()
    if cxx is None:
        check("a C++ compiler is available", False, "no clang++, c++ or g++ on PATH")
        return
    work = tempfile.mkdtemp(prefix="fix-0116-4-")
    try:
        cpp = os.path.join(work, "warm.cpp")
        exe = os.path.join(work, "warm")
        with open(cpp, "w", encoding="utf-8") as f:
            f.write(HARNESS.replace("@@BLOCK@@", block))
        build = subprocess.run([cxx, "-std=c++17", "-O0", "-o", exe, cpp],
                               capture_output=True, text=True, timeout=120)
        if build.returncode != 0:
            check("the faked warm block compiles", False, (build.stdout + build.stderr).strip()[:400])
            return
        check("the faked warm block compiles", True)

        cases = []

        line, err = run(exe, "resident")
        cases.append(("a range whose pages are all in the cache is skipped", line, err,
                      lambda o: o["reads"] == 0 and o["skipped"] == 1 and o["overflow"] == 0))

        line, err = run(exe, "part")
        cases.append(("a range whose first quarter is in the cache is still read", line, err,
                      lambda o: o["reads"] == 1 and o["skipped"] == 0 and o["read_bytes"] == 65536))

        line, err = run(exe, "unaligned")
        cases.append(("an unaligned range is read when its last pages are not in the cache", line, err,
                      lambda o: o["reads"] == 1 and o["skipped"] == 0 and o["read_bytes"] == 40960))

        line, err = run(exe, "big")
        cases.append(("a 512 * 16384 byte range never fills the 512 byte vec", line, err,
                      lambda o: o["overflow"] == 0 and o["reads"] == 1 and o["skipped"] == 0))

        for name, line, err, want in cases:
            if line is None:
                check(name, False, "the fake warm worker did not run: " + (err or ""))
                continue
            try:
                obs = parse(line)
            except ValueError:
                check(name, False, "cannot read the observation: " + line)
                continue
            check(name, want(obs), line)
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
    sys.exit(1 if fails else 0)
