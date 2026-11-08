#!/usr/bin/env python3
"""src/core/expert_source.cpp:2404 writes through the d11 selftest mapping before checking it.

d11_selftest (reached only when STRATA_D11_SELFTEST names an experts file and /tmp/d11_selftest is
touched) maps SL * 8 bytes (about 15 MB) at line 2404 and the very next line memsets all of it and
mlocks it.  Nothing between the mmap and the write looks at the result, so an allocation failure
under memory pressure turns into a write through (void*) -1 and the engine dies: the diagnostic path
reports nothing, it takes the process down.

The right answer is to look at the result of the mmap and, when it is MAP_FAILED, report it and bail
out of that loop iteration: no write through the pointer, no mlock, no munmap, no blob handed to the
reader, and the idle poll keeps running.

This test takes the d11_selftest function out of the tree under test and compiles it against fakes
for the file calls, the mmap family, the clock pacing and the MacPull reader (no GPU, no model, no
real experts file, no real mapping).  It then runs the function twice:
  - ordinary input (the mapping succeeds): one map, one unmap, twelve timed rounds of the reader,
    every handed buffer inside the mapping, and one byte really written per buffer;
  - planted input (the mapping fails): today the process dies in the memset; with the guard it
    reports the failure on stderr and goes back to the idle poll.
Run from the tree root: python3 tests/test_fix_0116_2.py
Prints one PASS or FAIL line per case and exits non-zero on any FAIL.  Temp folders only.
"""
import os
import shutil
import subprocess
import sys
import tempfile

SRC_REL = os.path.join("src", "core", "expert_source.cpp")
SIG = "static void d11_selftest("
SL = 1867776
MAP_BYTES = SL * 8
RUN_CALLS = 4 * 3 * 280          # modes x blob counts x timings: one reader call per timed round
DST_COUNT = 4 * 280 * (1 + 2 + 3)   # the calls hand over 1, 2 or 3 buffers

HARNESS = r"""
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#define off_t long
#define MAP_FAILED ((void*) -1)
#define MAP_ANON 0x1000
#define MAP_PRIVATE 0x0002
#define PROT_READ 0x1
#define PROT_WRITE 0x2
#define F_OK 0
#define O_RDONLY 0
#define O_CLOEXEC 0x1000000
#define SEEK_END 2
#define F_NOCACHE 48

static bool g_fail_map = false;
static void* g_base = nullptr;
static long long g_bytes = 0;
static int g_maps = 0, g_unmaps = 0, g_mlocks = 0, g_closes = 0;
static int g_access = 0, g_usleep = 0, g_idle = 0;
static int g_run_calls = 0, g_bad_dst = 0, g_ok_writes = 0;
static int g_munmap_bad = 0, g_mlock_bad = 0;

static void report() {
    std::printf("maps=%d unmaps=%d mlocks=%d closes=%d access_calls=%d usleep_calls=%d idle_polls=%d "
                "run_calls=%d bad_dst=%d ok_writes=%d munmap_bad=%d mlock_bad=%d bytes=%lld\n",
                g_maps, g_unmaps, g_mlocks, g_closes, g_access, g_usleep, g_idle,
                g_run_calls, g_bad_dst, g_ok_writes, g_munmap_bad, g_mlock_bad, g_bytes);
    std::fflush(stdout);
}

static void* fake_mmap(void*, size_t len, int, int, int, long) {
    ++g_maps;
    if (g_fail_map) { g_base = MAP_FAILED; g_bytes = 0; return MAP_FAILED; }
    void* p = std::calloc(1, len);
    g_base = p; g_bytes = p != nullptr ? (long long) len : 0;
    return p;
}
static int fake_munmap(void* p, size_t) { ++g_unmaps; if (p == MAP_FAILED) ++g_munmap_bad; std::free(p); return 0; }
static int fake_mlock(const void* p, size_t) { ++g_mlocks; if (p == MAP_FAILED) ++g_mlock_bad; return 0; }
static int fake_access(const char*, int) { ++g_access; return g_access == 1 ? 0 : -1; }   // a trigger file once, then idle
static int fake_unlink(const char*) { return 0; }
static int fake_open(const char*, int, ...) { return 7; }
static int fake_fcntl(int, int, ...) { return 0; }
static long fake_lseek(int, long, int) { return (long) 64 * 4096; }   // 64 blobs of the fake layout
static int fake_close(int) { ++g_closes; return 0; }
static void fake_usleep(unsigned us) {
    ++g_usleep;
    if (us != 500000) return;   // the pacing inside a timed round
    if (++g_idle >= 2) { report(); std::exit(0); }   // the loop is idle again, nothing left to time
}
static void fake_srand(unsigned) {}
static int fake_rand() { return 12345; }

struct FakeMacPull {
    void run_to(int, const std::vector<std::pair<uint64_t, uint64_t> >& jobs, const std::vector<uint8_t*>& dsts) {
        ++g_run_calls;
        if (jobs.size() != dsts.size()) ++g_bad_dst;
        for (size_t i = 0; i < dsts.size(); ++i) {
            uint8_t* d = dsts[i];
            if (g_base == nullptr || g_base == MAP_FAILED || d == nullptr ||
                (long long) (d - (uint8_t*) g_base) >= g_bytes) { ++g_bad_dst; continue; }
            *d = 0x5a;   // a real write into the slot the selftest handed over
            ++g_ok_writes;
        }
        for (size_t i = 1; i < dsts.size(); ++i) if (dsts[i] == dsts[i - 1]) ++g_bad_dst;
    }
};
static FakeMacPull g_pull;
static FakeMacPull& mac_pull() { return g_pull; }

namespace strata { namespace kernels { namespace cpu {
struct FakeLayout { uint64_t max_blob = 4096; };
inline FakeLayout expert_layout() { return FakeLayout(); }
} } }

#define mmap fake_mmap
#define munmap fake_munmap
#define mlock fake_mlock
#define access fake_access
#define unlink fake_unlink
#define open fake_open
#define fcntl fake_fcntl
#define lseek fake_lseek
#define close fake_close
#define usleep fake_usleep
#define srand fake_srand
#define rand fake_rand

@@BLOCK@@

int main(int argc, char** argv) {
    g_fail_map = argc > 1 && std::strcmp(argv[1], "fail") == 0;
    d11_selftest(std::string("/tmp/experts.bin"));
    report();
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


def brace_end(text, open_index):
    depth = 0
    for i in range(open_index, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return i + 1
    return None


def selftest_block(src):
    """The d11_selftest function, exactly as the tree writes it."""
    a = src.find(SIG)
    if a < 0:
        return None, "the d11 selftest function is gone from " + SRC_REL
    end = brace_end(src, src.index("{", a))
    if end is None:
        return None, "the d11 selftest function has no closing brace"
    return src[a:end], None


def compiler():
    for name in (os.environ.get("CXX"), "clang++", "c++", "g++"):
        if name and shutil.which(name):
            return name
    return None


def run(program, *args):
    """One run of the faked selftest: (exit code, stdout report or None, stderr)."""
    r = subprocess.run([program, *args], capture_output=True, text=True, timeout=300)
    out = r.stdout.strip()
    return r.returncode, (out if out else None), r.stderr


def parse(line):
    got = {}
    for part in line.split():
        k, _, v = part.partition("=")
        got[k] = int(v)
    return got


def main():
    root = find_root()
    src_path = os.path.join(root, SRC_REL)
    if not os.path.isfile(src_path):
        check("the tree holds " + SRC_REL, False, src_path + " is missing")
        return
    src = open(src_path, encoding="utf-8").read()
    block, err = selftest_block(src)
    if block is None:
        check("the d11 selftest function is readable", False, err)
        return
    check("the d11 selftest function is readable", True)

    cxx = compiler()
    if cxx is None:
        check("a C++ compiler is available", False, "no clang++, c++ or g++ on PATH")
        return
    work = tempfile.mkdtemp(prefix="fix-0116-2-")
    try:
        cpp = os.path.join(work, "selftest.cpp")
        exe = os.path.join(work, "selftest")
        with open(cpp, "w", encoding="utf-8") as f:
            f.write(HARNESS.replace("@@BLOCK@@", block))
        build = subprocess.run([cxx, "-std=c++17", "-O0", "-o", exe, cpp],
                               capture_output=True, text=True, timeout=300)
        if build.returncode != 0:
            check("the faked selftest compiles", False, (build.stdout + build.stderr).strip()[:400])
            return
        check("the faked selftest compiles", True)

        good_rc, good_line, good_err = run(exe, "map")
        good = None
        if good_rc == 0 and good_line:
            try:
                good = parse(good_line)
            except ValueError:
                good = None

        check("the ordinary selftest maps once and unmaps once",
              good is not None and good["maps"] == 1 and good["unmaps"] == 1,
              "exit %d, %s%s" % (good_rc, good_line or "no report", (" " + good_err.strip()[:200]) if good_err.strip() else ""))

        check("the ordinary selftest times %d reader calls and writes every handed buffer" % RUN_CALLS,
              good is not None and good["run_calls"] == RUN_CALLS and good["bad_dst"] == 0
              and good["ok_writes"] == DST_COUNT and good["bytes"] == MAP_BYTES,
              "exit %d, %s%s" % (good_rc, good_line or "no report", (" " + good_err.strip()[:200]) if good_err.strip() else ""))

        bad_rc, bad_line, bad_err = run(exe, "fail")
        bad = None
        if bad_rc == 0 and bad_line:
            try:
                bad = parse(bad_line)
            except ValueError:
                bad = None
        where = "exit %d%s, %s" % (bad_rc, " (killed by a signal)" if bad_rc < 0 else "",
                                   bad_line or "no report")

        check("a failed selftest mapping does not kill the process",
              bad_rc == 0 and bad is not None, where)

        check("a failed selftest mapping is reported on stderr",
              bad is not None and bool(bad_err.strip()), where + ", stderr %r" % bad_err.strip()[:200])

        check("a failed selftest mapping is never locked and never unmapped",
              bad is not None and bad["mlock_bad"] == 0 and bad["munmap_bad"] == 0 and bad["unmaps"] == 0,
              where)

        check("a failed selftest mapping hands no buffer to the reader",
              bad is not None and bad["run_calls"] == 0 and bad["ok_writes"] == 0,
              where)
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
    sys.exit(1 if fails else 0)
