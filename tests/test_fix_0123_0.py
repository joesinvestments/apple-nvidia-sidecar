#!/usr/bin/env python3
"""src/core/expert_source.cpp hands out at most 512 blob pointers but never says so.

FileExpertSource::prefetch (STRATA_PF_MAC, mode 2) ends with a loop that fills PfMac::ptrs, the
const void* array the Metal prefill kernels read.  The loop is bounded by the array (512) and by n,
so a call that names more than 512 experts fills the first 512 entries, leaves the rest of the call
unresolved and returns with PfMac::bad false: the caller reads as if every expert had been placed.
The entry points (strata_pf_mac_compute, strata_pf_mac_submit, strata_pf_mac_submit_h) cap nd at 512
so today no caller reaches it, but prefetch is the shared ExpertSource path and nothing in the loop
itself refuses the wider call.  The right answer for n > 512 is a bad call (P.bad), not a silent
partial hand-out: the pointer array is fixed at 512, so resolving more is out of bounds.

This test compiles the real hand-out block from the tree under test against a small fake PfMac,
Metal lane and blob arena (no GPU, no model, no file mapping) and runs it:
  - four experts resolve to their own blobs and the call is not bad (ordinary input);
  - a full 512-expert call resolves all 512 and is not bad (the boundary still works);
  - one unresolvable expert marks the call bad (the null check still works);
  - a 600-expert call must be bad, or resolve all 600: today it reports good after 512.
Run from the tree root: python3 tests/test_fix_0123_0.py
Prints one PASS or FAIL line per case and exits non-zero on any FAIL.  Temp folders only.
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile

SRC_REL = os.path.join("src", "core", "expert_source.cpp")
ANCHOR = "if (P.rd.short_reads.load() != 0) { P.bad = true; return; }"
LOOP = re.compile(r"for \(int64_t i = 0; i < n[^\n]*\{\n")

HARNESS = r"""
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

struct FakeReader { std::atomic<int64_t> short_reads{0}; };
struct FakeP {
    FakeReader rd;
    const void* ptrs[512];
    bool bad = false;
    std::vector<int16_t> slot_of[4];
    std::vector<uint8_t*> sptr;
    int per_set = 0;
    int64_t tier_hits = 0, lru_hits = 0;
    FakeP() {
        for (int i = 0; i < 512; ++i) ptrs[i] = nullptr;
        for (int k = 0; k < 4; ++k) slot_of[k].assign(4096, (int16_t) -1);
    }
};
struct FakeM { uint8_t* slots = nullptr; size_t stride = 0; };

static FakeP P;
static FakeM M;
static int s = 0;
static int64_t layer = 0, n_expert_ = 1024, n = 0;
static const int64_t* experts = nullptr;
static std::vector<int64_t> g_ids;
static std::vector<uint8_t> g_arena;
static std::vector<size_t> g_missing;

bool in_tier(size_t index) {
    for (size_t x : g_missing) if (x == index) return false;
    return true;
}
const uint8_t* resident_blob(size_t index) { return g_arena.data() + index * 16; }
int in_lru(size_t) { return -1; }
uint32_t d11_skew_lam(int64_t, int64_t) { return 0; }
bool d12_pf_on() { return false; }

static void hand_out() {
@@BLOCK@@
}

int main(int argc, char** argv) {
    const char* what = argc > 1 ? argv[1] : "tier";
    int cnt = argc > 2 ? std::atoi(argv[2]) : 4;
    const bool miss = std::strcmp(what, "miss") == 0;
    g_arena.assign((size_t) n_expert_ * 16 + 16, 0);
    if (miss) cnt = 3;
    for (int i = 0; i < cnt; ++i) g_ids.push_back(miss && i == 1 ? 5 : i);
    if (miss) g_missing.push_back((size_t) layer * (size_t) n_expert_ + 5);
    experts = g_ids.data();
    n = (int64_t) g_ids.size();
    hand_out();
    int resolved = 0, mismatch = 0;
    for (int64_t i = 0; i < n && i < 512; ++i) {
        const size_t index = (size_t) layer * (size_t) n_expert_ + (size_t) g_ids[(size_t) i];
        const void* want = in_tier(index) ? (const void*) resident_blob(index) : (const void*) nullptr;
        if (P.ptrs[i] != nullptr) ++resolved;
        if (P.ptrs[i] != want) ++mismatch;
    }
    std::printf("bad=%d resolved=%d mismatch=%d n=%lld\n", P.bad ? 1 : 0, resolved, mismatch, (long long) n);
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


def hand_out_block(src):
    """The short-read guard and the pointer hand-out loop, exactly as the tree writes them."""
    a = src.find(ANCHOR)
    if a < 0:
        return None, "the pf-mac short-read guard is gone from " + SRC_REL
    m = LOOP.search(src, a)
    if m is None:
        return None, "the pf-mac pointer hand-out loop is gone from " + SRC_REL
    end = brace_end(src, src.index("{", m.start()))
    if end is None:
        return None, "the pf-mac pointer hand-out loop has no closing brace"
    return src[a:end], None


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
        out[k] = int(v)
    return out


def main():
    root = find_root()
    src_path = os.path.join(root, SRC_REL)
    if not os.path.isfile(src_path):
        check("the tree holds " + SRC_REL, False, src_path + " is missing")
        return
    src = open(src_path, encoding="utf-8").read()
    block, err = hand_out_block(src)
    if block is None:
        check("the pointer hand-out block is readable", False, err)
        return
    check("the pointer hand-out block is readable", True)

    cxx = compiler()
    if cxx is None:
        check("a C++ compiler is available", False, "no clang++, c++ or g++ on PATH")
        return
    work = tempfile.mkdtemp(prefix="fix-0123-0-")
    try:
        cpp = os.path.join(work, "handout.cpp")
        exe = os.path.join(work, "handout")
        with open(cpp, "w", encoding="utf-8") as f:
            f.write(HARNESS.replace("@@BLOCK@@", block))
        build = subprocess.run([cxx, "-std=c++17", "-O0", "-o", exe, cpp],
                               capture_output=True, text=True, timeout=120)
        if build.returncode != 0:
            check("the faked hand-out block compiles", False,
                  (build.stdout + build.stderr).strip()[:400])
            return
        check("the faked hand-out block compiles", True)

        cases = []

        line, err = run(exe, "tier", "4")
        cases.append(("four experts resolve to their own blobs", line, err,
                      lambda o: o["bad"] == 0 and o["resolved"] == 4 and o["mismatch"] == 0))

        line, err = run(exe, "tier", "512")
        cases.append(("a full 512-expert hand-out resolves all 512", line, err,
                      lambda o: o["bad"] == 0 and o["resolved"] == 512 and o["mismatch"] == 0))

        line, err = run(exe, "miss")
        cases.append(("an unresolvable expert marks the call bad", line, err,
                      lambda o: o["bad"] == 1 and o["resolved"] == 2 and o["mismatch"] == 0))

        line, err = run(exe, "tier", "600")
        cases.append(("a 600-expert call is bad, not a silent 512 hand-out", line, err,
                      lambda o: o["bad"] == 1 or o["resolved"] == o["n"]))

        for name, line, err, want in cases:
            if line is None:
                check(name, False, "the fake hand-out did not run: " + (err or ""))
                continue
            try:
                obs = parse(line)
            except ValueError:
                check(name, False, "cannot read the observation: " + line)
                continue
            ok = want(obs)
            detail = "%s (bad=%d resolved=%d of %d)" % (line, obs["bad"], obs["resolved"], obs["n"])
            check(name, ok, detail)
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
    sys.exit(1 if fails else 0)
