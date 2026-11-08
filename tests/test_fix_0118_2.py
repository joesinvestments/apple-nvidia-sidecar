#!/usr/bin/env python3
"""tests/test_fix_0118_2.py - a failed exl3m_unregister must not free the Metal slot sets.

src/core/expert_source.cpp:2922 (strata_pf_mac_release, the STRATA_PF_MAC prompt lane's teardown when the sets
are not lent to the decode LRU and the exchange is off) read

    P.unreg(ml().ctx, P.slots);
    const size_t bytes = (size_t) pf_nsets() * (size_t) P.per_set * P.stride;
    munlock(P.slots, bytes);
    munmap(P.slots, bytes);

exl3m_unregister returns int and the return was dropped, so a registration that the Metal context refused to
drop is followed by munlock and munmap all the same: the lane is left registered on memory that is no longer
mapped, and P.slots is cleared, so no later release can retry the unregister. The right answer is to keep the
mapping and P.slots set when the unregister fails, report it, and free only once the unregister has succeeded.

These cases take the real strata_pf_mac_release out of the tree under test, compile it against a fake PfMac,
Metal lane and munlock/munmap pair (no GPU, no model, no file mapping) and run it. The fake unregister keeps
the registration when it is asked to fail and clears it when it succeeds; the fake munmap records that the
mapping is gone:
  - a release whose unregister succeeds frees the mapping and clears the registration (ordinary input);
  - a failed unregister keeps the mapping, the registration and P.slots (today it frees them: each row of the
    report below is the harm);
  - a failed unregister is reported;
  - the release after a failed one retries the unregister and frees once it succeeds;
  - lent slots go back to the decode LRU and the mapping is kept;
  - under the exchange (static residency) the mapping is kept for the next prompt;
  - a release with no slots does nothing.
Run from the tree root: python3 tests/test_fix_0118_2.py
Prints one PASS or FAIL line per case and exits non-zero on any FAIL. Temp folders only.
"""
import os
import shutil
import subprocess
import sys
import tempfile

SRC_REL = os.path.join("src", "core", "expert_source.cpp")
# the Apple arm of the file writes the body on its own lines; the #else stub is "extern \"C\" void strata_pf_mac_release() {}"
ANCHOR = 'extern "C" void strata_pf_mac_release() {\n'

HARNESS = r"""
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static uint8_t g_arena[4096];

struct FakeReader { void finish(int) {} };
struct FakePf {
    uint8_t* slots = nullptr;
    size_t stride = 64;
    int per_set = 4;
    int tag[4] = {-1, -1, -1, -1};
    std::vector<uint8_t*> sptr;
    std::vector<int> borrowed;
    bool lent = false;
    int (*unreg)(void*, const void*) = nullptr;
    FakeReader rd;
};
struct FakeM {
    void* ctx = (void*) 0x1000;
    bool xch_on = false;
    std::vector<int64_t> key;
    std::vector<uint32_t> stamp;
    std::vector<int> lru;
    void lru_push_back(int s) { lru.push_back(s); }
};
// the names the real release names its two globals by
using PfMac = FakePf;
using MetalLane = FakeM;
static FakePf g_p;
static FakeM g_m;
FakePf& pfm() { return g_p; }
FakeM& ml() { return g_m; }
int pf_nsets() { return 2; }

static int g_unreg_calls = 0, g_unreg_fail = 0;
static int g_munlock = 0, g_munmap = 0, g_freed = 0, g_registered = 0;

// the Metal context's side: a failed unregister leaves the registration pointing at the base
static int fake_unreg(void*, const void* base) {
    ++g_unreg_calls;
    if (g_unreg_fail > 0) { --g_unreg_fail; return -7; }
    if (base == (const void*) g_arena) g_registered = 0;
    return 0;
}
extern "C" int munlock(const void*, size_t) { ++g_munlock; return 0; }
extern "C" int munmap(void* p, size_t) { ++g_munmap; if (p == (void*) g_arena) g_freed = 1; return 0; }

// taken from src/core/expert_source.cpp by tests/test_fix_0118_2.py
@@BODY@@

int main(int argc, char** argv) {
    const char* what = argc > 1 ? argv[1] : "ok";
    const int calls = argc > 2 ? std::atoi(argv[2]) : 1;
    g_p.unreg = fake_unreg;
    g_p.slots = (uint8_t*) g_arena;
    g_p.sptr.assign(3, (uint8_t*) g_arena);
    for (int k = 0; k < 4; ++k) g_p.tag[k] = 7;
    g_m.key.assign(8, -1);
    g_m.stamp.assign(8, 0);
    g_registered = 1;
    if (std::strcmp(what, "none") == 0) { g_p.slots = nullptr; g_registered = 0; }
    if (std::strcmp(what, "fail") == 0) g_unreg_fail = 1;
    if (std::strcmp(what, "retry") == 0) g_unreg_fail = 1;
    if (std::strcmp(what, "lent") == 0) { g_p.lent = true; g_p.borrowed = {3, 5}; }
    if (std::strcmp(what, "xch") == 0) g_m.xch_on = true;
    for (int i = 0; i < calls; ++i) strata_pf_mac_release();
    std::printf("unreg=%d munlock=%d munmap=%d freed=%d registered=%d slots=%d sptr=%d borrowed=%d lru=%d tag0=%d\n",
                g_unreg_calls, g_munlock, g_munmap, g_freed, g_registered, g_p.slots != nullptr ? 1 : 0,
                (int) g_p.sptr.size(), (int) g_p.borrowed.size(), (int) g_m.lru.size(), g_p.tag[0]);
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


def release_body(src):
    """The real strata_pf_mac_release, exactly as the tree writes it."""
    a = src.find(ANCHOR)
    if a < 0:
        return None, "strata_pf_mac_release is gone from " + SRC_REL
    end = brace_end(src, src.index("{", a))
    if end is None:
        return None, "strata_pf_mac_release has no closing brace"
    return src[a:end], None


def compiler():
    for name in (os.environ.get("CXX"), "clang++", "c++", "g++"):
        if name and shutil.which(name):
            return name
    return None


def run(program, *args):
    r = subprocess.run([program, *args], capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        return None, None, (r.stdout + r.stderr).strip()
    return r.stdout.strip(), r.stderr, None


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
    body, err = release_body(src)
    if body is None:
        check("the pf-mac release is readable", False, err)
        return
    check("the pf-mac release is readable", True)

    cxx = compiler()
    if cxx is None:
        check("a C++ compiler is available", False, "no clang++, c++ or g++ on PATH")
        return
    work = tempfile.mkdtemp(prefix="fix-0118-2-")
    try:
        cpp = os.path.join(work, "release.cpp")
        exe = os.path.join(work, "release")
        with open(cpp, "w", encoding="utf-8") as f:
            f.write(HARNESS.replace("@@BODY@@", body))
        build = subprocess.run([cxx, "-std=c++17", "-O0", "-o", exe, cpp],
                               capture_output=True, text=True, timeout=120)
        if build.returncode != 0:
            check("the faked release compiles", False, (build.stdout + build.stderr).strip()[:400])
            return
        check("the faked release compiles", True)

        cases = []

        line, _, err = run(exe, "ok", "1")
        cases.append(("a release that unregisters frees the mapping and clears the registration", line, err,
                      lambda o: o["unreg"] == 1 and o["munlock"] == 1 and o["munmap"] == 1 and o["freed"] == 1
                      and o["registered"] == 0 and o["slots"] == 0 and o["sptr"] == 0 and o["tag0"] == -1))

        line, _, err = run(exe, "fail", "1")
        cases.append(("a failed unregister keeps the slot mapping, the registration and the slots", line, err,
                      lambda o: o["unreg"] == 1 and o["munlock"] == 0 and o["munmap"] == 0 and o["freed"] == 0
                      and o["registered"] == 1 and o["slots"] == 1))

        line, stderr, err = run(exe, "fail", "1")
        cases.append(("a failed unregister is reported", line, err,
                      lambda o: stderr is not None and "unregister" in stderr.lower(),
                      "stderr was empty" if not (stderr or "").strip() else "stderr: " + (stderr or "").strip()[:200]))

        line, _, err = run(exe, "retry", "2")
        cases.append(("the release after a failed one retries the unregister and frees once it succeeds", line, err,
                      lambda o: o["unreg"] == 2 and o["munmap"] == 1 and o["freed"] == 1 and o["registered"] == 0
                      and o["slots"] == 0 and o["sptr"] == 0))

        line, _, err = run(exe, "lent", "1")
        cases.append(("lent slots go back to the decode LRU and the mapping is kept", line, err,
                      lambda o: o["unreg"] == 0 and o["freed"] == 0 and o["slots"] == 0 and o["sptr"] == 0
                      and o["borrowed"] == 0 and o["lru"] == 2))

        line, _, err = run(exe, "xch", "1")
        cases.append(("under the exchange the mapping and the slots are kept for the next prompt", line, err,
                      lambda o: o["unreg"] == 0 and o["freed"] == 0 and o["slots"] == 1 and o["sptr"] == 3
                      and o["registered"] == 1 and o["tag0"] == -1))

        line, _, err = run(exe, "none", "1")
        cases.append(("a release with no slots does nothing", line, err,
                      lambda o: o["unreg"] == 0 and o["munlock"] == 0 and o["munmap"] == 0 and o["freed"] == 0
                      and o["slots"] == 0))

        for name, line, err, want, *detail in cases:
            if line is None and err:
                check(name, False, "the fake release did not run: " + err)
                continue
            try:
                obs = parse(line or "")
            except ValueError:
                check(name, False, "cannot read the observation: " + (line or ""))
                continue
            ok = want(obs)
            extra = detail[0] if detail and not ok else ""
            check(name, ok, (line or "") + ("" if not extra else " (" + extra + ")"))
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
    sys.exit(1 if fails else 0)
