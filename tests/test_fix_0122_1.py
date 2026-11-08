#!/usr/bin/env python3
"""tests/test_fix_0122_1.py - a failed locked RAM tier registration is not recorded as success (fix-0122#1).

src/core/expert_source.cpp:3789 (today) has the reg_tier lambda read

    auto reg_tier = [&] {
        if (complement_ready_ && !M.comp_reg) {
            M.comp_reg = true;
            const int64_t rr = M.reg(M.ctx, complement_host_, (size_t) complement_bytes_);
            std::fprintf(stderr, "strata exl3-metal: locked RAM tier registered: %lld bytes\\n", (long long) rr);
        }
    };

comp_reg is set before M.reg runs and rr is never inspected, so a registration that fails (rr <= 0) is
printed and then treated as registered: the next reg_tier call skips it and the later !M.comp_reg retry
paths never run. The tier is then used unregistered forever. The same file's other registration path
(src/core/expert_source.cpp:3844) checks M.reg(...) <= 0, so a non-positive answer is a real failure.

This test cuts the reg_tier lambda out of the tree as it is written and compiles it against a small fake
Metal lane and locked host tier (no GPU, no model, no mapping) that can answer success or failure:
  - a successful registration is recorded and a second call does not register again (ordinary input, the
    same before and after the fix);
  - a failed registration is not recorded as success, so comp_reg stays false;
  - a failed registration is retried by the next call, and a retry that succeeds is recorded;
  - an unready complement registers nothing and stays unregistered.
Run from the copy root: python3 tests/test_fix_0122_1.py
Prints one PASS or FAIL line per case and exits nonzero on any FAIL. Temp folders only.
"""
import os
import shutil
import subprocess
import sys
import tempfile

SRC_REL = os.path.join("src", "core", "expert_source.cpp")
ANCHOR = "auto reg_tier = [&] {"

HARNESS = r"""
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

static int g_calls = 0;
static int64_t g_rc = 0;
static size_t g_last_bytes = 0;

static int64_t fake_reg(void* ctx, const void* host, size_t bytes) {
    (void) ctx; (void) host;
    ++g_calls;
    g_last_bytes = bytes;
    return g_rc;
}

struct FakeMetal {
    void* ctx = (void*) 0x1;
    bool comp_reg = false;
    int64_t (*reg)(void*, const void*, size_t) = fake_reg;
};

struct Subject {
    bool complement_ready_ = true;
    void* complement_host_ = (void*) 0x2000;
    int64_t complement_bytes_ = 4096;
    FakeMetal lane;

    void call() {
        FakeMetal& M = lane;
@@BLOCK@@
        reg_tier();
    }
};

int main(int argc, char** argv) {
    const char* mode = argc > 1 ? argv[1] : "ok";
    Subject s;
    g_calls = 0;
    if (std::strcmp(mode, "ok") == 0) {
        g_rc = 16;
        s.call();
        s.call();
    } else if (std::strcmp(mode, "fail") == 0) {
        g_rc = 0;
        s.call();
    } else if (std::strcmp(mode, "retry") == 0) {
        g_rc = -1;
        s.call();
        g_rc = 32;
        s.call();
    } else if (std::strcmp(mode, "notready") == 0) {
        s.complement_ready_ = false;
        g_rc = 16;
        s.call();
    } else {
        std::fprintf(stderr, "unknown mode %s\n", mode);
        return 2;
    }
    std::printf("mode=%s calls=%d comp_reg=%d bytes=%lld\n", mode, g_calls,
                s.lane.comp_reg ? 1 : 0, (long long) g_last_bytes);
    return 0;
}
"""

fails = 0


def check(name, ok, detail=""):
    global fails
    print(("PASS " if ok else "FAIL ") + name + (": " + detail if not ok and detail else ""))
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


def reg_tier_block(src):
    """The reg_tier lambda exactly as the tree writes it."""
    a = src.find(ANCHOR)
    if a < 0:
        return None, "the reg_tier lambda is gone from " + SRC_REL
    end = brace_end(src, src.index("{", a))
    if end is None:
        return None, "the reg_tier lambda has no closing brace"
    semi = src.find(";", end)
    if semi >= 0 and src[end:semi].strip() == "":
        end = semi + 1
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
        out[k] = v
    return out


def main():
    root = find_root()
    src_path = os.path.join(root, SRC_REL)
    if not os.path.isfile(src_path):
        check("the tree holds " + SRC_REL, False, src_path + " is missing")
        return 1
    src = open(src_path, encoding="utf-8").read()
    block, err = reg_tier_block(src)
    if block is None:
        check("the reg_tier lambda is readable", False, err)
        return 1
    check("the reg_tier lambda is readable", True)

    cxx = compiler()
    if cxx is None:
        check("a C++ compiler is available", False, "no clang++, c++ or g++ on PATH")
        return 1
    work = tempfile.mkdtemp(prefix="fix-0122-1-")
    try:
        cpp = os.path.join(work, "reg_tier.cpp")
        exe = os.path.join(work, "reg_tier")
        with open(cpp, "w", encoding="utf-8") as f:
            f.write(HARNESS.replace("@@BLOCK@@", block))
        build = subprocess.run([cxx, "-std=c++17", "-O0", "-o", exe, cpp],
                               capture_output=True, text=True, timeout=120)
        if build.returncode != 0:
            check("the faked reg_tier lambda compiles", False,
                  (build.stdout + build.stderr).strip()[:400])
            return 1
        check("the faked reg_tier lambda compiles", True)

        cases = [
            ("a successful registration is recorded and not repeated", "ok",
             lambda o: o["calls"] == "1" and o["comp_reg"] == "1"),
            ("a failed registration is not recorded as success", "fail",
             lambda o: o["calls"] == "1" and o["comp_reg"] == "0"),
            ("a failed registration is retried and the retry sticks", "retry",
             lambda o: o["calls"] == "2" and o["comp_reg"] == "1"),
            ("an unready complement registers nothing", "notready",
             lambda o: o["calls"] == "0" and o["comp_reg"] == "0"),
        ]
        for name, mode, want in cases:
            line, err = run(exe, mode)
            if line is None:
                check(name, False, "the fake reg_tier did not run: " + (err or ""))
                continue
            try:
                obs = parse(line)
            except ValueError:
                check(name, False, "cannot read the observation: " + line)
                continue
            check(name, want(obs), "%s (calls=%s comp_reg=%s)" %
                  (line, obs.get("calls"), obs.get("comp_reg")))
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
