#!/usr/bin/env python3
"""tests/test_fix_0115_1.py - the expert-read pool's spin hint must build on x86_64 macOS too.

src/core/expert_source.cpp:2261 (today):
    static inline void spin_pause() { __asm__ volatile("yield" ::: "memory"); }
`yield` is an AArch64 instruction.  The function sits in MacPull (2244), the expert-read pool the Metal lane pulls its
experts through, and both spin waits call it: start()'s worker loop (2302) and finish()'s wait for the last piece
(2351).  On an x86_64 macOS build clang's assembler rejects the mnemonic, so the translation unit does not build
(CMakeLists.txt:708 puts it in strata_engine): no engine and no served replies on an Intel Mac.

These cases take the spin_pause definition as the tree writes it (braces and all), put it in a MacPull-shaped struct
and build it for the host and for x86_64 macOS, then run the pool's two spin waits over fake counters: the wait must
still stop when the next generation arrives, when the last piece lands, and at its deadline.  The x86_64 build cases
fail on today's code on the mnemonic and pass once spin_pause uses a per-arch hint (the tree already has one:
include/strata/platform/cpu_relax.hpp, `yield` on aarch64, `_mm_pause` on x86).  No model, no GPU, no weights.
Prints one PASS or FAIL line per case and exits nonzero on any FAIL.  Run: python3 tests/test_fix_0115_1.py
"""
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = os.getcwd()   # the tree under test: every test here runs with cwd at the copy root, never at the test file
SOURCE = os.path.join('src', 'core', 'expert_source.cpp')
INCLUDE = os.path.join(ROOT, 'include')
X86 = ['-arch', 'x86_64'] if platform.system() == 'Darwin' else []

PROGRAM = r"""// built by tests/test_fix_0115_1.py: the expert-read pool's spin hint over fake counters.
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <thread>

#include "strata/platform/cpu_relax.hpp"

struct MacPull {
%BODY%
};

// start() (expert_source.cpp:2295-2306): a worker polls for the next generation for a while after its last job.
static void await_generation(std::atomic<uint64_t>& gen_a, uint64_t seen, int us) {
    const auto end = std::chrono::steady_clock::now() + std::chrono::microseconds(us);
    for (unsigned c = 0; gen_a.load(std::memory_order_acquire) == seen; ++c) {
        MacPull::spin_pause();
        if ((c & 63) == 63 && std::chrono::steady_clock::now() >= end) break;
    }
}

// finish() (expert_source.cpp:2348-2356): the last pieces are in flight on spinning workers, so poll for them.
static bool await_pieces(const std::atomic<size_t>& completed, size_t n, int us) {
    const auto end = std::chrono::steady_clock::now() + std::chrono::microseconds(us);
    for (unsigned c = 0; completed.load(std::memory_order_acquire) < n; ++c) {
        MacPull::spin_pause();
        if ((c & 63) == 63 && std::chrono::steady_clock::now() >= end) break;
    }
    return completed.load(std::memory_order_acquire) >= n;
}

static long long us_since(const std::chrono::steady_clock::time_point& t0) {
    return (long long) std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count();
}

int main(int argc, char** argv) {
    const char* mode = argc > 1 ? argv[1] : "";
    if (std::strcmp(mode, "generation") == 0) {
        std::atomic<uint64_t> gen_a{7};
        std::thread t([&] { std::this_thread::sleep_for(std::chrono::microseconds(1000)); gen_a.store(8, std::memory_order_release); });
        const auto t0 = std::chrono::steady_clock::now();
        await_generation(gen_a, 7, 200000);
        t.join();
        if (gen_a.load() != 8) { std::printf("GEN_NOT_SEEN\n"); return 3; }
        std::printf("GEN_SEEN us=%lld\n", us_since(t0));
        return 0;
    }
    if (std::strcmp(mode, "landed") == 0) {
        std::atomic<size_t> completed{0};
        const size_t n = 4;
        std::thread t([&] { std::this_thread::sleep_for(std::chrono::microseconds(1000)); completed.store(n, std::memory_order_release); });
        const auto t0 = std::chrono::steady_clock::now();
        const bool ok = await_pieces(completed, n, 200000);
        t.join();
        if (!ok) { std::printf("MISSED\n"); return 3; }
        std::printf("LANDED us=%lld\n", us_since(t0));
        return 0;
    }
    if (std::strcmp(mode, "deadline") == 0) {
        std::atomic<size_t> completed{0};
        const auto t0 = std::chrono::steady_clock::now();
        const bool ok = await_pieces(completed, 4, 2000);
        const long long us = us_since(t0);
        if (ok) { std::printf("SAW_PIECES_THAT_NEVER_LANDED\n"); return 3; }
        if (us > 500000) { std::printf("NO_DEADLINE us=%lld\n", us); return 3; }
        std::printf("GAVE_UP us=%lld\n", us);
        return 0;
    }
    std::printf("BAD_MODE\n");
    return 4;
}
"""


def code_only(line):
    """The line with comments and string/char literal contents removed, for brace counting."""
    out, i, q = [], 0, None
    while i < len(line):
        ch = line[i]
        if q:
            if ch == '\\':
                i += 2
                continue
            if ch == q:
                q = None
            i += 1
            continue
        if ch in '"\'':
            q = ch
            i += 1
            continue
        if ch == '/' and line[i + 1:i + 2] == '/':
            break
        out.append(ch)
        i += 1
    return ''.join(out)


def definition(text):
    """The spin_pause definition exactly as the tree writes it, however many lines and directives it spans."""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if not re.search(r'\bvoid\s+spin_pause\s*\(', line):
            continue
        depth, opened, out = 0, False, []
        for j in range(i, min(len(lines), i + 40)):
            out.append(lines[j])
            for ch in code_only(lines[j]):
                if ch == '{':
                    depth += 1
                    opened = True
                elif ch == '}':
                    depth -= 1
            if opened and depth == 0:
                return '\n'.join(out)
        return None
    return None


def first_error(r):
    for line in (r.stderr + r.stdout).splitlines():
        if 'error' in line.lower():
            return line.strip()[:300]
    return (r.stderr + r.stdout).strip().splitlines()[-1][:300] if (r.stderr + r.stdout).strip() else 'no output'


def main():
    results = []
    def case(name, ok, detail=''):
        results.append(bool(ok))
        print(('PASS ' if ok else 'FAIL ') + name + ('' if ok or not detail else ': ' + detail))

    cxx = next((c for c in (os.environ.get('CXX'), 'clang++', 'c++', 'g++') if c and shutil.which(c)), None)
    if not cxx:
        case('a C++ compiler is available', False, 'no CXX, clang++, c++ or g++ on PATH')
        print('FAIL 1 of 1 case')
        return 1

    path = os.path.join(ROOT, SOURCE)
    try:
        with open(path, encoding='utf-8') as f:
            text = f.read()
    except OSError as e:
        case('the expert source the pool is built from is in the tree', False, str(e)[:200])
        print('FAIL 1 of 1 case')
        return 1

    body = definition(text)
    case('the MacPull spin_pause definition is in ' + SOURCE, body is not None,
         'no void spin_pause(...) { ... } definition found')
    if body is None:
        print('FAIL 1 of 1 case')
        return 1

    work = tempfile.mkdtemp(prefix='sce0115-')
    try:
        src = os.path.join(work, 'pool_spin.cpp')
        with open(src, 'w', encoding='utf-8') as f:
            f.write(PROGRAM.replace('%BODY%', body))
        base = [cxx, '-std=c++17', '-O2', '-I', INCLUDE]

        # The ordinary input: the host's own build, and the pool's two waits over fake counters.
        host_exe = os.path.join(work, 'pool_spin_host')
        r = subprocess.run(base + [src, '-o', host_exe], capture_output=True, text=True, timeout=180)
        built_host = r.returncode == 0 and os.path.exists(host_exe)
        case('the host build of the pool spin hint builds', built_host, first_error(r) if not built_host else '')

        def run_mode(mode):
            if not built_host:
                return None
            return subprocess.run([host_exe, mode], capture_output=True, text=True, timeout=120)

        r = run_mode('generation')
        case('the pool start wait sees the next generation and stops',
             r is not None and r.returncode == 0, 'build failed' if r is None else first_error(r))
        case_output = '' if r is None or r.returncode else r.stdout.strip()
        if case_output:
            print('#' + case_output)
        r = run_mode('landed')
        case('the pool finish wait sees the last piece land and stops',
             r is not None and r.returncode == 0, 'build failed' if r is None else first_error(r))
        if r is not None and r.returncode == 0:
            print('#' + r.stdout.strip())
        r = run_mode('deadline')
        case('the pool finish wait gives up at its deadline',
             r is not None and r.returncode == 0, 'build failed' if r is None else first_error(r))
        if r is not None and r.returncode == 0:
            print('#' + r.stdout.strip())

        # The planted input: the same definition compiled for x86_64 macOS, the build the finding is about.
        obj = os.path.join(work, 'pool_spin_x86.o')
        r = subprocess.run(base + X86 + ['-c', src, '-o', obj], capture_output=True, text=True, timeout=180)
        ok_obj = r.returncode == 0 and os.path.exists(obj)
        case('an x86_64 macOS build of the pool spin hint assembles', ok_obj, first_error(r) if not ok_obj else '')

        x86_exe = os.path.join(work, 'pool_spin_x86')
        r = subprocess.run(base + X86 + [src, '-o', x86_exe], capture_output=True, text=True, timeout=180)
        ok_link = r.returncode == 0 and os.path.exists(x86_exe)
        case('an x86_64 macOS build of both pool spin waits links', ok_link, first_error(r) if not ok_link else '')
    finally:
        shutil.rmtree(work, ignore_errors=True)

    bad = results.count(False)
    if bad:
        print('FAIL %d of %d cases' % (bad, len(results)))
        return 1
    print('PASS all %d cases' % len(results))
    return 0


if __name__ == '__main__':
    sys.exit(main())
