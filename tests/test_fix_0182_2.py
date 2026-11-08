#!/usr/bin/env python3
"""tests/test_fix_0182_2.py - a dump that failed must not claim the region was dumped.

src/core/verify.cpp:4150-4159 (STRATA_OLD_DUMP) sets the static old_dumped = true before it opens the dump
file and writes it. When std::fopen returns nullptr or std::fwrite comes back short, old_dumped is still true, so
old_dump_win (src/core/verify.cpp:3858) is false in every later window: the dump is never retried and nothing says so.
An operator who makes the destination writable, or frees the disk, gets no dump at all.

These cases take the real window decision and the real dump block out of src/core/verify.cpp, compile them unchanged
against the real include/strata/xch/xch_layout.h (no GPU, no model, no weights) and run the built program over three
windows with a planted destination:
  - a std::fopen that fails leaves the window undumped and the next window retries the dump;
  - a std::fwrite that comes back short (RLIMIT_FSIZE) leaves it undumped and the next window retries;
  - a verified write dumps once and reports once, the ordinary input, the same before and after the fix;
  - T below two is not a dump window at all.
Prints one PASS or FAIL line per case and exits nonzero on any FAIL. Run: python3 tests/test_fix_0182_2.py
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = os.getcwd()   # the tree under test: every test here runs with cwd at the copy root, never at the test file
SOURCE = os.path.join('src', 'core', 'verify.cpp')
OPEN_ANCHOR = re.compile(r'^    if \(old_dump_win && old_region\.size\(\) == \(size_t\) XCH_REGION_BYTES\) \{')

PROGRAM = r"""// built by tests/test_fix_0182_2.py from the real src/core/verify.cpp
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/resource.h>
#include <sys/stat.h>
#include <vector>
#include "strata/xch/xch_layout.h"

struct Model { int64_t n_embd = (int64_t) XCH_H; int64_t n_layers = (int64_t) XCH_N_LAYERS; } g;
struct Shape { int64_t k = (int64_t) XCH_K; } ss;
static int T = 2;

static void run_window(int w) {
@DECL@
    if (old_dump_win && old_region.size() == (size_t) XCH_REGION_BYTES) {   // verify.cpp:4036: this window's layers fill the region
        for (size_t i = 0; i < old_region.size(); i += (size_t) XCH_SLOT_BYTES) old_region[i] = 0x5a;
    }
@WRITE@
    std::printf("window %d dumped=%d\n", w, old_dumped ? 1 : 0);
}

static long long file_size(const char* p) {
    struct stat st;
    return ::stat(p, &st) == 0 ? (long long) st.st_size : -1;
}

static int first_byte(const char* p) {
    std::FILE* const f = std::fopen(p, "rb");
    if (f == nullptr) return -1;
    const int c = std::fgetc(f);
    std::fclose(f);
    return c == EOF ? -1 : c;
}

static struct rlimit saved_limit;

static void limit_set(const rlim_t cur) {
    struct rlimit r;
    ::getrlimit(RLIMIT_FSIZE, &r);
    saved_limit = r;
    r.rlim_cur = cur;
    if (::setrlimit(RLIMIT_FSIZE, &r) != 0) std::printf("limit failed\n");
}

int main(int argc, char** argv) {
    const char* const mode = argc > 1 ? argv[1] : "ok";
    const char* const path = argc > 2 ? argv[2] : "dump.bin";
    const char* const dir = argc > 3 ? argv[3] : "";
    std::signal(SIGXFSZ, SIG_IGN);
    ::setenv("STRATA_OLD_DUMP", path, 1);
    if (std::strcmp(mode, "t1") == 0) T = 1;
    if (std::strcmp(mode, "short") == 0) limit_set(0);
    run_window(1);
    if (std::strcmp(mode, "short") == 0) { struct rlimit r = saved_limit; ::setrlimit(RLIMIT_FSIZE, &r); }
    if (std::strcmp(mode, "fopen") == 0 && dir[0] != '\0') ::mkdir(dir, 0777);
    run_window(2);
    run_window(3);
    std::printf("region=%llu\n", (unsigned long long) XCH_REGION_BYTES);
    std::printf("file=%lld\n", file_size(path));
    std::printf("mark=%d\n", first_byte(path));
    return 0;
}
"""

CASES = [
    dict(name='a failed open leaves the window undumped and the next window retries', mode='fopen',
         windows={1: 0, 2: 1, 3: 1}, dumped=True, reports=1),
    dict(name='a short write leaves the window undumped and the next window retries', mode='short',
         windows={1: 0, 2: 1, 3: 1}, dumped=True, reports=1),
    dict(name='a verified write dumps once and reports once', mode='ok',
         windows={1: 1, 2: 1, 3: 1}, dumped=True, reports=1),
    dict(name='T below two is not a dump window', mode='t1',
         windows={1: 0, 2: 0, 3: 0}, dumped=False, reports=0),
]


def check(name, ok, detail=''):
    global fails
    print(f"{'PASS' if ok else 'FAIL'} {name}" + ('' if ok else f': {detail}'))
    if not ok:
        fails += 1


def code_only(line):
    """The line with string and character literals and any // comment blanked, for brace counting."""
    out = []
    i = 0
    while i < len(line):
        if line.startswith('//', i):
            break
        c = line[i]
        if c in '"\'':
            quote = c
            i += 1
            while i < len(line):
                if line[i] == '\\':
                    i += 2
                    continue
                if line[i] == quote:
                    i += 1
                    break
                i += 1
            out.append('""')
            continue
        out.append(c)
        i += 1
    return ''.join(out)


def brace_delta(line):
    code = code_only(line)
    return code.count('{') - code.count('}')


def extract(lines):
    """The real dump decision (old_dumped, old_region, old_dump_win) and the real dump block, as source lines."""
    decl_start = next((i for i, l in enumerate(lines) if l.strip() == 'static bool old_dumped = false;'), None)
    if decl_start is None:
        return None, None, 'src/core/verify.cpp has no static bool old_dumped = false; (dump state gone)'
    decl_end = next((i for i, l in enumerate(lines) if i >= decl_start and 'old_region.assign((size_t) XCH_REGION_BYTES, 0)' in l), None)
    if decl_end is None:
        return None, None, 'src/core/verify.cpp has no old_region.assign((size_t) XCH_REGION_BYTES, 0) after the dump state'
    write_start = next((i for i, l in enumerate(lines) if OPEN_ANCHOR.match(l)), None)
    if write_start is None:
        return None, None, 'src/core/verify.cpp has no four-space if (old_dump_win && old_region.size() == (size_t) XCH_REGION_BYTES) {'
    depth = 0
    write_end = None
    for i in range(write_start, len(lines)):
        depth += brace_delta(lines[i])
        if depth == 0:
            write_end = i
            break
    if write_end is None:
        return None, None, 'the dump block at line %d has no closing brace' % (write_start + 1)
    return lines[decl_start:decl_end + 1], lines[write_start:write_end + 1], ''


def compiler():
    for name in ('g++', 'clang++', 'c++'):
        path = shutil.which(name)
        if path:
            return path
    return None


def build(cxx, work, decl, write):
    src = os.path.join(work, 'dump_window.cpp')
    exe = os.path.join(work, 'dump_window')
    text = PROGRAM.replace('@DECL@', '\n'.join(decl)).replace('@WRITE@', '\n'.join(write))
    if 'STRATA_OLD_DUMP' not in text or 'old_dumped' not in text:
        return None, 'the extracted block lost STRATA_OLD_DUMP or old_dumped'
    with open(src, 'w', encoding='utf-8') as f:
        f.write(text)
    cmd = [cxx, '-std=c++17', '-O0', '-I', os.path.join(ROOT, 'include'), '-o', exe, src]
    built = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if built.returncode != 0:
        errors = [l.strip() for l in (built.stdout + built.stderr).splitlines() if ' error: ' in l]
        return None, 'does not build (' + (errors[0] if errors else 'see the compile log') + ')'
    return exe, ''


def run_case(exe, work, case):
    """Three windows against a planted destination; the stdout, the dump reports and the file are the result."""
    path = os.path.join(work, case['mode'] + '.bin')
    if case['mode'] == 'fopen':
        missing = os.path.join(work, 'late')
        path = os.path.join(missing, 'dump.bin')
    else:
        missing = ''
    env = dict(os.environ)
    env.pop('STRATA_OLD_DUMP', None)
    ran = subprocess.run([exe, case['mode'], path, missing], capture_output=True, text=True, timeout=300, env=env)
    if ran.returncode != 0:
        return 'the built program exited %d: %s' % (ran.returncode, (ran.stdout + ran.stderr).strip()[-200:])
    got = {}
    for line in ran.stdout.splitlines():
        m = re.fullmatch(r'window (\d+) dumped=(\d)', line)
        if m:
            got[int(m.group(1))] = int(m.group(2))
    if got != case['windows']:
        return 'the windows report %s, expected %s' % (got, case['windows'])
    reports = sum(1 for line in ran.stderr.splitlines() if line.startswith('strata old dump:'))
    if reports != case['reports']:
        return 'the run printed %d "strata old dump" report(s), expected %d' % (reports, case['reports'])
    want_region = next((int(l.split('=')[1]) for l in ran.stdout.splitlines() if l.startswith('region=')), 0)
    size = next((int(l.split('=')[1]) for l in ran.stdout.splitlines() if l.startswith('file=')), -1)
    mark = next((int(l.split('=')[1]) for l in ran.stdout.splitlines() if l.startswith('mark=')), -1)
    if case['dumped'] and size != want_region:
        return 'the dump file holds %d bytes, expected the region\'s %d' % (size, want_region)
    if case['dumped'] and mark != 0x5a:
        return 'the dump file starts with byte %d, expected the region\'s first byte %d' % (mark, 0x5a)
    if not case['dumped'] and size != -1:
        return 'a file was written (%d bytes) where no dump window exists' % size
    return ''


fails = 0


def main():
    cxx = compiler()
    if cxx is None:
        check('a C++ compiler is available for the built program', False, 'no g++, clang++ or c++ on PATH')
        return 1
    try:
        with open(SOURCE, encoding='utf-8') as f:
            lines = f.read().splitlines()
    except OSError as e:
        check('src/core/verify.cpp is readable', False, str(e))
        return 1
    decl, write, why = extract(lines)
    if why:
        check('the dump state and dump block are in src/core/verify.cpp', False, why)
        return 1
    work = tempfile.mkdtemp(prefix='fix-0182-2-')
    try:
        exe, why = build(cxx, work, decl, write)
        if exe is None:
            check('the real dump block builds against include/strata/xch/xch_layout.h', False, why)
            return 1
        for case in CASES:
            why = run_case(exe, work, case)
            check(case['name'], not why, why)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())
