#!/usr/bin/env python3
"""tests/test_tgcu_returns_driver_int.py - a tgcu hand-off must hand the driver's status back.

include/strata/platform/tgcu.hpp:25-27 cast tgcuMappedPush, tgcuMappedPull and tgcuMappedManage to void, so a failed
hand-off is silent: the caller cannot see it and the engine carries on as if host and GPU mirrored each other. These
cases compile the real header against a stub driver (no GPU, no model, no weights) and run the built program:
  - a failing driver comes back as that nonzero code from push, pull and manage;
  - a succeeding driver comes back as 0;
  - the null pointer and empty length guards still skip the driver and return 0;
  - the non-tgcu build still compiles, links with no driver at all, and its no-ops return 0;
  - no caller in the tree drops the status of a raw push, pull or manage call.
Prints one PASS or FAIL line per case and exits nonzero on any FAIL. Run: python3 tests/test_tgcu_returns_driver_int.py
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = os.getcwd()   # the tree under test: every test here runs with cwd at the copy root, never at the test file
HEADER = os.path.join('include', 'strata', 'platform', 'tgcu.hpp')
SKIP_DIRS = {'.git', 'logs', 'notes', 'third_party', 'node_modules', '.venv', 'build', '__pycache__', '.cache'}
SOURCE_EXT = ('.cpp', '.cc', '.cxx', '.hpp', '.hh', '.h')

STUB = """
static int push_calls = 0, pull_calls = 0, manage_calls = 0;
static int push_rc = 0, pull_rc = 0, manage_rc = 0;
extern "C" int tgcuMappedPush(const void* p, size_t n) { (void) p; (void) n; ++push_calls; return push_rc; }
extern "C" int tgcuMappedPull(void* p, size_t n) { (void) p; (void) n; ++pull_calls; return pull_rc; }
extern "C" int tgcuMappedManage(const void* p) { (void) p; ++manage_calls; return manage_rc; }
"""

PROGRAM = """// built by tests/test_tgcu_returns_driver_int.py
#include <cstddef>
#include <cstdio>
#include <cstring>
#include "strata/platform/tgcu.hpp"
%s
int main() {
    char buf[16];
    std::memset(buf, 0, sizeof buf);
%s
    return 0;
}
"""

CASES = [
    dict(name='push returns the driver\'s failing status', tgcu=True, stubs=True, want='rc=-7 calls=1', body='''
    push_rc = -7;
    int rc = strata::tgcu::push(buf, sizeof buf);
    std::printf("rc=%d calls=%d\\n", rc, push_calls);
'''),
    dict(name='pull returns the driver\'s failing status', tgcu=True, stubs=True, want='rc=-9 calls=1', body='''
    pull_rc = -9;
    int rc = strata::tgcu::pull(buf, sizeof buf);
    std::printf("rc=%d calls=%d\\n", rc, pull_calls);
'''),
    dict(name='manage returns the driver\'s failing status', tgcu=True, stubs=True, want='rc=-11 calls=1', body='''
    manage_rc = -11;
    int rc = strata::tgcu::manage(buf);
    std::printf("rc=%d calls=%d\\n", rc, manage_calls);
'''),
    dict(name='a succeeding hand-off returns 0', tgcu=True, stubs=True, want='rc=0,0,0 calls=1,1,1', body='''
    push_rc = 0; pull_rc = 0; manage_rc = 0;
    int a = strata::tgcu::push(buf, sizeof buf);
    int b = strata::tgcu::pull(buf, sizeof buf);
    int c = strata::tgcu::manage(buf);
    std::printf("rc=%d,%d,%d calls=%d,%d,%d\\n", a, b, c, push_calls, pull_calls, manage_calls);
'''),
    dict(name='the null pointer and empty length guards still skip the driver', tgcu=True, stubs=True,
         want='calls=0,0,0', body='''
    strata::tgcu::push(nullptr, 8);
    strata::tgcu::push(buf, 0);
    strata::tgcu::pull(nullptr, 4);
    strata::tgcu::pull(buf, 0);
    strata::tgcu::manage(nullptr);
    std::printf("calls=%d,%d,%d\\n", push_calls, pull_calls, manage_calls);
'''),
    dict(name='the non-tgcu build still compiles and needs no driver', tgcu=False, stubs=False, want='built', body='''
    strata::tgcu::push(buf, sizeof buf);
    strata::tgcu::pull(buf, sizeof buf);
    strata::tgcu::manage(buf);
    std::printf("built\\n");
'''),
    dict(name='the non-tgcu no-ops return 0', tgcu=False, stubs=False, want='rc=0,0,0', body='''
    int a = strata::tgcu::push(buf, sizeof buf);
    int b = strata::tgcu::pull(buf, sizeof buf);
    int c = strata::tgcu::manage(buf);
    std::printf("rc=%d,%d,%d\\n", a, b, c);
'''),
]


def compiler():
    for name in ('g++', 'clang++', 'c++'):
        path = shutil.which(name)
        if path:
            return path
    return None


def build_and_run(cxx, work, case):
    stem = re.sub(r'\W+', '_', case['name']).strip('_')
    src = os.path.join(work, stem + '.cpp')
    exe = os.path.join(work, stem)
    with open(src, 'w', encoding='utf-8') as f:
        f.write(PROGRAM % (STUB if case['stubs'] else '', case['body']))
    cmd = [cxx, '-std=c++17', '-O0']
    if case['tgcu']:
        cmd.append('-DSTRATA_TGCU')
    cmd += ['-I', os.path.join(ROOT, 'include'), '-o', exe, src]
    built = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    if built.returncode != 0:
        errors = [l.strip() for l in (built.stdout + built.stderr).splitlines() if ' error: ' in l]
        return False, 'does not build against the header (' + (errors[0] if errors else 'see the compile log').replace('"', "'") + ')'
    run = subprocess.run([exe], capture_output=True, text=True, timeout=120)
    got = run.stdout.strip()
    if run.returncode != 0:
        return False, 'the built program exited %d: %s' % (run.returncode, got[-200:])
    if got != case['want']:
        return False, 'the program printed %r, expected %r' % (got, case['want'])
    return True, ''


def strip_comments(text):
    """Replace comments and string literals with blanks so a call inside either is not read as code."""
    out = []
    i = 0
    n = len(text)
    while i < n:
        c = text[i]
        if text.startswith('//', i):
            j = text.find('\n', i)
            i = n if j < 0 else j
        elif text.startswith('/*', i):
            j = text.find('*/', i + 2)
            i = n if j < 0 else j + 2
            out.append(' ')
        elif c == '"' or c == "'":
            quote = c
            i += 1
            while i < n:
                if text[i] == '\\':
                    i += 2
                    continue
                if text[i] == quote or text[i] == '\n':
                    i += 1
                    break
                i += 1
            out.append('""')
        else:
            out.append(c)
            i += 1
    return ''.join(out)


CALL = re.compile(r'strata\s*::\s*tgcu\s*::\s*(?:push|pull|manage)\s*\(')


def close_paren(text, start):
    depth = 0
    i = start
    while i < len(text):
        if text[i] in '([{':
            depth += 1
        elif text[i] in ')]}':
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def drops_status(text, match):
    """True when this call is a whole expression statement, so its int return value is thrown away."""
    close = close_paren(text, match.end() - 1)
    if close < 0:
        return False
    depth = 0
    i = close + 1
    while i < len(text):
        c = text[i]
        if c in '([{':
            depth += 1
        elif c in ')]}':
            if depth == 0:
                break
            depth -= 1
        elif c == ';' and depth == 0:
            break
        i += 1
    if text[close + 1:i].strip():   # part of a larger expression: a condition, an argument, a comma, an assignment
        return False
    start = max(text.rfind(';', 0, match.start()), text.rfind('{', 0, match.start()), text.rfind('}', 0, match.start()))
    head = text[start + 1:match.start()].replace('(void)', ' ').strip()
    while True:
        if head.startswith('else'):
            head = head[4:].strip()
        guard = re.match(r'(?:if|while|for|switch)\s*\(', head)
        if not guard:
            break
        j = close_paren(head, guard.end() - 1)
        if j < 0:
            break
        head = head[j + 1:].strip()
    return head == ''


def callers():
    found, dropped = [], []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for name in filenames:
            if not name.endswith(SOURCE_EXT):
                continue
            path = os.path.join(dirpath, name)
            rel = os.path.relpath(path, ROOT)
            if rel == HEADER:
                continue
            try:
                with open(path, encoding='utf-8', errors='replace') as f:
                    text = strip_comments(f.read())
            except OSError:
                continue
            for match in CALL.finditer(text):
                line = text.count('\n', 0, match.start()) + 1
                found.append((rel, line))
                if drops_status(text, match):
                    dropped.append((rel, line))
    return found, dropped


def main():
    if not os.path.isfile(os.path.join(ROOT, HEADER)):
        print('FAIL run this test from the copy root: ' + HEADER + ' is not in ' + ROOT)
        return 1
    cxx = compiler()
    if cxx is None:
        print('FAIL no C++ compiler was found (looked for g++, clang++, c++)')
        return 1
    work = tempfile.mkdtemp(prefix='tgcu-returns-')
    failures = 0
    try:
        for case in CASES:
            ok, detail = build_and_run(cxx, work, case)
            print(('PASS ' if ok else 'FAIL ') + case['name'] + (': ' + detail if detail else ''))
            failures += 0 if ok else 1
    finally:
        shutil.rmtree(work, ignore_errors=True)
    found, dropped = callers()
    where = ', '.join('%s (%d)' % (f, sum(1 for g, _ in found if g == f)) for f in sorted({g for g, _ in found}))
    if not found:
        print('FAIL no caller of strata::tgcu::push, pull or manage was found in the tree')
        failures += 1
    elif dropped:
        listed = ', '.join('%s:%d' % (f, l) for f, l in dropped[:8]) + (' and %d more' % (len(dropped) - 8) if len(dropped) > 8 else '')
        print('FAIL these call sites drop the driver status: ' + listed)
        failures += 1
    else:
        print('PASS every caller keeps the driver status (' + str(len(found)) + ' call site(s): ' + where + ')')
    print('%d case(s) failed' % failures if failures else 'all cases passed')
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
