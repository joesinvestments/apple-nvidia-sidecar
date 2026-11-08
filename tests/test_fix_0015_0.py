#!/usr/bin/env python3
"""tests/test_fix_0015_0.py - a prefill that outgrows the staging buffer must stage into the buffer it grew.

exl3-metal/exl3_metal.mm:852 captures `id<MTLBuffer> sbuf = s->stage;` and the non-NVFP4 growth branch at 856 then
reassigns `s->stage = mkbuf(c, nstaged * c->blob)` without touching `sbuf`.  The memcpy at 862 therefore writes
nstaged*blen bytes into the old, smaller buffer, and `addr[]` at 864 points into that old buffer instead of the new
one, so the kernels read what the next allocation happens to hold.  The NVFP4 branch just above reassigns sbuf and
is correct.

These cases compile the real staging block from the .mm against a fake Metal (no GPU, no model, no weights) and run
the built program.  Each buffer has host memory with a canary past its stated length, so a write past the end shows
up, and addr[] is read back: it must land in the buffer that `s->stage` holds now, holding the staged blob.
  - a first staged prefill that outgrows a one-blob stage buffer;
  - a growth to four staged blobs with one registered expert mixed in (its address must not be staged);
  - a staged prefill that fits the stage buffer (the same result before and after the fix);
  - the NVFP4 staging lane, which already reassigns sbuf, so it behaves the same before and after the fix.
Prints one PASS or FAIL line per case and exits nonzero on any FAIL.  Run: python3 tests/test_fix_0015_0.py
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile

MM = os.path.join('exl3-metal', 'exl3_metal.mm')

CASES = [
    dict(name='a staged prefill that outgrows a one-blob stage buffer', pre=1, nu=3, nreg=0, nv4=False, stages=True),
    dict(name='a growth to four staged blobs with one registered expert mixed in', pre=2, nu=5, nreg=1, nv4=False, stages=True),
    dict(name='a staged prefill that fits the stage buffer', pre=4, nu=2, nreg=0, nv4=False, stages=True),
    dict(name='the NVFP4 staging lane still grows and stages correctly', pre=1, nu=3, nreg=0, nv4=True, stages=True),
]

PROGRAM = r'''// built by tests/test_fix_0015_0.py
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <vector>

// A fake Metal where each buffer is host memory with a canary past its stated length, so a write past the end shows.
static const size_t GUARD = 4096;
static const size_t NVFP4_BLOB = 96;   // a stub length: the branch logic, not the real blob size, is what is under test

struct DevBuf {
    size_t n = 0;                       // the buffer's stated length
    std::vector<unsigned char> mem;     // n + GUARD bytes
    uint64_t gpu = 0;                   // its fake GPU address
    int handle = 0;
};

static std::vector<std::shared_ptr<DevBuf>> g_all;
static uint64_t g_next_gpu = 0x100000;
static int g_next_handle = 1;

struct Buf {
    std::shared_ptr<DevBuf> d;
    unsigned char *contents = nullptr;
    uint64_t gpuAddress = 0;
    size_t length = 0;
    Buf() {}
    explicit Buf(const std::shared_ptr<DevBuf> &x)
        : d(x), contents(x->mem.data()), gpuAddress(x->gpu), length(x->n) {}
};

static Buf make_buf(size_t n) {
    auto d = std::make_shared<DevBuf>();
    d->n = n;
    d->mem.assign(n + GUARD, 0xA5);
    d->gpu = g_next_gpu;
    g_next_gpu += 0x1000000;
    d->handle = g_next_handle++;
    g_all.push_back(d);
    return Buf(d);
}

struct Ctx { size_t blob = 0; };
struct Slot { Buf stage; size_t stage_blobs = 0; Buf p_nvst; size_t p_nvst_n = 0; };

static Buf mkbuf(Ctx *, size_t n) { return make_buf(n); }
static double now_ms() { return 0.0; }

// the registered region: a blob pointer inside it resolves to a GPU address in g_reg_buf
static unsigned char g_reg[8192];
static Buf g_reg_buf;
static bool resolve(Ctx *, const void *p, uint64_t &a, Buf &buf, size_t n) {
    const unsigned char *q = (const unsigned char *)p;
    if (q >= g_reg && q + n <= g_reg + sizeof g_reg) {
        a = g_reg_buf.gpuAddress + (uint64_t)(q - g_reg);
        buf = g_reg_buf;
        return true;
    }
    return false;
}

@BLOCK@

static bool guards_ok() {
    for (auto &d : g_all)
        for (size_t i = d->n; i < d->mem.size(); i++)
            if (d->mem[i] != 0xA5) return false;
    return true;
}

static DevBuf *find_gpu(uint64_t a, size_t n) {
    for (auto &d : g_all)
        if (a >= d->gpu && a + n <= d->gpu + d->n) return d.get();
    return nullptr;
}

int main(int argc, char **argv) {
    if (argc < 5) return 2;
    const size_t pre = (size_t)std::atol(argv[1]);   // staged blobs already in the staging buffer
    const int nu = std::atoi(argv[2]);               // used experts
    const int nreg = std::atoi(argv[3]);             // of them covered by a registered region
    const bool nv4 = std::atoi(argv[4]) != 0;
    const size_t BLEN = 64;
    Ctx c;
    c.blob = BLEN;
    const size_t blen = nv4 ? NVFP4_BLOB : c.blob;
    Slot s;
    g_reg_buf = make_buf(sizeof g_reg);
    std::vector<std::vector<unsigned char>> data((size_t)nu, std::vector<unsigned char>(blen));
    for (int u = 0; u < nu; u++)
        for (size_t i = 0; i < blen; i++) data[(size_t)u][i] = (unsigned char)(0x11 + u * 7 + (int)i);
    std::vector<const void *> blobs((size_t)nu, nullptr);
    for (int u = 0; u < nu; u++) {
        if (u < nreg) {
            unsigned char *p = g_reg + (size_t)u * 128;
            std::memcpy(p, data[(size_t)u].data(), blen);
            blobs[(size_t)u] = p;
        } else {
            blobs[(size_t)u] = data[(size_t)u].data();
        }
    }
    Buf *targetp = nv4 ? &s.p_nvst : &s.stage;
    if (pre > 0) {
        *targetp = make_buf(pre * blen);
        if (nv4) s.p_nvst_n = pre; else s.stage_blobs = pre;
        std::memset(targetp->contents, 0x22, pre * blen);
    }
    std::vector<int32_t> uexp;
    for (int u = 0; u < nu; u++) uexp.push_back(u);
    std::vector<uint64_t> addr((size_t)nu, 0);
    std::vector<Buf> used;
    size_t nstaged = 0;
    double tc = 0;
    {   // the caller's setup pass: registered blobs resolve to their own GPU address and are not staged
        Buf buf;
        for (int u = 0; u < nu; u++)
            if (resolve(&c, blobs[(size_t)u], addr[(size_t)u], buf, blen)) used.push_back(buf);
            else nstaged++;
    }
    std::map<int, std::vector<unsigned char>> before;
    for (auto &d : g_all) before[d->handle] = d->mem;

    run_stage(&c, &s, nv4, nu, blobs.data(), uexp, addr, used, nstaged, blen, tc);

    Buf target = nv4 ? s.p_nvst : s.stage;
    bool addr_in_stage = true, addr_matches = true, bytes_at_addr = true, old_untouched = true;
    int k = 0;
    for (int u = 0; u < nu; u++) {
        if (u < nreg) continue;
        const uint64_t a = addr[(size_t)u];
        if (!(a >= target.gpuAddress && a + blen <= target.gpuAddress + target.length)) addr_in_stage = false;
        if (a != target.gpuAddress + (uint64_t)k * blen) addr_matches = false;
        DevBuf *d = find_gpu(a, blen);
        if (!d || std::memcmp(d->mem.data() + (a - d->gpu), data[(size_t)u].data(), blen) != 0) bytes_at_addr = false;
        k++;
    }
    for (auto &d : g_all) {   // a buffer the call retired must not be written at all
        if (d == target.d) continue;
        auto it = before.find(d->handle);
        if (it != before.end() && it->second != d->mem) old_untouched = false;
    }
    bool reg_ok = true;
    for (int u = 0; u < nreg; u++)
        if (addr[(size_t)u] != g_reg_buf.gpuAddress + (uint64_t)u * 128) reg_ok = false;
    const bool used_stage = !used.empty() && used.back().d == target.d;
    std::printf("nu=%d\n", nu);
    std::printf("nstaged=%zu\n", nstaged);
    std::printf("addr_in_stage=%d\n", (int)addr_in_stage);
    std::printf("addr_matches=%d\n", (int)addr_matches);
    std::printf("bytes_at_addr=%d\n", (int)bytes_at_addr);
    std::printf("old_untouched=%d\n", (int)old_untouched);
    std::printf("guards=%d\n", (int)guards_ok());
    std::printf("used_stage=%d\n", (int)used_stage);
    std::printf("reg_ok=%d\n", (int)reg_ok);
    return 0;
}
'''

CHECKS = ('addr_in_stage', 'addr_matches', 'bytes_at_addr', 'old_untouched', 'guards', 'used_stage', 'reg_ok')


def find_root():
    """The tree under test: the copy prove_test runs us in, else the one this file sits in."""
    for base in (os.getcwd(), os.path.dirname(os.path.dirname(os.path.abspath(__file__)))):
        if os.path.isfile(os.path.join(base, MM)):
            return base
    return os.getcwd()


def blank_comments(text):
    """Blank comments and string literals at the same length, so an offset into the copy is an offset into the source."""
    out = list(text)
    i, n = 0, len(text)
    while i < n:
        if text.startswith('//', i):
            j = text.find('\n', i)
            j = n if j < 0 else j
            for k in range(i, j):
                out[k] = ' '
            i = j
        elif text.startswith('/*', i):
            j = text.find('*/', i + 2)
            j = n if j < 0 else j + 2
            for k in range(i, j):
                if out[k] != '\n':
                    out[k] = ' '
            i = j
        elif text[i] in '"\'':
            quote = text[i]
            i += 1
            while i < n:
                if text[i] == '\\':
                    i += 2
                    continue
                if text[i] == quote or text[i] == '\n':
                    out[i] = out[i] if text[i] == '\n' else ' '
                    i += 1
                    break
                out[i] = ' '
                i += 1
        else:
            i += 1
    return ''.join(out)


def close_brace(code, open_at):
    """The index of the '}' matching the '{' at open_at, or -1."""
    depth = 0
    for i in range(open_at, len(code)):
        if code[i] == '{':
            depth += 1
        elif code[i] == '}':
            depth -= 1
            if depth == 0:
                return i
    return -1


BLOCK_HEAD = re.compile(r'if\s*\(\s*nstaged\s*\)\s*\{')


def staging_block(text):
    """The `if (nstaged) { ... }` block that stages into sbuf, and the line it starts on."""
    code = blank_comments(text)
    for m in BLOCK_HEAD.finditer(code):
        open_at = code.index('{', m.start())
        end = close_brace(code, open_at)
        if end < 0:
            continue
        body = code[m.start():end + 1]
        if 'sbuf' in body and 'memcpy' in body and 'resolve' in body:
            line = code.count('\n', 0, m.start()) + 1
            return text[m.start():end + 1], line
    return None, 0


def compiler():
    for name in ('g++', 'clang++', 'c++'):
        path = shutil.which(name)
        if path:
            return path
    return None


def build(cxx, work, block):
    body = re.sub(r'\bid\s*<\s*MTLBuffer\s*>', 'Buf', block)
    wrapper = ('static void run_stage(Ctx *c, Slot *s, bool nv4, int nu, const void *const *blobs,\n'
               '                      const std::vector<int32_t> &uexp, std::vector<uint64_t> &addr,\n'
               '                      std::vector<Buf> &used, size_t &nstaged, size_t blen, double &tc) {\n'
               + body + '\n}\n')
    src = os.path.join(work, 'stage_fake.cpp')
    exe = os.path.join(work, 'stage_fake')
    with open(src, 'w', encoding='utf-8') as f:
        f.write(PROGRAM.replace('@BLOCK@', wrapper))
    cmd = [cxx, '-std=c++17', '-O0', '-o', exe, src]
    built = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if built.returncode != 0:
        errors = [l.strip() for l in (built.stdout + built.stderr).splitlines() if ' error: ' in l]
        return None, 'the real staging block does not build against the fake Metal (' + \
            (errors[0] if errors else 'see the compile log').replace('"', "'") + ')'
    return exe, ''


def run_case(exe, case):
    argv = [exe, str(case['pre']), str(case['nu']), str(case['nreg']), '1' if case['nv4'] else '0']
    r = subprocess.run(argv, capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        return False, 'the built program exited %d: %s' % (r.returncode, r.stdout.strip()[-200:])
    got = {}
    for line in r.stdout.splitlines():
        if '=' in line:
            k, v = line.split('=', 1)
            got[k.strip()] = v.strip()
    want_nstaged = str(case['nu'] - case['nreg'])
    if got.get('nstaged') != want_nstaged:
        return False, 'the program counted nstaged=%r, expected %r' % (got.get('nstaged'), want_nstaged)
    if case['stages']:
        bad = [k for k in CHECKS if got.get(k) != '1']
        if bad:
            return False, 'not staged into the grown buffer: ' + ', '.join('%s=%s' % (k, got.get(k)) for k in bad)
    return True, ''


def main():
    root = find_root()
    path = os.path.join(root, MM)
    if not os.path.isfile(path):
        print('FAIL run this test from the copy root: ' + MM + ' is not in ' + root)
        return 1
    with open(path, encoding='utf-8', errors='replace') as f:
        block, line = staging_block(f.read())
    if block is None:
        print('FAIL ' + MM + ' has no if (nstaged) { ... } block that stages into sbuf')
        return 1
    cxx = compiler()
    if cxx is None:
        print('FAIL no C++ compiler was found (looked for g++, clang++, c++)')
        return 1
    work = tempfile.mkdtemp(prefix='fix-0015-')
    failures = 0
    try:
        exe, why = build(cxx, work, block)
        if exe is None:
            print('FAIL the staging block at ' + MM + ':' + str(line) + ' ' + why)
            return 1
        for case in CASES:
            ok, detail = run_case(exe, case)
            print(('PASS ' if ok else 'FAIL ') + case['name'] + (': ' + detail if detail else ''))
            failures += 0 if ok else 1
    finally:
        shutil.rmtree(work, ignore_errors=True)
    print('%d case(s) failed' % failures if failures else 'all cases passed')
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
