#!/usr/bin/env python3
"""test_fix_0280_0: the Mac cold-expert lane's block at Dm's end must hold exactly the rows rows_cpu counts.

# What the code does (src/prefill/prefill.cpp). A layer's CPU share marks its experts in on_cpu and adds their rows to
# rows_cpu; rows_cpu is the block at Dm's end (off[]'s last block) and its start is T * K - rows_cpu. The STRATA_PF_MAC
# block takes that same block for the layer's cold experts: it sets on_cpu to the cold experts only, but keeps adding on
# top of the share's count. rows_cpu then counts rows the block does not hold (the share's, once more), while off[], the
# Mac's rows_res, the lockstep combine's T * K - rows_cpu and rows_local (T * K - rows_peer - rows_cpu) all trust it. A
# Mac layer runs no CPU thread (`if (rows_cpu > 0 && !mac_layer)`), so the share's rows are not computed anywhere: they
# must not be counted.
#
# The test is behavioral and needs no GPU, model or network. It compiles the shipped marking block and the shipped off[]
# build verbatim (located by anchors, braces counted) inside a small fake of the layer state (routing counts, resident
# mask, share, the Mac lane's cold set), then checks the arithmetic that the code later reads: rows_cpu against the rows
# the marked experts hold, the block's start against T * K - rows_cpu, rows_local against that start, and the marked rows
# against the tail of off[].
#
# Run: python3 tests/test_fix_0280_0.py
"""
import os
import subprocess
import sys
import tempfile



def tree_root():
    """The tree under test: the working directory (the harness runs the test there), else the copy this file sits in."""
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for d in (os.getcwd(), here):
        if os.path.isfile(os.path.join(d, 'src', 'prefill', 'prefill.cpp')):
            return d
    return here


SOURCE = os.path.join(tree_root(), 'src', 'prefill', 'prefill.cpp')

# name, T, K, routing counts per expert, the experts not in VRAM (cold: the share's candidates and the Mac's), the share
# (double: the measured share, as STRATA_PREFILL_CPU_SHARE unset leaves it), cpu_maybe (the share may run this layer),
# mac_x_issued (STRATA_PF_MAC issued this layer's cold experts).
CASES = [
    dict(name='share only: no Mac lane, the share marks two experts',
         T=2, K=4, cnt=[2, 3, 0, 3], cold=[0, 1, 2, 3], share=0.5, cpu_maybe=True, mac_x_issued=False),
    dict(name='Mac only: the share is not armed, the Mac takes the block',
         T=2, K=4, cnt=[2, 3, 0, 3], cold=[1, 3], share=0.5, cpu_maybe=False, mac_x_issued=True),
    dict(name='share and Mac: the share marks one cold expert, the Mac takes the whole cold set',
         T=2, K=4, cnt=[2, 3, 0, 3], cold=[1, 3], share=0.5, cpu_maybe=True, mac_x_issued=True),
    dict(name='share armed but takes nothing, Mac takes the block',
         T=2, K=4, cnt=[2, 3, 0, 3], cold=[1, 3], share=0.0, cpu_maybe=True, mac_x_issued=True),
    dict(name='share and Mac on a wider layer: three of the Mac\'s rows, four cold experts',
         T=3, K=4, cnt=[1, 2, 4, 0, 3, 2], cold=[0, 2, 4, 5], share=0.5, cpu_maybe=True, mac_x_issued=True),
]

HARNESS = r'''
// Built by tests/test_fix_0280_0.py. The two marked sections are pasted from src/prefill/prefill.cpp as shipped: the
// routing counts, the resident mask, the share and the Mac lane's cold set come from the case, everything else is a fake.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <utility>
#include <vector>

namespace strata { namespace kernels { namespace cpu {
inline constexpr int32_t MAXT = 64;   // the shipped cap on an expert's tokens in the share; these cases stay below it
} } }

struct G { int32_t n_expert; };

struct Src {   // every expert's blob is stable, pinned and not transient: the source's own answers, faked
    const uint8_t* blob_stable(int32_t, int32_t e) const { static const uint8_t b[64] = {0}; return b + (size_t) e; }
    bool pinned(int32_t, int32_t) const { return true; }
    bool transient(int32_t, int32_t) const { return false; }
};

struct M {
    G* g = nullptr;
    Src* src = nullptr;
    std::vector<int32_t> cnt, off, mac_cold;
    const int32_t* host_res = nullptr;   // >= 0: in VRAM, < 0: cold
    const void* cache = nullptr;
    double cpu_share_now = 0.5;
};

static bool cpu_arm = true;         // auto: this layer may share
static double share_env = -1.0;     // STRATA_PREFILL_CPU_SHARE unset: the measured share
static double cpu_share_env() { return share_env; }

struct Case {
    const char* name;
    int32_t T, K, n_expert;
    int32_t cnt[8];
    int32_t cold[8];
    int cold_n;
    double share;
    bool cpu_maybe, mac_x_issued;
};

static const Case CASES[] = {
@@CASES@@
};

static void run(const Case& c) {
    G g; g.n_expert = c.n_expert;
    Src src;
    M m;
    m.g = &g; m.src = &src;
    m.cnt.assign(c.cnt, c.cnt + c.n_expert);
    m.off.assign((size_t) c.n_expert + 1, 0);
    std::vector<int32_t> host_res((size_t) c.n_expert, 0);
    for (int i = 0; i < c.cold_n; ++i) host_res[(size_t) c.cold[i]] = -1;
    m.host_res = host_res.data();
    share_env = c.share;
    const bool cpu_maybe = c.cpu_maybe;
    const bool mac_x_issued = c.mac_x_issued;
    const int32_t l = 0;
    const int64_t T = c.T, K = c.K;
    std::vector<char> on_cpu;
    std::vector<const uint8_t*> cpu_blob;
    int64_t rows_cpu = 0, n_cpu = 0, n_stream = 0;
    bool mac_layer = false;   // the Mac took this layer's block: rows_cpu counts its rows and no CPU thread runs

@@BLOCK_A@@

    std::vector<char> on_peer;   // no peer in these cases
    int64_t rows_local = T * K, rows_peer = 0;

@@BLOCK_B@@

    std::printf("CASE %s|%lld|%lld|%d|%lld|%lld|%d|", c.name, (long long) T, (long long) K, (int) c.n_expert,
                (long long) rows_cpu, (long long) rows_local, mac_layer ? 1 : 0);
    for (int32_t e = 0; e < c.n_expert; ++e) std::printf("%d,", (!on_cpu.empty() && on_cpu[(size_t) e]) ? 1 : 0);
    std::printf("|");
    for (int32_t e = 0; e <= c.n_expert; ++e) std::printf("%d,", (int) m.off[(size_t) e]);
    std::printf("|");
    for (int32_t e = 0; e < c.n_expert; ++e) std::printf("%d,", (int) m.cnt[(size_t) e]);
    std::printf("\n");
}

int main() {
    for (const Case& c : CASES) run(c);
    return 0;
}
'''


def strip(line):
    return line.strip()


def close_of(lines, start):
    """The line of the brace that closes the block opened on line `start` (braces in these blocks are code only)."""
    depth = 0
    for i in range(start, len(lines)):
        depth += lines[i].count('{') - lines[i].count('}')
        if depth <= 0:
            return i
    return -1


def extract(lines):
    """The two shipped blocks, or a reason they could not be found."""
    mac = [i for i, s in enumerate(lines) if strip(s).startswith('if (mac_x_issued) {')]
    if len(mac) != 1:
        return None, None, 'src/prefill/prefill.cpp: %d blocks open with "if (mac_x_issued) {" (want 1)' % len(mac)
    mac_end = close_of(lines, mac[0])
    cpu = [i for i in range(mac[0]) if strip(lines[i]) == 'if (cpu_maybe) {']
    if not cpu:
        return None, None, 'src/prefill/prefill.cpp: no "if (cpu_maybe) {" above the Mac block'
    block_a = ''.join(lines[cpu[-1]:mac_end + 1])
    r0 = [i for i, s in enumerate(lines) if strip(s) == 'int32_t r = 0;']
    if not r0:
        return None, None, 'src/prefill/prefill.cpp: no "int32_t r = 0;" (the off[] build)'
    last = [i for i, s in enumerate(lines) if strip(s) == 'rows_local -= rows_cpu;']
    if not last:
        return None, None, 'src/prefill/prefill.cpp: no "rows_local -= rows_cpu;" (the off[] build)'
    start = r0[-1] - 1
    if strip(lines[start]) != '{':
        return None, None, 'src/prefill/prefill.cpp: the off[] build does not open with a brace at line %d' % (start + 1)
    end = last[-1] + 1
    if strip(lines[end]) != '}':
        return None, None, 'src/prefill/prefill.cpp: the off[] build does not close with a brace at line %d' % (end + 1)
    return block_a, ''.join(lines[start:end + 1]), None


def case_table():
    out = []
    for c in CASES:
        cnt = ', '.join(str(v) for v in c['cnt'])
        cold = ', '.join(str(v) for v in c['cold'])
        out.append('    {"%s", %d, %d, %d, {%s}, {%s}, %d, %r, %s, %s},'
                   % (c['name'].replace('"', ''), c['T'], c['K'], len(c['cnt']), cnt, cold, len(c['cold']),
                      float(c['share']), 'true' if c['cpu_maybe'] else 'false',
                      'true' if c['mac_x_issued'] else 'false'))
    return '\n'.join(out)


def build(tmp, block_a, block_b):
    src = os.path.join(tmp, 'harness.cpp')
    exe = os.path.join(tmp, 'harness')
    text = HARNESS.replace('@@CASES@@', case_table()).replace('@@BLOCK_A@@', block_a).replace('@@BLOCK_B@@', block_b)
    with open(src, 'w') as f:
        f.write(text)
    cxx = os.environ.get('CXX') or 'c++'
    r = subprocess.run([cxx, '-std=c++17', '-O1', '-o', exe, src], capture_output=True, text=True)
    if r.returncode != 0:
        return None, (r.stderr or r.stdout).strip()[-600:]
    return exe, None


def numbers(field):
    return [int(v) for v in field.split(',') if v != '']


def main():
    with open(SOURCE) as f:
        lines = f.readlines()
    block_a, block_b, why = extract(lines)
    if why:
        print('FAIL every case: %s' % why)
        return 1
    tmp = tempfile.mkdtemp(prefix='fix-0280-')
    try:
        exe, err = build(tmp, block_a, block_b)
        if err:
            print('FAIL every case: the shipped blocks do not compile in the fake: %s' % err)
            return 1
        r = subprocess.run([exe], capture_output=True, text=True, timeout=60)
        rows = [ln for ln in r.stdout.splitlines() if ln.startswith('CASE ')]
        if r.returncode != 0 or len(rows) != len(CASES):
            print('FAIL every case: the harness printed %d of %d lines (exit %d): %s'
                  % (len(rows), len(CASES), r.returncode, (r.stderr or r.stdout).strip()[-300:]))
            return 1
    finally:
        for name in os.listdir(tmp):
            try:
                os.remove(os.path.join(tmp, name))
            except OSError:
                pass
        os.rmdir(tmp)
    bad = 0
    for case, line in zip(CASES, rows):
        _, T, K, n_expert, rows_cpu, rows_local, mac_layer, marks, off, cnt = line.split('CASE ')[1].split('|')
        T, K, n_expert = int(T), int(K), int(n_expert)
        rows_cpu, rows_local = int(rows_cpu), int(rows_local)
        mac_layer = mac_layer == '1'
        on = numbers(marks)
        off, cnt = numbers(off), numbers(cnt)
        expert_rows = T * K
        marked = [e for e in range(n_expert) if on[e]]
        block_rows = sum(cnt[e] for e in marked)
        block_start = min((off[e] for e in marked), default=expert_rows)
        problems = []
        if rows_cpu != block_rows:
            problems.append('rows_cpu counts %d rows, the block holds %d (experts %s)'
                            % (rows_cpu, block_rows, marked))
        if block_start != expert_rows - rows_cpu:
            problems.append('the block starts at %d, the consumers look at T*K - rows_cpu = %d'
                            % (block_start, expert_rows - rows_cpu))
        if rows_local != block_start:
            problems.append('rows_local is %d, the rows before the block are %d' % (rows_local, block_start))
        if rows_local != expert_rows - block_rows:
            problems.append('rows_local is %d, the GPU rows are %d' % (rows_local, expert_rows - block_rows))
        if mac_layer and not set(marked) <= set(case['cold']):
            problems.append('the Mac took the block, but it marks experts outside its cold set: %s' % (marked,))
        seen = []
        for e in range(n_expert):
            seen.extend(range(off[e], off[e] + cnt[e]))
        if sorted(seen) != list(range(expert_rows)):
            problems.append('off[] does not cover 0..%d once (it covers %s)' % (expert_rows - 1, sorted(seen)))
        if problems:
            bad += 1
            print('FAIL %s: %s' % (case['name'], '; '.join(problems)))
        else:
            print('PASS %s: rows_cpu %d = the block\'s %d rows from %d, rows_local %d, off[] covers T*K'
                  % (case['name'], rows_cpu, block_rows, block_start, rows_local))
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
