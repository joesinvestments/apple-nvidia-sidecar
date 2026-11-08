#!/usr/bin/env python3
"""tests/test_fix_0334_0.py - the guided refill's pieces must stop at the region's end, not the cache's.

src/program/generate.cpp:17244 plans the refill's copies in pieces of d7.piece slots:
    auto piece_range = [&](int k, int32_t& a, int32_t& b) { a = rb + k * P; b = std::min<int32_t>(slots, a + P); };
The last piece is clamped to the cache's slot count `slots` instead of the region's last copied slot rb + Kc.  The
lazy refill (d18) copies Kc of the region's nsl slots before the first token and leaves the rest pending, and the
region need not be the cache's last, so the last piece can end past rb + Kc: assign[s - rb] (17250/17255) is read
past its nsl entries, slots meant to stay pending are uploaded, and soff(b - 1) (17271) can run past the staging
buffer sized at 17241 from min(P, nsl) slots.

These cases call the region's piece arithmetic as the refill uses it (include/strata/program/refill_pieces.hpp: no
model, no GPU, no weights) and run the copy loop over a fake cache: slots as offsets, assign[] with one expert per
region slot, and a staging buffer sized from the widest piece.  On today's code the header is not there, so every
case fails to build; with the fix the pieces tile [rb, rb + Kc) exactly.
Prints one PASS or FAIL line per case and exits nonzero on any FAIL.  Run: python3 tests/test_fix_0334_0.py
"""
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.getcwd()   # the tree under test: every test here runs with cwd at the copy root, never at the test file
HEADER = os.path.join('include', 'strata', 'program', 'refill_pieces.hpp')

CASES = [
    'a region that is not the cache last stops at its own end',
    'a lazy region copies only its Kc slots and leaves the rest pending',
    'a region that ends with the cache copies all of it',
    'a piece larger than the region is one piece',
    'a piece of one slot tiles the region',
    'every region and piece size keeps the region bounds',
]

PROGRAM = r"""// built by tests/test_fix_0334_0.py: the guided refill's pieces over a fake expert cache.
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include "strata/program/refill_pieces.hpp"

using strata::program::refill_piece;
using strata::program::refill_pieces;
using strata::program::refill_span_slots;

namespace {

const int64_t kBlob = 1024;        // bytes of one expert blob in the fake cache
const int64_t kSlotBytes = 300000; // the fake cache's slot stride: a piece's span is visible in bytes

int64_t soff(int32_t s) { return (int64_t) s * kSlotBytes; }

std::string num(int32_t v) { return std::to_string(v); }

// The refill's copy loop: assign[] holds the region's nsl experts, only its first `bound` slots are copied, and buf is
// the staging buffer sized from the widest piece.  A piece that leaves the region throws from assign.at().
std::string run(int32_t base, std::size_t bound, std::size_t nsl, int32_t piece, std::string& detail) {
    const int32_t slots = base + (int32_t) nsl;
    std::vector<int32_t> assign(nsl, -1);
    for (std::size_t i = 0; i < nsl; ++i) assign[i] = 1000 + (int32_t) i;
    const int np = refill_pieces(bound, piece);
    const std::size_t want_np = (bound + (std::size_t) piece - 1) / (std::size_t) piece;
    if ((std::size_t) np != want_np) return "piece count " + num(np) + ", the region copies " + std::to_string(want_np) + " pieces";
    const int32_t span_slots = refill_span_slots(bound, piece);
    if (span_slots < 1 || (std::size_t) span_slots > (bound < (std::size_t) piece ? bound : (std::size_t) piece))
        return "staging span " + num(span_slots) + " slots for a region of " + std::to_string(bound) + " in pieces of " + num(piece);
    const int64_t span_max = soff(base + span_slots - 1) - soff(base) + kBlob;
    std::vector<unsigned char> buf((std::size_t) span_max);
    int32_t prev_b = base;
    int64_t copied = 0;
    std::string ends;
    for (int k = 0; k < np; ++k) {
        int32_t a = -1, b = -1;
        refill_piece(base, bound, piece, k, a, b);
        if (a != prev_b) return "piece " + num(k) + " starts at slot " + num(a) + ", the previous piece ended at " + num(prev_b);
        if (b <= a) return "piece " + num(k) + " covers no slot: [" + num(a) + "," + num(b) + ")";
        if (b - a > piece) return "piece " + num(k) + " is " + num(b - a) + " slots, more than the " + num(piece) + " of a piece";
        if (b > base + (int32_t) bound) return "piece " + num(k) + " ends at slot " + num(b) + ", past the region's last copied slot " + num(base + (int32_t) bound);
        if (b > slots) return "piece " + num(k) + " ends at slot " + num(b) + ", past the cache's last slot " + num(slots);
        const int64_t span = soff(b - 1) - soff(a) + kBlob;
        if (span > span_max) return "piece " + num(k) + "'s span " + std::to_string(span) + " exceeds the staging buffer's " + std::to_string(span_max) + " bytes";
        for (int32_t s = a; s < b; ++s) {
            const int32_t i = assign.at((std::size_t) (s - base));   // the out of range read of today's clamp
            if (i < 0) return "slot " + num(s) + " has no expert";
            const std::size_t off = (std::size_t) (soff(s) - soff(a));
            if (off + (std::size_t) kBlob > buf.size()) return "slot " + num(s) + "'s blob runs past the staging buffer";
        }
        copied += b - a;
        prev_b = b;
        ends += (ends.empty() ? "" : ",") + num(b);
    }
    if (copied != (int64_t) bound) return "copied " + std::to_string(copied) + " slots, the region copies " + std::to_string(bound);
    if (np > 0 && prev_b != base + (int32_t) bound) return "the last piece ends at slot " + num(prev_b) + ", the region's last copied slot is " + num(base + (int32_t) bound);
    detail = "region [" + num(base) + "," + num(base + (int32_t) bound) + ") of " + std::to_string(nsl) + " slots, pieces of " + num(piece) + ": ends at " + (ends.empty() ? "nothing" : ends) + ", staging span " + std::to_string(span_max) + " bytes";
    return "";
}

int fails = 0;

void one(const char* name, int32_t base, std::size_t bound, std::size_t nsl, int32_t piece) {
    std::string why, detail;
    try {
        why = run(base, bound, nsl, piece, detail);
    } catch (const std::exception& e) {
        why = e.what();
    }
    if (why.empty()) std::printf("CASE %s PASS: %s\n", name, detail.c_str());
    else { std::printf("CASE %s FAIL: %s\n", name, why.c_str()); ++fails; }
}

}  // namespace

int main() {
    // the planted input: slot 64 starts a region of 10 slots in a 200-slot cache, 3 pieces of 4 - a clamp to the cache
    // ends the last piece at slot 76 instead of 74 and reads assign[10] and assign[11] of a 10-entry region
    one("a region that is not the cache last stops at its own end", 64, 10, 10, 4);
    // the lazy refill's shape: 5 of 300 slots before the first token, the other 295 stay pending for the decode windows
    one("a lazy region copies only its Kc slots and leaves the rest pending", 0, 5, 300, 4);
    // the ordinary input: the region is the cache's last and all of it is copied, so the clamp lands on the same slot
    one("a region that ends with the cache copies all of it", 100, 64, 64, 7);
    // one piece holds the whole region
    one("a piece larger than the region is one piece", 7, 3, 3, 8);
    // the smallest piece
    one("a piece of one slot tiles the region", 5, 5, 5, 1);
    // every shape: the pieces tile the region, nothing is read past it, the staging buffer holds each piece
    {
        std::string why, detail;
        int shapes = 0;
        for (int32_t base : {0, 64, 4032})
            for (std::size_t nsl : {(std::size_t) 1, (std::size_t) 9, (std::size_t) 300})
                for (std::size_t bound = 1; bound <= nsl; ++bound)
                    for (int32_t piece : {1, 2, 3, 4, 7, 16}) {
                        ++shapes;
                        std::string w = run(base, bound, nsl, piece, detail);
                        if (!w.empty() && why.empty()) why = "region " + std::to_string(nsl) + " slots at " + std::to_string(base) + ", bound " + std::to_string(bound) + ", piece " + std::to_string(piece) + ": " + w;
                    }
        if (why.empty()) std::printf("CASE %s PASS: %d regions of 1 to 300 slots, pieces of 1 to 16, all tiling their region\n", "every region and piece size keeps the region bounds", shapes);
        else { std::printf("CASE %s FAIL: %s\n", "every region and piece size keeps the region bounds", why.c_str()); ++fails; }
    }
    return fails == 0 ? 0 : 1;
}
"""


def build_and_run(tmp):
    """Compile the case program against the tree's header and return {case: (ok, detail)}; None when it will not build."""
    src = os.path.join(tmp, 'refill_pieces_test.cpp')
    exe = os.path.join(tmp, 'refill_pieces_test')
    with open(src, 'w') as f:
        f.write(PROGRAM)
    cxx = shutil.which('c++') or shutil.which('clang++') or shutil.which('g++')
    if cxx is None:
        return None, 'no C++ compiler on this machine'
    b = subprocess.run([cxx, '-std=c++20', '-O1', '-I', os.path.join(ROOT, 'include'), '-o', exe, src],
                       capture_output=True, text=True)
    if b.returncode != 0:
        errs = [l.strip() for l in (b.stderr or b.stdout).splitlines() if 'error' in l]
        return None, 'the region arithmetic does not build: ' + (errs[0] if errs else ' '.join((b.stderr or b.stdout).strip().splitlines()[-3:]))[:600]
    r = subprocess.run([exe], capture_output=True, text=True)
    out = {}
    for line in (r.stdout + r.stderr).splitlines():
        if not line.startswith('CASE '):
            continue
        rest = line[len('CASE '):]
        for name in CASES:
            for mark in (' PASS: ', ' FAIL: '):
                if rest.startswith(name + mark):
                    out[name] = (mark.strip().strip(':') == 'PASS', rest[len(name) + len(mark):])
    return out, ''


def main():
    if not os.path.isfile(HEADER):
        got, err = None, f'the tree has no {HEADER} (run this from the repository root)'
    else:
        tmp = tempfile.mkdtemp(prefix='refill-pieces-')
        try:
            got, err = build_and_run(tmp)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    fails = 0
    for name in CASES:
        if got is None:
            ok, detail = False, err
        elif name not in got:
            ok, detail = False, 'the case program printed no result for it'
        else:
            ok, detail = got[name]
        print(f'{"PASS" if ok else "FAIL"} {name}' + ('' if ok else f': {detail}'))
        if not ok:
            fails += 1
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())
