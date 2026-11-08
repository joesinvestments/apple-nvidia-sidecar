#!/usr/bin/env python3
"""A chunk that stop_at_chunk ends inside a lockstep group is a clean stop, not a group failure.

src/prefill/prefill.cpp:3631-3634: a first-stage chunk whose turn comes after the read has given way sets
stopped_at_ and returns true, before it takes any of its lockstep turns.  The AbortOnFail guard at 3618 sees ok
still false on the way out and aborts the group, which is what wakes the group's other chunks: their wait()
fails and they return false.  The group loop at 6920-6921 counts every one of those as a chunk failure, so the
read comes back false with "prefill: a chunk of the lockstep group failed" although every chunk stopped where
stop_at_chunk asked, and src/program/generate.cpp:18017 takes that as the batch's error instead of giving way at
stopped_at.  The group cannot go on after a chunk stops (its turns would never come), so the stop has to be told
apart from a real failure.

This test builds the real Lockstep, the chunk body's guard line and stop branch, and the group loop's join and
result block, lifted from src/prefill/prefill.cpp into a small program with fakes for the device, the prompt and
stop_at_chunk (no CUDA, no model, no weights), and runs the paths a group takes:
  - the stop ends a group cleanly: the read succeeds and reports stopped_at (FAILS today: a group failure);
  - an ordinary group read still reads every chunk;
  - a chunk that fails for a reason of its own still fails the group;
  - without a group the stop at a chunk boundary is unchanged;
  - a later stage's chunk is not stopped (its rows come from the stage before).

Run: python3 tests/test_fix_0266_0.py
Prints one PASS or FAIL line per case and exits non-zero on any FAIL.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

LINE = re.compile(r"case=(?P<case>\S+) read_ok=(?P<read_ok>-?\d+) stopped_at=(?P<stopped_at>-?\d+) "
                  r"err=(?P<err>.*)")

SRC_REL = Path("src") / "prefill" / "prefill.cpp"
LOCKSTEP_MARK = "struct Lockstep {"
GUARD_MARK = "struct AbortOnFail {"
STOP_MARK = "if (c0 > 0 && hand_in_ == nullptr && stop_at_chunk && stop_at_chunk()) {"
DRIVER_START = "std::vector<char> ok((size_t) gn, 0);"
DRIVER_END = "if (hand_in_ == nullptr && stopped_at_ >= 0) break;"

HARNESS = r"""// built by tests/test_fix_0266_0.py: the real Lockstep, the chunk body's guard and stop branch, and the
// group loop's result block, run against fakes for the device, the prompt and stop_at_chunk.
#include <algorithm>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace core { struct OnDevice { explicit OnDevice(void*) {} }; }

@LOCKSTEP@

struct Fake {   // the Prefill state the lifted code reads, with the device and the prompt faked
    struct Fake& m = *this;                // the lifted group block says m.device
    void* device = nullptr;
    const int32_t* hand_in_ = nullptr;      // a later stage's chunk: its rows come from the stage before
    std::function<bool()> stop_at_chunk;    // Prefill::stop_at_chunk
    int64_t stopped_at_ = -1;               // Prefill::stopped_at_
    int64_t Tc = 8;                         // the tokens of one chunk
    int64_t n = 24;                         // the prompt's tokens
    int64_t nl = 1;                         // the layers the group's turns cover
    int fail_chunk = -1;                    // this chunk fails for a reason of its own (no stop asked)
    std::string err;

    bool chunk_body(int64_t c0, Lockstep* ls, int lj) {
@GUARD@
        auto bind = [&](int64_t pass) -> bool {   // this chunk's turn, as the real body's bind does
            if (ls == nullptr) return true;
            return ls->wait(pass, lj);
        };
        if (!bind(0)) return false;
@STOP@
        if (lj == fail_chunk) { err = "planted: this chunk failed"; return false; }
        if (ls == nullptr) { lockstep_guard.ok = true; return true; }
        ls->done();                               // the setup turn
        for (int64_t step = 1; step <= ls->sTail(nl); ++step) {
            if (!ls->wait(step, lj)) return false;
            ls->done();
        }
        lockstep_guard.ok = true;
        return true;
    }

    bool run_group(int gn) {   // the group arm of the chunk loop; g0 = Tc: the group's first chunk is not the prompt's first
        for (int64_t g0 = Tc; g0 < n;) {
            Lockstep ls(gn, nl, 0, false);
@DRIVER@
            g0 += (int64_t) gn * Tc;
        }
        return true;
    }

    bool run_single() {   // the `if (gn == 1)` arm: no group and no lockstep
        for (int64_t g0 = 0; g0 < n;) {
            if (!chunk_body(g0, nullptr, 0)) return false;
            if (hand_in_ == nullptr && stopped_at_ >= 0) break;
            g0 += Tc;
        }
        return true;
    }
};

int main(int argc, char** argv) {
    const std::string which = argc > 1 ? argv[1] : "";
    int32_t rows[8] = {0};
    Fake f;
    bool read_ok = true;
    if (which == "group-stop") {            // a BYIELD ends the read at the group's first chunk
        f.stop_at_chunk = [] { return true; };
        read_ok = f.run_group(3);
    } else if (which == "group-plain") {    // an ordinary group read: nothing stops it
        read_ok = f.run_group(3);
    } else if (which == "group-fail") {     // a chunk that fails for a reason of its own
        f.fail_chunk = 1;
        read_ok = f.run_group(3);
    } else if (which == "single-stop") {    // no group: the stop at a chunk boundary
        f.stop_at_chunk = [] { return true; };
        read_ok = f.run_single();
    } else if (which == "stage-stop") {     // a later stage's chunk: the stop does not apply to it
        f.stop_at_chunk = [] { return true; };
        f.hand_in_ = rows;
        read_ok = f.run_group(3);
    } else {
        std::printf("case=%s unknown\n", which.c_str());
        return 2;
    }
    std::printf("case=%s read_ok=%d stopped_at=%lld err=%s\n", which.c_str(), read_ok ? 1 : 0,
                (long long) f.stopped_at_, f.err.empty() ? "-" : f.err.c_str());
    return 0;
}
"""

CASES = [
    # name, the group failure today, and what the read must report once a stopped chunk is not a failure
    ("group-stop", dict(read_ok=0, stopped_at=8, err="prefill: a chunk of the lockstep group failed"),
     dict(read_ok=1, stopped_at=8, err="-"),
     "a stop ends the group's read cleanly at the chunk it stopped at"),
    ("group-plain", dict(read_ok=1, stopped_at=-1, err="-"), dict(read_ok=1, stopped_at=-1, err="-"),
     "an ordinary group read reads every chunk and stops nowhere"),
    ("group-fail", dict(read_ok=0, stopped_at=-1, err="planted: this chunk failed"),
     dict(read_ok=0, stopped_at=-1, err="planted: this chunk failed"),
     "a chunk that fails for its own reason still fails the group"),
    ("single-stop", dict(read_ok=1, stopped_at=8, err="-"), dict(read_ok=1, stopped_at=8, err="-"),
     "the stop at a chunk boundary without a group is unchanged"),
    ("stage-stop", dict(read_ok=1, stopped_at=-1, err="-"), dict(read_ok=1, stopped_at=-1, err="-"),
     "a later stage's chunk keeps reading when stop_at_chunk says stop"),
]


def find_root() -> Path:
    """The tree under test: the copy prove_test runs us in, else the one this file sits in."""
    for base in (Path.cwd(), Path(__file__).resolve().parent.parent):
        if (base / SRC_REL).is_file():
            return base
    return Path.cwd()


def block_from(lines, start):
    """From lines[start] to the line that closes its first brace, comments and literals skipped."""
    depth, seen, out, in_comment = 0, False, [], False
    for line in lines[start:]:
        out.append(line)
        i, n = 0, len(line)
        while i < n:
            if in_comment:
                j = line.find("*/", i)
                if j < 0:
                    i = n
                    continue
                in_comment, i = False, j + 2
                continue
            if line.startswith("//", i):
                break
            if line.startswith("/*", i):
                in_comment, i = True, i + 2
                continue
            c = line[i]
            if c in "\"'":
                quote, i = c, i + 1
                while i < n:
                    if line[i] == "\\":
                        i += 2
                        continue
                    if line[i] == quote:
                        i += 1
                        break
                    i += 1
                continue
            if c == "{":
                depth, seen = depth + 1, True
            elif c == "}":
                depth -= 1
                if seen and depth == 0:
                    return "\n".join(out)
            i += 1
    return None


def line_of(lines, marker):
    for i, line in enumerate(lines):
        if marker in line:
            return i
    return None


def lifted(text):
    """The four pieces of src/prefill/prefill.cpp this test runs, or None with the piece that is gone."""
    lines = text.splitlines()
    pieces = {}
    at = line_of(lines, LOCKSTEP_MARK)
    if at is None:
        return None, LOCKSTEP_MARK
    pieces["LOCKSTEP"] = block_from(lines, at)
    at = line_of(lines, GUARD_MARK)
    if at is None:
        return None, GUARD_MARK
    pieces["GUARD"] = lines[at].strip()
    at = line_of(lines, STOP_MARK)
    if at is None:
        return None, STOP_MARK
    pieces["STOP"] = block_from(lines, at)
    at = line_of(lines, DRIVER_START)
    if at is None:
        return None, DRIVER_START
    end = None
    for i in range(at, len(lines)):
        if lines[i].strip() == DRIVER_END:
            end = i
            break
    if end is None:
        return None, DRIVER_END
    pieces["DRIVER"] = "\n".join(lines[at:end + 1])
    for key, value in pieces.items():
        if not value:
            return None, key
    return pieces, ""


def compiler():
    for name in ("g++", "clang++", "c++"):
        for path in ("/usr/bin", "/usr/local/bin", "/opt/homebrew/bin"):
            exe = os.path.join(path, name)
            if os.access(exe, os.X_OK):
                return exe
        for path in os.environ.get("PATH", "").split(os.pathsep):
            exe = os.path.join(path, name)
            if path and os.access(exe, os.X_OK):
                return exe
    return None


def build(work, pieces, cxx):
    program = HARNESS
    for key, value in pieces.items():
        program = program.replace("@" + key + "@", value)
    src = os.path.join(work, "chunk_body.cpp")
    exe = os.path.join(work, "chunk_body")
    with open(src, "w", encoding="utf-8") as f:
        f.write(program)
    built = subprocess.run([cxx, "-std=c++17", "-O0", "-pthread", "-o", exe, src],
                           capture_output=True, text=True, timeout=180)
    if built.returncode != 0:
        errors = [l.strip() for l in (built.stdout + built.stderr).splitlines() if " error: " in l]
        return None, errors[0] if errors else (built.stdout + built.stderr).strip()[-300:]
    return exe, ""


def run_case(exe, name):
    """-> the fields the built program printed, or None and why not."""
    try:
        done = subprocess.run([exe, name], capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired:
        return None, "the run hung: a chunk waited for turns that never came"
    line = done.stdout.strip().splitlines()
    if done.returncode != 0 or not line:
        return None, f"the run exited {done.returncode}: {(done.stdout + done.stderr).strip()[-200:]}"
    found = LINE.fullmatch(line[-1].strip())
    if found is None or found.group("case") != name:
        return None, f"the run printed {line[-1]!r}"
    return dict(read_ok=int(found.group("read_ok")), stopped_at=int(found.group("stopped_at")),
                err=found.group("err")), ""


def main() -> int:
    fails = 0

    def case(name, ok, detail=""):
        nonlocal fails
        print(f"{'PASS' if ok else 'FAIL'} {name}" + ("" if ok else f": {detail}"))
        if not ok:
            fails += 1

    root = find_root()
    source = root / SRC_REL
    if not source.is_file():
        case("the tree under test has its prefill source", False, f"{source} is missing")
        return 1
    pieces, missing = lifted(source.read_text(encoding="utf-8"))
    if pieces is None:
        case("the prefill chunk body is where this test reads it", False, f"no {missing!r} in {SRC_REL}")
        return 1

    cxx = compiler()
    if cxx is None:
        case("a C++ compiler builds the fakes", False, "no g++, clang++ or c++ on PATH")
        return 1
    work = tempfile.mkdtemp(prefix="sce0266-")
    exe, why = build(work, pieces, cxx)
    if exe is None:
        case("the lifted chunk body builds", False, why)
        return 1

    for name, today, fixed, title in CASES:
        fields, why = run_case(exe, name)
        if fields is None:
            case(title, False, why)
            continue
        got = {k: fields[k] for k in ("read_ok", "stopped_at", "err")}
        if got == fixed:
            case(title, True)
        elif got == today:
            case(title, False, f"today's code: {got}, want {fixed}")
        else:
            case(title, False, f"got {got}, want {fixed}")
    shutil.rmtree(work, ignore_errors=True)
    print(f"{fails} failure(s)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
