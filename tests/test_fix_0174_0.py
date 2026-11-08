#!/usr/bin/env python3
"""Verifier::refresh_ar must scan every layer when the stage says "to the last" (-1).

src/core/verify.cpp: the per-layer zero-doorbell branch loops `for (int64_t l = lb_; l < le_ && !off; ++l)`,
but le_ is a sentinel: the header (`int64_t lb_ = 0, le_ = -1;  // set_stage ... (-1: to the last)`) and the
function's own all-resident path (`le_ < 0 ? g_->n_layers : le_`) both say -1 means every layer.  With le_ == -1
the per-layer loop body never runs, so a layer whose routable expert left VRAM is not seen: off stays false,
ar_off_ is forced false, and the next window runs the zero-doorbell graph while experts are out of VRAM.  The
fix bounds the loop by `lend = (le_ < 0 ? g_->n_layers : le_)`.

No GPU, model or weights here: the cases compile the real refresh_ar body (extracted from the tree under test,
the same way test_tgcu_returns_driver_int.py compiles the real header against a stub driver) against a small
fake Verifier, then run it on planted residency tables.  The sentinel cases are the harm; the explicit-range
cases are ordinary input and must read the same before and after the fix.

Run: python3 tests/test_fix_0174_0.py
Prints one PASS or FAIL line per case and exits nonzero on any FAIL.
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

SIGNATURE = "void Verifier::refresh_ar() {"


def find_root():
    """The tree under test: prove_test runs us with cwd at the copy root, else use the tree we sit in."""
    rel = Path("src") / "core" / "verify.cpp"
    for base in (Path.cwd(), Path(__file__).resolve().parent.parent):
        if (base / rel).is_file():
            return base
    return Path.cwd()


def extract_body(src, signature):
    """The text between the function's braces, with string, character and comment text skipped."""
    start = src.find(signature)
    if start < 0:
        raise ValueError("no " + signature + " in the tree under test")
    i = src.index("{", start)
    depth = 0
    k = i
    in_str = in_chr = in_line = in_block = False
    while k < len(src):
        c = src[k]
        nxt = src[k + 1] if k + 1 < len(src) else ""
        if in_line:
            if c == "\n":
                in_line = False
        elif in_block:
            if c == "*" and nxt == "/":
                in_block = False
                k += 1
        elif in_str:
            if c == "\\":
                k += 1
            elif c == '"':
                in_str = False
        elif in_chr:
            if c == "\\":
                k += 1
            elif c == "'":
                in_chr = False
        elif c == "/" and nxt == "/":
            in_line = True
            k += 1
        elif c == "/" and nxt == "*":
            in_block = True
            k += 1
        elif c == '"':
            in_str = True
        elif c == "'":
            in_chr = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return src[i + 1:k]
        k += 1
    raise ValueError("refresh_ar has no closing brace")


# name, lyar_, all_resident_, ar_off_ at entry, EXL3 layout, resident-only routing,
# n_layers, n_expert, lb_, le_, host residency table, ar_l_ table, keep mask, expected ar_off_
CASES = [
    # The harm: the documented "to the last" sentinel.  A miss in the last layer must turn the graph off.
    {"name": "sentinel_miss_found", "lyar": True, "all_res": False, "off0": False, "exl3": True,
     "resident_only": False, "n_layers": 4, "n_expert": 2, "lb": 0, "le": -1,
     "res": [0, 0, 0, 0, 0, 0, -1, 0], "ar_l": [1, 1, 1, 1], "keep": [], "want": True},
    # The harm on an already-off graph: the sentinel scan must keep it off while a layer still has a miss.
    {"name": "sentinel_keeps_off_on_miss", "lyar": True, "all_res": False, "off0": True, "exl3": True,
     "resident_only": False, "n_layers": 4, "n_expert": 2, "lb": 0, "le": -1,
     "res": [-1, 0, 0, 0, 0, 0, 0, 0], "ar_l": [1, 1, 1, 1], "keep": [], "want": True},
    # A miss on an expert the keep mask excludes is not a miss, on the sentinel path too.
    {"name": "sentinel_masked_miss_ignored", "lyar": True, "all_res": False, "off0": False, "exl3": True,
     "resident_only": False, "n_layers": 4, "n_expert": 2, "lb": 0, "le": -1,
     "res": [0, 0, -1, 0, 0, 0, 0, 0], "ar_l": [1, 1, 1, 1],
     "keep": [0, 0, -1, 0, 0, 0, 0, 0], "want": False},
    # Ordinary input, explicit range: the same answer before and after the fix.
    {"name": "explicit_range_miss_found", "lyar": True, "all_res": False, "off0": False, "exl3": True,
     "resident_only": False, "n_layers": 4, "n_expert": 2, "lb": 1, "le": 3,
     "res": [0, 0, 0, 0, 0, -1, 0, 0], "ar_l": [1, 1, 1, 1], "keep": [], "want": True},
    {"name": "explicit_range_outside_ignored", "lyar": True, "all_res": False, "off0": False, "exl3": True,
     "resident_only": False, "n_layers": 4, "n_expert": 2, "lb": 1, "le": 3,
     "res": [-1, 0, 0, 0, 0, 0, -1, 0], "ar_l": [1, 1, 1, 1], "keep": [], "want": False},
    # The function's all-resident path already reads the sentinel; that ordinary answer must not change.
    {"name": "non_lyar_sentinel_scans", "lyar": False, "all_res": True, "off0": False, "exl3": False,
     "resident_only": False, "n_layers": 4, "n_expert": 2, "lb": 0, "le": -1,
     "res": [0, 0, 0, 0, 0, 0, -1, 0], "ar_l": [], "keep": [], "want": True},
]


HARNESS_HEAD = r"""
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

struct ModelGeometry {
    int64_t n_layers = 0;
    int64_t n_expert = 0;
};
struct Exl3Gpu {
    std::vector<int32_t> keep_host;
};
static Exl3Gpu g_xg;
static bool g_xg_present = false;
static bool g_resident_only = false;
static Exl3Gpu* exl3_gpu(std::string&) { return g_xg_present ? &g_xg : nullptr; }
static bool exl3_resident_only() { return g_resident_only; }

namespace strata { namespace kernels { namespace cpu {
struct ExpertLayout { bool exl3 = false; };
static ExpertLayout g_expert_layout;
inline const ExpertLayout& expert_layout() { return g_expert_layout; }
} } }

struct Verifier {
    bool lyar_ = false;
    bool all_resident_ = false;
    bool ar_off_ = false;
    const ModelGeometry* g_ = nullptr;
    const int32_t* h_res_ = nullptr;
    std::vector<uint8_t> ar_l_;
    int64_t lb_ = 0, le_ = -1;

    void refresh_ar() {
"""

HARNESS_TAIL = r"""
    }
};

struct TestCase {
    const char* name;
    bool lyar, all_res, off0, exl3, resident_only;
    int64_t n_layers, n_expert, lb, le;
    std::vector<int32_t> res;
    std::vector<uint8_t> ar_l;
    std::vector<int32_t> keep;
};

int main() {
    std::vector<TestCase> cases = {
%s
    };
    for (auto& c : cases) {
        strata::kernels::cpu::g_expert_layout.exl3 = c.exl3;
        g_resident_only = c.resident_only;
        g_xg_present = !c.keep.empty();
        g_xg.keep_host = c.keep;
        ModelGeometry geom;
        geom.n_layers = c.n_layers;
        geom.n_expert = c.n_expert;
        Verifier v;
        v.lyar_ = c.lyar;
        v.all_resident_ = c.all_res;
        v.ar_off_ = c.off0;
        v.g_ = &geom;
        v.h_res_ = c.res.empty() ? nullptr : c.res.data();
        v.ar_l_ = c.ar_l;
        v.lb_ = c.lb;
        v.le_ = c.le;
        v.refresh_ar();
        std::printf("%%s %%d\n", c.name, v.ar_off_ ? 1 : 0);
    }
    return 0;
}
"""


def cpp_int_list(values):
    return "{" + ", ".join(str(int(v)) for v in values) + "}"


def cpp_case(c):
    return ('        {"%s", %s, %s, %s, %s, %s, %d, %d, %d, %d, %s, %s, %s},'
            % (c["name"], str(c["lyar"]).lower(), str(c["all_res"]).lower(), str(c["off0"]).lower(),
               str(c["exl3"]).lower(), str(c["resident_only"]).lower(), c["n_layers"], c["n_expert"],
               c["lb"], c["le"], cpp_int_list(c["res"]), cpp_int_list(c["ar_l"]), cpp_int_list(c["keep"])))


def build_harness(root, body, work):
    source = HARNESS_HEAD + body + (HARNESS_TAIL % "\n".join(cpp_case(c) for c in CASES))
    cpp = os.path.join(work, "refresh_ar_harness.cpp")
    exe = os.path.join(work, "refresh_ar_harness")
    with open(cpp, "w", encoding="utf-8") as f:
        f.write(source)
    cxx = os.environ.get("CXX") or shutil.which("c++") or shutil.which("clang++") or shutil.which("g++")
    if cxx is None:
        raise RuntimeError("no C++ compiler (c++, clang++, g++) on PATH")
    r = subprocess.run([cxx, "-std=c++17", "-O0", "-w", "-o", exe, cpp], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("the harness did not compile: " + (r.stdout + r.stderr).strip()[-600:])
    return exe


def main():
    root = find_root()
    src_path = root / "src" / "core" / "verify.cpp"
    results = []

    def case(name, ok, detail=""):
        results.append(bool(ok))
        print(("PASS " if ok else "FAIL ") + name + ("" if ok else ": " + detail))

    if not src_path.is_file():
        print("FAIL tree: " + str(src_path) + " is missing")
        return 1
    try:
        body = extract_body(src_path.read_text(encoding="utf-8"), SIGNATURE)
    except ValueError as exc:
        print("FAIL extract: " + str(exc))
        return 1

    work = tempfile.mkdtemp(prefix="fix-0174-")
    try:
        exe = build_harness(root, body, work)
        run = subprocess.run([exe], capture_output=True, text=True, timeout=60)
    except (RuntimeError, subprocess.TimeoutExpired) as exc:
        print("FAIL build: " + str(exc))
        return 1
    finally:
        shutil.rmtree(work, ignore_errors=True)

    got = {}
    for line in (run.stdout + run.stderr).splitlines():
        m = re.fullmatch(r"([A-Za-z0-9_]+) ([01])", line.strip())
        if m:
            got[m.group(1)] = m.group(2) == "1"
    if run.returncode != 0 or len(got) != len(CASES):
        print("FAIL run: exit %d, parsed %d of %d case lines: %s"
              % (run.returncode, len(got), len(CASES), (run.stdout + run.stderr).strip()[-400:]))
        return 1

    for c in CASES:
        if c["name"] not in got:
            case(c["name"], False, "the harness printed no result")
            continue
        case(c["name"], got[c["name"]] == c["want"],
             "ar_off_=%s, expected %s" % (got[c["name"]], c["want"]))

    failed = results.count(False)
    print("%d of %d cases pass" % (len(results) - failed, len(results)))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
