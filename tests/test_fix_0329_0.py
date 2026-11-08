#!/usr/bin/env python3
"""The batch_mtp admission sync reports a failed sync instead of a silent false.

src/program/generate.cpp:14336, in copy_to_slot's MTP branch, the draft-KV copy is followed by
    if (cudaDeviceSynchronize() != cudaSuccess) return false;
with the cudaError_t thrown away and `e` untouched, so an async fault from the copy above is
reported to the caller as a plain failure with an empty message (the caller prints
"strata serve: batch admission failed: %s" at line 19938 and clears the slot's cache).  Every
other sync check in the same lambda (14289, 14306, 14311) sets `e` from a descriptive message.

This test takes the MTP branch itself out of the source, compiles it unchanged as host C++ against
a fake CUDA runtime and fake drafters, and runs five admissions through it:
  - a sync fault on the device to device path and one on the host image path must come back with
    `e` naming the fault cudaGetErrorString(status) reports for the status the sync returned;
  - an ordinary admission, a failed K/V copy and a busy slot drafter give the same answer before
    and after the fix (the fix only adds the missing message on the sync failure).

Run: python3 tests/test_fix_0329_0.py
Prints one PASS or FAIL line per case and exits non-zero on any FAIL.  No network, no GPU, no model.
"""
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

SRC_REL = Path("src") / "program" / "generate.cpp"
LAMBDA = "auto copy_to_slot = [&]"
BRANCH = "if (batch_mtp) {"


def find_root():
    """The tree under test: the copy prove_test runs us in, else the one this file sits in."""
    for base in (Path.cwd(), Path(__file__).resolve().parent.parent):
        if (base / SRC_REL).is_file():
            return base
    return Path.cwd()


ROOT = find_root()
SRC = ROOT / SRC_REL

# The fake CUDA runtime and the fake drafters the MTP branch is run against, in front of @@BLOCK@@.
HARNESS = r"""
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

// cudaDeviceSynchronize's fault is planted; cudaGetErrorString names the status it returned.
enum cudaError { cudaSuccess = 0, cudaErrorIllegalAddress = 700, cudaErrorLaunchFailure = 719 };
typedef enum cudaError cudaError_t;

static cudaError_t fake_sync_status = cudaSuccess;
static int fake_sync_calls = 0;

static cudaError_t cudaDeviceSynchronize() { ++fake_sync_calls; return fake_sync_status; }
static const char* cudaGetErrorString(cudaError_t e) {
    switch (e) {
        case cudaSuccess: return "no error";
        case cudaErrorIllegalAddress: return "an illegal memory address was encountered";
        case cudaErrorLaunchFailure: return "unspecified launch failure";
    }
    return "unknown error";
}

namespace strata { namespace core {
struct OnDevice { explicit OnDevice(int d = 0) : dev(d) {} int dev; };
struct KvState { int kv_mode = 0; };
struct ConversationKv { };
struct KvGeometry { };
}}

static bool fake_copy_ok = true;
static std::string fake_copy_msg;

namespace strata { namespace core {
inline bool fake_copy(std::string& e) { if (!fake_copy_ok) e = fake_copy_msg; return fake_copy_ok; }
template <class From, class To, class Geom, class N, class Err>
bool conversation_kv_copy_device(From&, To&, const Geom&, N, bool, Err& e) { return fake_copy(e); }
template <class Image, class From, class Geom, class N, class Err>
bool conversation_kv_save(Image&, From&, const Geom&, N, bool, Err& e) { return fake_copy(e); }
template <class Image, class To, class Geom, class N, class Err>
bool conversation_kv_restore(Image&, To&, const Geom&, N, bool, Err& e) { return fake_copy(e); }
}}

struct FakeDrafter {
    strata::core::KvState kv;
    int dev = 0;
    int64_t prompt_len = -1;
    bool idle_ok = true;
    std::string idle_msg;
    int device() const { return dev; }
    bool idle(std::string& e) { if (!idle_ok) e = idle_msg; return idle_ok; }
    const strata::core::KvState& kv_state() const { return kv; }
    strata::core::KvState& kv_state() { return kv; }
    void set_prompt_len(int64_t n) { prompt_len = n; }
};

static FakeDrafter mtp;                       // the solo drafter the admission copies from
static FakeDrafter slot0;                     // slot 0's drafter, the copy's target
static std::vector<FakeDrafter*> slot_mtp = {&slot0};
static const bool batch_mtp = true;
static strata::core::KvGeometry draft_geometry;

// copy_to_slot's MTP branch, unchanged, with the loop's b, upto and e in scope.
static bool admit(int b, int64_t upto, std::string& e) {
@@BLOCK@@
    return true;
}

static void run(const char* name, bool host_path, cudaError_t sync_status, bool copy_ok,
                const char* copy_msg, bool slot_idle, int64_t upto) {
    mtp = FakeDrafter();
    slot0 = FakeDrafter();
    mtp.kv.kv_mode = host_path ? 1 : 0;
    slot0.kv.kv_mode = host_path ? 1 : 0;
    slot0.idle_ok = slot_idle;
    slot0.idle_msg = "the slot's drafter is busy";
    fake_sync_status = sync_status;
    fake_sync_calls = 0;
    fake_copy_ok = copy_ok;
    fake_copy_msg = copy_msg != nullptr ? copy_msg : "";
    std::string e;
    const bool ret = admit(0, upto, e);
    std::printf("CASE %s\nRET %d\nSYNC %d\nPROMPT %lld\nERR %s\n", name, ret ? 1 : 0, fake_sync_calls,
                (long long) slot0.prompt_len, e.c_str());
    std::fflush(stdout);
}

int main() {
    run("sync fault on the device copy path", false, cudaErrorIllegalAddress, true, "", true, 5);
    run("sync fault on the host image path", true, cudaErrorLaunchFailure, true, "", true, 7);
    run("ordinary admission", false, cudaSuccess, true, "", true, 5);
    run("a failed K/V copy keeps its message", false, cudaSuccess, false,
        "conversation kv copy: the device copy failed", true, 5);
    run("a busy slot drafter keeps its message", false, cudaSuccess, true, "", false, 5);
    return 0;
}
"""

ILLEGAL = "an illegal memory address was encountered"
LAUNCH = "unspecified launch failure"


def fail(msg):
    print("FAIL " + msg)
    return False


def ok(msg):
    print("PASS " + msg)
    return True


def extract_block(text):
    """The MTP branch of copy_to_slot, braces included, or a raise naming what is missing."""
    start = text.find(LAMBDA)
    if start < 0:
        raise ValueError("copy_to_slot is not in " + str(SRC_REL))
    open_at = text.find(BRANCH, start)
    if open_at < 0:
        raise ValueError("copy_to_slot has no " + BRANCH + " branch")
    i = text.index("{", open_at)
    depth = 0
    while i < len(text):
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[open_at:i + 1]
        i += 1
    raise ValueError("the " + BRANCH + " branch is not closed")


def build(block, tmp):
    cpp = Path(tmp) / "admit.cpp"
    cpp.write_text(HARNESS.replace("@@BLOCK@@", block), encoding="utf-8")
    exe = Path(tmp) / "admit"
    cxx = shutil.which("clang++") or shutil.which("c++") or shutil.which("g++")
    if cxx is None:
        raise RuntimeError("no C++ compiler")
    r = subprocess.run([cxx, "-std=c++17", "-O1", "-w", str(cpp), "-o", str(exe)],
                       capture_output=True, text=True, timeout=300)
    if r.returncode != 0:
        raise RuntimeError("the admission block does not compile against the fake runtime: "
                           + (r.stderr or r.stdout).strip()[-800:])
    return exe


def parse(out):
    cases = {}
    lines = out.splitlines()
    i = 0
    while i < len(lines):
        if not lines[i].startswith("CASE "):
            i += 1
            continue
        name = lines[i][5:]
        fields = {}
        i += 1
        while i < len(lines) and not lines[i].startswith("CASE "):
            line = lines[i]
            if line == "ERR":
                fields["err"] = ""
            elif line.startswith("ERR "):
                fields["err"] = line[4:]
            else:
                k, _, v = line.partition(" ")
                if k and v:
                    fields[k] = v
            i += 1
        cases[name] = fields
    return cases


def expect(results, name, cases, ret=None, sync=None, prompt=None, err=None, err_has=None):
    got = cases.get(name)
    if got is None:
        results.append(fail(name + ": the case did not run"))
        return
    problems = []
    if ret is not None and got.get("RET") != str(ret):
        problems.append("returned " + got.get("RET", "?") + ", want " + str(ret))
    if sync is not None and got.get("SYNC") != str(sync):
        problems.append("sync called " + got.get("SYNC", "?") + " times, want " + str(sync))
    if prompt is not None and got.get("PROMPT") != str(prompt):
        problems.append("set the prompt length to " + got.get("PROMPT", "?") + ", want " + str(prompt))
    text = got.get("err", "")
    if err is not None and text != err:
        problems.append("reported " + (repr(text) if text else "no message") + ", want " + repr(err))
    if err_has is not None and err_has not in text:
        problems.append("reported " + (repr(text) if text else "no message")
                        + ", which does not name the fault " + repr(err_has))
    if problems:
        results.append(fail(name + ": " + "; ".join(problems)))
    else:
        results.append(ok(name + (": " + text if text else "")))


def main():
    results = []
    if not SRC.is_file():
        return [fail(str(SRC_REL) + " is missing")]
    try:
        block = extract_block(SRC.read_text(encoding="utf-8", errors="replace"))
    except ValueError as e:
        return [fail(str(e))]
    results.append(ok("read copy_to_slot's MTP branch from src/program/generate.cpp (%d lines)"
                      % (block.count("\n") + 1)))
    tmp = tempfile.mkdtemp(prefix="fix-0329-0-")
    try:
        try:
            exe = build(block, tmp)
        except (RuntimeError, subprocess.TimeoutExpired) as e:
            return [fail(str(e))]
        r = subprocess.run([str(exe)], capture_output=True, text=True, timeout=120)
        if r.returncode != 0:
            return [fail("the admissions did not run: " + (r.stdout + r.stderr).strip()[-500:])]
        cases = parse(r.stdout)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    expect(results, "sync fault on the device copy path", cases, ret=0, sync=1, prompt=-1, err_has=ILLEGAL)
    expect(results, "sync fault on the host image path", cases, ret=0, prompt=-1, err_has=LAUNCH)
    expect(results, "ordinary admission", cases, ret=1, prompt=5, err="")
    expect(results, "a failed K/V copy keeps its message", cases, ret=0, sync=0,
           err="conversation kv copy: the device copy failed")
    expect(results, "a busy slot drafter keeps its message", cases, ret=0,
           err="the slot's drafter is busy")
    return results


if __name__ == "__main__":
    out = main()
    sys.exit(0 if all(out) else 1)
