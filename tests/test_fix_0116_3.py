"""tests/test_fix_0116_3.py - the d11 selftest survives a file smaller than one blob (fix-0116#3).

    python3 tests/test_fix_0116_3.py

src/core/expert_source.cpp decides the number of blobs once per selftest pass:

    const long nblobs = (long) (sz / (off_t) BL);

and then picks every read offset with

    const uint64_t off = (uint64_t) ((((long) rand() << 15) ^ rand()) % nblobs) * BL;

A file shorter than one blob, or an lseek that fails and returns -1, makes nblobs 0, so the
first offset divides by zero: undefined behaviour, a SIGFPE on x86 and a phantom read on a
host whose divide returns 0. The guard belongs between the two lines.

No GPU, model, server or hardware is touched. Each case compiles the real d11_selftest body,
copied out of src/core/expert_source.cpp in this tree, against stand-ins for MacPull, access,
unlink and the CPU expert layout (a 4 KiB blob). The one-pass loop keeps a pass finite; the
program prints jobs=<n> bad=<n> and exits non-zero when a read range leaves the file. It is
built with UndefinedBehaviorSanitizer and no recovery, so today's modulo-by-zero aborts the
program instead of silently reading blob 0, which is how the harm shows on Apple Silicon.

Every case prints one PASS or FAIL line; any FAIL exits non-zero.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path.cwd() if (Path.cwd() / "src" / "core" / "expert_source.cpp").is_file() else Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src" / "core" / "expert_source.cpp"
BL = 4096

CXX = shutil.which("clang++") or shutil.which("c++") or shutil.which("g++")
results: list[bool] = []

PREAMBLE = """\
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fcntl.h>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <utility>
#include <vector>

namespace strata { namespace kernels { namespace cpu {
struct Layout { std::uint64_t max_blob; };
inline Layout expert_layout() { return Layout{4096}; }
}}}
"""

STANDINS = """\
static long g_jobs = 0;
static long g_bad = 0;
static std::uint64_t g_size = 0;
struct MacPull {
    void run_to(int fd, const std::vector<std::pair<std::uint64_t, std::uint64_t>>& jobs,
                const std::vector<std::uint8_t*>& dst) {
        (void) fd;
        if (jobs.size() != dst.size()) ++g_bad;
        for (const auto& j : jobs) {
            ++g_jobs;
            if (j.first >= g_size) ++g_bad;   // a read that starts past EOF picks the wrong blob
        }
    }
};
inline MacPull& mac_pull() { static MacPull p; return p; }
static int fake_access(const char*, int) { return 0; }
static int fake_unlink(const char*) { return 0; }
#define access fake_access
#define unlink fake_unlink
#define usleep(x) ((void) (x))
"""

MAIN = """\
int main(int argc, char** argv) {
    if (argc != 2) return 2;
    struct stat st;
    if (::stat(argv[1], &st) == 0) g_size = (std::uint64_t) st.st_size;
    d11_selftest(std::string(argv[1]));
    std::printf("jobs=%ld bad=%ld\\n", g_jobs, g_bad);
    return g_bad == 0 ? 0 : 1;
}
"""


def case(name, ok, detail=""):
    results.append(bool(ok))
    print(("PASS " if ok else "FAIL ") + name + (f": {detail}" if detail else ""))


def extract_selftest(text):
    """Return the real d11_selftest body with its endless loop turned into one pass."""
    lines = text.splitlines()
    try:
        start = next(i for i, line in enumerate(lines) if line.startswith("static void d11_selftest("))
    except StopIteration:
        return None
    end = next((i for i in range(start + 1, len(lines)) if lines[i] == "}"), None)
    if end is None:
        return None
    body = "\n".join(lines[start:end + 1])
    if "for (;;) {" not in body:
        return None
    return body.replace("for (;;) {", "for (int once = 1; once; once = 0) {", 1)


def first_line(text):
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    return lines[0][:240] if lines else ""


def build(tmp):
    text = SOURCE.read_text(encoding="utf-8", errors="replace")
    body = extract_selftest(text)
    if body is None:
        return None, f"d11_selftest not found in {SOURCE}"
    src = tmp / "harness.cpp"
    src.write_text(PREAMBLE + STANDINS + "\n" + body + "\n" + MAIN, encoding="utf-8")
    exe = tmp / "harness.bin"
    cmd = [CXX, "-std=c++17", "-O1", "-fsanitize=undefined", "-fno-sanitize-recover=all",
           str(src), "-o", str(exe)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        return None, first_line(r.stderr or r.stdout)
    return exe, ""


def run(exe, path):
    r = subprocess.run([str(exe), str(path)], capture_output=True, text=True)
    out = r.stdout.strip()
    jobs = None
    for word in out.split():
        if word.startswith("jobs="):
            jobs = int(word[5:])
    return r.returncode, jobs, (r.stdout + r.stderr).strip(), "runtime error" in (r.stderr + r.stdout)


def main():
    if not CXX:
        case("a C++ compiler is available", False, "no clang++, c++ or g++ on PATH")
        return 1
    if not SOURCE.is_file():
        case("the expert source is present", False, str(SOURCE))
        return 1
    with tempfile.TemporaryDirectory(prefix="fix-0116-3-") as tmpdir:
        tmp = Path(tmpdir)
        exe, err = build(tmp)
        if exe is None:
            case("the selftest harness builds", False, err or "build failed")
            return 1

        # A file smaller than one blob has no blob to pick. Today the first offset divides by
        # zero and the sanitizer aborts; with the guard the pass is skipped and the program exits 0.
        small = [
            ("a zero-byte file skips the selftest", b""),
            ("a file smaller than one blob skips the selftest", bytes(range(100))),
        ]
        for i, (name, blob) in enumerate(small):
            path = tmp / f"small{i}.bin"
            path.write_bytes(blob)
            rc, jobs, text, ub = run(exe, path)
            ok = rc == 0 and not ub and jobs == 0
            case(name, ok, f"exit {rc}, jobs {jobs}, {'undefined behaviour' if ub else first_line(text)}")

        # An ordinary file of whole blobs gives the same answer before and after: every read range
        # sits inside the file and all 6720 offsets of a one-pass schedule are submitted.
        ordinary = [
            ("exactly one blob reads blob 0", BL),
            ("ten blobs read offsets inside the file", 10 * BL),
        ]
        for i, (name, size) in enumerate(ordinary):
            path = tmp / f"whole{i}.bin"
            path.write_bytes(bytes([65 + i]) * size)
            rc, jobs, text, ub = run(exe, path)
            ok = rc == 0 and not ub and jobs == 6720
            case(name, ok, f"exit {rc}, jobs {jobs}, {'undefined behaviour' if ub else first_line(text)}")

        # A real artifact in this tree (the expert profile the server loads) is larger than one stub
        # blob: the schedule is unchanged by the guard, before and after.
        sample = ROOT / "data" / "expert-profile.bin"
        if sample.is_file():
            rc, jobs, text, ub = run(exe, sample)
            ok = rc == 0 and not ub and jobs == 6720
            case("a real expert profile artifact keeps its schedule", ok,
                 f"exit {rc}, jobs {jobs}, {'undefined behaviour' if ub else first_line(text)}")

    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
