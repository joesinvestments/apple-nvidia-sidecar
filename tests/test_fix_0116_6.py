#!/usr/bin/env python3
"""tests/test_fix_0116_6.py - mac_pull_fd must open the path it is given, not the first path it saw.

src/core/expert_source.cpp:2434 caches the experts file descriptor in a function static:
    static int fd = -1;
    if (fd < 0) {
        fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    }
    if (g_pull_path.empty()) g_pull_path = path;
so the descriptor is opened once for the process.  A later call with a different expert file returns the first
file's descriptor and leaves g_pull_path on the first path: every mac_pull read, and every per-thread reopen in
use_fd (it opens g_pull_path when the fd matches g_pull_fd_main), then targets the wrong file.  No caller guards
paths_.front() against changing between calls.

This test compiles the real mac_pull_fd from the tree under test against the two statics it writes (g_pull_path
and g_pull_fd_main) and calls it with real files in a temp folder: no GPU, no model, no weights, no pack.
  - one path still opens and reads that file, and the same path twice keeps the first descriptor (ordinary input,
    the same before and after the change);
  - a second call with a different file must return a descriptor that reads the second file (today it reads the
    first file's bytes);
  - a third path after two others must read that file as well (today the first file again);
  - a call for a missing file must fail instead of handing back the first file's descriptor, and a later call for
    a real file must recover;
  - g_pull_path must follow the path the descriptor was opened for (use_fd reopens that path per thread).
Prints one PASS or FAIL line per case and exits nonzero on any FAIL.  Run from the copy root:
python3 tests/test_fix_0116_6.py
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile

SRC_REL = os.path.join("src", "core", "expert_source.cpp")
SIG = "int mac_pull_fd(const std::string& path)"

HARNESS = r"""
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <mutex>
#include <string>
#include <unistd.h>

#ifndef O_CLOEXEC
#define O_CLOEXEC 0
#endif
#ifndef F_NOCACHE
#define F_NOCACHE 48
#endif

// the two process-wide statics mac_pull_fd writes, as src/core/expert_source.cpp declares them
static std::string g_pull_path;
static std::atomic<int> g_pull_fd_main{-1};

@@FUNC@@

static std::string read_at(int fd) {
    char buf[512];
    if (fd < 0) return "<no fd>";
    const ssize_t n = ::pread(fd, buf, sizeof buf - 1, 0);
    if (n < 0) return "<pread failed>";
    buf[n] = 0;
    std::string s(buf);
    while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) s.pop_back();
    return s;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <label> <path>...\n", argv[0]);
        return 2;
    }
    std::printf("LABEL %s\n", argv[1]);
    for (int i = 2; i < argc; ++i) {
        const int fd = mac_pull_fd(argv[i]);
        std::printf("CALL %d %d %s\n", i - 2, fd, read_at(fd).c_str());
    }
    std::printf("PATH %s\n", g_pull_path.c_str());
    std::printf("MAIN %d\n", g_pull_fd_main.load());
    return 0;
}
"""


class Fails:
    def __init__(self):
        self.count = 0

    def check(self, name, ok, detail=""):
        print(("PASS " if ok else "FAIL ") + name + ("" if ok else ": " + detail))
        if not ok:
            self.count += 1


def find_root():
    """The tree under test: the copy prove_test runs us in, else the one this file sits in."""
    for base in (os.getcwd(), os.path.dirname(os.path.dirname(os.path.abspath(__file__)))):
        if os.path.isfile(os.path.join(base, SRC_REL)):
            return base
    return os.getcwd()


def brace_end(text, open_index):
    depth = 0
    for i in range(open_index, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return i + 1
    return None


def mac_pull_fd_text(src):
    """The whole mac_pull_fd definition, exactly as the tree writes it."""
    a = src.find(SIG)
    if a < 0:
        return None, "mac_pull_fd is gone from " + SRC_REL
    a = src.rfind("\n", 0, a) + 1
    body = src.find("{", src.find(")", a))
    if body < 0:
        return None, "mac_pull_fd has no body in " + SRC_REL
    end = brace_end(src, body)
    if end is None:
        return None, "mac_pull_fd has no closing brace in " + SRC_REL
    return src[a:end], None


def compiler():
    for name in (os.environ.get("CXX"), "clang++", "c++", "g++"):
        if name and shutil.which(name):
            return name
    return None


def parse(out):
    """{label, calls: [(fd, content)], path} from the harness's lines."""
    got = {"label": None, "calls": [], "path": None, "main": None}
    for line in out.splitlines():
        if line.startswith("CALL "):
            rest = line[len("CALL "):]
            i, _, rest = rest.partition(" ")
            fd, _, content = rest.partition(" ")
            if not i.isdigit() or not re.fullmatch(r"-?\d+", fd):
                return None
            got["calls"].append((int(fd), content))
        elif line.startswith("PATH "):
            got["path"] = line[len("PATH "):]
        elif line.startswith("MAIN "):
            got["main"] = line[len("MAIN "):].strip()
    return got


def main():
    fails = Fails()
    root = find_root()
    src_path = os.path.join(root, SRC_REL)
    if not os.path.isfile(src_path):
        fails.check("the tree holds " + SRC_REL, False, src_path + " is missing")
        return 1
    with open(src_path, encoding="utf-8") as f:
        src = f.read()
    func, err = mac_pull_fd_text(src)
    if func is None:
        fails.check("the mac_pull_fd descriptor cache is readable", False, err)
        return 1
    fails.check("the mac_pull_fd descriptor cache is readable", True)

    cxx = compiler()
    if cxx is None:
        fails.check("a C++ compiler is available", False, "no clang++, c++ or g++ on PATH")
        return 1

    work = tempfile.mkdtemp(prefix="fix-0116-6-")
    try:
        cpp = os.path.join(work, "pull_fd.cpp")
        exe = os.path.join(work, "pull_fd")
        with open(cpp, "w", encoding="utf-8") as f:
            f.write(HARNESS.replace("@@FUNC@@", func))
        build = subprocess.run([cxx, "-std=c++17", "-O0", "-o", exe, cpp],
                               capture_output=True, text=True, timeout=180)
        if build.returncode != 0:
            detail = [l.strip() for l in (build.stdout + build.stderr).splitlines() if " error: " in l]
            fails.check("the real mac_pull_fd compiles against the two statics", False,
                        (detail[0] if detail else (build.stdout + build.stderr).strip()[:400]))
            return 1
        fails.check("the real mac_pull_fd compiles against the two statics", True)

        # the planted expert files: two packs, a third, and a path that is not there
        packs = {}
        for name, text in (("a", "expert file A first blob"),
                           ("b", "expert file B first blob"),
                           ("c", "expert file C first blob")):
            path = os.path.join(work, "experts-%s.bin" % name)
            with open(path, "wb") as f:
                f.write(text.encode("utf-8"))
            packs[name] = (path, text)
        missing = os.path.join(work, "experts-gone.bin")

        def run(label, *paths):
            r = subprocess.run([exe, label, *paths], capture_output=True, text=True, timeout=120)
            if r.returncode != 0:
                return None, "the fake did not run (exit %d): %s" % (r.returncode, (r.stdout + r.stderr).strip()[-300:])
            got = parse(r.stdout)
            if got is None:
                return None, "cannot read the observation: " + r.stdout.strip()[:300]
            return got, None

        cases = []

        def add(name, label, paths, want):
            got, err = run(label, *paths)
            if got is None:
                fails.check(name, False, err)
                return
            ok, detail = want(got)
            fails.check(name, ok, detail)

        # ordinary input: one file, its descriptor and g_pull_path, the same before and after the change
        add("one expert file opens and reads that file", "single", [packs["a"][0]],
            lambda g: (len(g["calls"]) == 1 and g["calls"][0][0] >= 0 and g["calls"][0][1] == packs["a"][1]
                       and g["path"] == packs["a"][0],
                       "calls=%r path=%r" % (g["calls"], g["path"])))

        # ordinary input: the cache still holds one descriptor for one path
        add("the same path twice keeps the first descriptor", "repeat", [packs["a"][0], packs["a"][0]],
            lambda g: (len(g["calls"]) == 2 and g["calls"][0][0] >= 0 and g["calls"][0][0] == g["calls"][1][0]
                       and [c[1] for c in g["calls"]] == [packs["a"][1], packs["a"][1]],
                       "calls=%r" % (g["calls"],)))

        # the planted input: a second expert file must be the one the descriptor reads
        add("a second expert file reads the second file", "switch", [packs["a"][0], packs["b"][0]],
            lambda g: (len(g["calls"]) == 2 and g["calls"][1][0] >= 0 and g["calls"][1][1] == packs["b"][1],
                       "calls=%r" % (g["calls"],)))

        # a third file after two others, not the first file again
        add("a third expert file reads the third file", "three",
            [packs["a"][0], packs["b"][0], packs["c"][0]],
            lambda g: (len(g["calls"]) == 3 and [c[1] for c in g["calls"]] == [packs["a"][1], packs["b"][1], packs["c"][1]],
                       "calls=%r" % (g["calls"],)))

        # a file that is not there must fail, not hand back the first file's descriptor; a real file after it recovers
        add("a missing expert file fails and a later file recovers", "missing",
            [packs["a"][0], missing, packs["c"][0]],
            lambda g: (len(g["calls"]) == 3 and g["calls"][1][0] < 0 and g["calls"][2][0] >= 0
                       and g["calls"][2][1] == packs["c"][1],
                       "calls=%r" % (g["calls"],)))

        # use_fd opens g_pull_path for a per-thread descriptor: it must name the file the current fd belongs to
        add("g_pull_path follows the file the descriptor was opened for", "path",
            [packs["a"][0], packs["b"][0]],
            lambda g: (g["path"] == packs["b"][0],
                       "path=%r, expected %r" % (g["path"], packs["b"][0])))
    finally:
        shutil.rmtree(work, ignore_errors=True)

    if fails.count:
        print("%d case(s) failed" % fails.count)
        return 1
    print("all cases passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
