#!/usr/bin/env python3
"""tests/test_fix_0097_3.py - cudaStreamCreate must hand out one handle per call and refuse a null out pointer.

sidecar-shim/tgcudart.cpp:907 (today) reads

    static uintptr_t g_next_stream = 0x5100;
    EXPORT cudaError_t cudaStreamCreate(cudaStream_t* s) { *s = (cudaStream_t)(g_next_stream += 16); return cudaSuccess; }

g_next_stream is a plain static, so two threads that create a stream at the same moment both read the same value,
add 16 and store it back: both callers get the same handle. s is also written without a null check, so a null out
pointer faults instead of returning cudaErrorInvalidValue. The declaration and the function are cut out of the
shim and built against a small fake CUDA runtime (no GPU, no model, no CPython), and the run plants the input:
  - sequential creates still step by 16 from 0x5110 (the ordinary path, the same before and after the fix);
  - a null out pointer is refused with cudaErrorInvalidValue through fail(), instead of dying;
  - a create after a refused null call still gets 0x5110 (a refused call consumes no handle);
  - 8 threads creating streams at once all get distinct handles.
Run: python3 tests/test_fix_0097_3.py
Prints one PASS or FAIL line per case and exits nonzero on any FAIL.
"""
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

REL = Path("sidecar-shim") / "tgcudart.cpp"


def find_root():
    """The tree under test: the copy prove_test runs us in, else the one this file sits in."""
    for base in (Path.cwd(), Path(__file__).resolve().parent.parent):
        if (base / REL).is_file():
            return base
    return Path.cwd()


ROOT = find_root()
SOURCE = ROOT / REL

# the fake cuda runtime: the types, the one error code and the shim's own fail() and lock, nothing else
HARNESS = r"""// built by tests/test_fix_0097_3.py
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <set>
#include <thread>
#include <vector>

#define EXPORT extern "C"
typedef int cudaError_t;
typedef struct CUstream_st* cudaStream_t;
enum { cudaSuccess = 0, cudaErrorInvalidValue = 1, cudaErrorUnknown = 999 };

static int g_fail_calls = 0;
static int g_last_err = cudaSuccess;
static cudaError_t fail(cudaError_t e, const char*) { ++g_fail_calls; g_last_err = (int) e; return e; }

std::recursive_mutex g_lock;   // the shim's own lock; a lock based fix uses it

// ---- cut out of sidecar-shim/tgcudart.cpp ----
@REGION@
// ---- end of the cut out code ----

static void run_seq(const char* name, int n) {
  std::vector<uintptr_t> h;
  int rc = 0;
  for (int i = 0; i < n; i++) {
    cudaStream_t s = nullptr;
    rc = (int) cudaStreamCreate(&s);
    if (rc != 0) break;
    h.push_back((uintptr_t) s);
  }
  std::set<uintptr_t> uniq(h.begin(), h.end());
  printf("case=%s rc=%d count=%d unique=%d fails=%d handles=", name, rc, (int) h.size(), (int) uniq.size(), g_fail_calls);
  for (size_t i = 0; i < h.size(); i++) printf("%s%llu", i ? "," : "", (unsigned long long) h[i]);
  printf("\n");
}

static int run_null() {
  cudaError_t rc = cudaStreamCreate(nullptr);
  printf("case=null rc=%d fails=%d last=%d\n", (int) rc, g_fail_calls, g_last_err);
  return 0;
}

static int run_after_null() {
  cudaStream_t s = nullptr;
  cudaError_t bad = cudaStreamCreate(nullptr);
  cudaError_t rc = cudaStreamCreate(&s);
  printf("case=after_null null_rc=%d rc=%d first=%llu fails=%d\n", (int) bad, (int) rc, (unsigned long long) (uintptr_t) s, g_fail_calls);
  return 0;
}

static int run_race() {
  const int nt = 8, per = 20000;
  std::vector<std::vector<uintptr_t>> got((size_t) nt);
  std::atomic<int> ready{0};
  std::atomic<bool> go{false};
  std::vector<std::thread> th;
  for (int i = 0; i < nt; i++) {
    th.emplace_back([&, i] {
      ready.fetch_add(1);
      while (!go.load()) std::this_thread::yield();
      for (int k = 0; k < per; k++) {
        cudaStream_t s = nullptr;
        cudaStreamCreate(&s);
        got[(size_t) i].push_back((uintptr_t) s);
      }
    });
  }
  while (ready.load() < nt) std::this_thread::yield();
  go.store(true);
  for (auto& t : th) t.join();
  std::set<uintptr_t> uniq;
  unsigned long long total = 0, dup = 0;
  for (auto& v : got) for (auto h : v) { total++; if (!uniq.insert(h).second) dup++; }
  printf("case=race total=%llu unique=%llu dup=%llu fails=%d\n", total, (unsigned long long) uniq.size(), dup, g_fail_calls);
  return 0;
}

int main(int argc, char** argv) {
  const char* mode = argc > 1 ? argv[1] : "seq";
  if (!strcmp(mode, "seq")) run_seq("seq", 8);
  else if (!strcmp(mode, "null")) return run_null();
  else if (!strcmp(mode, "after_null")) return run_after_null();
  else if (!strcmp(mode, "race")) return run_race();
  else { fprintf(stderr, "unknown mode %s\n", mode); return 2; }
  return 0;
}
"""

CASES = ["seq", "null", "after_null", "race"]


def strip_noise(text):
    """Blank comments and literals in place, so an index into the result is an index into the source."""
    out = list(text)

    def blank(a, b):
        for k in range(a, min(b, len(out))):
            if out[k] != "\n":
                out[k] = " "

    i, n = 0, len(text)
    while i < n:
        if text.startswith("//", i):
            j = text.find("\n", i)
            j = n if j < 0 else j
            blank(i, j)
            i = j
        elif text.startswith("/*", i):
            j = text.find("*/", i + 2)
            j = n if j < 0 else j + 2
            blank(i, j)
            i = j
        elif text[i] in "\"'":
            start = i
            quote = text[i]
            i += 1
            while i < n:
                if text[i] == "\\":
                    i += 2
                    continue
                if text[i] == quote or text[i] == "\n":
                    i += 1
                    break
                i += 1
            blank(start, i)
        else:
            i += 1
    return "".join(out)


def extract(path):
    """The real g_next_stream declaration and cudaStreamCreate definition, cut out of the shim."""
    text = path.read_text(encoding="utf-8", errors="replace")
    clean = strip_noise(text)
    m = re.search(r"^[ \t]*(?:EXPORT[ \t]+)?cudaError_t[ \t]+cudaStreamCreate[ \t]*\(", clean, re.M)
    if not m:
        return None
    start = clean.find("{", m.end())
    if start < 0:
        return None
    depth = 0
    end = -1
    for i in range(start, len(clean)):
        if clean[i] == "{":
            depth += 1
        elif clean[i] == "}":
            depth -= 1
            if depth == 0:
                end = i
                break
    if end < 0:
        return None
    decl = None
    for dm in re.finditer(r"^[ \t]*[^\n]*\bg_next_stream\b[^\n]*;[ \t]*$", clean[:m.start()], re.M):
        decl = dm
    if decl is None:
        return None
    region = text[decl.start():end + 1].strip()
    if "g_next_stream" not in region or "cudaStreamCreate(" not in region:
        return None
    return region


def compiler():
    for name in ("g++", "clang++", "c++"):
        path = shutil.which(name)
        if path:
            return path
    return None


def build(cxx, region, work):
    src = Path(work) / "stream_create.cpp"
    exe = Path(work) / "stream_create"
    src.write_text(HARNESS.replace("@REGION@", region), encoding="utf-8")
    cmd = [cxx, "-std=c++17", "-O0", "-pthread", "-o", str(exe), str(src)]
    built = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    if built.returncode != 0:
        errors = [l.strip() for l in (built.stdout + built.stderr).splitlines() if " error: " in l]
        return None, "does not build (" + (errors[0] if errors else "see the compile log").replace('"', "'") + ")"
    return str(exe), ""


def run(exe, mode):
    """The program, its output, and how it ended; a fault is reported, not raised."""
    r = subprocess.run([exe, mode], capture_output=True, text=True, timeout=120)
    rows = {}
    for line in r.stdout.splitlines():
        if line.startswith("case="):
            parts = dict(p.split("=", 1) for p in line.split() if "=" in p)
            rows[parts["case"]] = parts
    return rows.get(mode), r.returncode, (r.stdout + r.stderr).strip()[-160:]


def died(rc, tail):
    if rc == 0:
        return ""
    if rc < 0:
        return "the process died on signal %d instead of returning: %s" % (-rc, tail)
    return "the process exited %d without a result: %s" % (rc, tail)


def check_seq(row):
    """None when right, else what it got wrong: handles 0x5110, 0x5120, ... in order."""
    want = [0x5110 + 16 * k for k in range(8)]
    if row["rc"] != "0":
        return "rc=%s, want cudaSuccess" % row["rc"]
    got = [int(v) for v in row["handles"].split(",") if v] if "handles" in row else []
    if got != want:
        return "handles %s, want %s" % (got, want)
    if row["fails"] != "0":
        return "went through fail() on the ordinary path (%s call(s))" % row["fails"]
    return None


def check_null(row, rc, tail):
    bad = died(rc, tail)
    if bad:
        return "a null out pointer " + bad
    if row is None:
        return "the program printed no line for this case"
    if row["rc"] == "0":
        return "cudaSuccess from a null out pointer"
    if row["rc"] != "1":
        return "rc=%s, want cudaErrorInvalidValue (1)" % row["rc"]
    if row["fails"] != "1":
        return "did not go through fail() (%s call(s))" % row["fails"]
    return None


def check_after_null(row, rc, tail):
    bad = died(rc, tail)
    if bad:
        return "the refused null call before it " + bad
    if row is None:
        return "the program printed no line for this case"
    if row["null_rc"] == "0":
        return "cudaSuccess from a null out pointer"
    if row["rc"] != "0":
        return "the next real create returned rc=%s, want cudaSuccess" % row["rc"]
    if int(row["first"]) != 0x5110:
        return "the next real create got handle 0x%x, want 0x5110 (a refused call must not consume a handle)" % int(row["first"])
    return None


def check_race(row, rc, tail):
    bad = died(rc, tail)
    if bad:
        return "concurrent creates " + bad
    if row is None:
        return "the program printed no line for this case"
    total, uniq, dup = int(row["total"]), int(row["unique"]), int(row["dup"])
    if total != 8 * 20000:
        return "created %d handle(s), want %d" % (total, 8 * 20000)
    if dup:
        return "%d of %d creates got a handle another caller already had (only %d distinct)" % (dup, total, uniq)
    return None


def main():
    if not SOURCE.is_file():
        print("FAIL run this test from the copy root: " + str(REL) + " is not in " + str(ROOT))
        return 1
    region = extract(SOURCE)
    if not region:
        print("FAIL cannot find the g_next_stream declaration and cudaStreamCreate in " + str(REL))
        return 1
    cxx = compiler()
    if cxx is None:
        print("FAIL no C++ compiler was found (looked for g++, clang++, c++)")
        return 1
    work = tempfile.mkdtemp(prefix="fix-0097-stream-")
    failures = 0
    try:
        exe, detail = build(cxx, region, work)
        if exe is None:
            print("FAIL the cut out cudaStreamCreate " + detail)
            return 1
        results = {mode: run(exe, mode) for mode in CASES}
        seq = results["seq"][0]
        wrong = check_seq(seq) if seq else "the program printed no line for this case"
        print(("PASS " if wrong is None else "FAIL ") + "8 creates in a row still get 0x5110, 0x5120, ... 0x5180"
              + (": " + wrong if wrong else ""))
        failures += 0 if wrong is None else 1
        row, rc, tail = results["null"]
        wrong = check_null(row, rc, tail)
        print(("PASS " if wrong is None else "FAIL ") + "a null out pointer returns cudaErrorInvalidValue through fail()"
              + (": " + wrong if wrong else ""))
        failures += 0 if wrong is None else 1
        row, rc, tail = results["after_null"]
        wrong = check_after_null(row, rc, tail)
        print(("PASS " if wrong is None else "FAIL ") + "a create after a refused null call still gets 0x5110"
              + (": " + wrong if wrong else ""))
        failures += 0 if wrong is None else 1
        row, rc, tail = results["race"]
        wrong = check_race(row, rc, tail)
        print(("PASS " if wrong is None else "FAIL ") + "8 threads creating 160000 streams at once get 160000 distinct handles"
              + (": " + wrong if wrong else ""))
        failures += 0 if wrong is None else 1
    finally:
        shutil.rmtree(work, ignore_errors=True)
    print("%d case(s) failed" % failures if failures else "all cases passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
