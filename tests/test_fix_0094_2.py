#!/usr/bin/env python3
"""tests/test_fix_0094_2.py - a failed upload must keep its pages dirty so a later flush retries them.

sidecar-shim/tgcudart.cpp:133-136 clear the dirty bits of a run, write-protect the pages and then upload them:

    while (j < t->npages && t->dirty[j].load(std::memory_order_relaxed)) { t->dirty[j].store(0, ...); j++; }
    mprotect((void*)(t->base + i * g_page), (j - i) * g_page, PROT_READ);
    call_i("h2d_raw", "(KKK)", ...);

The return value of h2d_raw is dropped, and g_any_dirty was already cleared at the top of the flush. A failed
upload therefore loses the host writes: nothing stays dirty and no later flush tries again, so the VRAM
mirror stays stale with no error and no retry. These cases cut the real flush_dirty out of the shim by brace
matching and run it over real page-aligned memory (mmap plus mprotect) with a stub h2d_raw that copies the
host bytes into a mirror and can be planted to fail (no GPU, no model, no weights):
  - a clean run of dirty pages uploads once, clears the bits and the bytes reach the mirror;
  - two runs split by a clean page are uploaded separately;
  - a failed upload keeps the run dirty and the pending flag set;
  - the next flush retries that run and the bytes reach the mirror;
  - a failure in the first run does not stop the second run in the same flush;
  - a page whose upload keeps failing is retried on every flush.
The clean cases give the same result before and after the fix.
Run: python3 tests/test_fix_0094_2.py
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

# the harness: real page-aligned host memory, a stub h2d_raw that copies host bytes to the mirror or fails,
# then the real flush_dirty cut out of the source above
HARNESS = r"""// built by tests/test_fix_0094_2.py
#include <atomic>
#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <sys/mman.h>
#include <unistd.h>

size_t g_page = 4096;

struct Tracked { uintptr_t base; size_t size; uint64_t mirror; size_t npages; std::atomic<uint8_t>* dirty; };
std::vector<Tracked*> g_tracked;
std::atomic<int> g_any_dirty{0};

static int g_calls = 0;         // h2d_raw calls in this phase
static int g_fail_call = -1;    // 0-based call that fails, -1 none, -2 every call
static unsigned char* g_host = nullptr;
static unsigned char* g_mirror = nullptr;
static Tracked* g_t = nullptr;
static size_t g_span = 0;

/* the shim's call into tgcu_api: copy the host bytes into the mirror, or fail and copy nothing */
long long call_i(const char* fn, const char* fmt, ...) {
  if (std::strcmp(fn, "h2d_raw") != 0) return 0;
  va_list ap; va_start(ap, fmt);
  unsigned long long dst = va_arg(ap, unsigned long long);
  unsigned long long src = va_arg(ap, unsigned long long);
  unsigned long long n = va_arg(ap, unsigned long long);
  va_end(ap);
  const int idx = g_calls++;
  if (g_fail_call == -2 || g_fail_call == idx) return -5;
  std::memcpy((void*)(uintptr_t)dst, (const void*)(uintptr_t)src, (size_t)n);
  return 0;
}

// ---- flush_dirty, cut out of sidecar-shim/tgcudart.cpp ----
@EXTRACTED@
// ---- end of the cut out function ----

static void teardown() {
  g_tracked.clear();
  g_any_dirty.store(0);
  if (g_t) { delete[] g_t->dirty; delete g_t; g_t = nullptr; }
  if (g_host) { mprotect(g_host, g_span, PROT_READ | PROT_WRITE); munmap(g_host, g_span); g_host = nullptr; }
  std::free(g_mirror); g_mirror = nullptr;
  g_span = 0;
}

/* a page-aligned host region in the state track_region leaves it: whole pages, dirty bits as planted, write-protected */
static void setup(size_t npages, const char* dirty) {
  teardown();
  g_page = (size_t) getpagesize();
  g_span = npages * g_page;
  void* p = mmap(nullptr, g_span, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
  if (p == MAP_FAILED) { std::printf("case=setup calls=0 dirty= any=0 mirror= (mmap failed)\n"); std::exit(2); }
  g_host = (unsigned char*) p;
  g_mirror = (unsigned char*) std::calloc(g_span, 1);
  g_t = new Tracked{(uintptr_t)g_host, g_span, (uint64_t)(uintptr_t)g_mirror, npages, new std::atomic<uint8_t>[npages]};
  for (size_t i = 0; i < npages; i++) {
    std::memset(g_host + i * g_page, (int)(0xA0 + i), g_page);   // every host page holds a different byte
    g_t->dirty[i].store(dirty[i] == '1' ? 1 : 0);
  }
  g_tracked.push_back(g_t);
  g_any_dirty.store(1, std::memory_order_release);
  mprotect(g_host, g_span, PROT_READ);
}

static void phase(int fail_call) { g_calls = 0; g_fail_call = fail_call; }

static void report(const char* id) {
  char bits[64], mb[64];
  const size_t np = g_t->npages;
  for (size_t i = 0; i < np && i < 63; i++) bits[i] = g_t->dirty[i].load() ? '1' : '0';
  bits[np < 63 ? np : 63] = 0;
  for (size_t i = 0; i < np && i < 63; i++) mb[i] = std::memcmp(g_mirror + i * g_page, g_host + i * g_page, g_page) == 0 ? '1' : '0';
  mb[np < 63 ? np : 63] = 0;
  std::printf("case=%s calls=%d dirty=%s any=%d mirror=%s\n", id, g_calls, bits, g_any_dirty.load(), mb);
}

int main() {
  setup(2, "11");
  phase(-1);
  flush_dirty();
  report("clean");

  setup(3, "101");
  phase(-1);
  flush_dirty();
  report("gaps");

  setup(1, "1");
  phase(0);
  flush_dirty();
  report("fail-state");

  phase(-1);
  flush_dirty();
  report("retry");

  setup(3, "101");
  phase(0);
  flush_dirty();
  report("later-run");

  setup(1, "1");
  phase(-2);
  flush_dirty();
  flush_dirty();
  flush_dirty();
  report("still-fail");

  teardown();
  return 0;
}
"""

# what each case must show: calls = h2d_raw calls in the phase, dirty = dirty bits after it,
# any = g_any_dirty, mirror = which pages the mirror holds the host bytes of
CASES = [
    dict(id="clean", want=dict(calls=1, dirty="00", any=0, mirror="11"),
         why="a clean run of two dirty pages uploads once, clears the bits and the bytes reach the mirror"),
    dict(id="gaps", want=dict(calls=2, dirty="000", any=0, mirror="101"),
         why="two runs split by a clean page are uploaded separately"),
    dict(id="fail-state", want=dict(calls=1, dirty="1", any=1, mirror="0"),
         why="a failed upload keeps its page dirty and the pending flag set"),
    dict(id="retry", want=dict(calls=1, dirty="0", any=0, mirror="1"),
         why="the next flush retries the failed upload and its bytes reach the mirror"),
    dict(id="later-run", want=dict(calls=2, dirty="100", any=1, mirror="001"),
         why="a failure in the first run does not stop the second run in the same flush"),
    dict(id="still-fail", want=dict(calls=3, dirty="1", any=1, mirror="0"),
         why="a page whose upload keeps failing is retried on every flush"),
]


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
    """The real flush_dirty definition, from its signature to its matching closing brace."""
    text = path.read_text(encoding="utf-8", errors="replace")
    clean = strip_noise(text)
    m = re.search(r"^[ \t]*void[ \t]+flush_dirty[ \t]*\([ \t]*\)[ \t]*\{", clean, re.M)
    if not m:
        return None
    start = clean.find("{", m.end() - 1)
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(clean)):
        if clean[i] == "{":
            depth += 1
        elif clean[i] == "}":
            depth -= 1
            if depth == 0:
                return text[m.start():i + 1].strip()
    return None


def compiler():
    for name in ("g++", "clang++", "c++"):
        path = shutil.which(name)
        if path:
            return path
    return None


def build_and_run(cxx, body, work):
    src = Path(work) / "flush_dirty.cpp"
    exe = Path(work) / "flush_dirty"
    src.write_text(HARNESS.replace("@EXTRACTED@", body), encoding="utf-8")
    cmd = [cxx, "-std=c++17", "-O0", "-o", str(exe), str(src)]
    built = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    if built.returncode != 0:
        errors = [l.strip() for l in (built.stdout + built.stderr).splitlines() if " error: " in l]
        return None, "does not build (" + (errors[0] if errors else "see the compile log").replace('"', "'") + ")"
    run = subprocess.run([str(exe)], capture_output=True, text=True, timeout=120)
    if run.returncode != 0:
        return None, "the built program exited %d: %s" % (run.returncode, (run.stdout + run.stderr).strip()[-200:])
    rows = {}
    for line in run.stdout.splitlines():
        if line.startswith("case="):
            rows[line.split()[0][5:]] = dict(part.split("=", 1) for part in line.split() if "=" in part)
    return rows, ""


def check(case, row):
    """None when the row is right, else what it got wrong."""
    if row is None:
        return "the harness printed no line for this case"
    for key in ("calls", "dirty", "any", "mirror"):
        want = str(case["want"][key])
        if row.get(key) != want:
            return "%s=%s, want %s" % (key, row.get(key), want)
    return None


def main():
    if not SOURCE.is_file():
        print("FAIL run this test from the copy root: " + str(REL) + " is not in " + str(ROOT))
        return 1
    body = extract(SOURCE)
    if not body:
        print("FAIL cannot find the flush_dirty definition in " + str(REL))
        return 1
    cxx = compiler()
    if cxx is None:
        print("FAIL no C++ compiler was found (looked for g++, clang++, c++)")
        return 1
    work = tempfile.mkdtemp(prefix="fix-0094-2-flush-")
    failures = 0
    try:
        rows, detail = build_and_run(cxx, body, work)
        if rows is None:
            print("FAIL the cut out flush_dirty " + detail)
            return 1
        for case in CASES:
            wrong = check(case, rows.get(case["id"]))
            print(("PASS " if wrong is None else "FAIL ") + case["why"] + (": " + wrong if wrong else ""))
            failures += 0 if wrong is None else 1
    finally:
        shutil.rmtree(work, ignore_errors=True)
    print("%d case(s) failed" % failures if failures else "all cases passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
