#!/usr/bin/env python3
"""tests/test_fix_0098_9.py - tgcuHsArm must not keep the segment list when it refuses the gate pointer.

sidecar-shim/tgcudart.cpp tgcuHsArm builds the segment list with hs_segs (a new reference), then reads the gate
pointer's region. When the four gate segments are mapped but the gate pointer is not, region_base(gate_p) is 0 and the
function refuses the call with -3 at

  if (!l || !gb) return -3;

and the only reference to the freshly made list is dropped without a Py_DECREF: every such call leaks one PyObject
list. The sibling tgcuHsPushPreNext decrefs both lists on exactly the same condition, so the omission is a leak, not
an ownership handover.

These cases compile the real region_base, hs_segs and tgcuHsArm out of sidecar-shim/tgcudart.cpp against a stand-in
for the Python C API that keeps a live list count (no Python, no CUDA, no GPU, no model), plant a mapped region in the
shim's host map and drive the calls:
  - a mapped gate must still reach hs_arm and answer with its int (ordinary input, unchanged by the fix);
  - a mapped gate must leave no list behind;
  - an unmapped gate must return -3 and leave no list behind (today it leaves one);
  - 64 refusals must not accumulate lists (today they leave 64);
  - a segment outside a mapped region must still return -3 without calling hs_arm;
  - two mapped segments must still reach hs_arm whole.
Prints one PASS or FAIL line per case and exits nonzero on any FAIL. Run: python3 tests/test_fix_0098_9.py
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = os.getcwd()   # the tree under test: every test here runs with cwd at the copy root, never at the test file
SHIM = os.path.join('sidecar-shim', 'tgcudart.cpp')

# the fake Python C API keeps a live list count instead of freeing memory: a list the shim allocates and never
# decrefs is still live when the call returns, which is the leak this test has to observe
PRELUDE = r"""
#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>

#define EXPORT
#define API_ENTER(name) ((void)0)

// the shim's host map: base -> (size, mirror va or 0, owned); region_base answers the base of the region a pointer
// sits in when that region has a mirror, and 0 for a pointer outside every region
struct HostAlloc { size_t size; uint64_t mirror; bool owned; };
static std::map<uintptr_t, HostAlloc> g_host;

struct PyObject { int kind; int refcnt; long long items; long long value; };
static const int PY_LIST = 1, PY_TUPLE = 2, PY_LONG = 3;

static int g_live_lists = 0;      // lists hs_segs made that no Py_DECREF has released yet
static int g_freed_lists = 0;
static long long g_arm_calls = 0;
static long long g_arm_items = -1;   // the length of the segment list hs_arm was handed
static long long g_arm_want = -1;    // the want flag hs_arm was handed
static long long g_arm_result = 7;   // what the fake hs_arm answers
static int g_live_base = 0, g_freed_base = 0;   // the counts at the start of a case, so each case sees its own lists

static int live_now() { return g_live_lists - g_live_base; }
static int freed_now() { return g_freed_lists - g_freed_base; }

static PyObject* PyList_New(long long n) {
  PyObject* o = new PyObject{PY_LIST, 1, n, 0};
  g_live_lists++;
  return o;
}
static PyObject* Py_BuildValue(const char*, ...) {   // the (base, offset, bytes) tuple each segment becomes
  static PyObject tuple{PY_TUPLE, 1000000, 0, 0};
  return &tuple;
}
static void Py_DECREF(PyObject* o) {
  if (--o->refcnt > 0) return;
  if (o->kind == PY_LIST) { g_live_lists--; g_freed_lists++; }
  if (o->kind != PY_TUPLE) delete o;
}
static void Py_XDECREF(PyObject* o) { if (o) Py_DECREF(o); }
static void PyList_SET_ITEM(PyObject*, long long, PyObject*) {}
static bool PyLong_Check(PyObject* o) { return o && o->kind == PY_LONG; }
static long long PyLong_AsLongLong(PyObject* o) { return o ? o->value : -1; }

// the fake Python side: hs_arm records what it was handed and answers g_arm_result, the int the real hs_arm returns
static bool call_obj(PyObject** out, const char* fn, const char* fmt, ...) {
  va_list ap; va_start(ap, fmt);
  if (!std::strcmp(fn, "hs_arm")) {
    g_arm_calls++;
    for (const char* f = fmt; f && *f; ++f) {
      if (*f == 'O') { PyObject* l = va_arg(ap, PyObject*); g_arm_items = l ? l->items : -1; }
      else if (*f == 'K') (void)va_arg(ap, unsigned long long);
      else if (*f == 'I') g_arm_want = (long long)va_arg(ap, unsigned);
      else if (*f == 'i') (void)va_arg(ap, int);
      else if (*f == 's') (void)va_arg(ap, const char*);
    }
  }
  va_end(ap);
  *out = new PyObject{PY_LONG, 1, 0, g_arm_result};
  return true;
}
"""

MAIN = r"""
static char g_seg[2][256];
static char g_gate[256];

static void reset() {
  g_host.clear();
  g_live_base = g_live_lists;
  g_freed_base = g_freed_lists;
  g_arm_calls = 0;
  g_arm_items = -1;
  g_arm_want = -1;
  g_arm_result = 7;
}

int main() {
  const size_t n1 = sizeof(g_seg[0]), n2 = sizeof(g_seg[1]);
  void* segs[2] = {g_seg[0], g_seg[1]};
  size_t ns[2] = {n1, n2};

  // the segments are mapped, the gate pointer is not: the gate alone is what refuses the call
  reset();
  g_host[(uintptr_t)g_seg[0]] = HostAlloc{n1, 0x1000000ull, false};
  int rc = tgcuHsArm(&segs[0], &ns[0], 1, g_gate, 0);
  std::printf("leak rc=%d live_after=%d freed=%d calls=%lld\n", rc, live_now(), freed_now(), g_arm_calls);

  // the same refusal 64 times: a leak per call adds up
  reset();
  g_host[(uintptr_t)g_seg[0]] = HostAlloc{n1, 0x1000000ull, false};
  for (int i = 0; i < 64; ++i) tgcuHsArm(&segs[0], &ns[0], 1, g_gate, 0);
  std::printf("repeat live_after=%d freed=%d calls=%lld\n", live_now(), freed_now(), g_arm_calls);

  // ordinary input: both regions mapped, so the call goes through to hs_arm and returns its answer
  reset();
  g_host[(uintptr_t)g_seg[0]] = HostAlloc{n1, 0x1000000ull, false};
  g_host[(uintptr_t)g_gate] = HostAlloc{sizeof(g_gate), 0x2000000ull, false};
  rc = tgcuHsArm(&segs[0], &ns[0], 1, g_gate, 5);
  std::printf("mapped rc=%d calls=%lld items=%lld want=%lld live_after=%d\n",
              rc, g_arm_calls, g_arm_items, g_arm_want, live_now());

  // two mapped segments: the list hs_segs made still holds both
  reset();
  g_host[(uintptr_t)g_seg[0]] = HostAlloc{n1, 0x1000000ull, false};
  g_host[(uintptr_t)g_seg[1]] = HostAlloc{n2, 0x3000000ull, false};
  g_host[(uintptr_t)g_gate] = HostAlloc{sizeof(g_gate), 0x2000000ull, false};
  rc = tgcuHsArm(&segs[0], &ns[0], 2, g_gate, 1);
  std::printf("two rc=%d calls=%lld items=%lld live_after=%d\n",
              rc, g_arm_calls, g_arm_items, live_now());

  // a segment outside every mapped region: hs_segs itself refuses, hs_arm is never called
  reset();
  g_host[(uintptr_t)g_gate] = HostAlloc{sizeof(g_gate), 0x2000000ull, false};
  rc = tgcuHsArm(&segs[0], &ns[0], 1, g_gate, 0);
  std::printf("segbad rc=%d calls=%lld live_after=%d\n", rc, g_arm_calls, live_now());
  return 0;
}
"""

NEEDED = ['region_base', 'hs_segs', 'tgcuHsArm']


def read_shim():
    with open(SHIM, encoding='utf-8', errors='replace') as f:
        return f.read()


def matching_brace(text, i):
    """The index of the brace that closes the one at i, with comments and literals skipped."""
    depth, n = 0, len(text)
    while i < n:
        c = text[i]
        if text.startswith('//', i):
            j = text.find('\n', i)
            i = n if j < 0 else j
        elif text.startswith('/*', i):
            j = text.find('*/', i + 2)
            i = n if j < 0 else j + 2
        elif c in '"\'':
            quote = c
            i += 1
            while i < n and text[i] != quote:
                i += 2 if text[i] == '\\' else 1
            i += 1
        else:
            if c == '{':
                depth += 1
            elif c == '}':
                depth -= 1
                if depth == 0:
                    return i
            i += 1
    return -1


def definition(text, name):
    """The whole definition of name, signature through its closing brace, or None."""
    pat = re.compile(r'\b' + re.escape(name) + r'\s*\(')
    off = 0
    for line in text.split('\n'):
        code = line.split('//')[0]
        m = pat.search(code)
        brace = code.find('{', m.end()) if m else -1
        if brace >= 0:
            close = matching_brace(text, off + brace)
            if close > 0:
                return text[off:close + 1]
        off += len(line) + 1
    return None


def harness(text):
    parts = [PRELUDE]
    for name in NEEDED:
        body = definition(text, name)
        if body is None:
            return None, name
        parts.append(body)
    parts.append(MAIN)
    return '\n'.join(parts), None


def compiler():
    for name in ('g++', 'clang++', 'c++'):
        path = shutil.which(name)
        if path:
            return path
    return None


def build(cxx, work, source):
    src = os.path.join(work, 'hs_arm.cpp')
    exe = os.path.join(work, 'hs_arm')
    with open(src, 'w', encoding='utf-8') as f:
        f.write(source)
    built = subprocess.run([cxx, '-std=c++17', '-O0', '-o', exe, src], capture_output=True, text=True, timeout=300)
    if built.returncode != 0:
        errors = [l.strip() for l in (built.stdout + built.stderr).splitlines() if ' error: ' in l]
        return None, (errors[0] if errors else 'see the compile log').replace('"', "'")
    return exe, ''


def run(exe):
    r = subprocess.run([exe], capture_output=True, text=True, timeout=120)
    return r.returncode, r.stdout + r.stderr


def parse(text):
    got = {}
    for line in text.splitlines():
        parts = line.split()
        if parts:
            got[parts[0]] = dict(p.split('=', 1) for p in parts[1:] if '=' in p)
    return got


def field(got, row, key):
    return got.get(row, {}).get(key)


def check_leak(got):
    """an unmapped gate: the call is refused with -3 and the list hs_segs made must be released."""
    if field(got, 'leak', 'rc') != '-3':
        return 'the unmapped gate answered %s, expected -3' % field(got, 'leak', 'rc')
    if field(got, 'leak', 'calls') != '0':
        return 'hs_arm was called for an unmapped gate'
    if field(got, 'leak', 'live_after') != '0':
        return 'the refused call left %s list(s) live (leaked), %s released' % (
            field(got, 'leak', 'live_after'), field(got, 'leak', 'freed'))
    return ''


def check_repeat(got):
    """64 refusals: the live list count has to come back to zero, not grow by one per call."""
    if field(got, 'repeat', 'calls') != '0':
        return 'hs_arm was called for an unmapped gate'
    if field(got, 'repeat', 'live_after') != '0':
        return '64 refused calls left %s list(s) live, expected 0' % field(got, 'repeat', 'live_after')
    return ''


def check_mapped(got):
    """ordinary input: the call goes through to hs_arm, answers with its int and keeps nothing."""
    if field(got, 'mapped', 'rc') != '7':
        return 'the mapped call answered %s, expected hs_arm\'s 7' % field(got, 'mapped', 'rc')
    if field(got, 'mapped', 'calls') != '1':
        return 'hs_arm was called %s time(s), expected 1' % field(got, 'mapped', 'calls')
    if field(got, 'mapped', 'items') != '1':
        return 'hs_arm was handed a list of %s segment(s), expected 1' % field(got, 'mapped', 'items')
    if field(got, 'mapped', 'want') != '5':
        return 'hs_arm was handed want=%s, expected 5' % field(got, 'mapped', 'want')
    if field(got, 'mapped', 'live_after') != '0':
        return 'the served call left %s list(s) live' % field(got, 'mapped', 'live_after')
    return ''


def check_two(got):
    """two mapped segments still reach hs_arm whole, and the list is released after the call."""
    if field(got, 'two', 'rc') != '7' or field(got, 'two', 'calls') != '1':
        return 'the two segment call answered rc=%s calls=%s' % (field(got, 'two', 'rc'), field(got, 'two', 'calls'))
    if field(got, 'two', 'items') != '2':
        return 'hs_arm was handed a list of %s segment(s), expected 2' % field(got, 'two', 'items')
    if field(got, 'two', 'live_after') != '0':
        return 'the served call left %s list(s) live' % field(got, 'two', 'live_after')
    return ''


def check_segbad(got):
    """a segment outside every mapped region: hs_segs refuses, hs_arm is never called, nothing is left."""
    if field(got, 'segbad', 'rc') != '-3':
        return 'the unmapped segment answered %s, expected -3' % field(got, 'segbad', 'rc')
    if field(got, 'segbad', 'calls') != '0':
        return 'hs_arm was called with a segment outside every mapped region'
    if field(got, 'segbad', 'live_after') != '0':
        return 'the refused call left %s list(s) live' % field(got, 'segbad', 'live_after')
    return ''


CASES = [
    dict(name='an unmapped gate pointer leaves no segment list behind', check=check_leak),
    dict(name='64 calls with an unmapped gate do not accumulate lists', check=check_repeat),
    dict(name='a mapped gate still reaches hs_arm and returns its answer', check=check_mapped),
    dict(name='two mapped segments still reach hs_arm whole', check=check_two),
    dict(name='a segment outside a mapped region still answers -3 without a call', check=check_segbad),
]


def main():
    if not os.path.isfile(SHIM):
        print('FAIL run this test from the copy root: ' + SHIM + ' is not in ' + ROOT)
        return 1
    source, missing = harness(read_shim())
    if source is None:
        print('FAIL sidecar-shim/tgcudart.cpp has no definition of ' + missing)
        return 1
    cxx = compiler()
    if cxx is None:
        print('FAIL no C++ compiler was found (looked for g++, clang++, c++)')
        return 1
    work = tempfile.mkdtemp(prefix='shim-hs-arm-')
    try:
        exe, detail = build(cxx, work, source)
        if exe is None:
            print('FAIL tgcuHsArm does not build against the fake Python surface (' + detail + ')')
            return 1
        rc, out = run(exe)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    if rc != 0:
        print('FAIL the built program exited %d: %s' % (rc, out.strip()[-300:]))
        return 1
    got = parse(out)
    failures = 0
    for case in CASES:
        detail = case['check'](got)
        print(('PASS ' if not detail else 'FAIL ') + case['name'] + (': ' + detail if detail else ''))
        failures += 0 if not detail else 1
    print('%d case(s) failed' % failures if failures else 'all cases passed')
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
