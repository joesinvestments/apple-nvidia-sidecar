#!/usr/bin/env python3
"""tests/test_fix_0097_4.py - the shim's capture bookkeeping must be read under g_lock.

sidecar-shim/tgcudart.cpp keeps its stream capture state in g_capturing, g_cap_streams and g_cap_events
(lines 47 to 50). The writers (cudaStreamBeginCapture and cudaStreamEndCapture) take g_lock through the
API_ENTER macro, and cudaStreamWaitEvent takes it explicitly. Three readers do not:

  cudaEventRecord (line 951) tests captured(s) before it takes the lock for the insert;
  cudaStreamIsCapturing (line 1010) and cudaStreamGetCaptureInfo_v2 (line 1015) read captured(s) with
  no lock at all.

A thread querying the status of a capture therefore reads g_capturing and g_cap_streams while another
thread begins or ends a capture under g_lock: a std::set read next to an assignment or a clear, which is
the data race fix-0097#4 names for the capture bookkeeping.

These cases compile the real captured(), cudaStreamBeginCapture, cudaStreamEndCapture,
cudaStreamWaitEvent, cudaEventRecord, cudaStreamIsCapturing and cudaStreamGetCaptureInfo_v2 out of the
shim (no GPU, no CUDA headers, no model, no Python) against instrumented copies of the three globals and
a mutex of the harness that reports, on the calling thread, whether its lock is held. Every instrumented
access records whether the lock was held: when it was not, the access happened without the lock and the
case fails. Each case also asserts the ordinary result, so the fix cannot trade the race for different
answers:
  - begin sets the state under the lock, end clears it under the lock, a wait forks under the lock;
  - an event recorded inside a capture is remembered under the lock and submits nothing;
  - an event recorded outside a capture is submitted as before (the same result before and after);
  - the two capture status queries read the state under the lock.
Run: python3 tests/test_fix_0097_4.py
Prints one PASS or FAIL line per case and exits nonzero on any FAIL.
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile

REL = os.path.join('sidecar-shim', 'tgcudart.cpp')

# the definitions cut out of the shim, in dependency order. The pattern is anchored on the whole name so
# cudaEventRecord does not also match cudaEventRecordWithFlags.
CUTS = [
    ('captured', r'^[ \t]*bool[ \t]+captured[ \t]*\('),
    ('cudaStreamWaitEvent', r'^[ \t]*(?:EXPORT[ \t]+)?cudaError_t[ \t]+cudaStreamWaitEvent[ \t]*\('),
    ('cudaEventRecord', r'^[ \t]*(?:EXPORT[ \t]+)?cudaError_t[ \t]+cudaEventRecord[ \t]*\('),
    ('cudaStreamBeginCapture', r'^[ \t]*(?:EXPORT[ \t]+)?cudaError_t[ \t]+cudaStreamBeginCapture[ \t]*\('),
    ('cudaStreamEndCapture', r'^[ \t]*(?:EXPORT[ \t]+)?cudaError_t[ \t]+cudaStreamEndCapture[ \t]*\('),
    ('cudaStreamIsCapturing', r'^[ \t]*(?:EXPORT[ \t]+)?cudaError_t[ \t]+cudaStreamIsCapturing[ \t]*\('),
    ('cudaStreamGetCaptureInfo_v2', r'^[ \t]*(?:EXPORT[ \t]+)?cudaError_t[ \t]+cudaStreamGetCaptureInfo_v2[ \t]*\('),
]

HARNESS = r"""// built by tests/test_fix_0097_4.py
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <set>
#include <string>
#include <vector>

typedef int cudaError_t;
typedef void* cudaStream_t;
typedef void* cudaEvent_t;
typedef void* cudaGraph_t;
typedef void* cudaGraphNode_t;

enum { cudaSuccess = 0, cudaErrorStreamCaptureUnsupported = 900, cudaErrorStreamCaptureInvalidated = 901 };
enum cudaStreamCaptureMode {
  cudaStreamCaptureModeGlobal = 0,
  cudaStreamCaptureModeThreadLocal = 1,
  cudaStreamCaptureModeRelaxed = 2
};
enum cudaStreamCaptureStatus {
  cudaStreamCaptureStatusNone = 0,
  cudaStreamCaptureStatusActive = 1,
  cudaStreamCaptureStatusInvalidated = 2
};

#define EXPORT

// the shim's lock, with a count of the locks this thread holds. std::lock_guard<TrackedMutex> is still the
// lock guard the shim writes, so the functions below are compiled exactly as they are written.
static thread_local int t_lock_depth = 0;
struct TrackedMutex {
  std::recursive_mutex m;
  void lock() { m.lock(); ++t_lock_depth; }
  void unlock() { --t_lock_depth; m.unlock(); }
};
TrackedMutex g_lock;

static std::vector<std::string> g_unlocked;
static int g_notes = 0;
static void note(const char* what) {
  ++g_notes;
  if (t_lock_depth <= 0) g_unlocked.push_back(what);
}

// the three globals, with every read and write recorded
struct Flag {
  int v = 0;
  operator int() const { note("g_capturing_read"); return v; }
  Flag& operator=(int x) { note("g_capturing_write"); v = x; return *this; }
};
struct Streams {
  std::set<cudaStream_t> s;
  Streams& operator=(std::initializer_list<cudaStream_t> il) { note("g_cap_streams_write"); s = il; return *this; }
  size_t count(cudaStream_t x) const { note("g_cap_streams_read"); return s.count(x); }
  void insert(cudaStream_t x) { note("g_cap_streams_write"); s.insert(x); }
  void clear() { note("g_cap_streams_write"); s.clear(); }
};
struct Events {
  std::set<cudaEvent_t> s;
  size_t count(cudaEvent_t x) const { note("g_cap_events_read"); return s.count(x); }
  void insert(cudaEvent_t x) { note("g_cap_events_write"); s.insert(x); }
  void clear() { note("g_cap_events_write"); s.clear(); }
};
Flag g_capturing;
Streams g_cap_streams;
Events g_cap_events;

struct Event { std::chrono::steady_clock::time_point t; int slot = -1; };
static const bool g_gpu_events = true;

static std::vector<std::string> g_calls;
static int g_event_slots = 0;
long long call_i(const char* fn, const char*, ...) {
  g_calls.push_back(fn);
  if (!std::strcmp(fn, "event_record")) return ++g_event_slots;
  return 0;
}
cudaError_t fail(cudaError_t e, const char*) { return e; }

// the shim's API_ENTER takes g_lock for the whole body, as it does at line 247 of the source
#define API_ENTER(name) std::lock_guard<TrackedMutex> _lk(g_lock); (void) (name)

// ---- the definitions cut out of sidecar-shim/tgcudart.cpp ----
@@BODIES@@
// ---- end of the cut out definitions ----

static cudaStream_t g_s = (cudaStream_t) 0x5100;     // the capturing stream
static cudaStream_t g_fork = (cudaStream_t) 0x5200;  // a stream that waits on a captured event
static cudaEvent_t g_e = (cudaEvent_t) 0xE100;

static void clear_notes() { g_unlocked.clear(); g_notes = 0; g_calls.clear(); }
static void plant(int capturing, std::initializer_list<cudaStream_t> streams, std::initializer_list<cudaEvent_t> events) {
  g_capturing.v = capturing;
  g_cap_streams.s = streams;
  g_cap_events.s = events;
  clear_notes();
}
static std::string join(const std::vector<std::string>& v, char sep) {
  std::string out;
  for (size_t i = 0; i < v.size(); ++i) { if (i) out += sep; out += v[i]; }
  return out;
}

static cudaError_t g_rc = 0;
static int g_status = -1;
static unsigned long long g_id = 0;
static long long g_graph = 0;
static int g_depsnull = 0;
static size_t g_ndeps = 0;
static int g_slot = -1;

static void finish() {
  std::printf("rc=%d cap=%d has_s=%d has_fork=%d has_e=%d status=%d id=%llu g=%lld deps=%d ndeps=%zu "
              "slot=%d notes=%d unlocked=%s calls=%s\n",
              (int) g_rc, g_capturing.v, (int) g_cap_streams.s.count(g_s), (int) g_cap_streams.s.count(g_fork),
              (int) g_cap_events.s.count(g_e), g_status, g_id, g_graph, g_depsnull, g_ndeps, g_slot, g_notes,
              join(g_unlocked, ';').c_str(), join(g_calls, ',').c_str());
}

int main(int argc, char** argv) {
  if (argc != 2) { std::printf("usage: %s case\n", argv[0]); return 2; }
  const std::string c = argv[1];
  if (c == "begin") {
    clear_notes();
    g_rc = cudaStreamBeginCapture(g_s, cudaStreamCaptureModeThreadLocal);
  } else if (c == "end") {
    plant(1, {g_s}, {g_e});
    cudaGraph_t graph = nullptr;
    g_rc = cudaStreamEndCapture(g_s, &graph);
    g_graph = (long long) (uintptr_t) graph;
  } else if (c == "wait") {
    plant(1, {g_s}, {g_e});
    g_rc = cudaStreamWaitEvent(g_fork, g_e, 0);
  } else if (c == "record_captured") {
    plant(1, {g_s}, {});
    g_rc = cudaEventRecord(g_e, g_s);
  } else if (c == "record_ordinary" || c == "record_ordinary_lock") {
    plant(0, {}, {});
    Event ev;
    g_rc = cudaEventRecord(&ev, g_fork);
    g_slot = ev.slot;
  } else if (c == "is_capturing") {
    plant(1, {g_s}, {});
    cudaStreamCaptureStatus st = cudaStreamCaptureStatusNone;
    g_rc = cudaStreamIsCapturing(g_s, &st);
    g_status = (int) st;
  } else if (c == "capture_info") {
    plant(1, {g_s}, {});
    cudaStreamCaptureStatus st = cudaStreamCaptureStatusNone;
    unsigned long long id = 0;
    cudaGraph_t graph = (cudaGraph_t) 1;
    const cudaGraphNode_t* deps = (const cudaGraphNode_t*) 1;
    size_t ndeps = 99;
    g_rc = cudaStreamGetCaptureInfo_v2(g_s, &st, &id, &graph, &deps, &ndeps);
    g_status = (int) st;
    g_id = id;
    g_graph = (long long) (uintptr_t) graph;
    g_depsnull = deps == nullptr ? 1 : 0;
    g_ndeps = ndeps;
  } else {
    std::printf("unknown case %s\n", argv[1]);
    return 2;
  }
  finish();
  return 0;
}
"""

# id: harness case. want: the ordinary result the case must keep. lock: the capture access must be under g_lock.
CASES = [
    dict(id='begin',
         name='cudaStreamBeginCapture sets the capture state under g_lock',
         want={'rc': '0', 'cap': '1', 'has_s': '1', 'has_fork': '0', 'has_e': '0'},
         lock=True),
    dict(id='end',
         name='cudaStreamEndCapture clears the capture state under g_lock',
         want={'rc': '0', 'cap': '0', 'has_s': '0', 'has_e': '0', 'g': '1'},
         lock=True),
    dict(id='wait',
         name='cudaStreamWaitEvent forks the capture under g_lock',
         want={'rc': '0', 'has_s': '1', 'has_fork': '1', 'has_e': '1'},
         lock=True),
    dict(id='record_captured',
         name='cudaEventRecord inside a capture keeps the event under g_lock and submits nothing',
         want={'rc': '0', 'has_e': '1', 'slot': '-1', 'calls': ''},
         lock=True),
    dict(id='record_ordinary',
         name='cudaEventRecord outside a capture records the event (the same result before and after)',
         want={'rc': '0', 'has_e': '0', 'slot': '1', 'calls': 'event_record'},
         lock=False),
    dict(id='record_ordinary_lock',
         name='cudaEventRecord outside a capture reads the capture state under g_lock',
         want={'rc': '0', 'has_e': '0', 'slot': '1', 'calls': 'event_record'},
         lock=True),
    dict(id='is_capturing',
         name='cudaStreamIsCapturing reads the capture state under g_lock',
         want={'rc': '0', 'status': '1'},
         lock=True),
    dict(id='capture_info',
         name='cudaStreamGetCaptureInfo_v2 reads the capture state under g_lock',
         want={'rc': '0', 'status': '1', 'id': '1', 'g': '0', 'deps': '1', 'ndeps': '0'},
         lock=True),
]


def strip_noise(text):
    """Blank comments and literals in place, so an index into the result is an index into the source."""
    out = list(text)

    def blank(a, b):
        for k in range(a, min(b, len(out))):
            if out[k] != '\n':
                out[k] = ' '

    i, n = 0, len(text)
    while i < n:
        if text.startswith('//', i):
            j = text.find('\n', i)
            j = n if j < 0 else j
            blank(i, j)
            i = j
        elif text.startswith('/*', i):
            j = text.find('*/', i + 2)
            j = n if j < 0 else j + 2
            blank(i, j)
            i = j
        elif text[i] in '"\'':
            start = i
            quote = text[i]
            i += 1
            while i < n:
                if text[i] == '\\':
                    i += 2
                    continue
                if text[i] == quote or text[i] == '\n':
                    i += 1
                    break
                i += 1
            blank(start, i)
        else:
            i += 1
    return ''.join(out)


def cut(text, clean, pattern):
    """One definition, from its signature line to its matching closing brace."""
    m = re.search(pattern, clean, re.M)
    if not m:
        return None
    start = clean.find('{', m.end())
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(clean)):
        if clean[i] == '{':
            depth += 1
        elif clean[i] == '}':
            depth -= 1
            if depth == 0:
                return text[m.start():i + 1].strip()
    return None


def compiler():
    for name in ('clang++', 'g++', 'c++'):
        path = shutil.which(name)
        if path:
            return path
    return None


def build_and_run(cxx, bodies, work):
    src = os.path.join(work, 'capture.cpp')
    exe = os.path.join(work, 'capture')
    with open(src, 'w', encoding='utf-8') as f:
        f.write(HARNESS.replace('@@BODIES@@', bodies))
    built = subprocess.run([cxx, '-std=c++17', '-O0', '-pthread', '-o', exe, src],
                           capture_output=True, text=True, timeout=180)
    if built.returncode != 0:
        errors = [l.strip() for l in (built.stdout + built.stderr).splitlines() if ' error: ' in l]
        return None, 'does not build (' + (errors[0] if errors else 'see the compile log').replace('"', "'") + ')'
    return exe, ''


def run_case(exe, case):
    run = subprocess.run([exe, case['id']], capture_output=True, text=True, timeout=120)
    if run.returncode != 0:
        return None, 'the built program exited %d: %s' % (run.returncode, (run.stdout + run.stderr).strip()[-200:])
    for line in run.stdout.splitlines():
        if line.startswith('rc='):
            return dict(part.split('=', 1) for part in line.split() if '=' in part), ''
    return None, 'the built program printed no result line'


def check(case, row):
    if row is None:
        return 'the harness printed no result'
    for key, want in case['want'].items():
        got = row.get(key, '<missing>')
        if got != want:
            return '%s=%s, want %s' % (key, got, want)
    if row.get('notes', '0') == '0':
        return 'the capture state was never touched, the case proves nothing'
    if case['lock'] and row.get('unlocked', ''):
        return 'read or wrote %s without g_lock' % row['unlocked']
    return None


def main():
    root = os.getcwd()
    source = os.path.join(root, REL)
    if not os.path.isfile(source):
        print('FAIL run this test from the copy root: ' + REL + ' is not in ' + root)
        return 1
    text = open(source, encoding='utf-8', errors='replace').read()
    clean = strip_noise(text)
    bodies = []
    for name, pattern in CUTS:
        body = cut(text, clean, pattern)
        if body is None:
            print('FAIL cannot find ' + name + ' in ' + REL)
            return 1
        bodies.append(body.replace('std::recursive_mutex', 'TrackedMutex'))
    cxx = compiler()
    if cxx is None:
        print('FAIL no C++ compiler was found (looked for clang++, g++, c++)')
        return 1
    work = tempfile.mkdtemp(prefix='fix-0097-capture-')
    failures = 0
    try:
        exe, detail = build_and_run(cxx, '\n\n'.join(bodies), work)
        if exe is None:
            print('FAIL the cut out capture functions ' + detail)
            return 1
        for case in CASES:
            row, detail = run_case(exe, case)
            wrong = check(case, row) if row is not None else detail
            print(('PASS ' if wrong is None else 'FAIL ') + case['name'] + (': ' + wrong if wrong else ''))
            failures += 0 if wrong is None else 1
    finally:
        shutil.rmtree(work, ignore_errors=True)
    print('%d case(s) failed' % failures if failures else 'all cases passed')
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
