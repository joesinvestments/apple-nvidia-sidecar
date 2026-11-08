#!/usr/bin/env python3
"""tests/test_fix_0096_4.py - a failed edge d2h copy in lazy_d2h must not be reported as success.

sidecar-shim/tgcudart.cpp lazy_d2h() keeps the page-aligned interior of a device-to-host copy in a VRAM shadow and
copies only the range edges straight to the host. The two edge calls discard the return of call_i("d2h", ...), so a
failed edge copy still mprotects and registers the range and returns true: memcpy_impl then reports cudaSuccess and
the host keeps its stale bytes where the edge copy should have landed. A checked edge copy has to free the shadow,
clear the Python error and return false, so memcpy_impl does the plain device-to-host copy instead.

These cases build the real tgcudart.cpp and run it. No GPU, no model and no weights are touched: the tgcu_api side
is a temp stub that records the copies, and its d2h either succeeds or fails on request:
  - when every d2h fails, the copy must come back as an error, not as success with stale edge bytes (FAILS today);
  - when only the edge copies fail, the destination must still hold the device bytes, not the stale host bytes
    (FAILS today);
  - when the edge copies succeed, the lazy path still fills both edges (ordinary input, same before and after);
  - a copy below TGCU_LAZY_MIN_MB still goes straight through the plain d2h (ordinary input, same before and after).
The harness prints integers and this file turns them into PASS or FAIL lines. Run from the copy root:
python3 tests/test_fix_0096_4.py
"""
import glob
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.getcwd()
SRC = os.path.join('sidecar-shim', 'tgcudart.cpp')
PATTERN = 0x5c

# name, stub mode, lazy minimum in MiB (None keeps the default), copy length in pages, and what a correct copy gives.
CASES = [
  ('every d2h failing is reported as an error', 'all', 0, 3,
   dict(rc='nonzero')),
  ('only the edge d2h failing still leaves the device bytes', 'edge', 0, 3,
   dict(rc='zero', head=PATTERN, tail=PATTERN)),
  ('successful edge copies still fill both edges', 'none', 0, 3,
   dict(rc='zero', head=PATTERN, tail=PATTERN)),
  ('a copy below the lazy minimum still uses the plain d2h', 'none', None, 1,
   dict(rc='zero', head=PATTERN, tail=PATTERN)),
]

HARNESS = r'''// built by tests/test_fix_0096_4.py: drives the real tgcudart.cpp against host memory it owns.
#include <cuda_runtime_api.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>

int main() {
  const size_t page = (size_t)getpagesize();
  const size_t pages = (size_t)atoi(getenv("COPY_PAGES"));
  const size_t n = pages * page;
  void* dev = nullptr;
  int a = cudaMalloc(&dev, n) == cudaSuccess;
  void* base = nullptr;
  if (posix_memalign(&base, page, (pages + 1) * page)) return 2;
  memset(base, 0xAA, (pages + 1) * page);          // the stale host bytes the copy must replace
  unsigned char* dst = (unsigned char*)base + 100;  // not page aligned: the lazy path copies both edges
  cudaError_t e = cudaMemcpy(dst, dev, n, cudaMemcpyDeviceToHost);
  printf("alloc %d\n", a);
  printf("rc %d\n", (int)e);
  printf("head %d\n", (int)dst[0]);
  printf("tail %d\n", (int)dst[n - 1]);
  return 0;
}
'''

STUB = r"""
import ctypes
import os

MODE = os.environ.get("STUB_D2H", "none")
WHOLE = int(os.environ.get("STUB_N", "0"))
PATTERN = int(os.environ.get("STUB_PATTERN", "92"))
next_va = [0x100000000]


def init():
    return 0


def malloc(n):
    va = next_va[0]
    next_va[0] += 0x10000000
    return va


def free(p):
    return 0


def d2d(dst, src, n, stream=0):
    return 0


def _fill(addr, n):
    ctypes.memset(ctypes.c_void_p(addr), PATTERN, n)


def d2h(dst, src, n):
    if MODE == "all":
        return -1
    if MODE == "edge" and n != WHOLE:
        return -1
    _fill(dst, n)
    return 0


def lazy_submit():
    return 0


def lazy_ok():
    return 1


def lazy_fetch(lo, shadow, n, sem):
    _fill(lo, n)
    return 0


def synchronize():
    return 0
"""


def cuda_include():
  """The CUDA headers this Mac carries: tgcudart.cpp includes cuda_runtime_api.h and cuda.h."""
  cands = []
  for var in ('CUDA_HOME', 'CUDA_HOST'):
    if os.environ.get(var):
      cands.append(os.path.join(os.environ[var], 'include'))
  for pat in ('~/cuda-*/include', '/usr/local/cuda/include',
              '/Developer/NVIDIA/CUDA-*/include'):
    cands.extend(sorted(glob.glob(os.path.expanduser(pat)), reverse=True))
  for d in cands:
    if os.path.isfile(os.path.join(d, 'cuda_runtime_api.h')) and os.path.isfile(os.path.join(d, 'cuda.h')):
      return d
  return None


def python_include():
  """A CPython prefix with both Python.h and its dylib, for the embedding the shim links against."""
  cands = []
  try:
    import sysconfig
    inc = sysconfig.get_paths().get('include')
    if inc:
      cands.append(inc)
  except Exception:
    pass
  for pat in ('~/.local/share/uv/python/*/include/python3*', '/opt/homebrew/include/python3*',
              '/usr/local/include/python3*',
              '/Library/Frameworks/Python.framework/Versions/*/include/python3*'):
    cands.extend(sorted(glob.glob(os.path.expanduser(pat)), reverse=True))
  for inc in cands:
    ver = os.path.basename(inc)                       # python3.11
    lib = os.path.join(os.path.dirname(os.path.dirname(inc)), 'lib')
    libs = glob.glob(os.path.join(lib, 'lib%s*.dylib' % ver))
    if os.path.isfile(os.path.join(inc, 'Python.h')) and libs:
      return inc, lib, ver
  return None, None, None


def compile_harness(work):
  """Compile tgcudart.cpp and the harness into one program. Returns (exe, error)."""
  cxx = os.environ.get('CXX') or shutil.which('clang++') or shutil.which('c++') or shutil.which('g++')
  if not cxx:
    return None, 'no C++ compiler was found'
  cu = cuda_include()
  if not cu:
    return None, 'no CUDA headers (cuda_runtime_api.h and cuda.h) were found'
  pinc, plib, pver = python_include()
  if not pinc:
    return None, 'no CPython headers and dylib were found for the embedding'
  src = os.path.join(ROOT, SRC)
  if not os.path.isfile(src):
    return None, 'run this test from the copy root: ' + SRC + ' is not in ' + ROOT
  common = [cxx, '-std=c++17', '-O0', '-Wno-everything', '-I', cu, '-I', pinc,
            '-I', os.path.join(ROOT, 'sidecar-shim')]
  ccpp = os.path.join(work, 'tgcudart.o')
  cmain = os.path.join(work, 'main.o')
  exe = os.path.join(work, 'lazyedge')
  harness = os.path.join(work, 'main.cpp')
  with open(harness, 'w') as f:
    f.write(HARNESS)
  for args, what in (([src, '-c', '-o', ccpp], SRC),
                     ([harness, '-c', '-o', cmain], 'the harness')):
    r = subprocess.run(common + args, capture_output=True, text=True, timeout=300)
    if r.returncode != 0:
      return None, 'compiling %s failed: %s' % (what, (r.stderr or r.stdout).strip()[:400])
  link = [cxx, '-Wl,-dead_strip', cmain, ccpp, '-L', plib, '-l' + pver, '-Wl,-rpath,' + plib, '-o', exe]
  r = subprocess.run(link, capture_output=True, text=True, timeout=300)
  if r.returncode != 0:
    return None, 'linking the harness failed: %s' % ((r.stderr or r.stdout).strip()[:400])
  return exe, None


def run_case(exe, stub, case):
  """Run one case against the built shim. Returns (got dict or None, stderr tail, exit code)."""
  name, mode, lazy_mb, pages, _ = case
  page = os.sysconf('SC_PAGE_SIZE')
  n = pages * page
  env = {k: v for k, v in os.environ.items() if not k.startswith('TGCU_')}
  env['PYTHONPATH'] = stub
  env['PYTHONDONTWRITEBYTECODE'] = '1'
  env['STUB_D2H'] = mode
  env['STUB_N'] = str(n)
  env['STUB_PATTERN'] = str(PATTERN)
  env['COPY_PAGES'] = str(pages)
  if lazy_mb is not None:
    env['TGCU_LAZY_MIN_MB'] = str(lazy_mb)
  r = subprocess.run([exe], env=env, capture_output=True, text=True, timeout=120)
  got = {}
  for line in r.stdout.splitlines():
    parts = line.split()
    if len(parts) == 2:
      try:
        got[parts[0]] = int(parts[1])
      except ValueError:
        pass
  return got, r.stderr.strip()[-300:], r.returncode


def main():
  work = tempfile.mkdtemp(prefix='sce0096-4-')
  try:
    stub = os.path.join(work, 'stub')
    os.makedirs(stub)
    with open(os.path.join(stub, 'tgcu_api.py'), 'w') as f:
      f.write(STUB)
    exe, err = compile_harness(work)
    if err:
      print('FAIL cannot build the harness: ' + err)
      return 1
    failures = 0
    for case in CASES:
      name, mode, lazy_mb, pages, want = case
      got, err, rc = run_case(exe, stub, case)
      if got is None or not got or 'rc' not in got:
        ok = False
        detail = 'the harness did not report a result (exit %d)%s' % (rc, (': ' + err) if err else '')
      else:
        ok = True
        detail = 'cudaMemcpy rc %d, head %d, tail %d' % (got.get('rc', -1), got.get('head', -1), got.get('tail', -1))
        if want['rc'] == 'nonzero' and got['rc'] == 0:
          ok = False
          detail = 'cudaMemcpy reported success (0) although the only copies it made failed'
        elif want['rc'] == 'zero' and got['rc'] != 0:
          ok = False
          detail = 'cudaMemcpy reported error %d for a copy that should succeed' % got['rc']
        for field in ('head', 'tail'):
          if field in want and ok and got.get(field) != want[field]:
            ok = False
            detail = 'the %s of the range is 0x%02x, the stale host byte, not the device byte 0x%02x' % (
              field, got.get(field, -1), want[field])
      print(('PASS ' if ok else 'FAIL ') + name + ': ' + detail)
      failures += 0 if ok else 1
    print('all cases passed' if not failures else '%d case(s) failed' % failures)
    return 1 if failures else 0
  finally:
    shutil.rmtree(work, ignore_errors=True)


if __name__ == '__main__':
  sys.exit(main())
