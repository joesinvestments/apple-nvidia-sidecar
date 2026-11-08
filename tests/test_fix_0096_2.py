#!/usr/bin/env python3
"""tests/test_fix_0096_2.py - cudaHostRegister fails only when map_host itself failed.

sidecar-shim/tgcudart.cpp cudaHostRegister() asks tgcu_api.map_host for a device mirror of the caller's buffer
when the caller passed cudaHostRegisterMapped. map_host answers -1 (or raises) when the mirror cannot be made,
and 0 when the region is deliberately left without one: tgcu_api.map_host returns 0 for a region over the
static limit ("beyond the limit (an expert arena) there is no device alias and the program takes its no-alias
path"), and the engine registers that arena and then keeps it in locked host memory.

So the two answers mean different things:
  - -1, or a raise, is a failure: cudaHostRegister must return cudaErrorMemoryAllocation, record no
    registration, and leave no Python error pending. Today's code gets this right (it clears the error and
    fails), and it must keep doing so;
  - 0 is success: the registration must be accepted and recorded with no mirror, so cudaHostUnregister accepts
    it and cudaHostGetDevicePointer answers "region not mirrored" instead of the registration being refused.
    Today's code refuses it with cudaErrorMemoryAllocation, so this case FAILS on today's code.

These cases build the real tgcudart.cpp and run it. No GPU, no model and no weights are touched, and the
tgcu_api side is stubbed in a temp folder by a small Python module whose map_host returns a planted answer.
The unmapped registration and the working map_host are ordinary input: the same result before and after.
The harness prints one integer per field and this file turns them into PASS or FAIL lines. Print one line
per case and exit nonzero on any FAIL.
Run from the copy root: python3 tests/test_fix_0096_2.py
"""
import glob
import os
import shutil
import subprocess
import sys
import tempfile

REL = os.path.join('sidecar-shim', 'tgcudart.cpp')
MIRROR = 0x40000000
SENTINEL = 0x1          # left in the caller's device pointer when a call must not write it
SMALL = 65536           # a mapped buffer under tgcu_api.MAPPED_MAX
ARENA = 32 << 20        # an expert arena over tgcudart's 8 MiB kMappedMax: entered through the static-mirror path
CUDA_SUCCESS = 0
CUDA_ERROR_MEMORY_ALLOCATION = 2

# name, the tgcu_api.map_host answer, the flags, the buffer size, and what a correct shim gives each call.
CASES = [
  ('a map_host that raises fails the mapped registration with cudaErrorMemoryAllocation', 'error', True, SMALL,
   dict(register='alloc', device='fail', unregister='fail', map_called=True)),
  ('a map_host that returns -1 fails the mapped registration with cudaErrorMemoryAllocation', 'neg', True, SMALL,
   dict(register='alloc', device='fail', unregister='fail', map_called=True)),
  ('an arena over the static limit registers with no mirror and returns success', 'zero', True, ARENA,
   dict(register='ok', device='fail', unregister='ok', map_called=True)),
  ('an unmapped registration needs no mirror and still succeeds', 'zero', False, SMALL,
   dict(register='ok', device='fail', unregister='ok', map_called=False)),
  ('a working map_host still yields the mirror as the device pointer', 'ok', True, SMALL,
   dict(register='ok', device='mirror', unregister='ok', map_called=True)),
]

HARNESS = r'''// built by tests/test_fix_0096_2.py: drives the real tgcudart.cpp, map_host stubbed by tgcu_api.py.
#include <Python.h>
#include <cuda_runtime_api.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>

int main(int argc, char** argv) {
  const char* mode = argc > 1 ? argv[1] : "ok";
  const bool mapped = argc > 2 && strcmp(argv[2], "mapped") == 0;
  const size_t N = argc > 3 ? (size_t) strtoull(argv[3], nullptr, 10) : 65536;
  (void) mode;   // tgcu_api.py reads it from TG_STUB_MAP
  const uintptr_t kSentinel = 0x1;   // SENTINEL in the Python file: unchanged means no pointer was written
  void* p = malloc(N);
  if (p == nullptr) { printf("register_rc -2\n"); return 2; }
  memset(p, 0, N);
  cudaError_t reg = cudaHostRegister(p, N, mapped ? cudaHostRegisterMapped : 0);
  int pending = 0;
  if (Py_IsInitialized()) {   // an unmapped registration never starts Python
    PyGILState_STATE g = PyGILState_Ensure();
    pending = PyErr_Occurred() != nullptr;
    PyErr_Clear();
    PyGILState_Release(g);
  }
  void* d = (void*)kSentinel;
  cudaError_t got = cudaHostGetDevicePointer(&d, p, 0);
  cudaError_t un = cudaHostUnregister(p);
  printf("register_rc %d\n", (int)reg);
  printf("device_rc %d\n", (int)got);
  printf("device_ptr %llu\n", (unsigned long long)(uintptr_t)d);
  printf("unregister_rc %d\n", (int)un);
  printf("pending_pyerr %d\n", pending);
  free(p);
  return 0;
}
'''

STUB = r'''import os

def _note(name):
  p = os.environ.get('TG_STUB_CALLS')
  if p:
    with open(p, 'a') as f:
      f.write(name + '\n')

def init():
  return 0

def map_host(hptr, size):
  _note('map_host')
  mode = os.environ.get('TG_STUB_MAP', 'ok')
  if mode == 'zero':
    return 0
  if mode == 'neg':
    return -1
  if mode == 'error':
    raise RuntimeError('map_host: planted failure')
  return 0x40000000

def unmap_host(hptr):
  _note('unmap_host')
  return 0
'''


def find_root():
  """The tree under test: the copy prove_test runs us in, else the one this file sits in."""
  for base in (os.getcwd(), os.path.dirname(os.path.dirname(os.path.abspath(__file__)))):
    if os.path.isfile(os.path.join(base, REL)):
      return base
  return os.getcwd()


ROOT = find_root()


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
  src = os.path.join(ROOT, REL)
  if not os.path.isfile(src):
    return None, 'run this test from the copy root: ' + REL + ' is not in ' + ROOT
  common = [cxx, '-std=c++17', '-O0', '-Wno-everything', '-I', cu, '-I', pinc,
            '-I', os.path.join(ROOT, 'sidecar-shim')]
  ccpp = os.path.join(work, 'tgcudart.o')
  cmain = os.path.join(work, 'main.o')
  exe = os.path.join(work, 'hostreg')
  harness = os.path.join(work, 'main.cpp')
  with open(harness, 'w') as f:
    f.write(HARNESS)
  for args, what in (([src, '-c', '-o', ccpp], REL),
                     ([harness, '-c', '-o', cmain], 'the harness')):
    r = subprocess.run(common + args, capture_output=True, text=True, timeout=300)
    if r.returncode != 0:
      return None, 'compiling %s failed: %s' % (what, (r.stderr or r.stdout).strip()[:400])
  link = [cxx, '-Wl,-dead_strip', cmain, ccpp, '-L', plib, '-l' + pver, '-Wl,-rpath,' + plib, '-o', exe]
  r = subprocess.run(link, capture_output=True, text=True, timeout=300)
  if r.returncode != 0:
    return None, 'linking the harness failed: %s' % ((r.stderr or r.stdout).strip()[:400])
  return exe, None


def run_case(exe, work, mode, mapped, size):
  """Run the harness once with a planted map_host answer. Returns the printed fields and the calls made."""
  calls = os.path.join(work, 'calls-%s-%s-%d' % (mode, 'mapped' if mapped else 'plain', size))
  if os.path.exists(calls):
    os.unlink(calls)
  env = {k: v for k, v in os.environ.items() if not k.startswith(('TGCU_', 'TG_STUB_'))}
  env['PYTHONPATH'] = os.path.join(work, 'stub')
  env['PYTHONDONTWRITEBYTECODE'] = '1'
  env['TG_STUB_MAP'] = mode
  env['TG_STUB_CALLS'] = calls
  r = subprocess.run([exe, mode, 'mapped' if mapped else 'plain', str(size)], cwd=work, env=env,
                     capture_output=True, text=True, timeout=120)
  got = {}
  for line in r.stdout.splitlines():
    parts = line.split()
    if len(parts) == 2:
      try:
        got[parts[0]] = int(parts[1])
      except ValueError:
        pass
  made = []
  if os.path.exists(calls):
    with open(calls) as f:
      made = [l.strip() for l in f if l.strip()]
  return got, made, r


def check(want, got, made):
  """Compare one run with the expected calls. Returns (ok, detail)."""
  if 'register_rc' not in got:
    return False, 'the harness did not report register_rc'
  reg = got['register_rc']
  if want['register'] == 'alloc':
    if reg != CUDA_ERROR_MEMORY_ALLOCATION:
      return False, ('cudaHostRegister answered %d after map_host failed, not cudaErrorMemoryAllocation (%d)'
                     % (reg, CUDA_ERROR_MEMORY_ALLOCATION))
  elif reg != CUDA_SUCCESS:
    return False, 'cudaHostRegister refused a registration that must succeed (%d)' % reg
  dev = got.get('device_rc')
  ptr = got.get('device_ptr')
  if want['device'] == 'fail':
    if dev == 0 or ptr != SENTINEL:
      return False, 'cudaHostGetDevicePointer succeeded or wrote a pointer for an unregistered region'
  else:
    if dev != 0 or ptr != MIRROR:
      return False, 'cudaHostGetDevicePointer did not return the mirror (%s, %s)' % (dev, ptr)
  un = got.get('unregister_rc')
  if want['unregister'] == 'fail' and un == 0:
    return False, 'cudaHostUnregister accepted a registration that failed'
  if want['unregister'] == 'ok' and un != 0:
    return False, 'cudaHostUnregister refused a registration that succeeded'
  called = 'map_host' in made
  if called != want['map_called']:
    return False, ('map_host was called for an unmapped registration' if called
                   else 'map_host was not called for a mapped registration')
  if got.get('pending_pyerr') != 0:
    return False, 'a Python error was left pending when cudaHostRegister returned'
  return True, 'as expected'


def main():
  work = tempfile.mkdtemp(prefix='sce0096b-')
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
    for name, mode, mapped, size, want in CASES:
      got, made, r = run_case(exe, work, mode, mapped, size)
      ok, detail = check(want, got, made)
      if not ok and not got:
        detail += ' (exit %d%s)' % (r.returncode, (': ' + r.stderr.strip()[-200:]) if r.stderr.strip() else '')
      print(('PASS ' if ok else 'FAIL ') + name + ': ' + detail)
      failures += 0 if ok else 1
    print('all cases passed' if not failures else '%d case(s) failed' % failures)
    return 1 if failures else 0
  finally:
    shutil.rmtree(work, ignore_errors=True)


if __name__ == '__main__':
  sys.exit(main())
