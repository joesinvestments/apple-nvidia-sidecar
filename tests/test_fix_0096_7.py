#!/usr/bin/env python3
"""tests/test_fix_0096_7.py - cudaMemcpy must wait for a device-to-device copy issued as cudaMemcpyDefault.

sidecar-shim/tgcudart.cpp memcpy_impl() resolves cudaMemcpyDefault from the two pointers (is_dev), but cudaMemcpy()
tests the caller's unresolved kind after the copy returns:
    if (e == cudaSuccess && kind == cudaMemcpyDeviceToDevice) call_i("synchronize", nullptr);
A cudaMemcpy(dst, src, n, cudaMemcpyDefault) with both pointers on the device is resolved to DeviceToDevice only
inside memcpy_impl, so that check still sees cudaMemcpyDefault and skips the synchronize the explicit DeviceToDevice
call performs; the d2d call is asynchronous, so the bytes can still be in flight when cudaMemcpy returns.

These cases build the real tgcudart.cpp and run it. No GPU, no model and no weights are touched, and the tgcu_api
side is a stub in a temp folder that appends every call it receives to a log, so both the copy and the synchronize
are visible:
  - cudaMemcpyDefault between two device buffers issues d2d and then synchronize. This case FAILS on today's code;
  - an explicit cudaMemcpyDeviceToDevice still issues d2d and then synchronize;
  - cudaMemcpyDefault host to host copies the bytes and issues no synchronize;
  - cudaMemcpyDefault host to device issues h2d and issues no synchronize.
The last three are ordinary input: the same result before and after the fix. The TGCU_ switches a serving host
exports (lazy shadows, zero copy) are passed through to the run, only the trace and census diagnostics are dropped,
so the same cases hold under the serving configuration. This file prints one PASS or FAIL line per case and exits
nonzero on any FAIL. Run from the copy root: python3 tests/test_fix_0096_7.py
"""
import glob
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.getcwd()   # the tree under test: every test here runs with cwd at the copy root
SRC = os.path.join('sidecar-shim', 'tgcudart.cpp')

# The cases: name, the case the harness runs, the copy call that must arrive, whether a synchronize must follow it,
# and whether the copy must have landed in host memory.
CASES = [
  ('cudaMemcpyDefault device to device synchronizes after d2d', 'default_d2d', 'd2d', True, False),
  ('an explicit cudaMemcpyDeviceToDevice synchronizes after d2d', 'explicit_d2d', 'd2d', True, False),
  ('cudaMemcpyDefault host to host copies and does not synchronize', 'default_h2h', None, False, True),
  ('cudaMemcpyDefault host to device does not synchronize', 'default_h2d', 'h2d', False, False),
]

HARNESS = r'''// built by tests/test_fix_0096_7.py: drives the real tgcudart.cpp against a stub tgcu_api.
#include <cuda_runtime_api.h>
#include <cstdio>
#include <cstring>

static const size_t N = 1 << 16;

static int bad(const char* what, cudaError_t e) {
  if (e == cudaSuccess) return 0;
  std::printf("failed %s %d\n", what, (int)e);
  return 1;
}

int main(int argc, char** argv) {
  const char* c = argc > 1 ? argv[1] : "";
  if (!std::strcmp(c, "default_d2d") || !std::strcmp(c, "explicit_d2d")) {
    void* a = nullptr; void* b = nullptr;
    if (bad("cudaMalloc(a)", cudaMalloc(&a, N)) || bad("cudaMalloc(b)", cudaMalloc(&b, N))) return 1;
    cudaMemcpyKind kind = !std::strcmp(c, "default_d2d") ? cudaMemcpyDefault : cudaMemcpyDeviceToDevice;
    if (bad("cudaMemcpy", cudaMemcpy(a, b, N, kind))) return 1;
  } else if (!std::strcmp(c, "default_h2h")) {
    static char src[N], dst[N];
    std::memset(src, 0x5a, sizeof src);
    std::memset(dst, 0, sizeof dst);
    if (bad("cudaMemcpy", cudaMemcpy(dst, src, N, cudaMemcpyDefault))) return 1;
    std::printf("copied %d\n", std::memcmp(dst, src, N) == 0);
  } else if (!std::strcmp(c, "default_h2d")) {
    static char src[N];
    std::memset(src, 0x5a, sizeof src);
    void* d = nullptr;
    if (bad("cudaMalloc", cudaMalloc(&d, N)) || bad("cudaMemcpy", cudaMemcpy(d, src, N, cudaMemcpyDefault))) return 1;
  } else {
    std::printf("failed unknown case %s\n", c);
    return 1;
  }
  std::printf("done\n");
  return 0;
}
'''

# Every call tgcudart.cpp makes into tgcu_api lands in id2log's file, in order, one name per line.
STUB = r'''
import os

_next = [0x100000000]

def _log(name):
    with open(os.environ["SCE0096_LOG"], "a") as f:
        f.write(name + "\n")

def init():
    return 0

def malloc(n):
    p = _next[0]
    _next[0] += (n + 0xffff) & ~0xffff
    _log("malloc")
    return p

def free(p):
    _log("free")
    return 0

def lazy_ok():
    return 0

def d2d(dst, src, n, captured):
    _log("d2d")
    return 0

def h2d(dst, src, n):
    _log("h2d")
    return 0

def d2h(dst, src, n):
    _log("d2h")
    return 0

def synchronize():
    _log("synchronize")
    return 0
'''


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
  exe = os.path.join(work, 'memcpysync')
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


def run_case(exe, stub, work, case):
  """Run one case and return (exit code, stdout, the calls tgcu_api received)."""
  log = os.path.join(work, case + '.log')
  with open(log, 'w'):
    pass
  # keep the TGCU_ switches a serving host exports (the lazy shadows, zero copy): only the diagnostics are dropped
  env = {k: v for k, v in os.environ.items()
         if k not in ('TGCU_TRACE', 'TGCU_VERBOSE', 'TGCU_CENSUS', 'TGCU_CALL_STATS', 'TGCU_CHECK_NAMES')}
  env['PYTHONPATH'] = stub
  env['PYTHONDONTWRITEBYTECODE'] = '1'
  env['SCE0096_LOG'] = log
  r = subprocess.run([exe, case], cwd=work, env=env, capture_output=True, text=True, timeout=120)
  with open(log) as f:
    calls = [l.strip() for l in f if l.strip()]
  return r.returncode, (r.stdout or '') + (r.stderr or ''), calls


def judge(name, case, after, want_sync, copied, rc, out, calls):
  if rc != 0 or 'done' not in out:
    return False, 'the harness failed (exit %d): %s' % (rc, ' '.join(out.split())[-200:])
  if after and after not in calls:
    return False, 'the copy never reached tgcu_api.%s (%s)' % (after, ', '.join(calls) or 'no call at all')
  if after and not want_sync and 'synchronize' in calls:
    return False, ('tgcu_api.synchronize was called for a host direction copy that must not need it (calls: %s)'
                   % ', '.join(calls))
  if copied and 'copied 1' not in out:
    return False, 'the host bytes were not copied'
  if want_sync:
    rest = calls[calls.index(after) + 1:]
    if 'synchronize' not in rest:
      return False, ('tgcu_api.%s returned but the copy did not synchronize: cudaMemcpy returned while the '
                     'asynchronous copy may still be in flight (calls: %s)' % (after, ', '.join(calls)))
    return True, 'the copy synchronized after ' + after
  if copied:
    return True, 'the host copy landed and no synchronize was issued'
  return True, 'the copy reached tgcu_api.%s and no synchronize was issued' % after


def main():
  work = tempfile.mkdtemp(prefix='sce0096-7-')
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
    for name, case, after, want_sync, copied in CASES:
      rc, out, calls = run_case(exe, stub, work, case)
      ok, detail = judge(name, case, after, want_sync, copied, rc, out, calls)
      print(('PASS ' if ok else 'FAIL ') + name + ': ' + detail)
      failures += 0 if ok else 1
    print('all cases passed' if not failures else '%d case(s) failed' % failures)
    return 1 if failures else 0
  finally:
    shutil.rmtree(work, ignore_errors=True)


if __name__ == '__main__':
  sys.exit(main())
