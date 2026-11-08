#!/usr/bin/env python3
"""tests/test_fix_0096_0.py - cudaFreeHost must not free a buffer the application owns.

sidecar-shim/tgcudart.cpp cudaHostRegister() records the caller's buffer in g_host with the same {size, mirror}
pair that host_alloc() records for its own allocations, and cudaFreeHost() reaches free(p) for every entry it
finds. A buffer the application allocated and registered is therefore released by cudaFreeHost(), which the
application still owns: heap corruption or a crash. Only memory that host_alloc() made may be freed there.

These cases build the real tgcudart.cpp and run it. No GPU, no model and no weights are touched, and the
tgcu_api side is stubbed in a temp folder so only the host-memory bookkeeping runs:
  - a buffer registered with cudaHostRegister() is still allocated after cudaFreeHost(): the next malloc of
    the same size must not hand the same address back. This case FAILS on today's code;
  - a buffer from cudaHostAlloc() is still freed by cudaFreeHost(), so that malloc does hand it back;
  - cudaHostUnregister() still forgets a registered buffer, so cudaFreeHost() refuses it and frees nothing;
  - a second cudaFreeHost() of the same registered buffer is refused and frees nothing.
The last three are ordinary input: the same result before and after the fix. The harness prints one integer per
case and this file turns it into a PASS or FAIL line. Print one line per case and exit nonzero on any FAIL.
Run from the copy root: python3 tests/test_fix_0096_0.py
"""
import glob
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.getcwd()
SRC = os.path.join('sidecar-shim', 'tgcudart.cpp')

# The cases: name, the field the harness prints, and the value a correct cudaFreeHost gives it.
CASES = [
  ('a registered application buffer is not freed by cudaFreeHost', 'registered_not_freed', 1),
  ('a cudaHostAlloc buffer is still freed by cudaFreeHost', 'hostalloc_freed', 1),
  ('an unregistered buffer is refused and not freed', 'unregistered_not_freed', 1),
  ('a second cudaFreeHost is refused and frees nothing', 'second_free_refused', 1),
]

HARNESS = r'''// built by tests/test_fix_0096_0.py: drives the real tgcudart.cpp against host memory it owns.
#include <cuda_runtime_api.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>

static const size_t N = 65536;

int main() {
  {  // a registered application buffer must survive cudaFreeHost
    void* p = malloc(N);
    cudaHostRegister(p, N, 0);
    cudaFreeHost(p);
    void* q = malloc(N);
    printf("registered_not_freed %d\n", q != p);
    if (q != p) free(p);   // with the fix p is still ours
    free(q);
  }
  {  // memory from cudaHostAlloc is ours to free and must still be freed
    void* p = nullptr;
    cudaError_t a = cudaHostAlloc(&p, N, 0);
    if (p) cudaFreeHost(p);
    void* q = malloc(N);
    printf("hostalloc_freed %d\n", p != nullptr && a == cudaSuccess && q == p);
    free(q);
  }
  {  // unregistering forgets the buffer, so cudaFreeHost refuses it and frees nothing
    void* p = malloc(N);
    cudaHostRegister(p, N, 0);
    cudaHostUnregister(p);
    cudaError_t f = cudaFreeHost(p);
    void* q = malloc(N);
    printf("unregistered_not_freed %d\n", f != cudaSuccess && q != p);
    free(p);
    free(q);
  }
  {  // the second cudaFreeHost of a registered buffer is refused
    void* p = malloc(N);
    cudaHostRegister(p, N, 0);
    cudaFreeHost(p);
    cudaError_t f = cudaFreeHost(p);
    void* q = malloc(N);
    printf("second_free_refused %d\n", f != cudaSuccess);
    if (q != p) free(p);
    free(q);
  }
  return 0;
}
'''

STUB = "def init():\n    return 0\n"


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
  exe = os.path.join(work, 'hostmem')
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


def main():
  work = tempfile.mkdtemp(prefix='sce0096-')
  try:
    stub = os.path.join(work, 'stub')
    os.makedirs(stub)
    with open(os.path.join(stub, 'tgcu_api.py'), 'w') as f:
      f.write(STUB)
    exe, err = compile_harness(work)
    if err:
      print('FAIL cannot build the harness: ' + err)
      return 1
    env = {k: v for k, v in os.environ.items() if not k.startswith('TGCU_')}
    env['PYTHONPATH'] = stub
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    r = subprocess.run([exe], cwd=work, env=env, capture_output=True, text=True, timeout=120)
    got = {}
    for line in r.stdout.splitlines():
      parts = line.split()
      if len(parts) == 2:
        try:
          got[parts[0]] = int(parts[1])
        except ValueError:
          pass
    failures = 0
    for name, field, want in CASES:
      if field not in got:
        ok, detail = False, 'the harness did not report %s (exit %d)%s' % (
          field, r.returncode, (': ' + r.stderr.strip()[-200:]) if r.stderr.strip() else '')
      elif got[field] == want:
        ok = True
        detail = ('the next malloc of the same size did not return the buffer'
                  if field == 'registered_not_freed' else 'as before')
      else:
        ok = False
        if field == 'registered_not_freed':
          detail = "cudaFreeHost released a buffer the application owns (malloc handed the same address back)"
        elif field == 'hostalloc_freed':
          detail = 'cudaFreeHost no longer released memory that cudaHostAlloc made'
        else:
          detail = 'the buffer was freed or the call was accepted when it must not be'
      print(('PASS ' if ok else 'FAIL ') + name + ': ' + detail)
      failures += 0 if ok else 1
    print('all cases passed' if not failures else '%d case(s) failed' % failures)
    return 1 if failures else 0
  finally:
    shutil.rmtree(work, ignore_errors=True)


if __name__ == '__main__':
  sys.exit(main())
