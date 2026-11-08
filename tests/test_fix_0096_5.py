#!/usr/bin/env python3
"""tests/test_fix_0096_5.py - a failed protection must not keep the stale host bytes.

sidecar-shim/tgcudart.cpp lazy_d2h() makes the page interior of a device-to-host destination PROT_NONE, then records
the range in g_lazy. The mprotect call is unchecked: when it fails (an unmapped or odd range, a VM limit) the pages
stay readable and writable, so the entry still claims a VRAM shadow and a later host read of that range returns the
stale host bytes instead of faulting the shadow in. Nothing ever downloads the device bytes.

These cases build the real tgcudart.cpp and run it against a stub tgcu_api (no GPU, no model, no weights). The shim's
mprotect is routed to a stub that can be made to fail, the way the real one fails on a range it cannot protect:
  - with the protection failing, the destination still ends up with the device bytes. This case FAILS on today's
    code: the range is kept lazy and the interior keeps the stale zeros it was filled with;
  - with the protection failing, the shadow malloc is handed back instead of leaked. This case FAILS on today's code;
  - a copy smaller than TGCU_LAZY_MIN_MB still copies every byte and leaves the range plain memory (same before and
    after the fix);
  - a copy that can be protected is still kept lazily: the interior is inaccessible and only the edges were copied
    (same before and after the fix).
The harness prints one integer per field and this file turns it into a PASS or FAIL line. Ordinary input gives the
same result before and after the fix. Print one line per case and exit nonzero on any FAIL.
Run from the copy root: python3 tests/test_fix_0096_5.py
"""
import glob
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.getcwd()
SRC = os.path.join('sidecar-shim', 'tgcudart.cpp')

# The shim's mprotect is sent here by a header force-included into tgcudart.cpp, so the failure path can be observed
# without a GPU. Every other call in the file goes straight to the real mprotect.
FAIL_MPROTECT = r'''// force-included into sidecar-shim/tgcudart.cpp by tests/test_fix_0096_5.py
#pragma once
#include <stddef.h>
#include <sys/mman.h>
extern "C" int test_mprotect(void* addr, size_t len, int prot);
extern int tgcu_test_mprotect_fail;
#define mprotect(addr, len, prot) test_mprotect((void*)(addr), (size_t)(len), (prot))
'''

HARNESS = r'''// built by tests/test_fix_0096_5.py: drives the real tgcudart.cpp device-to-host path on host memory.
#include <cuda_runtime_api.h>
#include <mach/mach.h>
#include <mach/mach_vm.h>
#include <sys/mman.h>
#include <unistd.h>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>

int tgcu_test_mprotect_fail = 0;

extern "C" int test_mprotect(void* addr, size_t len, int prot) {
  if (tgcu_test_mprotect_fail) { errno = ENOMEM; return -1; }   // what a range the shim cannot protect returns
  return mprotect(addr, len, prot);
}

// 1 when the page at at is inaccessible, 0 when it is readable and writable
static int protected_here(uintptr_t at) {
  mach_vm_address_t base = (mach_vm_address_t)at, a = base;
  mach_vm_size_t sz = 0;
  vm_region_basic_info_data_64_t info;
  mach_msg_type_number_t cnt = VM_REGION_BASIC_INFO_COUNT_64;
  mach_port_t obj = MACH_PORT_NULL;
  if (mach_vm_region(mach_task_self(), &a, &sz, VM_REGION_BASIC_INFO_64, (vm_region_info_t)&info, &cnt, &obj) != KERN_SUCCESS)
    return -1;
  if (a > base) return 0;
  return info.protection == VM_PROT_NONE ? 1 : 0;
}

int main(int argc, char** argv) {
  const std::string mode = argc > 1 ? argv[1] : "";
  const size_t page = (size_t)getpagesize();
  const size_t n = mode == "small" ? (64u << 10) : (4u << 20);
  const size_t len = n + 2 * page;
  char* base = (char*)mmap(nullptr, len, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
  if (base == MAP_FAILED) { printf("error 3\n"); return 3; }
  memset(base, 0x00, len);
  char* dst = base + 100;   // 100 stale bytes at each end: both edges are real copies either way
  if (mode == "protect_fails") tgcu_test_mprotect_fail = 1;
  cudaError_t rc = cudaMemcpy(dst, (const void*)(uintptr_t)0x700000000000ull, n, cudaMemcpyDeviceToHost);
  printf("rc %d\n", (int)rc);
  printf("edge_byte %d\n", (int)(unsigned char)dst[0]);
  printf("interior_prot %d\n", protected_here((uintptr_t)base + page));
  if (mode != "protect_works") printf("interior_byte %d\n", (int)(unsigned char)base[page]);
  return 0;
}
'''

STUB = r'''"""tests/test_fix_0096_5.py stub: the tgcu_api side of the shim, on the host, with no GPU."""
import ctypes
import os

LOG = os.environ.get('TGCU_STUB_LOG', '')
_shadow = [0x1000000]


def _log(line):
    if LOG:
        with open(LOG, 'a') as f:
            f.write(line + '\n')


def init():
    return 0


def lazy_ok():
    return 1


def malloc(size):
    _shadow[0] += 0x1000000
    _log('malloc %d' % size)
    return _shadow[0]


def free(a):
    _log('free %d' % a)
    return 0


def d2d(dst, src, size, cap=0):
    _log('d2d %d %d' % (dst, size))
    return 0


def d2h(dst, src, size):
    _log('d2h %d %d' % (dst, size))
    ctypes.memset(dst, 0xAB, size)   # the device bytes: afterwards every byte of the range reads 0xAB
    return 0


def lazy_submit():
    _log('lazy_submit')
    return 1


def synchronize():
    return 0
'''

# name, the harness run it reads, the field it checks and the value a correct shim gives it
CASES = [
  dict(name='a failed protection still delivers the device bytes', mode='protect_fails', field='interior_byte', want=0xAB,
       why='the interior kept its stale host bytes: the range was kept lazy and nothing faulted in the shadow'),
  dict(name='a failed protection hands the shadow back', mode='protect_fails', field='frees', want=1,
       why='the shadow malloc was kept after the protection failed'),
  dict(name='a failed protection still reports success', mode='protect_fails', field='rc', want=0,
       why='the copy failed where a plain d2h would have carried it'),
  dict(name='a copy under the lazy minimum copies every byte', mode='small', field='interior_byte', want=0xAB,
       why='the interior was not copied'),
  dict(name='a copy under the lazy minimum leaves plain memory', mode='small', field='interior_prot', want=0,
       why='the small copy was protected'),
  dict(name='a copy that can be protected keeps its interior lazy', mode='protect_works', field='interior_prot', want=1,
       why='the lazy path was dropped although the protection succeeded'),
  dict(name='a copy that can be protected copies only the edges', mode='protect_works', field='edge_byte', want=0xAB,
       why='the edge bytes were not copied'),
]


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
  cxx = os.environ.get('CXX') or shutil.which('clang++') or shutil.which('c++')
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
  exe = os.path.join(work, 'lazyd2h')
  hdr = os.path.join(work, 'fail_mprotect.h')
  harness = os.path.join(work, 'main.cpp')
  with open(hdr, 'w') as f:
    f.write(FAIL_MPROTECT)
  with open(harness, 'w') as f:
    f.write(HARNESS)
  for args, what in (([src, '-include', hdr, '-c', '-o', ccpp], SRC),
                     ([harness, '-c', '-o', cmain], 'the harness')):
    r = subprocess.run(common + args, capture_output=True, text=True, timeout=300)
    if r.returncode != 0:
      return None, 'compiling %s failed: %s' % (what, (r.stderr or r.stdout).strip()[:400])
  link = [cxx, '-Wl,-dead_strip', cmain, ccpp, '-L', plib, '-l' + pver, '-Wl,-rpath,' + plib, '-o', exe]
  r = subprocess.run(link, capture_output=True, text=True, timeout=300)
  if r.returncode != 0:
    return None, 'linking the harness failed: %s' % ((r.stderr or r.stdout).strip()[:400])
  return exe, None


def run_mode(exe, work, stub, mode):
  """Run one case mode and return its printed fields and the calls the stub tgcu_api saw."""
  log = os.path.join(work, 'calls-%s.log' % mode)
  if os.path.exists(log):
    os.remove(log)
  env = {k: v for k, v in os.environ.items() if not k.startswith('TGCU_')}
  env['PYTHONPATH'] = stub
  env['PYTHONDONTWRITEBYTECODE'] = '1'
  env['TGCU_LAZY_D2H_MB'] = '64'      # a budget that keeps the 4 MiB range
  env['TGCU_LAZY_MIN_MB'] = '1'       # the default, named so the sizes do not move under an inherited setting
  env['TGCU_STUB_LOG'] = log
  r = subprocess.run([exe, mode], cwd=work, env=env, capture_output=True, text=True, timeout=120)
  got = {}
  for line in r.stdout.splitlines():
    parts = line.split()
    if len(parts) == 2:
      try:
        got[parts[0]] = int(parts[1])
      except ValueError:
        pass
  calls = []
  if os.path.isfile(log):
    with open(log) as f:
      calls = f.read().splitlines()
  got['frees'] = sum(1 for c in calls if c.startswith('free '))
  return got, calls, r


def main():
  work = tempfile.mkdtemp(prefix='sce0096-5-')
  try:
    stub = os.path.join(work, 'stub')
    os.makedirs(stub)
    with open(os.path.join(stub, 'tgcu_api.py'), 'w') as f:
      f.write(STUB)
    exe, err = compile_harness(work)
    if err:
      print('FAIL cannot build the harness: ' + err)
      return 1
    runs = {}
    failures = 0
    for case in CASES:
      if case['mode'] not in runs:
        runs[case['mode']] = run_mode(exe, work, stub, case['mode'])
      got, calls, r = runs[case['mode']]
      name = case['name']
      if case['field'] not in got:
        ok = False
        detail = 'the harness printed no %s (exit %d)%s' % (
          case['field'], r.returncode, (': ' + r.stderr.strip()[-200:]) if r.stderr.strip() else '')
      elif got[case['field']] == case['want']:
        ok = True
        detail = 'as expected'
      else:
        ok = False
        detail = '%s (read %d, wanted %d)' % (case['why'], got[case['field']], case['want'])
      print(('PASS ' if ok else 'FAIL ') + name + ': ' + detail)
      failures += 0 if ok else 1
    print('all cases passed' if not failures else '%d case(s) failed' % failures)
    return 1 if failures else 0
  finally:
    shutil.rmtree(work, ignore_errors=True)


if __name__ == '__main__':
  sys.exit(main())
