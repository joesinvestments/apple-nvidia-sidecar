#!/usr/bin/env python3
"""tests/test_fix_0096_3.py - tgcuSysmemAlloc must not report success when the host address fails.

sidecar-shim/tgcudart.cpp tgcuSysmemAlloc() checks the sysmem_alloc result (va) but stores the sysmem_host result
without any check. When tgcu_api.sysmem_host() answers 0 or -1 the caller gets a bogus host pointer and
cudaSuccess while the sysmem block stays allocated (the caller at src/program/generate.cpp only sees success and
never frees it). host_alloc() already checks the same call and frees the block on failure.

These cases build the real tgcudart.cpp and run it against a fake tgcu_api in a temp folder. No GPU, no model and
no weights are touched; only the sysmem bookkeeping runs:
  - a failed host-address lookup (-1) is an error, not cudaSuccess. This case FAILS on today's code;
  - a zero host-address lookup is an error too. This case FAILS on today's code;
  - the sysmem block is given back with sysmem_free when the lookup fails. This case FAILS on today's code;
  - a failed lookup hands back no bogus host pointer. This case FAILS on today's code;
  - a successful lookup still returns the host and device addresses;
  - the device address of a successful lookup is still released by tgcuSysmemFree;
  - a failed sysmem_alloc is still an error and never looks up a host address.
The last three are ordinary input: the same result before and after the fix. Print one PASS or FAIL line per case
and exit nonzero on any FAIL. Run from the copy root: python3 tests/test_fix_0096_3.py
"""
import glob
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.getcwd()
SRC = os.path.join('sidecar-shim', 'tgcudart.cpp')
SIZE = 65536
VA = 0x7f0000000000        # what the fake sysmem_alloc hands back
HOST = 0x7f0000010000      # what the fake sysmem_host hands back on success
SENTINEL = 1               # the harness fills both out pointers with 1 before the call
BOGUS = 0xffffffffffffffff  # what a -1 lookup becomes as a host pointer

HARNESS = r'''// built by tests/test_fix_0096_3.py: drives the real tgcudart.cpp against a fake tgcu_api.
#include <cstddef>
#include <cstdint>
#include <cstdio>

extern "C" int tgcuSysmemAlloc(size_t size, void** host, void** dev);
extern "C" int tgcuSysmemFree(void* dev);

int main(int argc, char** argv) {
  void* host = (void*)(uintptr_t)1;   // sentinel: only tgcuSysmemAlloc may replace it
  void* dev = (void*)(uintptr_t)1;
  const int rc = tgcuSysmemAlloc((size_t)65536, &host, &dev);
  int frc = -999;
  if (argc > 1 && argv[1][0] == 'f' && rc == 0 && dev) frc = tgcuSysmemFree(dev);
  std::printf("rc=%d host=%llx dev=%llx frc=%d\n", rc,
              (unsigned long long)(uintptr_t)host, (unsigned long long)(uintptr_t)dev, frc);
  return 0;
}
'''

STUB = '''import os

LOG = os.environ.get('SCE0096_LOG') or ''


def _log(name, *args):
    if not LOG:
        return
    with open(LOG, 'a', encoding='utf-8') as f:
        f.write(name + ' ' + ' '.join(str(a) for a in args) + '\\n')


def _num(name, default):
    try:
        return int(os.environ.get(name, ''), 0)
    except ValueError:
        return default


def init():
    return 0


def sysmem_alloc(size):
    _log('sysmem_alloc', size)
    return _num('SCE0096_ALLOC', 0x7f0000000000)


def sysmem_host(va):
    _log('sysmem_host', va)
    return _num('SCE0096_HOST', 0x7f0000010000)


def sysmem_free(va):
    _log('sysmem_free', va)
    return 0
'''

# name, fake sysmem_alloc reply, fake sysmem_host reply, harness frees the result, check(state) -> (ok, detail)
SYS_FAIL = -1
LOOKUP_FAIL = -1
LOOKUP_ZERO = 0


def state(alloc, host, free, log):
    return dict(alloc=alloc, host=host, free=free, log=log)


def calls(log):
    out = []
    for line in log.splitlines():
        parts = line.split()
        if len(parts) == 2:
            try:
                out.append((parts[0], int(parts[1])))
            except ValueError:
                pass
    return out


def want_rc_failure(s):
    if s['rc'] == 0:
        return False, 'tgcuSysmemAlloc returned cudaSuccess (0) after the host-address lookup failed, so the caller ' \
                      'carries on with a bogus pointer and never frees the block'
    return True, 'reported error %d' % s['rc']


def want_free(s):
    freed = [v for (n, v) in calls(s['log']) if n == 'sysmem_free']
    if not freed:
        return False, 'the sysmem block (va %#x) was leaked: sysmem_free was never called' % s['va']
    if freed != [s['va']]:
        return False, 'sysmem_free was called with %s, expected only the leaked block %#x' % (
            ', '.join(hex(v) for v in freed), s['va'])
    return True, 'the block %#x went back through sysmem_free' % s['va']


def want_no_bogus_host(s):
    if s['host'] == BOGUS:
        return False, 'the -1 lookup became host %#x and was still handed back as a pointer' % s['host']
    if s['host'] not in (SENTINEL, 0):
        return False, 'a failed lookup handed back host %#x' % s['host']
    return True, 'no bogus host pointer'


def want_success(s):
    if s['rc'] != 0:
        return False, 'a valid lookup returned error %d' % s['rc']
    if s['host'] != HOST or s['dev'] != VA:
        return False, 'returned host %#x dev %#x, expected host %#x dev %#x' % (s['host'], s['dev'], HOST, VA)
    names = [n for (n, _) in calls(s['log'])]
    if names != ['sysmem_alloc', 'sysmem_host']:
        return False, 'the tgcu_api calls were %s, expected sysmem_alloc then sysmem_host' % (names or 'none')
    return True, 'host %#x and dev %#x' % (s['host'], s['dev'])


def want_free_ok(s):
    if s['rc'] != 0 or s['frc'] != 0:
        return False, 'alloc returned %d and tgcuSysmemFree returned %d' % (s['rc'], s['frc'])
    freed = [v for (n, v) in calls(s['log']) if n == 'sysmem_free']
    if freed != [VA]:
        return False, 'sysmem_free was called with %s, expected the device address %#x' % (
            ', '.join(hex(v) for v in freed) or 'nothing', VA)
    return True, 'the device address %#x was released' % VA


def want_alloc_failure(s):
    if s['rc'] == 0:
        return False, 'a failed sysmem_alloc still returned cudaSuccess'
    names = [n for (n, _) in calls(s['log'])]
    if names != ['sysmem_alloc']:
        return False, 'the tgcu_api calls were %s, expected only sysmem_alloc' % (names or 'none')
    return True, 'reported error %d and looked up no host address' % s['rc']


CASES = [
    ('a failed host-address lookup is an error, not a success', VA, LOOKUP_FAIL, False, want_rc_failure),
    ('a zero host-address lookup is an error too', VA, LOOKUP_ZERO, False, want_rc_failure),
    ('the sysmem block is released when the host lookup fails', VA, LOOKUP_FAIL, False, want_free),
    ('a failed lookup hands back no bogus host pointer', VA, LOOKUP_FAIL, False, want_no_bogus_host),
    ('a successful lookup still returns the host and device addresses', VA, HOST, False, want_success),
    ('the device address of a successful lookup is still released', VA, HOST, True, want_free_ok),
    ('a failed sysmem_alloc is still an error and looks up no host address', SYS_FAIL, HOST, False, want_alloc_failure),
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
    exe = os.path.join(work, 'sysmem')
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


def run_case(exe, work, index, alloc, host):
    """Run the harness once with the fake replies and return what it printed plus the tgcu_api call log."""
    log_path = os.path.join(work, 'case%02d.log' % index)
    env = {k: v for k, v in os.environ.items() if not k.startswith('TGCU_')}
    env['PYTHONPATH'] = os.path.join(work, 'stub')
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    env['SCE0096_LOG'] = log_path
    env['SCE0096_ALLOC'] = str(alloc)
    env['SCE0096_HOST'] = str(host)
    r = subprocess.run([exe, 'free' if index == 5 else ''], cwd=work, env=env,
                       capture_output=True, text=True, timeout=120)
    log = ''
    if os.path.isfile(log_path):
        with open(log_path, encoding='utf-8') as f:
            log = f.read()
    s = state(alloc, host, index == 5, log)
    s['va'] = VA
    for line in r.stdout.splitlines():
        parts = line.split()
        if len(parts) == 4 and parts[0].startswith('rc='):
            s['rc'] = int(parts[0][3:])
            s['host_out'] = int(parts[1][5:], 16)
            s['dev_out'] = int(parts[2][4:], 16)
            s['frc'] = int(parts[3][4:])
    s['host'] = s.get('host_out')
    s['dev'] = s.get('dev_out')
    s['stdout'] = r.stdout.strip()
    s['returncode'] = r.returncode
    s['stderr'] = r.stderr.strip()
    return s


def main():
    work = tempfile.mkdtemp(prefix='sce0096-3-')
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
        for index, (name, alloc, host, free, check) in enumerate(CASES):
            s = run_case(exe, work, index, alloc, host)
            if 'rc' not in s:
                ok, detail = False, 'the harness printed nothing it could parse (exit %d)%s' % (
                    s['returncode'], (': ' + s['stderr'][-200:]) if s['stderr'] else '')
            else:
                ok, detail = check(s)
            print(('PASS ' if ok else 'FAIL ') + name + ': ' + detail)
            failures += 0 if ok else 1
        print('all cases passed' if not failures else '%d case(s) failed' % failures)
        return 1 if failures else 0
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == '__main__':
    sys.exit(main())
