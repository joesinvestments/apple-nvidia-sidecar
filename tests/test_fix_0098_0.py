#!/usr/bin/env python3
"""tests/test_fix_0098_0.py - the shim's clock rate must be the installed card's, not a literal.

sidecar-shim/tgcudart.cpp:1054 answers cudaDevAttrClockRate with *v = 1695000 for every device, and
src/program/generate.cpp:6087 feeds that number into the layer split's SMs x clock cost model (it falls
back only on an error or a non-positive answer).  So on any card but the one the literal came from (RTX
5090 config/sidecar.json env, 3090 defaults) the planner prices a layer against a clock that card does not
have: 1.695 GHz on a card whose boost clock is 2.4 GHz.  The shim's geometry resolver (sidecar-shim/geom.py,
fix-0004) already carries the fields the backend cannot report, so the clock belongs there too.

These cases take the real env_int, dev_field, dev_sm, fill_props and cudaDeviceGetAttribute out of
sidecar-shim/tgcudart.cpp, compile them against a stub tgcu (no GPU, no CUDA headers, no model) and run
the built program on planted geometry and environment:
  - a config that names a different card's clock is answered by cudaDeviceGetAttribute, not 1695000;
  - the selected serving config (config/sidecar.json) names its card's clock;
  - with no setting the answer is still 1695000 (ordinary input unchanged);
  - bus width, L2, SM count and warp size still answer from the geometry.
Prints one PASS or FAIL line per case and exits nonzero on any FAIL. Run: python3 tests/test_fix_0098_0.py
"""
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.getcwd()   # the tree under test: every test here runs with cwd at the copy root, never at the test file
SHIM = os.path.join('sidecar-shim', 'tgcudart.cpp')
GEOM = os.path.join('sidecar-shim', 'geom.py')
CONFIG = os.path.join('config', 'sidecar.json')

CLOCK_KEY = 'clock_mhz'
CLOCK_ENV = 'TGCU_DEV_CLOCK_MHZ'
CONSTANT_KHZ = 1695000                       # cudaDevAttrClockRate's literal today: 1.695 GHz, one card's clock
DEVICE_KEYS = ('sm', 'sms', 'mem_gib')       # what tgcu's device answers; L2, bus width and clock are the config's fallback

SIGNATURES = (
    'static int env_int(',
    'static int dev_field(',
    'static int dev_sm(',
    'void fill_props(',
    'EXPORT cudaError_t cudaDeviceGetAttribute(',
)

# the harness supplies what the extracted code needs from libcudart and tgcu: the error type, every attribute
# name the switch reads, the device property record and the geometry query.  The shim's own code is copied in
# as it is written, so the arithmetic and the index order under test are the shim's, not the test's.
HARNESS = r"""
#include <cstdarg>
#include <climits>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#define EXPORT
#define API_ENTER(name)

typedef int cudaError_t;
typedef int cudaDeviceAttr;
static const cudaError_t cudaSuccess = 0;

enum {
  cudaDevAttrMaxThreadsPerBlock = 1,
  cudaDevAttrMaxGridDimX = 2,
  cudaDevAttrMaxGridDimY = 3,
  cudaDevAttrMaxGridDimZ = 4,
  cudaDevAttrMaxBlockDimX = 5,
  cudaDevAttrMaxBlockDimY = 6,
  cudaDevAttrMaxBlockDimZ = 7,
  cudaDevAttrMaxSharedMemoryPerBlock = 8,
  cudaDevAttrWarpSize = 10,
  cudaDevAttrMaxRegistersPerBlock = 12,
  cudaDevAttrClockRate = 13,
  cudaDevAttrMultiProcessorCount = 16,
  cudaDevAttrIntegrated = 18,
  cudaDevAttrCanMapHostMemory = 19,
  cudaDevAttrMemoryClockRate = 36,
  cudaDevAttrGlobalMemoryBusWidth = 37,
  cudaDevAttrL2CacheSize = 38,
  cudaDevAttrMaxThreadsPerMultiProcessor = 39,
  cudaDevAttrUnifiedAddressing = 41,
  cudaDevAttrPciBusId = 50,
  cudaDevAttrComputeCapabilityMajor = 75,
  cudaDevAttrComputeCapabilityMinor = 76,
  cudaDevAttrMaxSharedMemoryPerMultiprocessor = 81,
  cudaDevAttrMaxRegistersPerMultiprocessor = 82,
  cudaDevAttrManagedMemory = 83,
  cudaDevAttrPageableMemoryAccess = 88,
  cudaDevAttrConcurrentManagedAccess = 89,
  cudaDevAttrCooperativeLaunch = 95,
  cudaDevAttrMaxSharedMemoryPerBlockOptin = 97,
  cudaDevAttrHostRegisterSupported = 99,
  cudaDevAttrMemoryPoolsSupported = 101,
  cudaDevAttrMaxBlocksPerMultiprocessor = 106,
  cudaDevAttrClusterLaunch = 120,
};

struct cudaDeviceProp {
  char name[256];
  size_t totalGlobalMem;
  size_t sharedMemPerBlock;
  int regsPerBlock;
  int warpSize;
  size_t memPitch;
  int maxThreadsPerBlock;
  int maxThreadsDim[3];
  int maxGridSize[3];
  size_t totalConstMem;
  int major;
  int minor;
  size_t textureAlignment;
  int multiProcessorCount;
  int integrated;
  int canMapHostMemory;
  int computeMode;
  int concurrentKernels;
  int pciBusID;
  int asyncEngineCount;
  int unifiedAddressing;
  int memoryBusWidth;
  int l2CacheSize;
  size_t persistingL2CacheMaxSize;
  int maxThreadsPerMultiProcessor;
  size_t sharedMemPerMultiprocessor;
  int regsPerMultiprocessor;
  int managedMemory;
  size_t sharedMemPerBlockOptin;
  size_t reservedSharedMemPerBlock;
  int maxBlocksPerMultiProcessor;
  int hostRegisterSupported;
  int globalL1CacheSupported;
  int localL1CacheSupported;
  int streamPrioritiesSupported;
  int cooperativeLaunch;
  int clockRate;
  int memoryClockRate;
};

// the device's answer per geom.py index: args are the geometry table the runner plants, zeros where tgcu
// reports nothing (the shim then takes its TGCU_DEV_* setting, as env_int reads the runner's environment)
static int g_geom[16];

std::string dev_name() { return "planted card"; }

long long call_i(const char* fn, const char* fmt, ...) {
  va_list ap; va_start(ap, fmt);
  int idx = -1;
  if (std::strcmp(fn, "dev_geom_int") == 0) idx = va_arg(ap, int);
  va_end(ap);
  if (idx < 0 || idx >= (int)(sizeof g_geom / sizeof g_geom[0])) return 0;
  return g_geom[idx];
}

// taken from sidecar-shim/tgcudart.cpp by tests/test_fix_0098_0.py
@@BODIES@@

int main(int argc, char** argv) {
  const int n = (int)(sizeof g_geom / sizeof g_geom[0]);
  for (int i = 1; i < argc && i <= n; ++i) g_geom[i - 1] = std::atoi(argv[i]);
  cudaDeviceProp p;
  fill_props(&p);
  struct Query { const char* key; cudaDeviceAttr a; } q[] = {
    {"clock", cudaDevAttrClockRate},
    {"memclock", cudaDevAttrMemoryClockRate},
    {"busw", cudaDevAttrGlobalMemoryBusWidth},
    {"l2", cudaDevAttrL2CacheSize},
    {"sms", cudaDevAttrMultiProcessorCount},
    {"sm_major", cudaDevAttrComputeCapabilityMajor},
    {"sm_minor", cudaDevAttrComputeCapabilityMinor},
    {"warp", cudaDevAttrWarpSize},
  };
  for (const Query& e : q) {
    int v = -1;
    const cudaError_t rc = cudaDeviceGetAttribute(&v, e.a, 0);
    std::printf("attr_%s=%d\n", e.key, v);
    std::printf("status_%s=%d\n", e.key, (int) rc);
  }
  return 0;
}
"""


def find_root():
    """The tree under test: the copy prove_test runs us in (cwd), else the one this file sits in."""
    for base in (os.getcwd(), os.path.dirname(os.path.dirname(os.path.abspath(__file__)))):
        if os.path.isfile(os.path.join(base, SHIM)):
            return base
    return os.getcwd()


ROOT = find_root()


def check(name, detail):
    print(('PASS ' if not detail else 'FAIL ') + name + ('' if not detail else ': ' + detail))
    return 0 if not detail else 1


def function_text(src, signature):
    """the function from its signature to its matching close brace, comments and literals skipped"""
    start = src.find(signature)
    while 0 <= start and src.rfind('\n', 0, start) >= 0 and '//' in src[src.rfind('\n', 0, start):start]:
        start = src.find(signature, start + len(signature))
    if start < 0:
        return None
    i = src.find('{', start)
    if i < 0:
        return None
    depth = 0
    while 0 <= i < len(src):
        if src.startswith('//', i):
            i = src.find('\n', i)
            if i < 0:
                return None
        elif src.startswith('/*', i):
            i = src.find('*/', i)
            if i < 0:
                return None
            i += 1
        elif src[i] in '"\'':
            q = src[i]
            i += 1
            while i < len(src) and src[i] != q:
                i += 2 if src[i] == '\\' else 1
        elif src[i] == '{':
            depth += 1
        elif src[i] == '}':
            depth -= 1
            if depth == 0:
                return src[start:i + 1]
        i += 1
    return None


def compiler():
    for name in ('clang++', 'c++', 'g++'):
        path = shutil.which(name)
        if path:
            return path
    return None


def build(cxx, work, bodies):
    src = os.path.join(work, 'shim_clock.cpp')
    exe = os.path.join(work, 'shim_clock')
    with open(src, 'w', encoding='utf-8') as f:
        f.write(HARNESS.replace('@@BODIES@@', bodies))
    built = subprocess.run([cxx, '-std=c++17', '-O0', '-o', exe, src], capture_output=True, text=True, timeout=300)
    if built.returncode != 0:
        errors = [l.strip() for l in (built.stdout + built.stderr).splitlines() if ' error: ' in l]
        return None, (errors[0] if errors else 'see the compile log').replace('"', "'")
    return exe, ''


def load_geom():
    if not os.path.isfile(os.path.join(ROOT, GEOM)):
        return None
    spec = importlib.util.spec_from_file_location('sidecar_geom', os.path.join(ROOT, GEOM))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def config_env():
    """The selected serving config's engine environment: the production sample this code reads."""
    with open(os.path.join(ROOT, CONFIG), encoding='utf-8') as f:
        cfg = json.load(f)
    return {str(k): str(v) for k, v in (cfg.get('env') or {}).items()}


def run(exe, geom, env):
    values = geom.dev_geom(env=env)
    keys = tuple(getattr(geom, 'KEYS', ()))
    table = [int(values.get(k, 0)) if k in DEVICE_KEYS else 0 for k in keys]   # the device answers only these
    args = [exe] + [str(v) for v in table]
    prog_env = dict(os.environ)
    for k in list(prog_env):
        if k.startswith('TGCU_DEV_'):
            del prog_env[k]
    prog_env.update(env)
    r = subprocess.run(args, capture_output=True, text=True, timeout=120, env=prog_env)
    if r.returncode != 0:
        return None, values, 'the built program exited %d: %s' % (r.returncode, (r.stdout + r.stderr).strip()[-200:])
    got = {}
    for line in r.stdout.splitlines():
        key, sep, value = line.partition('=')
        if sep:
            got[key.strip()] = value.strip()
    return got, values, ''


def want_clock(values, env):
    """the clock the geometry resolver names for this environment, in kHz, or None when it names none"""
    if CLOCK_KEY not in values:
        return None
    try:
        return int(values[CLOCK_KEY]) * 1000
    except (TypeError, ValueError):
        return None


def case_different_card(got, values, env):
    want = want_clock(values, env)
    if want is None:
        return ('sidecar-shim/geom.py resolves no %s from %s, so cudaDevAttrClockRate can only answer its literal'
                % (CLOCK_KEY, CLOCK_ENV))
    if got.get('status_clock') != '0':
        return 'cudaDevAttrClockRate returned status %s, expected 0' % got.get('status_clock')
    have = int(got['attr_clock'])
    if have != want:
        return 'cudaDevAttrClockRate answered %d kHz, the configured card runs at %d kHz' % (have, want)
    return ''



def case_production_config(got, values, env):
    named = str(env.get(CLOCK_ENV) or '').strip()
    if not named:
        return ('config/sidecar.json names no %s, so the selected RTX 5090 config still gets the shim\'s 1.695 GHz '
                'literal' % CLOCK_ENV)
    try:
        want = int(named) * 1000
    except ValueError:
        return 'config/sidecar.json has %s=%r, not a clock in MHz' % (CLOCK_ENV, named)
    if want == CONSTANT_KHZ:
        return 'config/sidecar.json carries the 1.695 GHz literal, not the installed card\'s clock'
    if want_clock(values, env) != want:
        return 'the geometry resolver reads %s MHz from the config, expected %d MHz' % (values.get(CLOCK_KEY), want / 1000)
    have = int(got['attr_clock'])
    if have != want:
        return 'cudaDevAttrClockRate answered %d kHz for the selected config, expected %d kHz' % (have, want)
    return ''


def case_unchanged_default(got, values, env):
    if got.get('status_clock') != '0':
        return 'cudaDevAttrClockRate returned status %s with no clock setting' % got.get('status_clock')
    have = int(got['attr_clock'])
    if have != CONSTANT_KHZ:
        return 'with no setting cudaDevAttrClockRate answers %d kHz, today\'s 1695000 must stand' % have
    return ''


def case_neighbours(got, values, env):
    if int(got['attr_busw']) != int(values['busw']):
        return 'cudaDevAttrGlobalMemoryBusWidth answers %s, the geometry says %s' % (got['attr_busw'], values['busw'])
    if int(got['attr_l2']) != int(values['l2_mib']) << 20:
        return 'cudaDevAttrL2CacheSize answers %s, the geometry says %s MiB' % (got['attr_l2'], values['l2_mib'])
    if int(got['attr_sms']) != int(values['sms']):
        return 'cudaDevAttrMultiProcessorCount answers %s, the geometry says %s' % (got['attr_sms'], values['sms'])
    if got['attr_warp'] != '32':
        return 'cudaDevAttrWarpSize answers %s, expected 32' % got['attr_warp']
    major, minor = int(values['sm']) // 10, int(values['sm']) % 10
    if (int(got['attr_sm_major']), int(got['attr_sm_minor'])) != (major, minor):
        return 'the compute capability answers %s.%s, the geometry says %d.%d' % (
            got['attr_sm_major'], got['attr_sm_minor'], major, minor)
    return ''


def outside_config():
    return {'TGCU_DEV_SM': '120', 'TGCU_DEV_SMS': '170', 'TGCU_DEV_MEM_GIB': '32',
            'TGCU_DEV_L2_MIB': '96', 'TGCU_DEV_BUSW': '512'}


CASES = [
    dict(name='a config that names another card\'s clock is answered, not the 1.695 GHz literal',
         env={'TGCU_DEV_CLOCK_MHZ': '2505'}, check=case_different_card),
    dict(name='the selected serving config names the installed card\'s clock',
         env=None, check=case_production_config),          # env=None: config/sidecar.json's own env
    dict(name='with no clock setting the shim still answers 1695000 kHz',
         env={}, check=case_unchanged_default),
    dict(name='bus width, L2, SM count, warp size and compute capability still answer from the geometry',
         env=outside_config(), check=case_neighbours),
]


def main():
    if not os.path.isfile(os.path.join(ROOT, SHIM)):
        print('FAIL run this test from the copy root: ' + SHIM + ' is not in ' + ROOT)
        return 1
    with open(os.path.join(ROOT, SHIM), encoding='utf-8', errors='replace') as f:
        src = f.read()
    bodies, missing = [], []
    for sig in SIGNATURES:
        text = function_text(src, sig)
        if text is None:
            missing.append(sig.strip())
        else:
            bodies.append(text)
    if missing:
        print('FAIL sidecar-shim/tgcudart.cpp has no definition of ' + ', '.join(missing))
        return 1
    cxx = compiler()
    if cxx is None:
        print('FAIL no C++ compiler was found (looked for clang++, c++, g++)')
        return 1
    geom = load_geom()
    if geom is None:
        print('FAIL ' + GEOM + ' is missing, so the shim has no geometry resolver for the clock')
        return 1
    work = tempfile.mkdtemp(prefix='shim-clock-')
    failures = 0
    try:
        exe, detail = build(cxx, work, '\n\n'.join(bodies))
        if exe is None:
            print('FAIL the shim\'s device functions do not build against the stub CUDA surface (' + detail + ')')
            return 1
        production = config_env()
        for case in CASES:
            env = dict(production if case['env'] is None else case['env'])
            got, values, detail = run(exe, geom, env)
            if got is None:
                failures += check(case['name'], detail)
                continue
            failures += check(case['name'], case['check'](got, values, env))
    finally:
        shutil.rmtree(work, ignore_errors=True)
    print('%d case(s) failed' % failures if failures else 'all cases passed')
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
