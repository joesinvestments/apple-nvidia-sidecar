#!/usr/bin/env python3
"""tests/test_fix_0098_6.py - the shim's cuDeviceGetAttribute must answer every attribute, not only VMM support.

sidecar-shim/tgcudart.cpp registers vmm_DeviceGetAttribute as the implementation of cuDeviceGetAttribute in the
cudaGetDriverEntryPoint table, and that function returned CUDA_ERROR_NOT_SUPPORTED for every attribute other than
CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED. A caller that resolves the driver entry point (src/core/vmm.cpp
does for the elastic K/V, and any library in the engine's process may) and asks for WARP_SIZE, MULTIPROCESSOR_COUNT or an
attribute the runtime table has no case for got an error instead of the value the runtime reports.

These cases compile the real functions out of sidecar-shim/tgcudart.cpp (env_int, dev_sm, fill_props, the runtime's
cudaDeviceGetAttribute, the VMM entry points and cudaGetDriverEntryPoint) against a stub CUDA surface (no GPU, no CUDA
headers, no Python), run the built program with the device identity of the selected our test machine config (serving.json:
TGCU_DEV_SM=120, TGCU_DEV_SMS=170, TGCU_DEV_L2_MIB=96) and drive them through the same cudaGetDriverEntryPoint path the
engine uses:
  - the VMM attribute still reports 1 through cuDeviceGetAttribute;
  - WARP_SIZE, MULTIPROCESSOR_COUNT and COMPUTE_CAPABILITY_MAJOR come back as the runtime's own values for that card,
    not CUDA_ERROR_NOT_SUPPORTED;
  - an attribute the runtime table has no case for is answered with the runtime's value, not refused;
  - cuMemGetAllocationGranularity is still served and an unknown name is still not;
  - TGCU_VMM=0 still hides the whole table.
Prints one PASS or FAIL line per case and exits nonzero on any FAIL. Run: python3 tests/test_fix_0098_6.py
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = os.getcwd()   # the tree under test: every test here runs with cwd at the copy root, never at the test file
SHIM = os.path.join('sidecar-shim', 'tgcudart.cpp')

# the stub CUDA surface: the driver and runtime attribute numbers are the CUDA headers' (cuda_runtime_api.h keeps
# cudaDevAttrX equal to CU_DEVICE_ATTRIBUTE_X), and fill_props reports the selected our test machine device (TGCU_DEV_SMS=170,
# TGCU_DEV_SM=120, TGCU_DEV_L2_MIB=96 in serving.json) so the delegated answers are the ones production sees.
PRELUDE = r"""
#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

typedef int cudaError_t;
typedef int CUresult;
typedef int CUdevice;
#define EXPORT
#define API_ENTER(name) ((void)0)

static const cudaError_t cudaSuccess = 0;
static const cudaError_t cudaErrorNotSupported = 801;
static const CUresult CUDA_SUCCESS = 0;
static const CUresult CUDA_ERROR_INVALID_VALUE = 1;
static const CUresult CUDA_ERROR_NOT_SUPPORTED = 801;

enum cudaDriverEntryPointQueryResult { cudaDriverEntryPointSuccess = 0, cudaDriverEntryPointSymbolNotFound = 1 };

enum CUdevice_attribute {
  CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 1,
  CU_DEVICE_ATTRIBUTE_WARP_SIZE = 10,
  CU_DEVICE_ATTRIBUTE_MAX_PITCH = 11,
  CU_DEVICE_ATTRIBUTE_CLOCK_RATE = 13,
  CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16,
  CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75,
  CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76,
  CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED = 102
};

enum cudaDeviceAttr {   // the names the runtime's attribute switch uses
  cudaDevAttrMaxThreadsPerBlock = 1,
  cudaDevAttrMaxBlockDimX = 2,
  cudaDevAttrMaxBlockDimY = 3,
  cudaDevAttrMaxBlockDimZ = 4,
  cudaDevAttrMaxGridDimX = 5,
  cudaDevAttrMaxGridDimY = 6,
  cudaDevAttrMaxGridDimZ = 7,
  cudaDevAttrMaxSharedMemoryPerBlock = 8,
  cudaDevAttrWarpSize = 10,
  cudaDevAttrMaxPitch = 11,
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
  cudaDevAttrConcurrentManagedAccess = 89,
  cudaDevAttrCooperativeLaunch = 95,
  cudaDevAttrMaxSharedMemoryPerBlockOptin = 97,
  cudaDevAttrHostRegisterSupported = 99,
  cudaDevAttrPageableMemoryAccess = 100,
  cudaDevAttrMemoryPoolsSupported = 101,
  cudaDevAttrMaxBlocksPerMultiprocessor = 106,
  cudaDevAttrClusterLaunch = 120
};

struct cudaDeviceProp {   // the fields the shim's fill_props and attribute switch touch
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
};

// the other names in the entry point table: this test needs their addresses, never their work
static CUresult vmm_GetAllocationGranularity() { return CUDA_SUCCESS; }
static CUresult vmm_AddressReserve() { return CUDA_SUCCESS; }
static CUresult vmm_AddressFree() { return CUDA_SUCCESS; }
static CUresult vmm_Create() { return CUDA_SUCCESS; }
static CUresult vmm_Release() { return CUDA_SUCCESS; }
static CUresult vmm_Map() { return CUDA_SUCCESS; }
static CUresult vmm_Unmap() { return CUDA_SUCCESS; }
static CUresult vmm_SetAccess() { return CUDA_SUCCESS; }
// F1 device geometry queries are outside this isolated driver-entry-point test.
static long long call_i(const char*, const char*, int) { return 0; }
static std::string dev_name() { return "stub device"; }
"""

MAIN = r"""
typedef CUresult (*AttrFn)(int*, CUdevice_attribute, CUdevice);

static void attr_line(const AttrFn attr, const char* row, CUdevice_attribute a, cudaDeviceAttr r) {
  int v = -1;
  const CUresult r0 = attr(&v, a, 0);
  int rv = -1;
  const cudaError_t r1 = cudaDeviceGetAttribute(&rv, r, 0);
  std::printf("%s status=%d value=%d\n%s_runtime value=%d err=%d\n", row, (int)r0, v, row, rv, (int)r1);
}

int main() {
  cudaDriverEntryPointQueryResult q = cudaDriverEntryPointSymbolNotFound;
  void* fn = nullptr;
  const cudaError_t e = cudaGetDriverEntryPoint("cuDeviceGetAttribute", &fn, 0, &q);
  std::printf("resolve err=%d q=%d fn=%d\n", (int)e, (int)q, fn != nullptr);
  void* gran = nullptr;
  const cudaError_t ge = cudaGetDriverEntryPoint("cuMemGetAllocationGranularity", &gran, 0, &q);
  std::printf("gran err=%d fn=%d\n", (int)ge, gran != nullptr);
  void* unknown = (void*)1;
  const cudaError_t ue = cudaGetDriverEntryPoint("cuMemCopyRange", &unknown, 0, &q);
  std::printf("unknown err=%d fn=%d\n", (int)ue, unknown != nullptr);
  if (fn == nullptr) {
    std::printf("vmm status=-1 value=-1\n");
    std::printf("warp status=-1 value=-1\nwarp_runtime value=-1 err=-1\n");
    std::printf("sms status=-1 value=-1\nsms_runtime value=-1 err=-1\n");
    std::printf("cc status=-1 value=-1\ncc_runtime value=-1 err=-1\n");
    std::printf("pitch status=-1 value=-1\npitch_runtime value=-1 err=-1\n");
    return 0;
  }
  const AttrFn attr = (AttrFn) fn;
  int v = -1;
  const CUresult vm = attr(&v, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, 0);
  std::printf("vmm status=%d value=%d\n", (int)vm, v);
  attr_line(attr, "warp", CU_DEVICE_ATTRIBUTE_WARP_SIZE, cudaDevAttrWarpSize);
  attr_line(attr, "sms", CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, cudaDevAttrMultiProcessorCount);
  attr_line(attr, "cc", CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, cudaDevAttrComputeCapabilityMajor);
  attr_line(attr, "pitch", CU_DEVICE_ATTRIBUTE_MAX_PITCH, cudaDevAttrMaxPitch);
  return 0;
}
"""

NEEDED = ['env_int', 'dev_field', 'dev_sm', 'fill_props', 'cudaDeviceGetAttribute', 'vmm_DeviceGetAttribute', 'vmm_entry',
          'cudaGetDriverEntryPoint']

# the selected our test machine config (serving.json, read only): the card identity the shim reports and the numbers the
# driver entry point has to agree with when it delegates an attribute to the runtime
SERVING = {
    'TGCU_DEV_NAME': 'NVIDIA GeForce RTX 5090 (tgcu APL)',
    'TGCU_DEV_SM': '120',
    'TGCU_DEV_SMS': '170',
    'TGCU_DEV_MEM_GIB': '32',
    'TGCU_DEV_L2_MIB': '96',
    'TGCU_DEV_BUSW': '512',
}


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
    src = os.path.join(work, 'driver_attr.cpp')
    exe = os.path.join(work, 'driver_attr')
    with open(src, 'w', encoding='utf-8') as f:
        f.write(source)
    built = subprocess.run([cxx, '-std=c++17', '-O0', '-o', exe, src], capture_output=True, text=True, timeout=300)
    if built.returncode != 0:
        errors = [l.strip() for l in (built.stdout + built.stderr).splitlines() if ' error: ' in l]
        return None, (errors[0] if errors else 'see the compile log').replace('"', "'")
    return exe, ''


def run(exe, env=None):
    e = dict(os.environ)
    e.pop('TGCU_VMM', None)
    e.update(env or {})
    r = subprocess.run([exe], capture_output=True, text=True, timeout=120, env=e)
    return r.returncode, r.stdout + r.stderr


def parse(text):
    got = {}
    for line in text.splitlines():
        parts = line.split()
        if not parts:
            continue
        got[parts[0]] = dict(p.split('=', 1) for p in parts[1:] if '=' in p)
    return got


def field(got, row, key):
    return got.get(row, {}).get(key)


def fancy(got, row):
    """The row's value is not a constant: it is the same answer the runtime gives for the same attribute."""
    status = field(got, row, 'status')
    if status != '0':
        return '%s status is %s, expected 0: a real attribute must not be refused' % (row, status)
    if field(got, row, 'value') != field(got, row + '_runtime', 'value'):
        return '%s value is %s, the runtime reports %s' % (row, field(got, row, 'value'), field(got, row + '_runtime', 'value'))
    if field(got, row, 'value') in (None, '0', '-1'):
        return '%s value is %s: the program did not carry the attribute' % (row, field(got, row, 'value'))
    return ''


def check_vmm(got):
    if field(got, 'vmm', 'status') != '0' or field(got, 'vmm', 'value') != '1':
        return 'the VMM attribute answered status %s value %s, expected status 0 value 1' % (
            field(got, 'vmm', 'status'), field(got, 'vmm', 'value'))
    return ''


def check_warp(got):
    return fancy(got, 'warp')


def check_sms(got):
    detail = fancy(got, 'sms')
    if detail:
        return detail
    if field(got, 'sms', 'value') != '170' or field(got, 'sms_runtime', 'value') != '170':
        return 'the serving config\'s 170 SMs came back as %s (runtime %s)' % (
            field(got, 'sms', 'value'), field(got, 'sms_runtime', 'value'))
    return ''


def check_cc(got):
    detail = fancy(got, 'cc')
    if detail:
        return detail
    if field(got, 'cc', 'value') != '12':
        return 'TGCU_DEV_SM=120 came back as compute capability %s' % field(got, 'cc', 'value')
    return ''


def check_pitch(got):
    """an attribute the runtime table has no case for: delegated, so it is answered with the runtime's value."""
    status = field(got, 'pitch', 'status')
    if status != '0':
        return 'pitch status is %s, expected 0: the driver entry point must not refuse it' % status
    if field(got, 'pitch', 'value') != field(got, 'pitch_runtime', 'value'):
        return 'pitch value is %s, the runtime reports %s' % (field(got, 'pitch', 'value'), field(got, 'pitch_runtime', 'value'))
    return ''


def check_table(got):
    if field(got, 'gran', 'fn') != '1':
        return 'cuMemGetAllocationGranularity is no longer served (err %s)' % field(got, 'gran', 'err')
    if field(got, 'unknown', 'fn') != '0' or field(got, 'unknown', 'err') in (None, '0'):
        return 'a name the shim does not serve is now reported as found'
    return ''


def check_off(got):
    if field(got, 'resolve', 'fn') != '0':
        return 'TGCU_VMM=0 left cuDeviceGetAttribute served'
    return ''


CASES = [
    dict(name='the VMM attribute still reports 1 through cuDeviceGetAttribute', check=check_vmm),
    dict(name='WARP_SIZE through cuDeviceGetAttribute is the runtime\'s answer', check=check_warp),
    dict(name='MULTIPROCESSOR_COUNT through cuDeviceGetAttribute is the runtime\'s answer', check=check_sms),
    dict(name='COMPUTE_CAPABILITY_MAJOR through cuDeviceGetAttribute is the runtime\'s answer', check=check_cc),
    dict(name='an attribute the runtime table has no case for is answered, not refused', check=check_pitch),
    dict(name='cuMemGetAllocationGranularity is still served and an unknown name is still not', check=check_table),
    dict(name='TGCU_VMM=0 still hides the driver entry points', check=check_off),
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
    work = tempfile.mkdtemp(prefix='shim-driver-attr-')
    try:
        exe, detail = build(cxx, work, source)
        if exe is None:
            print('FAIL the shim\'s driver entry points do not build against the stub CUDA surface (' + detail + ')')
            return 1
        rc, out = run(exe, SERVING)
        rc_off, out_off = run(exe, dict(SERVING, TGCU_VMM='0'))
    finally:
        shutil.rmtree(work, ignore_errors=True)
    if rc != 0:
        print('FAIL the built program exited %d: %s' % (rc, out.strip()[-300:]))
        return 1
    if rc_off != 0:
        print('FAIL the built program exited %d with TGCU_VMM=0: %s' % (rc_off, out_off.strip()[-300:]))
        return 1
    got, off = parse(out), parse(out_off)
    failures = 0
    for case in CASES:
        detail = case['check'](off if case['check'] is check_off else got)
        print(('PASS ' if not detail else 'FAIL ') + case['name'] + (': ' + detail if detail else ''))
        failures += 0 if not detail else 1
    print('%d case(s) failed' % failures if failures else 'all cases passed')
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
