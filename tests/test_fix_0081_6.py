#!/usr/bin/env python3
"""tests/test_fix_0081_6.py - a vmm handle must not go back to the physical pool while a page table still points at it.

sidecar-shim/tgcu.py (vmm_release, reached from tgcudart's cuMemRelease for the elastic K/V, --kv-grow) flushes the queued
umaps with the comment "a page that is still mapped must not go back to the pool", but then calls pfree(h - 1) without
looking at vmm_maps: a still-mapped chunk's physical page is handed back to tinygrad's pool while the page tables still
point at it (the next palloc hands the same page to another buffer), and the out-of-memory handle vmm_create returns (0)
reaches pfree(-1). These cases drive the real Context.vmm_* methods on a stub physical pool and page-table writer (no GPU,
no tinygrad, no model):
  - a still-mapped handle is rejected and its page stays allocated;
  - the handle 0 from vmm_create's out-of-memory path is rejected, never freed as pfree(-1);
  - the ordinary create, map, unmap, release still frees the page exactly once (same before and after the fix);
  - a map that was only queued and then cancelled by unmap still allows the release (the guard looks at written page tables, not at the queue).
Prints one PASS or FAIL line per case and exits nonzero on any FAIL. Run: python3 tests/test_fix_0081_6.py
"""
import importlib.util
import os
import sys
import types

ROOT = os.getcwd()   # the tree under test: every test here runs with cwd at the copy root, never at the test file
SHIM = os.path.join('sidecar-shim', 'tgcu.py')
VMM_GRAN = 2 << 20
VA = 0x7f00_0000_0000


def load_tgcu():
    """Import sidecar-shim/tgcu.py with tinygrad stubbed out: the module only needs Device, Buffer*, round_up and data64_le."""
    tinygrad = types.ModuleType('tinygrad')
    tinygrad.Device = {}
    device = types.ModuleType('tinygrad.device')
    device.Buffer = type('Buffer', (), {})
    device.BufferSpec = type('BufferSpec', (), {})
    device.BufferStorage = type('BufferStorage', (), {})
    helpers = types.ModuleType('tinygrad.helpers')
    helpers.round_up = lambda n, amt: (n + amt - 1) // amt * amt
    helpers.data64_le = lambda b: int.from_bytes(b, 'little')
    tinygrad.device, tinygrad.helpers = device, helpers
    names = ('tinygrad', 'tinygrad.device', 'tinygrad.helpers')
    saved = {n: sys.modules.get(n) for n in names}
    sys.modules.update(dict(zip(names, (tinygrad, device, helpers))))
    mod = None
    try:
        spec = importlib.util.spec_from_file_location('tgcu_under_test', SHIM)
        mod = importlib.util.module_from_spec(spec)
        sys.modules['tgcu_under_test'] = mod   # the dataclasses in the shim look their module up while it is being executed
        spec.loader.exec_module(mod)
    finally:
        sys.modules.pop('tgcu_under_test', None)
        for n, m in saved.items():
            if m is None:
                sys.modules.pop(n, None)
            else:
                sys.modules[n] = m
    return mod


class FakeMM:
    """tinygrad's device memory manager: palloc hands out 2 MiB chunks of physical memory, pfree returns one to the pool."""

    def __init__(self):
        self.next_pa = 0x1000_0000
        self.freed = []
        self.full = False
        self.range_mapped = 0

    def palloc(self, size, align=0, zero=False):
        if self.full:
            raise MemoryError('physical pool full')
        pa, self.next_pa = self.next_pa, self.next_pa + VMM_GRAN
        return pa

    def pfree(self, pa):
        self.freed.append(pa)

    def on_range_mapped(self):
        self.range_mapped += 1


def make_ctx(mod):
    """A Context with only the state the vmm_* methods use; the page-table writer is recorded instead of written."""
    mm = FakeMM()
    ctx = mod.Context.__new__(mod.Context)
    ctx.iface = types.SimpleNamespace(dev_impl=types.SimpleNamespace(mm=mm))
    ctx.pending = []
    ctx.sem_value, ctx.vmm_synced = 0, 0
    ctx.vmm_phys = 0
    ctx.vmm_maps = {}
    ctx.vmm_pend_map = {}
    ctx.vmm_pend_unmap = []
    ctx.writes = []
    ctx._vmm_write = lambda vas, pas: ctx.writes.append((list(vas), None if pas is None else list(pas)))
    return ctx, mm


def release(ctx, h):
    """vmm_release, telling a rejection apart from a silent return."""
    try:
        ctx.vmm_release(h)
    except AssertionError as e:
        return e
    return None


def case_still_mapped_handle(mod):
    """A handle whose page is in the page tables must be rejected, and the page must stay allocated."""
    ctx, mm = make_ctx(mod)
    h = ctx.vmm_create()
    pa = h - 1
    ctx.vmm_map(VA, h)
    ctx.vmm_flush()
    if ctx.vmm_maps != {VA: pa}:
        return False, 'the case did not set up a mapped page: vmm_maps is %r' % (ctx.vmm_maps,)
    err = release(ctx, h)
    if err is None:
        bad = mm.freed[0] if mm.freed else None
        return False, ('releasing a still-mapped handle returned quietly: pfree(%s) with the page table at %#x still pointing at %#x'
                       % (('%#x' % bad) if bad is not None else 'not called', VA, pa))
    if mm.freed:
        return False, 'the release was rejected but the page was freed anyway: pfree(%#x)' % mm.freed[0]
    if ctx.vmm_phys != VMM_GRAN:
        return False, 'the release was rejected but vmm_phys changed to %d' % ctx.vmm_phys
    return True, ''


def case_out_of_memory_handle(mod):
    """vmm_create returns 0 when the pool is full: releasing that handle must not reach pfree(-1)."""
    ctx, mm = make_ctx(mod)
    mm.full = True
    h = ctx.vmm_create()
    mm.full = False
    if h != 0:
        return False, 'vmm_create returned %#x with the pool full, the case cannot plant handle 0' % h
    err = release(ctx, h)
    if err is None:
        return False, 'releasing the out-of-memory handle 0 returned quietly: pfree(%d) was called' % (mm.freed[0] if mm.freed else 0)
    if mm.freed:
        return False, 'the release of handle 0 was rejected but pfree(%d) still ran' % mm.freed[0]
    if ctx.vmm_phys != 0:
        return False, 'the release of handle 0 was rejected but vmm_phys changed to %d' % ctx.vmm_phys
    return True, ''


def case_contract_following_release(mod):
    """Create, map, unmap, release: the page goes back exactly once and both page-table writes still happen."""
    ctx, mm = make_ctx(mod)
    h = ctx.vmm_create()
    pa = h - 1
    ctx.vmm_map(VA, h)
    ctx.vmm_flush()
    ctx.vmm_unmap(VA)
    err = release(ctx, h)
    if err is not None:
        return False, 'the ordinary unmap-then-release was rejected: %s' % err
    if mm.freed != [pa]:
        return False, 'the page was freed as %r, expected [%#x]' % (mm.freed, pa)
    if ctx.vmm_maps:
        return False, 'the page is still in the page tables: %r' % (ctx.vmm_maps,)
    if ctx.vmm_phys != 0:
        return False, 'vmm_phys is %d after the release, expected 0' % ctx.vmm_phys
    if ctx.writes != [([VA], [pa]), ([VA], None)]:
        return False, 'the page-table writes were %r, expected the map then the unmap of %#x' % (ctx.writes, VA)
    return True, ''


def case_cancelled_queued_map(mod):
    """A map that never reached the page tables and was cancelled by unmap must not block the release."""
    ctx, mm = make_ctx(mod)
    h = ctx.vmm_create()
    pa = h - 1
    ctx.vmm_map(VA, h)   # queued only
    ctx.vmm_unmap(VA)    # the queued map is dropped, nothing is in the page tables
    err = release(ctx, h)
    if err is not None:
        return False, 'the release was rejected although no page table points at the page: %s' % err
    if mm.freed != [pa]:
        return False, 'the page was freed as %r, expected [%#x]' % (mm.freed, pa)
    return True, ''


CASES = [
    ('a still-mapped handle is rejected and its page stays allocated', case_still_mapped_handle),
    ('the out-of-memory handle 0 is rejected, not freed as pfree(-1)', case_out_of_memory_handle),
    ('create, map, unmap, release frees the page exactly once', case_contract_following_release),
    ('a queued map cancelled by unmap still allows the release', case_cancelled_queued_map),
]


def main():
    if not os.path.isfile(SHIM):
        print('FAIL run this test from the copy root: ' + SHIM + ' is not in ' + ROOT)
        return 1
    mod = load_tgcu()
    failures = 0
    for name, case in CASES:
        try:
            ok, detail = case(mod)
        except Exception as e:   # a case that blows up is a failing case, not a broken run
            ok, detail = False, '%s: %s' % (type(e).__name__, e)
        print(('PASS ' if ok else 'FAIL ') + name + (': ' + detail if detail else ''))
        failures += 0 if ok else 1
    print('%d case(s) failed' % failures if failures else 'all cases passed')
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
