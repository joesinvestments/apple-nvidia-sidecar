#!/usr/bin/env python3
"""The conversation pool's eviction save must use a configurable free-space floor.

serve/convpool.py:62 only saves an evicted conversation when the disk has 51 GiB
free, hard-coded, so a smaller disk silently never snapshots.  This test builds a
pool over a temp folder with a fake engine and a stubbed disk_usage: a floor the
caller sets is honoured, and the default keeps today's 51 GiB guard.

    python3 tests/test_fix_0047_0.py
"""
import collections
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path.cwd()                         # prove_test runs the test with cwd at the tree under test
sys.path.insert(0, str(ROOT))

from serve import convpool  # noqa: E402


GiB = 1 << 30
Usage = collections.namedtuple('Usage', 'total used free')


class FakeDisk:
    """Patch convpool's disk_usage to report the free space the case plants."""

    def __init__(self, free):
        self.free = free

    def __enter__(self):
        self.saved = convpool.shutil.disk_usage
        convpool.shutil.disk_usage = lambda path: Usage(1 << 40, 0, self.free)
        return self

    def __exit__(self, *exc):
        convpool.shutil.disk_usage = self.saved
        return False


class FakeEngine:
    """The one call the save branch makes: session_file('save', path)."""

    batch = 0

    def __init__(self):
        self.calls = []

    def session_file(self, action, path):
        self.calls.append((action, path))
        if action == 'save':
            Path(path).write_bytes(b'snapshot')
            return {'tokens': 123, 'ms': 4.0}
        return {'tokens': 123, 'ms': 2.0}


def evict(directory, free, **kwargs):
    """-> (pool, engine) after a 30000-token conversation is replaced by another."""
    engine = FakeEngine()
    with FakeDisk(free):
        pool = convpool.Pool(directory, 1, **kwargs)
        pool.before(engine, [{'role': 'user', 'content': 'old'}], 30000)
        pool.before(engine, [{'role': 'user', 'content': 'new'}], 30000)
    return pool, engine


def main():
    results = []

    def case(name, ok, detail=''):
        results.append(bool(ok))
        print(('PASS ' if ok else 'FAIL ') + name + ('' if ok else f': {detail}'))

    def check(name, free_gib, want_saves, **kwargs):
        directory = tempfile.mkdtemp(prefix='convpool-floor-')
        try:
            pool, engine = evict(directory, int(free_gib * GiB), **kwargs)
            saves = pool.counts['saves']
            files = list(Path(directory).glob('convpool-*.bin'))
            case(name, saves == want_saves and len(files) == want_saves,
                 f'saves={saves} files={len(files)} calls={engine.calls}')
        except Exception as exc:          # today's pool has no floor argument
            case(name, False, f'{type(exc).__name__}: {exc}')
        finally:
            shutil.rmtree(directory, ignore_errors=True)

    check('a low configured floor saves on a 10 GiB disk', 10, 1, min_free_gib=0)
    check('a configured floor above free space skips the save', 10, 0, min_free_gib=20)
    check('a floor equal to free space still saves', 10, 1, min_free_gib=10)
    check('the default floor keeps the 51 GiB guard', 10, 0)
    check('the default floor saves with ample free space', 100, 1)

    failed = results.count(False)
    print(f'{len(results) - failed} of {len(results)} cases pass')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
