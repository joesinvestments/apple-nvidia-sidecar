#!/usr/bin/env python3
"""The conversation pool's free-space floor comes from the config, and a skipped save says so.

fix-0047#0 gave serve/convpool.py a floor before it snapshots an evicted conversation, but
serve/server.py:10498 still builds Pool(svc.slot_save_path, cap): the config's convpool_min_free_gib is
read by nobody, so a rig whose disk keeps less than 51 GiB free never snapshots and no line says why.
The fix reads convpool_min_free_gib (default 51) into the pool and has Pool.before() write exactly one
line to stderr when a save is skipped for lack of free space.

This test builds the pool from a config and drives the real Pool with a fake engine and a stubbed
disk_usage: a configured floor reaches the pool, no key keeps 51 GiB, a save under the floor is skipped
with one line on stderr, and a save above the floor writes as before.  No model, no GPU, no network,
no server.

    python3 tests/test_convpool_min_free_config.py
"""
import collections
import contextlib
import io
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path.cwd()                         # prove_test runs the test with cwd at the tree under test
sys.path.insert(0, str(ROOT))

from serve import convpool  # noqa: E402
from serve import server as serve_server  # noqa: E402

GiB = 1 << 30
Usage = collections.namedtuple('Usage', 'total used free')
results = []


def case(name, body):
    """Run one case: `body()` -> (ok, detail); any exception is a FAIL line, never a crash."""
    try:
        ok, detail = body()
    except Exception as exc:
        ok, detail = False, f'{type(exc).__name__}: {exc}'
    results.append(bool(ok))
    print(('PASS ' if ok else 'FAIL ') + name + ('' if ok else ': ' + str(detail)))


class FakeDisk:
    """Patch convpool's disk_usage to report the free space a case plants."""

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

    def __init__(self):
        self.calls = []

    def session_file(self, action, path):
        self.calls.append((action, Path(path).name))
        if action == 'save':
            Path(path).write_bytes(b'native snapshot')
        return {'tokens': 40000, 'ms': 3.0}


def pool_from(cfg, directory):
    """-> the Pool serve/server.py builds for this config, or None when the disk tier is off."""
    return serve_server.convpool_from_config(cfg, directory)


def evict(pool, free):
    """-> (engine, stdout, stderr) after a 30000-token conversation is replaced by another on `free` bytes."""
    engine, out, err = FakeEngine(), io.StringIO(), io.StringIO()
    with FakeDisk(free), contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        pool.before(engine, [{'role': 'user', 'content': 'old'}], 30000)
        pool.before(engine, [{'role': 'user', 'content': 'new'}], 30000)
    return engine, out.getvalue(), err.getvalue()


def skip_lines(text):
    """The lines that report a skipped save."""
    return [l for l in text.splitlines() if 'convpool' in l and 'skip' in l.lower()]


def skip_body(directory, cfg, free):
    """Build the pool for `cfg` (or a plain default-floor pool), evict on `free` bytes, expect a skip + one line."""
    pool = convpool.Pool(directory, 10) if cfg is None else pool_from(cfg, directory)
    if pool is None:
        return False, 'the config built no pool'
    engine, out, err = evict(pool, free)
    files = sorted(p.name for p in Path(directory).glob('convpool-*'))
    skipped = skip_lines(err)
    ok = (pool.counts['saves'] == 0 and not files and not engine.calls
          and len(skipped) == 1 and skipped[0].startswith('[strata] convpool:')
          and not skip_lines(out))
    return ok, (f'saves={pool.counts["saves"]} files={files} calls={engine.calls} '
                f'skip stderr={skipped} stdout={out!r}')


def save_body(directory, cfg, free):
    """Build the pool for `cfg`, evict on `free` bytes, expect one snapshot, its meta and no skip line."""
    pool = pool_from(cfg, directory)
    if pool is None:
        return False, 'the config built no pool'
    engine, out, err = evict(pool, free)
    bins = sorted(Path(directory).glob('convpool-*.bin'))
    metas = sorted(Path(directory).glob('convpool-*.json'))
    ok = (pool.counts['saves'] == 1 and len(bins) == 1 and len(metas) == 1
          and [c[0] for c in engine.calls] == ['save'] and metas[0].is_file()
          and not skip_lines(err + out))
    return ok, (f'saves={pool.counts["saves"]} bins={[p.name for p in bins]} '
                f'metas={[p.name for p in metas]} calls={engine.calls} err={err!r}')


def temporary():
    return tempfile.TemporaryDirectory(prefix='convpool-config-')


def main():
    low = {'convpool_cap_gib': 10, 'convpool_min_free_gib': 10}
    plain = {'convpool_cap_gib': 10}

    def floor_case(cfg, want):
        def body():
            with temporary() as directory:
                pool = pool_from(cfg, directory)
                return (pool is not None and pool.min_free == want,
                        f'min_free={getattr(pool, "min_free", None)} want={want}')
        return body

    case('a config floor of 10 GiB reaches the pool', floor_case(low, 10 * GiB))
    case('a config without the key keeps the 51 GiB floor', floor_case(plain, 51 * GiB))

    def cap_off():
        with temporary() as directory:
            pool = pool_from({'convpool_cap_gib': 0, 'convpool_min_free_gib': 10}, directory)
            return pool is None, f'pool={pool!r}'
    case('a config with the cap off builds no pool', cap_off)

    def plain_skip():
        with temporary() as directory:
            return skip_body(directory, None, 10 * GiB)
    case('a save under the 51 GiB floor is skipped and logs one line to stderr', plain_skip)

    def low_skip():
        with temporary() as directory:
            return skip_body(directory, low, 5 * GiB)
    case('a save under a configured 10 GiB floor is skipped and logs one line to stderr', low_skip)

    def low_save():
        with temporary() as directory:
            return save_body(directory, low, 10 * GiB)
    case('a configured 10 GiB floor still saves on a 10 GiB disk', low_save)

    def plain_save():
        with temporary() as directory:
            return save_body(directory, plain, 100 * GiB)
    case('a save above the floor writes the snapshot as before', plain_save)

    failed = results.count(False)
    print(f'{len(results) - failed} of {len(results)} cases pass')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
