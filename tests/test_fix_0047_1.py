#!/usr/bin/env python3
"""The conversation pool's save, restore, miss and error lines go to stderr, never to stdout.

serve/convpool.py prints '[strata] convpool: ...' with print() on every save, restore, miss and error
(lines 75, 78, 93 and 96).  Pool.before() runs on the server's request path (serve/server.py), so a
library that writes to stdout pollutes the server's own output and there is no way to silence it.  The
fix keeps the same '[strata] convpool: ...' text, on stderr.

This test drives the real Pool with a fake engine and a stubbed disk in temp folders: each path must
keep its count and its snapshot result, keep its line (same text, on stderr), and write nothing to
stdout.  Ordinary inputs (a batched engine, no snapshot) behave the same before and after.

    python3 tests/test_fix_0047_1.py
"""
import contextlib
import io
import json
import sys
import tempfile
import types
from pathlib import Path

ROOT = Path.cwd() if (Path.cwd() / "serve" / "convpool.py").is_file() else Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from serve import convpool  # noqa: E402

BIG_FREE = 200 * (1 << 30)        # the save path only runs with 51 GiB free; the host under test has 11
PREFIX = "[strata] convpool: "
OLD = [{"role": "user", "content": "old conversation"}]
NEW = [{"role": "user", "content": "new conversation"}]
PAIR = [{"role": "user", "content": "first"}, {"role": "assistant", "content": "second"}]
results = []


def case(name, ok, detail=""):
    results.append(bool(ok))
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else ": " + str(detail)))


class FakeEngine:
    """The two calls Pool makes: session_file('save', path) and session_file('restore', path)."""

    batch = 0

    def __init__(self, fail=None):
        self.fail = fail
        self.calls = []

    def session_file(self, action, path):
        self.calls.append(action)
        if self.fail is not None:
            raise self.fail
        Path(path).write_bytes(b"native snapshot")
        return {"tokens": 42, "ms": 3}


@contextlib.contextmanager
def disk_free(free):
    original = convpool.shutil.disk_usage
    convpool.shutil.disk_usage = lambda path: types.SimpleNamespace(free=free)
    try:
        yield
    finally:
        convpool.shutil.disk_usage = original


@contextlib.contextmanager
def both_streams():
    """-> (stdout text, stderr text) written while the block runs."""
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        yield out, err


def active_pool(directory, messages=OLD, tokens=40000):
    """A pool whose single slot already holds `messages`, saved at `tokens` prompt tokens."""
    pool = convpool.Pool(directory, 1)
    pool.active = (convpool.identity(messages), convpool.prefix(messages), tokens, len(messages))
    return pool


def snapshot(pool, messages, count):
    """The .bin plus meta .json an earlier save left for `messages`, taken when it had `count` messages."""
    file = pool.directory / ("convpool-" + convpool.identity(messages) + ".bin")
    file.write_bytes(b"native snapshot")
    meta = {"prefix": convpool.prefix(messages[:count]), "messages": count, "tokens": 40000}
    file.with_suffix(".json").write_text(json.dumps(meta))
    return file


def kept(name, out, err, want):
    """One path's line: the same text, on stderr, and nothing on stdout."""
    lines = err.getvalue().splitlines()
    case(name + " keeps its line text on stderr", PREFIX + want in lines, lines)
    case(name + " writes nothing to stdout", out.getvalue() == "", repr(out.getvalue()))


def main():
    # The save path: a conversation with 40000 prompt tokens is evicted by a different one.
    with tempfile.TemporaryDirectory() as tmp:
        pool = active_pool(tmp)
        engine = FakeEngine()
        with disk_free(BIG_FREE), both_streams() as (out, err):
            pool.before(engine, NEW, 7)
        saved = pool.directory / ("convpool-" + convpool.identity(OLD) + ".bin")
        case("an active eviction save keeps its count and snapshot",
             pool.counts["saves"] == 1 and engine.calls == ["save"] and saved.is_file()
             and saved.with_suffix(".json").is_file(),
             (dict(pool.counts), engine.calls, saved.is_file()))
        kept("an active eviction save", out, err, "save on active eviction")

    # The error path: the native save fails.
    with tempfile.TemporaryDirectory() as tmp:
        pool = active_pool(tmp)
        engine = FakeEngine(fail=RuntimeError("native save failed"))
        with disk_free(BIG_FREE), both_streams() as (out, err):
            pool.before(engine, NEW, 4)
        case("a failed save keeps its error count and no snapshot",
             pool.counts["errors"] == 1 and not list(pool.directory.glob("convpool-*")),
             (dict(pool.counts), [p.name for p in pool.directory.glob("convpool-*")]))
        kept("a failed save", out, err, "save failed (native save failed)")

    # The restore path: a snapshot that matches the conversation prefix comes back.
    with tempfile.TemporaryDirectory() as tmp:
        pool = convpool.Pool(tmp, 1)
        snapshot(pool, PAIR, 1)
        engine = FakeEngine()
        with disk_free(BIG_FREE), both_streams() as (out, err):
            pool.before(engine, PAIR, 9)
        case("a matching restore keeps its count and result",
             pool.counts["restores"] == 1 and engine.calls == ["restore"] and pool.counts["misses"] == 0,
             (dict(pool.counts), engine.calls))
        kept("a matching restore", out, err, "restored matching conversation")

    # The miss path: nothing planted for this conversation.
    with tempfile.TemporaryDirectory() as tmp:
        pool = convpool.Pool(tmp, 1)
        engine = FakeEngine()
        with disk_free(BIG_FREE), both_streams() as (out, err):
            pool.before(engine, NEW, 3)
        case("a miss keeps its count and calls nothing",
             pool.counts["misses"] == 1 and engine.calls == [], (dict(pool.counts), engine.calls))
        case("a miss says so on stderr with its reason",
             err.getvalue().startswith(PREFIX + "miss (") and "No such file" in err.getvalue(),
             repr(err.getvalue()))
        case("a miss writes nothing to stdout", out.getvalue() == "", repr(out.getvalue()))

    # Ordinary input: a batched engine never uses the single-slot disk tier and writes no line.
    with tempfile.TemporaryDirectory() as tmp:
        engine = FakeEngine()
        engine.batch = 4
        pool = convpool.Pool(tmp, 1)
        with disk_free(BIG_FREE), both_streams() as (out, err):
            pool.before(engine, NEW, 5)
        case("a batched engine is left alone with no line on either stream",
             pool.active is None and engine.calls == [] and out.getvalue() == "" and err.getvalue() == "",
             (pool.active, engine.calls, repr(out.getvalue()), repr(err.getvalue())))

    failed = results.count(False)
    print(f"{len(results) - failed} of {len(results)} cases pass")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
