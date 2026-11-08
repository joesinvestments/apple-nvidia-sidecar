#!/usr/bin/env python3
"""serve/convpool.py: an exact-length snapshot is a valid prefix match and must restore.

Pool.before() saves a conversation when a different one evicts it, writing meta['messages'] as the
conversation length at that moment (line 67).  When the same conversation comes back with no new
messages, len(messages) equals that saved count and prefix(messages[:count]) still matches, so the
snapshot is a valid restore.  Today's guard at line 85 rejects it with "count >= len(messages)",
counting a miss and reading cold.  The fix makes that comparison "count > len(messages)".

This test calls the real Pool with a fake engine that writes and restores session files in a temp
folder.  No model, no GPU, no network, no server.  Ordinary inputs (a shorter matching prefix, a
mismatched prefix, a longer snapshot, no snapshot at all) behave the same before and after.

    python3 tests/test_fix_0047_2.py
"""
import json
import sys
import tempfile
import types
from pathlib import Path
from unittest import mock

ROOT = Path.cwd() if (Path.cwd() / "serve" / "convpool.py").is_file() else Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from serve import convpool  # noqa: E402

BIG_FREE = 200 * (1 << 30)        # the save path needs 51 GiB free; the host under test has 11
results = []


def case(name, ok, detail=""):
    results.append(bool(ok))
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else ": " + str(detail)))


class FakeEngine:
    """The two calls Pool makes: session_file('save', path) and session_file('restore', path)."""

    def __init__(self):
        self.calls = []

    def session_file(self, action, path):
        self.calls.append((action, Path(path).name))
        if action == "save":
            Path(path).write_bytes(b"native snapshot")
        return {"tokens": 64, "ms": 3}


def planted(directory, messages, count, digest=None, tokens=40000):
    """Write the .bin plus the meta .json the save path leaves for a conversation, with `count` messages."""
    key = convpool.identity(messages)
    file = Path(directory) / ("convpool-" + key + ".bin")
    file.write_bytes(b"native snapshot")
    meta = {"prefix": convpool.prefix(messages[:count]) if digest is None else digest,
            "messages": count, "tokens": tokens}
    file.with_suffix(".json").write_text(json.dumps(meta))
    return file


def pool_in(tmp):
    return convpool.Pool(tmp, 1)


def resumed(messages, plant, tokens=123):
    """-> (pool counts, engine) after one Pool.before() against the planted snapshot."""
    with tempfile.TemporaryDirectory() as tmp:
        pool = pool_in(tmp)
        engine = FakeEngine()
        plant(tmp, messages)
        pool.before(engine, messages, tokens)
        return dict(pool.counts), engine, pool.active


def main():
    conversation = [{"role": "user", "content": "trace the build"},
                    {"role": "assistant", "content": "which target?"},
                    {"role": "user", "content": "the release one"}]

    # The harm: a snapshot saved at the conversation's current length is the same conversation resumed.
    counts, engine, active = resumed(conversation,
                                     lambda tmp, m: planted(tmp, m, len(m)))
    case("exact-length snapshot is restored",
         counts["restores"] == 1 and [c[0] for c in engine.calls] == ["restore"], (counts, engine.calls))
    case("exact-length snapshot is not counted as a miss", counts["misses"] == 0, counts)
    case("exact-length restore leaves the conversation active", active is not None and active[3] == len(conversation),
         active)

    # Ordinary input: a shorter matching prefix restores before and after.
    shorter = [{"role": "user", "content": "trace the build"},
               {"role": "assistant", "content": "which target?"},
               {"role": "user", "content": "the release one"},
               {"role": "assistant", "content": "building"}]
    counts, engine, _ = resumed(shorter, lambda tmp, m: planted(tmp, m, 2))
    case("shorter matching prefix is restored", counts["restores"] == 1 and counts["misses"] == 0,
         (counts, engine.calls))

    # Ordinary input: a mismatched prefix is still a miss.
    counts, engine, _ = resumed(shorter, lambda tmp, m: planted(tmp, m, 2, digest="0" * 64))
    case("mismatched prefix is a miss", counts["restores"] == 0 and counts["misses"] == 1,
         (counts, engine.calls))

    # Ordinary input: a snapshot with more messages than the conversation is still a miss.
    counts, engine, _ = resumed(shorter, lambda tmp, m: planted(tmp, m, len(m) + 1))
    case("longer snapshot is a miss", counts["restores"] == 0 and counts["misses"] == 1,
         (counts, engine.calls))

    # Ordinary input: no snapshot on disk is still a miss.
    with tempfile.TemporaryDirectory() as tmp:
        pool = pool_in(tmp)
        engine = FakeEngine()
        pool.before(engine, shorter, 123)
        case("missing snapshot is a miss", pool.counts["restores"] == 0 and pool.counts["misses"] == 1,
             (dict(pool.counts), engine.calls))

    # End to end: eviction saves the conversation at its current length, then the same conversation resumes.
    first = [{"role": "user", "content": "alpha"}]
    other = [{"role": "user", "content": "beta"}]
    with tempfile.TemporaryDirectory() as tmp:
        pool = pool_in(tmp)
        engine = FakeEngine()
        pool.active = (convpool.identity(first), convpool.prefix(first), 40000, len(first))
        with mock.patch.object(convpool.shutil, "disk_usage", return_value=types.SimpleNamespace(free=BIG_FREE)):
            pool.before(engine, other, 9)             # beta evicts alpha, saving it
        saved = (Path(tmp) / ("convpool-" + convpool.identity(first) + ".json"))
        meta = json.loads(saved.read_text())
        case("eviction saves the conversation length", meta["messages"] == len(first)
             and meta["prefix"] == convpool.prefix(first), meta)
        pool.active = None                            # a restart: the snapshot is all that is left
        misses_before = pool.counts["misses"]         # beta's own start had nothing to restore
        pool.before(engine, first, 10)                # same conversation, no new messages
        case("evicted conversation resumes from its snapshot",
             pool.counts["restores"] == 1 and pool.counts["misses"] == misses_before, dict(pool.counts))

    failed = results.count(False)
    print(f"{len(results) - failed} of {len(results)} cases pass")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
