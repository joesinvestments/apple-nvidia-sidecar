#!/usr/bin/env python3
"""A conversation pool that raises in finished() must not leave the request busy.

serve/server.py:6799 calls self.convpool.finished(...) just ahead of st["busy"] = False and the tail/tool
pops (6800-6802), so a disk tier that raised there skipped all three: the conversation stayed busy with the
finished answer's tail and tool still in its status, and the error escaped the request.  This test runs the
real Service against the mock engine with a stub pool (no model, no GPU, no network): a raising pool must
leave the request finished and its status settled, a pool that returns must not change the reply, and the
failure must be reported rather than escaping.

    python3 tests/test_convpool_finished_raises_not_busy.py
"""
import contextlib
import io
import sys
import threading
from pathlib import Path

ROOT = Path.cwd()                         # prove_test runs the test with cwd at the tree under test
sys.path.insert(0, str(ROOT))

CTX = 4096
ANSWER = "hello"
POOL_ERROR = "the conversation pool is gone"


class StubPool:
    """The two calls Service makes into the pool: before() at the start, finished() when the request settles."""

    def __init__(self, error=None):
        self.error, self.calls = error, []

    def before(self, engine, messages, prompt_tokens):
        pass

    def finished(self, messages, prompt_tokens, finish):
        self.calls.append((messages, prompt_tokens, finish))
        if self.error is not None:
            raise self.error


def run_request(pool):
    """-> (service, events, the exception that escaped the request or None, what it printed, messages, ids)."""
    from serve.frontend import ChatTemplate
    from serve.server import ByteTokenizer, MockEngine, Service
    tok = ByteTokenizer()
    svc = Service(MockEngine(tok, ANSWER, max_context=CTX), tok,
                  ChatTemplate(ROOT / "serve" / "chat_template.jinja"))
    svc.convpool = pool
    messages = [{"role": "user", "content": "hi"}]
    ids = tok.encode("hi")
    events, escaped, printed = [], None, io.StringIO()
    stream = svc.run(ids, False, None, 30, {"messages": messages}, threading.Event())
    try:
        with contextlib.redirect_stdout(printed):
            for ev in stream:
                events.append(ev)
    except Exception as exc:              # noqa: BLE001 - a pool error escaping the request is the harm
        escaped = exc
    return svc, events, escaped, printed.getvalue(), messages, ids


def answer_of(events):
    return "".join(ev.text or "" for kind, ev in events if kind == "event")


def done_of(events):
    return events[-1][1] if events and events[-1][0] == "done" else None


def main():
    results = []

    def case(name, ok, detail=""):
        results.append(bool(ok))
        print(("PASS " if ok else "FAIL ") + name + ("" if ok else f": {detail}"))

    raised = StubPool(RuntimeError(POOL_ERROR))
    svc, events, escaped, printed, messages, ids = run_request(raised)
    case("a pool that raises does not escape the request", escaped is None, repr(escaped))
    case("the reply still finishes", (done_of(events) or {}).get("finish") == "stop", done_of(events))
    case("the status is no longer busy", svc.status.get("busy") is False, dict(svc.status))
    case("the tail and the tool are gone", "tail" not in svc.status and "tool" not in svc.status, dict(svc.status))
    case("the pool's failure is reported", POOL_ERROR in printed, printed[-300:])
    case("the pool still sees the finished turn", raised.calls == [(messages, len(ids), "stop")], raised.calls)

    good = StubPool()
    svc_pool, events_pool, escaped_pool, _, messages_pool, ids_pool = run_request(good)
    svc_plain, events_plain, escaped_plain, _, _, _ = run_request(None)
    case("a pool that returns does not change the reply",
         escaped_pool is None and escaped_plain is None and done_of(events_pool) == done_of(events_plain)
         and answer_of(events_pool) == answer_of(events_plain),
         (done_of(events_pool), done_of(events_plain), answer_of(events_pool), answer_of(events_plain)))
    case("a pool that returns leaves the status settled",
         svc_pool.status.get("busy") is False and "tail" not in svc_pool.status and "tool" not in svc_pool.status,
         dict(svc_pool.status))
    case("a pool that returns sees the finished turn",
         good.calls == [(messages_pool, len(ids_pool), "stop")] and list(svc_pool.history)[-1]["finish"] == "stop",
         (good.calls, list(svc_pool.history)))

    failed = results.count(False)
    print(f"{len(results) - failed} of {len(results)} cases pass")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
