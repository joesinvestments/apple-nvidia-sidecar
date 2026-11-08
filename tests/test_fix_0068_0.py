"""An early stream close after the answer is complete must still remember the reply for prefix replay.

serve/server.py:6879 calls self.remember_reply(...) after the stream's last yields (the parser's finish events at
6852-6853 and the held stop tail at 6877-6878).  A client that closes the stream at one of those yields never
resumes the generator to that line, so a reply the model did finish is silently not recorded and the next turn
reads the whole reply again.  This test runs the real Service against the mock engine with the prefix replay cache
on (no model, no GPU, no network): the same request read to its end records one reply, and one closed right after
its last content event must record the same reply.

    python3 tests/test_fix_0068_0.py
"""
import sys
import threading
from pathlib import Path

ROOT = Path.cwd()                         # prove_test runs the test with cwd at the tree under test
sys.path.insert(0, str(ROOT))

from serve.frontend import ChatTemplate  # noqa: E402
from serve.prefix_replay import PrefixReplay, PromptIds  # noqa: E402
from serve.server import ByteTokenizer, MockEngine, Service  # noqa: E402

CTX = 4096
HELD = "hello <tool"                      # ends in a partial <tool_call> opener, held to the parser's finish
PLAIN = "hello"


def request(script):
    """A Service with the replay cache on and one request's ids (a PromptIds, as prepare() hands to run())."""
    tok = ByteTokenizer()
    svc = Service(MockEngine(tok, script, max_context=CTX), tok,
                  ChatTemplate(ROOT / "serve" / "chat_template.jinja"))
    svc.replay = PrefixReplay(svc.stop_ids, pin_min=0, decode=tok.decode)
    ids = PromptIds(tok.encode("hi"))
    ids.n_messages = 1
    return svc, tok, ids


def read(script, stop_at=None, close_after=None):
    """Run one request -> (service, events, done event, the text sent, the exception that escaped or None).

    stop_at: close the stream as soon as the sent text is this (before the generator resumes past that yield).
    close_after: close before reading any event at all.
    """
    svc, tok, ids = request(script)
    stream = svc.run(ids, False, None, 30, {}, threading.Event())
    events, done, text, escaped = [], None, "", None
    try:
        if close_after is not None:
            next(stream)
            stream.close()
            return svc, events, done, text, None
        for kind, ev in stream:
            events.append((kind, ev))
            if kind == "event":
                text += ev.text or ""
            elif kind == "done":
                done = ev
                break
            if stop_at is not None and text == stop_at:
                stream.close()
                break
    except Exception as exc:              # noqa: BLE001 - a stream that raises at close is the harm
        escaped = exc
    return svc, events, done, text, escaped


def main():
    results = []

    def case(name, ok, detail=""):
        results.append(bool(ok))
        print(("PASS " if ok else "FAIL ") + name + ("" if ok else f": {detail}"))

    # the harm: the answer is complete (its held tail is the last event) and the client closes there
    svc, events, done, text, escaped = read(HELD, stop_at=HELD)
    records = list(svc.replay.records)
    case("a close after the last content event does not raise", escaped is None, repr(escaped))
    case("the client saw the whole answer before closing", done is None and text == HELD, (done, text))
    case("a reply the model finished is remembered even when the stream is closed at its last event",
         len(records) == 1, records)
    case("the remembered reply is the tokens the model generated",
         bool(records) and records[0].out == ByteTokenizer().encode(HELD), records)
    case("the remembered prompt is the request's own and keeps its message count",
         bool(records) and records[0].ids == list(ByteTokenizer().encode("hi")) and records[0].n_messages == 1,
         records)

    # ordinary input: read to the end, one record, same before and after
    svc, events, done, text, escaped = read(PLAIN)
    records = list(svc.replay.records)
    case("a stream read to its end finishes normally", escaped is None and (done or {}).get("finish") == "stop",
         (escaped, done))
    case("a stream read to its end records the reply once", len(records) == 1, records)
    case("the reply read to its end is the same one the early close remembers",
         bool(records) and records[0].out == ByteTokenizer().encode(PLAIN), records)

    # a close before any content: the turn did not finish, so nothing is remembered
    svc, events, done, text, escaped = read(PLAIN, close_after=True)
    case("a close before any content raises nothing and remembers nothing",
         escaped is None and done is None and not svc.replay.records, (escaped, done, svc.replay.records))

    failed = results.count(False)
    print(f"{len(results) - failed} of {len(results)} cases pass")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
