"""serve/prefix_replay.py - keep the engine's conversation cache hitting across agent turns (no GPU, no engine).

Why this exists.  The engine reuses the session it holds only when ALL of its tokens (the last prompt + what it
generated) are an exact prefix of the next prompt; otherwise it falls back to the checkpoint at the end of the last
prompt, so the whole previous reply is read again (a 7,952-token tool call = 19 s of prefill on the Mac mini).  A
client does not send back the model's tokens, it sends back a parsed message, and the chat template renders that
message again: trimmed reasoning and content, JSON arguments re-serialised (1.50 -> 1.5, ["a","b"] -> ["a", "b"]).
The text the model sees differs from what it wrote by a token or two, and the cache misses at that point.

What it does.
  * PrefixReplay.record(): after a clean request, remember the prompt ids, the ids the model generated and a
    fingerprint of the message the client received (reasoning, content, tool calls with their parsed arguments).
  * PrefixReplay.apply(): on the next request, when its ids start with a remembered prompt and the assistant message
    that follows in the history has the SAME fingerprint (the client sent back exactly what it was given), the
    template's re-rendering of that message in the ids is replaced by the ids the model really generated.  The model's
    own tokens are the faithful rendering of its message, and they are what the engine holds, so the session matches
    and only the new turn (the tool result) is read.  A message the client changed is never touched.
  * PrefixReplay.sibling_pin(): a request that continues no remembered prompt (a new chat) but shares a long prefix
    with a recent one (the same agent harness in another directory: Pi puts <project_context> and <cwd> after
    thousands of tokens of identical text) asks the engine to checkpoint and pin that shared prefix (`pin=N`, the
    existing strata_prefix mechanism), so the next new chat reads only what differs.
"""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass


class PromptIds(list):
    """The prompt ids Service.prepare() returns: a list that also knows how many messages rendered them, so that
    Service.run() can record the request.  Any arithmetic on it (ids + more) gives a plain list: not recorded."""
    n_messages: int | None = None


def text_of(content) -> str:
    """The text of a template message's content (a string, a list of text parts, None)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p if isinstance(p, str) else (p.get("text") or "") for p in content
                       if isinstance(p, str) or (isinstance(p, dict) and "text" in p))
    return ""


def canon(args) -> str:
    return json.dumps(args, sort_keys=True, ensure_ascii=False, default=str)


def fingerprint(reasoning: str | None, content: str | None, calls) -> tuple:
    """(reasoning, content, ((name, arguments), ...)): what an assistant message says, without the whitespace the
    chat template trims around the reasoning and the content."""
    return ((reasoning or "").strip(), (content or "").strip(),
            tuple((name, canon(args)) for name, args in calls))


def message_fingerprint(message: dict) -> tuple:
    """The fingerprint of an assistant message of the template's messages (the client's history)."""
    calls = []
    for c in message.get("tool_calls") or []:
        fn = c.get("function", c)
        calls.append((fn.get("name"), fn.get("arguments") if fn.get("arguments") is not None else {}))
    return fingerprint(message.get("reasoning_content"), text_of(message.get("content")), calls)


def common_prefix(a, b, block: int = 512) -> int:
    """How many leading elements two sequences share (blocks of equal ids are skipped in C)."""
    n = min(len(a), len(b))
    i = 0
    while i + block <= n and a[i:i + block] == b[i:i + block]:
        i += block
    while i < n and a[i] == b[i]:
        i += 1
    return i


@dataclass
class Record:
    ids: list            # the prompt as sent to the engine
    out: list            # the ids the model generated, without the stop token
    n_messages: int      # how many template messages rendered `ids`
    fp: tuple            # fingerprint of the assistant message the client received


class PrefixReplay:
    def __init__(self, stop_ids, max_records: int = 6, pin_min: int = 2048, decode=None):
        self.stop_ids = frozenset(int(t) for t in stop_ids)
        self.max_records = max(1, int(max_records))
        self.pin_min = int(pin_min)            # 0: no sibling pin
        self.decode = decode                   # ids -> text, for the log's diagnostics (optional)
        self.records: list[Record] = []        # newest last
        self.lock = threading.Lock()
        self.stats = {"replayed": 0, "exact": 0, "changed": 0, "new_chat": 0}

    # -------------------------------------------------------------------------------------------- recording
    def record(self, ids, out, n_messages, fp) -> None:
        out = list(out)
        while out and out[-1] in self.stop_ids:
            out.pop()
        if not out or n_messages is None:
            return
        rec = Record(list(ids), out, int(n_messages), fp)
        with self.lock:
            # the chat moved on: the older prompt of the same conversation is a prefix of this one
            self.records = [r for r in self.records
                            if not (len(r.ids) <= len(rec.ids) and rec.ids[:len(r.ids)] == r.ids)]
            self.records.append(rec)
            del self.records[:-self.max_records]

    # -------------------------------------------------------------------------------------------- applying
    def apply(self, ids, messages) -> tuple[PromptIds, str | None]:
        """-> (ids for the engine, a log line or None).  The ids are the request's own unless the last reply of a
        remembered conversation is in them and the client sent it back unchanged."""
        out = PromptIds(ids)
        out.n_messages = len(messages)
        with self.lock:
            records = list(reversed(self.records))
        note = None
        for rec in records:
            p = len(rec.ids)
            if len(ids) <= p or ids[:p] != rec.ids:
                continue
            at = rec.n_messages
            msg = messages[at] if at < len(messages) else None
            if not isinstance(msg, dict) or msg.get("role") != "assistant":
                return out, None
            j = next((k for k in range(p, len(ids)) if ids[k] in self.stop_ids), None)
            if j is None:
                return out, None
            span = list(ids[p:j])
            if span == rec.out:
                self.stats["exact"] += 1
                return out, None
            if message_fingerprint(msg) != rec.fp:
                self.stats["changed"] += 1
                return out, ("prefix: the client changed the last reply (it is not what was sent), so it is read "
                             f"again: {len(span)} tokens re-rendered from {p}")
            first = common_prefix(span, rec.out)
            new = PromptIds(list(ids[:p]) + rec.out + list(ids[j:]))
            new.n_messages = out.n_messages
            self.stats["replayed"] += 1
            return new, (f"prefix replay: the model's own {len(rec.out)} reply tokens go back into the prompt "
                         f"(the template re-rendered {len(span)}; they differ from token {first} of the reply"
                         + (f": {self._show(rec.out, first)!r} vs {self._show(span, first)!r}" if self.decode else "")
                         + ")")
        return out, note

    def _show(self, ids, at, width=6):
        try:
            return self.decode(list(ids[max(0, at - 2):at + width]))
        except Exception:  # noqa: BLE001 - a log line only
            return "?"

    # -------------------------------------------------------------------------------------------- sibling pin
    def overlap(self, ids) -> tuple[bool, int] | None:
        """How this request relates to the remembered prompts: (continues one of them, the most leading ids it shares
        with any); None when nothing is remembered."""
        with self.lock:
            records = list(self.records)
        if not records:
            return None
        best = 0
        for rec in records:
            if len(ids) > len(rec.ids) and ids[:len(rec.ids)] == rec.ids:
                return True, len(rec.ids)
            best = max(best, common_prefix(ids, rec.ids))
        return False, best

    def pin_for(self, ids, shared: int) -> int | None:
        """The engine's `pin=N` for a NEW chat that shares `shared` leading ids with a recent prompt of another chat:
        a checkpoint there lets the chat after this one read only what differs.  None when the overlap is short (a
        checkpoint costs ~0.3 s and a slot) or leaves no tail to read."""
        if self.pin_min <= 0 or shared < self.pin_min or shared >= len(ids) - 1:
            return None
        self.stats["new_chat"] += 1
        return shared

    def sibling_pin(self, ids) -> int | None:
        o = self.overlap(ids)
        return None if o is None or o[0] else self.pin_for(ids, o[1])
