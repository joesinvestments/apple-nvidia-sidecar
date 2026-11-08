"""tests/test_fix_0070_1.py - a selected exchange fault answers /v1/responses in the Responses error shape.

    python3 tests/test_fix_0070_1.py

serve/server.py, in the ValueError branch of Handler._post, answers a selected exchange fault ("verify: exchange
window failed: ..." while r5g4-recovery.flag sits beside the server) with a 503. On the /v1/responses path it wrote
{"error": {"type", "message"}}, the plain shape, while the 400 in the same branch and the EngineNonFinite 503 go
through serve.responses.error_body: {"error": {"message", "type", "param", "code"}}. A Responses client that reads
error.param or error.code on that 503 gets a body it cannot parse. The fix sends the fault through error_body too,
and only on /v1/responses.

No server and no socket: the real request handler runs over a stub Service (MockEngine and ByteTokenizer stand in
for the model and the hardware) and writes its answer to a buffer. pathlib.Path is redirected to a temp folder for
the length of each request, so the recovery flag and marker never touch the tree, and the recovery worker runs in
place. Every case prints one PASS or FAIL line; any FAIL exits non-zero.
"""
from __future__ import annotations

import email.message
import io
import json
import shutil
import sys
import tempfile
import threading
import types
from pathlib import Path
from unittest import mock

ROOT = Path.cwd() if (Path.cwd() / "serve" / "server.py").is_file() else Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from serve import server  # noqa: E402
from serve.server import ByteTokenizer, MockEngine, Service, make_handler  # noqa: E402

CTX = 4096
FAULT = "verify: exchange window failed: the selected exchange window closed"
RESPONSES_KEYS = ["code", "message", "param", "type"]        # serve.responses.error_body
PLAIN_KEYS = ["message", "type"]
results: list[bool] = []


def case(name, ok, detail=""):
    results.append(bool(ok))
    print(("PASS " if ok else "FAIL ") + name + (f": {detail}" if detail else ""))


def service():
    """A Service over the scripted engine and the byte tokenizer (no GPU, no pack, no weights)."""
    tok = ByteTokenizer()
    svc = Service(MockEngine(tok, "hi", max_context=CTX), tok, None)
    svc.engine.restart = lambda: None            # MockEngine has none: the recovery worker's restart is a no-op
    return svc


def temp_path(folder):
    """pathlib.Path, but with_name names a file in folder: the recovery flag and marker stay in a temp folder."""
    class RedirectedPath:
        def __init__(self, *parts):
            self._path = Path(*parts)

        def with_name(self, name):
            return folder / name

        def __getattr__(self, name):
            return getattr(self._path, name)

    return RedirectedPath


class SyncThread:
    """threading.Thread, but start() runs the target in place: the recovery worker's Path lookup happens inside the
    redirected Path, and no thread outlives the request."""

    def __init__(self, target=None, daemon=None, **kwargs):
        self.target = target

    def start(self):
        self.target()


def faulted_handler(svc):
    """The server's own handler, with the generation methods raising the selected exchange fault instead of running
    the model."""
    base = make_handler(svc)

    class Faulted(base):
        def _responses(self, req):
            raise ValueError(FAULT)

        def _openai(self, req):
            raise ValueError(FAULT)

        def _anthropic(self, req):
            raise ValueError(FAULT)

    return Faulted


def post(path, svc, body, flags):
    """POST body to path through the real handler, without a socket: (status, the JSON answer). The recovery flag is
    looked for in the temp folder flags."""
    h = object.__new__(faulted_handler(svc))
    raw = json.dumps(body).encode()
    h.path, h.command = path, "POST"
    h.request_version, h.requestline = "HTTP/1.0", f"POST {path} HTTP/1.0"
    h.headers = email.message.Message()
    h.headers["Content-Type"] = "application/json"
    h.headers["Content-Length"] = str(len(raw))
    h.rfile = io.BytesIO(raw)
    h.wfile, h._headers_buffer = io.BytesIO(), []
    h.close_connection, h.record = False, None
    h.answer_started, h.cancel_event = False, None
    h.server = types.SimpleNamespace(server_address=("127.0.0.1", 8095))
    h.client_address = ("127.0.0.1", 50000)
    with mock.patch.object(server, "Path", temp_path(flags)), mock.patch.object(threading, "Thread", SyncThread):
        h.do_POST()
    head, _, payload = h.wfile.getvalue().partition(b"\r\n\r\n")
    return int(head.split(b" ")[1]), json.loads(payload)


def fault_cases(tmp):
    flags = Path(tmp) / "flags"
    flags.mkdir(parents=True, exist_ok=True)
    (flags / "r5g4-recovery.flag").write_text("on\n", encoding="utf-8")
    svc = service()

    code, body = post("/v1/responses", svc, {"model": "m", "input": "hi"}, flags)
    error = body.get("error") or {}
    case("/v1/responses exchange fault says the Responses error shape",
         code == 503 and sorted(error) == RESPONSES_KEYS and error.get("type") == "server_error"
         and error.get("message") == FAULT,
         f"status {code}, error keys {sorted(error)}, type {error.get('type')!r}")

    code_400, body_400 = post("/v1/responses", svc, [], flags)
    error_400 = body_400.get("error") or {}
    case("an ordinary 400 on /v1/responses keeps the Responses shape",
         code_400 == 400 and sorted(error_400) == RESPONSES_KEYS
         and error_400.get("type") == "invalid_request_error",
         f"status {code_400}, error keys {sorted(error_400)}")

    code_msg, body_msg = post("/v1/messages", svc, {"model": "m", "messages": []}, flags)
    error_msg = body_msg.get("error") or {}
    case("the exchange fault on /v1/messages keeps the plain error shape",
         code_msg == 503 and sorted(error_msg) == PLAIN_KEYS and error_msg.get("type") == "server_error",
         f"status {code_msg}, error keys {sorted(error_msg)}")

    quiet = Path(tmp) / "quiet"
    quiet.mkdir()
    code_nf, body_nf = post("/v1/responses", svc, {"model": "m", "input": "hi"}, quiet)
    error_nf = body_nf.get("error") or {}
    case("without the flag the fault is an ordinary 400",
         code_nf == 400 and sorted(error_nf) == RESPONSES_KEYS and error_nf.get("message") == FAULT,
         f"status {code_nf}, error keys {sorted(error_nf)}")


def main():
    tmp = tempfile.mkdtemp(prefix="fix-0070-1-")
    try:
        fault_cases(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"{sum(results)}/{len(results)} cases pass")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
