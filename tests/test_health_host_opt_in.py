"""tests/test_health_host_opt_in.py - /health names the machine only when the run config asks (health_host).

    python3 tests/test_health_host_opt_in.py

serve/server.py:8198 puts socket.gethostname() in every /health and /api/health body. /health answers without an API
key, so a port reachable from the LAN or through a tunnel tells every caller which machine it is. The fix: the body
carries "host" only when the config's "health_host" is true, off by default (main sets it as svc.health_host), and
the rest of the body stays exactly as clients read it today.

No server is started and no socket is opened: the server's own request handler is built over a stub Service
(MockEngine and ByteTokenizer stand in for the model and the hardware) and handed one request the way the HTTP
server would, with its answer written to a buffer. socket.gethostname is planted, so the value can be checked and is
never this Mac's name. Every case prints one PASS or FAIL line; any FAIL exits non-zero.
"""
from __future__ import annotations

import email.message
import io
import json
import sys
import types
from pathlib import Path
from unittest import mock

ROOT = Path.cwd() if (Path.cwd() / "serve" / "server.py").is_file() else Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from serve import server  # noqa: E402
from serve.server import ByteTokenizer, MockEngine, Service, make_handler  # noqa: E402

PLANTED = "planted-studio-12"        # the machine name the handler would report; never this Mac's name
CTX = 4096
BASE = {"status": "ok", "max_context": CTX, "model": "qwen3.8-flash-next", "images": False, "api_key": False,
        "image_generation": False, "loaded": True, "service": "strata"}
results: list[bool] = []


def case(name, ok, detail=""):
    results.append(bool(ok))
    print(("PASS " if ok else "FAIL ") + name + (f": {detail}" if detail else ""))


def service(health_host=None):
    """A Service as main builds it, over the scripted engine and the byte tokenizer (no GPU, no pack, no weights).
    health_host is left to the code under test (the config default) unless a case asks for the opt-in."""
    tok = ByteTokenizer()
    svc = Service(MockEngine(tok, "hi", max_context=CTX), tok, None)
    if health_host is not None:
        svc.health_host = health_host
    return svc


def get(path, svc):
    """GET path through the server's own request handler, without a socket: the answer lands in a buffer."""
    handler_cls = make_handler(svc)
    h = object.__new__(handler_cls)
    h.path, h.command = path, "GET"
    h.request_version, h.requestline = "HTTP/1.0", f"GET {path} HTTP/1.0"
    h.headers, h.rfile = email.message.Message(), io.BytesIO(b"")
    h.wfile, h._headers_buffer = io.BytesIO(), []
    h.close_connection, h.record = False, None
    h.server = types.SimpleNamespace(server_address=("127.0.0.1", 8095))
    h.client_address = ("127.0.0.1", 50000)
    h.do_GET()
    head, _, body = h.wfile.getvalue().partition(b"\r\n\r\n")
    return int(head.split(b" ")[1]), json.loads(body)


def main():
    with mock.patch.object(server.socket, "gethostname", return_value=PLANTED):
        code_off, off = get("/health", service())
        case("default /health has no host", "host" not in off, f"host is {off.get('host')!r}")
        # F6: "build" is always present and its value depends on the checkout's stamp, so it is required, not pinned.
        case("default /health keeps the body clients read", "build" in off and "tree" in off and {k: v for k, v in off.items() if k not in ("build", "tree")} == BASE, f"body is {off}")
        code_alias, alias = get("/api/health", service())
        case("default /api/health has no host and the same body", "host" not in alias and alias == off,
             f"body is {alias}")
        code_on, on = get("/health", service(health_host=True))
        case("health_host true names the machine under host", on.get("host") == PLANTED, f"host is {on.get('host')!r}")
        code_alias_on, alias_on = get("/api/health", service(health_host=True))
        case("health_host true names it on /api/health too", alias_on.get("host") == PLANTED,
             f"host is {alias_on.get('host')!r}")
        case("health_host true adds only host", "build" in on and {k: v for k, v in on.items() if k not in ("host", "build", "tree")} == BASE, f"body is {on}")
        case("every answer is still 200", {code_off, code_alias, code_on, code_alias_on} == {200},
             f"{code_off}, {code_alias}, {code_on}, {code_alias_on}")
    print(f"{sum(results)}/{len(results)} cases pass")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
