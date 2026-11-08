"""tests/test_health_build_id.py - /health and /api/health name the build they run (F6).

    python3 tests/test_health_build_id.py

tools/build-sidecar.sh writes build/sidecar/BUILD_ID beside the binary: the 12-character commit of the tree, plus
"-dirty" when git status --porcelain shows changes. serve/server.py reads that file once at start (never by running
git while it serves) and puts "build" in every /health and /api/health body, always, not behind health_host; a
missing, unreadable or empty stamp reads "unknown".

No server is started and no socket is opened: the request handler is built over a stub Service (MockEngine and
ByteTokenizer stand in for the model and the hardware) and handed one request the way the HTTP server would, with its
answer written to a buffer. The stamp helper runs only in a temp git repo, and the BUILD_ID file lives in a temp
folder. Every case prints one PASS or FAIL line; any FAIL exits non-zero.
"""
from __future__ import annotations

import email.message
import io
import json
import re
import shutil
import subprocess
import sys
import tempfile
import types
from pathlib import Path
from unittest import mock

ROOT = Path.cwd() if (Path.cwd() / "serve" / "server.py").is_file() else Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from serve import server  # noqa: E402
from serve.server import ByteTokenizer, MockEngine, Service, make_handler  # noqa: E402

CTX = 4096
STAMP = "a1b2c3d4e5f6"               # the stamp a build would leave; never a commit of this tree
GIT = ["git", "-c", "user.name=test", "-c", "user.email=test@users.noreply.github.com"]
results: list[bool] = []


def case(name, ok, detail=""):
    results.append(bool(ok))
    print(("PASS " if ok else "FAIL ") + name + (f": {detail}" if detail else ""))


def run(cmd, cwd=None):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)


def repo_with_one_commit(tmp):
    """A temp git repo holding tracked.txt, committed once. Returns (repo, error)."""
    repo = Path(tmp) / "repo"
    repo.mkdir(parents=True)
    (repo / "tracked.txt").write_text("one\n", encoding="utf-8")
    for cmd in (["git", "init", "-q"], ["git", "add", "-A"], GIT + ["commit", "-qm", "base"]):
        r = run(cmd, cwd=repo)
        if r.returncode != 0:
            return None, f"{' '.join(cmd)}: {r.stderr.strip()[:200]}"
    return repo, ""


def stamp(repo, outdir):
    """Run tools/stamp-build.sh REPO OUTDIR and read back OUTDIR/BUILD_ID. Returns (stamp, error)."""
    helper = ROOT / "tools" / "stamp-build.sh"
    if not helper.is_file():
        return None, f"{helper} is missing"
    r = run(["bash", str(helper), str(repo), str(outdir)])
    if r.returncode != 0:
        return None, f"exit {r.returncode}: {(r.stderr or r.stdout).strip()[:200]}"
    path = Path(outdir) / "BUILD_ID"
    if not path.is_file():
        return None, f"{path} was not written"
    return path.read_text(encoding="utf-8").strip(), ""


def service_for(build_path):
    """A Service as main builds it, over the scripted engine and the byte tokenizer (no GPU, no pack, no weights).
    BUILD_ID_PATH is pointed at the temp stamp while the Service reads it, so the stamp file is never a real one."""
    tok = ByteTokenizer()
    with mock.patch.object(server, "BUILD_ID_PATH", build_path, create=True):
        return Service(MockEngine(tok, "hi", max_context=CTX), tok, None)


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


def stamp_cases(tmp):
    repo, err = repo_with_one_commit(tmp)
    if repo is None:
        case("stamp is the 12-character commit", False, err)
        case("a changed tracked file adds -dirty", False, err)
        return
    head = run(["git", "rev-parse", "HEAD"], cwd=repo).stdout.strip()
    got, err = stamp(repo, Path(tmp) / "out")
    case("stamp is the 12-character commit",
         got == head[:12] and re.fullmatch(r"[0-9a-f]{12}", got or "") is not None,
         f"stamp {got!r}, commit {head[:12]!r}" + (f", {err}" if err else ""))
    (repo / "tracked.txt").write_text("two\n", encoding="utf-8")
    got, err = stamp(repo, Path(tmp) / "out-dirty")
    case("a changed tracked file adds -dirty", got == head[:12] + "-dirty",
         f"stamp {got!r}" + (f", {err}" if err else ""))


def health_cases(tmp):
    tmp.mkdir(parents=True, exist_ok=True)
    path = Path(tmp) / "BUILD_ID"
    path.write_text(f"  {STAMP}\n", encoding="utf-8")
    svc = service_for(path)
    code, body = get("/health", svc)
    case("/health reports the stamp as build", code == 200 and body.get("build") == STAMP,
         f"build is {body.get('build')!r}")
    code_alias, alias = get("/api/health", svc)
    case("/api/health reports the same build", code_alias == 200 and alias.get("build") == STAMP,
         f"build is {alias.get('build')!r}")
    case("build is present with health_host off", svc.health_host is False and body.get("build") == STAMP,
         f"health_host is {svc.health_host!r}, build is {body.get('build')!r}")
    path.write_text("ffffffffffff\n", encoding="utf-8")
    _, later = get("/health", svc)
    case("the stamp is read once at start", later.get("build") == STAMP, f"build is {later.get('build')!r}")

    missing = Path(tmp) / "missing" / "BUILD_ID"
    _, missing_body = get("/health", service_for(missing))
    case("a missing stamp is unknown", missing_body.get("build") == "unknown",
         f"build is {missing_body.get('build')!r}")
    empty = Path(tmp) / "empty"
    empty.write_text(" \n\t\n", encoding="utf-8")
    _, empty_body = get("/api/health", service_for(empty))
    case("an empty stamp is unknown", empty_body.get("build") == "unknown", f"build is {empty_body.get('build')!r}")
    unreadable = Path(tmp) / "unreadable"
    unreadable.mkdir()
    _, unreadable_body = get("/health", service_for(unreadable))
    case("an unreadable stamp is unknown", unreadable_body.get("build") == "unknown",
         f"build is {unreadable_body.get('build')!r}")


def main():
    tmp = tempfile.mkdtemp(prefix="health-build-id-")
    try:
        stamp_cases(Path(tmp) / "stamp")
        health_cases(Path(tmp) / "health")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"{sum(results)}/{len(results)} cases pass")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
