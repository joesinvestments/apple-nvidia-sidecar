"""tests/test_health_build_tree.py - /health and /api/health name the source tree the build came from (F8).

    python3 tests/test_health_build_tree.py

tools/stamp-build.sh (called by tools/build-sidecar.sh, which writes the F6 stamp) writes OUTDIR/BUILD_TREE beside
OUTDIR/BUILD_ID: the first 12 hex of git rev-parse HEAD^{tree}, plus "-dirty" when a tracked file is modified, the same
rule BUILD_ID uses for the commit. serve/server.py reads that file once at start, next to read_build_id(), and puts
"tree" in every /health and /api/health body: the stamp, or "unknown" when the file is missing. "build" does not
change.

No server is started and no socket is opened: the request handler is built over a stub Service (MockEngine and
ByteTokenizer stand in for the model and the hardware) and handed one request the way the HTTP server would, with its
answer written to a buffer. The stamp helper runs only in a temp git repo, and the BUILD_ID and BUILD_TREE files live
in a temp folder. Every case prints one PASS or FAIL line; any FAIL exits non-zero.
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
STAMP = "a1b2c3d4e5f6"               # the build stamp a build would leave; never a commit of this tree
TREE = "0f1e2d3c4b5a"                # the source tree stamp a build would leave; never a tree of this repo
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
    """Run tools/stamp-build.sh REPO OUTDIR and read back OUTDIR/BUILD_ID and OUTDIR/BUILD_TREE.
    Returns ((build, tree), error); a missing file reads None."""
    helper = ROOT / "tools" / "stamp-build.sh"
    if not helper.is_file():
        return (None, None), f"{helper} is missing"
    r = run(["bash", str(helper), str(repo), str(outdir)])
    if r.returncode != 0:
        return (None, None), f"exit {r.returncode}: {(r.stderr or r.stdout).strip()[:200]}"
    out = []
    for name in ("BUILD_ID", "BUILD_TREE"):
        path = Path(outdir) / name
        out.append(path.read_text(encoding="utf-8").strip() if path.is_file() else None)
    return tuple(out), ""


def service_for(build_path, tree_path):
    """A Service as main builds it, over the scripted engine and the byte tokenizer (no GPU, no pack, no weights).
    BUILD_ID_PATH and BUILD_TREE_PATH are pointed at the temp stamps while the Service reads them, so a real stamp
    file is never read."""
    tok = ByteTokenizer()
    with mock.patch.object(server, "BUILD_ID_PATH", build_path, create=True), \
            mock.patch.object(server, "BUILD_TREE_PATH", tree_path, create=True):
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
        for name in ("stamp writes the source tree beside the commit", "a changed tracked file adds -dirty to the tree",
                     "the build stamp does not change"):
            case(name, False, err)
        return
    head = run(["git", "rev-parse", "HEAD"], cwd=repo).stdout.strip()
    tree = run(["git", "rev-parse", "HEAD^{tree}"], cwd=repo).stdout.strip()
    (build, got), err = stamp(repo, Path(tmp) / "out")
    case("stamp writes the source tree beside the commit",
         got == tree[:12] and re.fullmatch(r"[0-9a-f]{12}", got or "") is not None,
         f"tree {got!r}, HEAD tree {tree[:12]!r}" + (f", {err}" if err else ""))
    case("the build stamp does not change", build == head[:12],
         f"build {build!r}, commit {head[:12]!r}" + (f", {err}" if err else ""))
    (repo / "tracked.txt").write_text("two\n", encoding="utf-8")
    (build, got), err = stamp(repo, Path(tmp) / "out-dirty")
    case("a changed tracked file adds -dirty to the tree", got == tree[:12] + "-dirty",
         f"tree {got!r}" + (f", {err}" if err else ""))


def health_cases(tmp):
    tmp.mkdir(parents=True, exist_ok=True)
    build_path, tree_path = Path(tmp) / "BUILD_ID", Path(tmp) / "BUILD_TREE"
    build_path.write_text(f"  {STAMP}\n", encoding="utf-8")
    tree_path.write_text(f" {TREE}\n", encoding="utf-8")
    svc = service_for(build_path, tree_path)
    code, body = get("/health", svc)
    case("/health reports the tree stamp as tree", code == 200 and body.get("tree") == TREE,
         f"tree is {body.get('tree')!r}")
    code_alias, alias = get("/api/health", svc)
    case("/api/health reports the same tree", code_alias == 200 and alias.get("tree") == TREE,
         f"tree is {alias.get('tree')!r}")
    case("the build stamp is still reported", body.get("build") == STAMP and alias.get("build") == STAMP,
         f"build is {body.get('build')!r}")
    tree_path.write_text(f"{TREE[:6]}999999\n", encoding="utf-8")
    _, later = get("/health", svc)
    case("the tree is read once at start", later.get("tree") == TREE, f"tree is {later.get('tree')!r}")

    missing = Path(tmp) / "missing" / "BUILD_TREE"
    _, missing_body = get("/health", service_for(build_path, missing))
    case("a missing tree stamp is unknown", missing_body.get("tree") == "unknown",
         f"tree is {missing_body.get('tree')!r}")
    case("a missing tree stamp leaves build alone", missing_body.get("build") == STAMP,
         f"build is {missing_body.get('build')!r}")
    empty = Path(tmp) / "empty"
    empty.write_text(" \n\t\n", encoding="utf-8")
    _, empty_body = get("/api/health", service_for(build_path, empty))
    case("an empty tree stamp is unknown", empty_body.get("tree") == "unknown", f"tree is {empty_body.get('tree')!r}")


def main():
    tmp = tempfile.mkdtemp(prefix="health-build-tree-")
    try:
        stamp_cases(Path(tmp) / "stamp")
        health_cases(Path(tmp) / "health")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"{sum(results)}/{len(results)} cases pass")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
