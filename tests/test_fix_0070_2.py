#!/usr/bin/env python3
"""Behavior test for fix-0070#2: a failed r5g4 engine restart must reach the logs, not only the marker file.

The handler's r5g4 worker is started from the ValueError branch. This test calls the real Handler._post with a
small fake Service: the exchange fault is a ValueError with the selected prefix, the engine restart either fails
(the planted case) or succeeds, and stdout, stderr and the logging records are captured around the call.

The failure case is the harm: today the restart exception is only written to r5g4-recovery.result, so the failure
is invisible in stdout, stderr and logging. With the fix in place the exception is also logged and the same run
shows it.
"""
import contextlib
import email.message
import io
import json
import logging
import os
import sys
import tempfile
import types
from pathlib import Path

ROOT = Path(os.getcwd())
if not (ROOT / "serve" / "server.py").is_file():       # run from another directory: fall back to the test's tree
    ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import serve.server as server  # noqa: E402

EXCHANGE_FAULT = "verify: exchange window failed: planted exchange fault"
RESTART_ERROR = "planted restart failure"


class _ImmediateThread:
    """threading.Thread that runs its target now: the marker and the log are written before the case is read."""

    def __init__(self, target=None, args=(), kwargs=None, daemon=None, name=None):
        self._target = target
        self._args = args
        self._kwargs = kwargs or {}

    def start(self):
        self._target(*self._args, **self._kwargs)


class _RecordHandler(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.lines = []

    def emit(self, record):
        self.lines.append(self.format(record))


class _Fifo:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _Engine:
    def __init__(self, fails):
        self.fails = fails
        self.calls = 0

    def restart(self):
        self.calls += 1
        if self.fails:
            raise RuntimeError(RESTART_ERROR)
        return True


class _Service:
    """Only what the /v1/chat/completions path reads before the exchange fault is raised."""

    api_key = None
    model = "mock"

    def __init__(self, restart_fails, begin_error):
        self.engine = _Engine(restart_fails)
        self.fifo = _Fifo()
        self.begin_error = begin_error
        self.released = 0

    def reserve_inference(self):
        return True

    def release_inference(self):
        self.released += 1

    def drop_embeddings(self):
        pass

    def begin_request(self, path, req):
        raise ValueError(self.begin_error)


def _run_case(flag_present, restart_fails, begin_error):
    with tempfile.TemporaryDirectory(prefix="sce0070-") as tmp:
        server_path = os.path.join(tmp, "server.py")
        Path(server_path).write_text("")
        if flag_present:
            Path(tmp, "r5g4-recovery.flag").write_text("")
        svc = _Service(restart_fails, begin_error)
        handler = object.__new__(server.make_handler(svc))
        handler.path = "/v1/chat/completions"
        handler.headers = email.message.Message()
        handler.record = None
        handler.watch_done = None
        handler.cancel_event = None
        handler.answer_started = False
        responses = []

        def record_json(code, obj, headers=None, retry_after=None):
            responses.append((code, obj))

        handler._json = record_json
        handler._body = lambda: b"{}"

        out, err = io.StringIO(), io.StringIO()
        records = _RecordHandler()
        records.setFormatter(logging.Formatter("%(levelname)s %(name)s %(message)s"))
        root = logging.getLogger()
        old_level = root.level
        root.addHandler(records)
        root.setLevel(logging.DEBUG)
        old_threading, old_file = server.threading, server.__file__
        server.threading = types.SimpleNamespace(Thread=_ImmediateThread)
        server.__file__ = server_path
        try:
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                handler._post()
        finally:
            server.threading, server.__file__ = old_threading, old_file
            root.removeHandler(records)
            root.setLevel(old_level)

        marker_path = Path(tmp, "r5g4-recovery.result")
        return {
            "responses": responses,
            "marker": marker_path.read_text() if marker_path.exists() else None,
            "visible": "\n".join([out.getvalue(), err.getvalue()] + records.lines),
            "engine": svc.engine,
            "released": svc.released,
        }


def _status(result):
    return result["responses"][0] if result["responses"] else (None, {})


def case_failed_restart_is_visible():
    result = _run_case(True, True, EXCHANGE_FAULT)
    problems = []
    code, body = _status(result)
    if code != 503:
        problems.append(f"the client answer was {code}, want 503")
    if result["engine"].calls != 1:
        problems.append(f"engine.restart ran {result['engine'].calls} times, want 1")
    if result["marker"] is None or "restart failed" not in result["marker"]:
        problems.append(f"the marker does not record the failed restart: {result['marker']!r}")
    if RESTART_ERROR not in result["visible"]:
        problems.append("the failed restart is invisible in stdout, stderr and logging")
    return problems


def case_successful_restart_is_quiet():
    result = _run_case(True, False, EXCHANGE_FAULT)
    problems = []
    code, body = _status(result)
    if code != 503:
        problems.append(f"the client answer was {code}, want 503")
    if result["engine"].calls != 1:
        problems.append(f"engine.restart ran {result['engine'].calls} times, want 1")
    if result["marker"] is None or "engine restarted" not in result["marker"]:
        problems.append(f"the marker does not record the restart: {result['marker']!r}")
    if RESTART_ERROR in result["visible"] or "restart failed" in result["visible"]:
        problems.append("a successful restart reported a failure")
    return problems


def case_ordinary_value_error_is_a_bad_request():
    result = _run_case(True, True, "messages: not an exchange fault")
    problems = []
    code, body = _status(result)
    if code != 400:
        problems.append(f"the client answer was {code}, want 400")
    elif body.get("error", {}).get("type") != "invalid_request_error":
        problems.append(f"the 400 body is {body!r}")
    if result["engine"].calls != 0:
        problems.append(f"engine.restart ran {result['engine'].calls} times on an ordinary bad request")
    if result["marker"] is not None:
        problems.append(f"an ordinary bad request wrote the restart marker: {result['marker']!r}")
    if "400 invalid request" not in result["visible"]:
        problems.append("the ordinary bad request was not logged")
    return problems


def case_exchange_fault_without_flag_is_a_bad_request():
    result = _run_case(False, True, EXCHANGE_FAULT)
    problems = []
    code, body = _status(result)
    if code != 400:
        problems.append(f"the client answer was {code}, want 400 without the opt-in flag")
    if result["engine"].calls != 0:
        problems.append(f"engine.restart ran {result['engine'].calls} times without the opt-in flag")
    if result["marker"] is not None:
        problems.append(f"the restart marker was written without the opt-in flag: {result['marker']!r}")
    if "400 invalid request" not in result["visible"]:
        problems.append("the refused exchange fault was not logged")
    return problems


CASES = [
    ("failed restart reaches the logs and the marker", case_failed_restart_is_visible),
    ("successful restart writes the marker and no failure", case_successful_restart_is_quiet),
    ("ordinary ValueError is still a 400 and never restarts", case_ordinary_value_error_is_a_bad_request),
    ("exchange fault without the opt-in flag is still a 400", case_exchange_fault_without_flag_is_a_bad_request),
]


def main():
    failed = 0
    for name, case in CASES:
        try:
            problems = case()
        except Exception as exc:                        # a broken case is a FAIL, never a silent pass
            problems = [f"the case raised {type(exc).__name__}: {exc}"]
        if problems:
            failed += 1
            print(f"FAIL: {name}: " + "; ".join(problems))
        else:
            print(f"PASS: {name}")
    print(json.dumps({"failed": failed, "cases": len(CASES)}))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
