"""tests/test_fix_0057_0.py - an encoder stopped on purpose is not reported alive (fix-0057#0).

    python3 tests/test_fix_0057_0.py

Vision.alive() in serve/server.py says True for a picture encoder that was ended after idle_unload_s, on purpose: that
encoder is not "down", the next picture starts it again.  Vision.unload() (POST /unload, the free-VRAM check before a
load, the end of an image job) stops the encoder on purpose too, and it left idle_unloaded set, so an encoder that had
idled out still answered alive() == True after unload(): Service._release_orphan_encoder and _vision_down() took a
deliberately stopped encoder for a running one.

Note that the idle loop ends the encoder through close(), which sets stopped: an idle-unloaded encoder is stopped and
alive() at once, so alive() cannot be gated on stopped (that would report every idle-unloaded encoder as down, and
_vision_down() would start it again with every load).

No encoder is spawned and no model is loaded: Vision is built with idle_unload_s set (nothing starts with the server),
its process is a small fake with poll(), the idle loop runs for the one second it sleeps, and Service stands over
MockEngine and ByteTokenizer.  Every case prints one PASS or FAIL line; any FAIL exits non-zero.
"""
from __future__ import annotations

import io
import shutil
import sys
import time
from pathlib import Path

ROOT = Path.cwd() if (Path.cwd() / "serve" / "server.py").is_file() else Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from serve.server import ByteTokenizer, MockEngine, Service, Vision  # noqa: E402

CTX = 4096
CFG = {"exe": "strata-vision", "mmproj": "mmproj.gguf", "model": "reader.gguf"}   # never spawned here
results: list[bool] = []
dirs: list[Path] = []


def case(name, ok, detail=""):
    results.append(bool(ok))
    print(("PASS " if ok else "FAIL ") + name + (f": {detail}" if detail else ""))


class FakeProc:
    """The encoder as alive() and close() see it: poll() is None while it runs, and close() writes QUIT and waits."""

    def __init__(self, running: bool = True):
        self.stdin = io.StringIO()
        self.running = running
        self.quit = None

    def poll(self):
        return None if self.running else 1

    def wait(self, timeout=None):
        self.running = False
        return 0

    def kill(self):
        self.running = False


def vision(idle_s: float) -> Vision:
    """A Vision built with idle_unload_s: it is not started with the server, so no process is spawned."""
    v = Vision(dict(CFG, idle_unload_s=idle_s))
    dirs.append(v.dir)
    return v


def wait_until(pred, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(0.05)
    return pred()


def ended_by_the_idle_loop():
    """A Vision whose encoder ran, then idled out: the loop in the code ends it, one second after the quiet spell."""
    v = vision(1)
    v.proc, v.stopped, v.idle_unloaded = FakeProc(), False, False
    v.last_used = time.monotonic() - 3600
    return v, wait_until(lambda: v.proc is None and v.idle_unloaded, 10)


class DeadEngine(MockEngine):
    """The engine is not running (POST /unload, or it died): Service.loaded() is False, and a load starts it."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.restarts = 0

    def alive(self):
        return False

    def restart(self):
        self.restarts += 1


def vision_cases():
    running = vision(60)
    running.proc, running.stopped, running.idle_unloaded = FakeProc(), False, False
    case("a running encoder is alive", running.alive() is True, f"alive() -> {running.alive()!r}")

    resident = Vision(dict(CFG), lazy=True)              # no idle_unload_s: it is started with the server and stays
    dirs.append(resident.dir)
    resident.proc, resident.stopped = FakeProc(), False
    up = resident.alive()
    resident.unload()
    case("a resident encoder is up, then down after unload()", up is True and resident.alive() is False,
         f"alive() was {up!r}, now {resident.alive()!r}")

    idle, ended = ended_by_the_idle_loop()
    case("the idle loop ends the encoder and leaves it up for the next picture",
         ended and idle.stopped and idle.alive() is True,
         f"stopped={idle.stopped}, idle_unloaded={idle.idle_unloaded}, alive() -> {idle.alive()!r}")

    idle.unload()
    case("unload() of an encoder that had idled out reports it down", idle.alive() is False,
         f"alive() -> {idle.alive()!r}")

    tok = ByteTokenizer()
    engine = DeadEngine(tok, "ok", max_context=CTX)
    svc = Service(engine, tok, None)
    svc.vision = idle
    again = svc._release_orphan_encoder()
    case("the free-VRAM check does not stop the stopped encoder again", again is False,
         f"_release_orphan_encoder() -> {again!r}")
    case("the Service sees the encoder as down, so the next load starts it", svc._vision_down() is True,
         f"_vision_down() -> {svc._vision_down()!r}")
    starts = []
    idle.restart = lambda: starts.append("vision")      # the encoder is never spawned here
    svc.load()
    case("the next load starts the stopped encoder again", starts == ["vision"] and engine.restarts == 1,
         f"vision starts {len(starts)}, engine starts {engine.restarts}")


def main():
    try:
        vision_cases()
    finally:
        for d in dirs:
            shutil.rmtree(d, ignore_errors=True)
    print(f"{sum(results)}/{len(results)} cases pass")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
