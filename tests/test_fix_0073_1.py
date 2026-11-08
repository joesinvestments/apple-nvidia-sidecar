#!/usr/bin/env python3
"""tests/test_fix_0073_1.py - fix-0073#1: serve/test_prefix_replay.py must not invent a private tokenizer path.

The replay tests used to fall back to a private model pack under the home directory when
STRATA_TEST_TOKENIZER was unset. The module must require the variable and skip the real-tokenizer
class when it is absent. Each case imports the module in a child interpreter from the tree under the
current directory, so the source file is never inspected as text.
"""
from __future__ import annotations

import json
import os
import site
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path.cwd()
# A redirected HOME also moves Python's user site-packages, where jinja2 lives on this machine, so
# keep the real one on the child's path.
USER_SITE = site.getusersitepackages()

# Runs in a fresh interpreter: TOK_DIR is computed at import time, and unittest's skip decorators
# leave __unittest_skip__ on the class, so one import reports both facts.
PROBE = r"""
import json, os, sys
sys.path.insert(0, os.getcwd())
import serve.test_prefix_replay as m
print(json.dumps({
    "tok_dir": None if m.TOK_DIR is None else str(m.TOK_DIR),
    "skipped": bool(getattr(m.RealTokenizer, "__unittest_skip__", False)),
}))
"""


def probe(home, tokenizer):
    """Import serve.test_prefix_replay with HOME set to `home` and STRATA_TEST_TOKENIZER set or absent."""
    env = {k: v for k, v in os.environ.items() if k != "STRATA_TEST_TOKENIZER"}
    env["HOME"] = str(home)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    paths = [p for p in (USER_SITE, os.environ.get("PYTHONPATH")) if p]
    if paths:
        env["PYTHONPATH"] = os.pathsep.join(paths)
    if tokenizer is not None:
        env["STRATA_TEST_TOKENIZER"] = str(tokenizer)
    r = subprocess.run([sys.executable, "-c", PROBE], cwd=str(ROOT), env=env,
                       capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.strip()[-500:])
    return json.loads(r.stdout.strip().splitlines()[-1])


def plant_pack(home):
    """The private pack layout the old default named, under a redirected HOME (a temp dir)."""
    pack = home / "models" / "qwen38fn" / "packs" / "iq3_s" / "tokenizer"
    pack.mkdir(parents=True)
    (pack / "vocab.json").write_text("{}", encoding="utf-8")
    return pack


def main():
    failures = 0

    def check(name, ok, detail=""):
        nonlocal failures
        if ok:
            print(f"PASS {name}")
        else:
            failures += 1
            print(f"FAIL {name}: {detail}")

    with tempfile.TemporaryDirectory(prefix="fix-0073-") as tmp:
        tmp = Path(tmp)
        home = tmp / "home"
        home.mkdir()
        planted = plant_pack(home)

        unset = probe(home, None)
        check("unset names no fallback tokenizer directory", unset["tok_dir"] is None,
              f"STRATA_TEST_TOKENIZER unset, TOK_DIR={unset['tok_dir']!r}")
        check("unset skips the real-tokenizer cases", unset["skipped"] is True,
              "STRATA_TEST_TOKENIZER unset, RealTokenizer not skipped")
        leaked = unset["tok_dir"] is not None and (
            "qwen38fn" in unset["tok_dir"] or "iq3_s" in unset["tok_dir"] or unset["tok_dir"].startswith(str(home)))
        check("unset never leaks the private pack location", not leaked,
              f"STRATA_TEST_TOKENIZER unset, TOK_DIR names the planted private pack: {unset['tok_dir']!r}")

        given = tmp / "given"
        given.mkdir()
        (given / "vocab.json").write_text("{}", encoding="utf-8")
        for name in ("merges.txt", "token_type.json", "chat_template.jinja"):
            (given / name).write_text("{}", encoding="utf-8")
        named = probe(home, given)
        check("set uses the named tokenizer directory", named["tok_dir"] == str(given) and not named["skipped"],
              f"STRATA_TEST_TOKENIZER={given}, TOK_DIR={named['tok_dir']!r}, skipped={named['skipped']}")

        empty = tmp / "empty"
        empty.mkdir()
        missing = probe(home, empty)
        check("set without vocab.json still skips", missing["skipped"] is True,
              f"STRATA_TEST_TOKENIZER={empty} has no vocab.json, RealTokenizer not skipped")

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
