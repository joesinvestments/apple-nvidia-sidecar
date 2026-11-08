#!/usr/bin/env python3
"""Behavior check for fix-0074#1: the real-tokenizer tests must skip when the tokenizer directory is
incomplete, because setUpClass reads vocab.json, merges.txt, token_type.json and chat_template.jinja.

Each case plants a temporary tokenizer directory and runs the existing RealTokenizer class with
STRATA_TEST_TOKENIZER pointed at it. A complete skip exits 0; today's guard only looks at vocab.json,
so a directory that holds vocab.json but lacks one of the other three raises FileNotFoundError instead
of skipping. The empty directory is the ordinary input: it skips before and after the change.
"""
import os
import shutil
import subprocess
import sys
import tempfile

TREE = os.getcwd()
MODULE = os.path.join(TREE, "serve", "test_prefix_replay.py")
TARGET = "serve.test_prefix_replay.RealTokenizer"

VOCAB = '{"a": 0, "b": 1}'
MERGES = "a b"
TYPES = "[1, 1]"
TEMPLATE = "{% for m in messages %}{{ m['content'] }}{% endfor %}"

CASES = [
    ("empty directory skips", {}),
    ("vocab.json only skips", {"vocab.json": VOCAB}),
    ("missing merges.txt skips", {"vocab.json": VOCAB, "token_type.json": TYPES,
                                  "chat_template.jinja": TEMPLATE}),
    ("missing token_type.json skips", {"vocab.json": VOCAB, "merges.txt": MERGES,
                                       "chat_template.jinja": TEMPLATE}),
    ("missing chat_template.jinja skips", {"vocab.json": VOCAB, "merges.txt": MERGES,
                                           "token_type.json": TYPES}),
]


def run_case(files):
    root = tempfile.mkdtemp(prefix="fix-0074-")
    try:
        for name, text in files.items():
            with open(os.path.join(root, name), "w", encoding="utf-8") as f:
                f.write(text)
        env = dict(os.environ, STRATA_TEST_TOKENIZER=root)
        r = subprocess.run([sys.executable, "-m", "unittest", TARGET, "-v"], cwd=TREE,
                           env=env, capture_output=True, text=True, timeout=120)
        return r.returncode, r.stdout + r.stderr
    finally:
        shutil.rmtree(root, ignore_errors=True)


def skipped(rc, out):
    return rc == 0 and "(skipped" in out and "FAILED" not in out


def main():
    if not os.path.isfile(MODULE):
        print("FAIL no serve/test_prefix_replay.py under " + TREE)
        return 1
    bad = 0
    for name, files in CASES:
        rc, out = run_case(files)
        if skipped(rc, out):
            print("PASS " + name)
        else:
            bad += 1
            tail = " | ".join(line.strip() for line in out.strip().splitlines()[-3:])
            print("FAIL %s: exit %d, the class did not skip: %s" % (name, rc, tail))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
