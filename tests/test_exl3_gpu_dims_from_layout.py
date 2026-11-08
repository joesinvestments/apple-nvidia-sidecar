#!/usr/bin/env python3
"""The EXL3 GPU context is sized from the model's layout, not from literals.

include/strata/core/exl3_gpu.hpp calls the external exl3s_create loader with the hidden size, the
expert intermediate size and the top-k.  Today those are the bare numbers 2560, 640, 10, so the
context would keep that shape for a layout that says otherwise (and a change to the compiled
geometry would leave the call silently disagreeing with it).  This test reads the create call,
evaluates its three dimension arguments against a planted layout, and requires them to follow it.
The stock layout (2560, 640, top-10) gives the same context before and after the fix.

Run: python3 tests/test_exl3_gpu_dims_from_layout.py
Prints one PASS or FAIL line per case and exits non-zero on any FAIL.  No network, no GPU, no model.
"""
from pathlib import Path
import re
import sys

def find_root():
    """The tree under test: the copy prove_test runs us in, else the one this file sits in."""
    rel = Path("include") / "strata" / "core" / "exl3_gpu.hpp"
    for base in (Path.cwd(), Path(__file__).resolve().parent.parent):
        if (base / rel).is_file():
            return base
    return Path.cwd()


ROOT = find_root()
SRC = ROOT / "include" / "strata" / "core" / "exl3_gpu.hpp"

# Layouts the call is evaluated against: a planted one whose three fields differ from the literals,
# and the stock model's geometry (the same numbers before and after the fix).
PLANTED = {"n_embd": 3072, "n_ff": 768, "n_expert_used": 8, "n_expert": 256}
STOCK = {"n_embd": 2560, "n_ff": 640, "n_expert_used": 10, "n_expert": 512}

# create(n_expert, hidden, n_ff, top_k, max_rows, off, max_blob, buf, len)
DIMS = [
    ("hidden size", 1, "n_embd"),
    ("expert intermediate width", 2, "n_ff"),
    ("top-k", 3, "n_expert_used"),
]

CAST = re.compile(
    r"\(\s*(?:const\s+)?(?:std::)?(?:unsigned\s+)?"
    r"(?:long\s+long|long|int|int32_t|int64_t|size_t|uint32_t|uint64_t)\s*\)"
)
IDENT = re.compile(r"[A-Za-z_]\w*(?:::[A-Za-z_]\w*)*(?:\.[A-Za-z_]\w*)*")


def fail(msg):
    print("FAIL " + msg)
    return False


def ok(msg):
    print("PASS " + msg)
    return True


def split_call(src, open_paren):
    """The top-level arguments of the call whose '(' is at open_paren, and the index after its ')'."""
    depth = 0
    i = open_paren
    while i < len(src):
        c = src[i]
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                break
        i += 1
    if depth != 0:
        return None, None
    body = src[open_paren + 1:i]
    args, cur, depth = [], "", 0
    for c in body:
        if c in "([":
            depth += 1
        elif c in ")]":
            depth -= 1
        if c == "," and depth == 0:
            args.append(cur.strip())
            cur = ""
        else:
            cur += c
    args.append(cur.strip())
    return args, i


def dim_value(expr, geom):
    """Evaluate one create argument against `geom`, by the layout field it names."""
    e = CAST.sub(" ", expr)

    def field(m):
        name = m.group(0)
        tail = re.split(r"::|\.", name)[-1]
        if tail == "top_k":
            return str(geom["n_expert_used"])
        if tail in geom:
            return str(geom[tail])
        raise ValueError("cannot size the context from " + name)

    e = IDENT.sub(field, e)
    if not re.fullmatch(r"[\s0-9()+\-*/%]+", e):
        raise ValueError("not an integer expression: " + expr)
    return int(eval(e, {"__builtins__": {}}, {}))


def main():
    results = []
    if not SRC.is_file():
        results.append(fail(str(SRC) + " is missing"))
        return results

    src = SRC.read_text(encoding="utf-8")
    m = re.search(r"r->ctx\s*=\s*create\s*\(", src)
    if m is None:
        results.append(fail("no r->ctx = create(...) call in include/strata/core/exl3_gpu.hpp"))
        return results
    args, _ = split_call(src, m.end() - 1)
    if args is None or len(args) < 8:
        results.append(fail("cannot read the create(...) arguments"))
        return results
    results.append(ok("found the EXL3 context create call (%d arguments)" % len(args)))

    for name, pos, field in DIMS:
        want = PLANTED[field]
        try:
            got = dim_value(args[pos], PLANTED)
        except ValueError as e:
            results.append(fail("%s: %s" % (name, e)))
            continue
        results.append(ok("%s follows the layout: %d" % (name, got)) if got == want
                       else fail("%s does not follow the layout: want %d, got %d" % (name, want, got)))

    same = True
    for name, pos, field in DIMS:
        want = STOCK[field]
        try:
            got = dim_value(args[pos], STOCK)
        except ValueError as e:
            results.append(fail("%s on the stock layout: %s" % (name, e)))
            same = False
            continue
        if got != want:
            results.append(fail("%s on the stock layout: want %d, got %d" % (name, want, got)))
            same = False
    if same:
        results.append(ok("stock layout still sizes the context 2560 x 640 x top-10"))

    return results


if __name__ == "__main__":
    out = main()
    sys.exit(0 if all(out) else 1)
