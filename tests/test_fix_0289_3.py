#!/usr/bin/env python3
"""fix-0289#3: the lockstep branch's final combine must apply the NVFP4 MMQ s_down (row_sd).

src/prefill/prefill.cpp:6588 takes the lockstep branch (`mac_layer && ls != nullptr`): it adds the Mac
lane's rows to the chunk's parked partial sum with moe_combine_peer16 (or moe_combine_peer).  The parked
sums are built at 6449 without s_down, and the branch's calls at 6590/6592 pass no row_sd, while the
non-lockstep Mac-lane calls at 6595-6599 pass m.row_sd for exactly the NVFP4 MMQ layer (use_mmq &&
!fused_l && lay.native && tail_off).  On such a layer the lockstep combine therefore scales nothing and
its sums come out with the s_down missing.

The test reads the two lockstep calls out of the source, evaluates the row_sd expression each one passes
under a planted NVFP4 MMQ layer (and under ordinary layers), and feeds the result into a CPU model of the
combine kernel (src/prefill/kernels.cu: moe_combine_peer16_kernel / moe_combine_peer_kernel) on planted
rows.  The planted NVFP4 MMQ cases fail while the argument is missing and pass once the branch passes the
same row_sd condition as the non-lockstep calls; the ordinary cases are unchanged either way.

    python3 tests/test_fix_0289_3.py
"""
import math
import re
import struct
import sys
from pathlib import Path

ROOT = Path.cwd()                         # prove_test runs the test with cwd at the tree under test
SOURCE = ROOT / "src" / "prefill" / "prefill.cpp"

K = 10                                    # moe_combine_peer16_kernel's fixed top-k unroll
OUT_OF_RANGE = 1 << 30


def fail(message):
    sys.exit(f"FAIL: {message}")


def call_arguments(text, name):
    """Every `name(...)` call in text as its argument-source list, parentheses balanced."""
    found, i = [], 0
    while True:
        start = text.find(name + "(", i)
        if start < 0:
            return found
        if start > 0 and (text[start - 1].isalnum() or text[start - 1] == "_"):
            i = start + 1
            continue
        depth, j = 0, start + len(name)
        while j < len(text):
            if text[j] == "(":
                depth += 1
            elif text[j] == ")":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        args, piece, depth = [], [], 0
        for ch in text[start + len(name) + 1:j]:
            if ch in "([{":
                depth += 1
            elif ch in ")]}":
                depth -= 1
            if ch == "," and depth == 0:
                args.append("".join(piece).strip())
                piece = []
            else:
                piece.append(ch)
        if "".join(piece).strip():
            args.append("".join(piece).strip())
        found.append(args)
        i = j + 1


def top_level(text, mark):
    depth = 0
    for i, ch in enumerate(text):
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        elif ch == mark and depth == 0:
            return i
    return -1


def to_python(expr):
    """The small subset of C++ the row_sd conditions use, as a Python expression."""
    expr = re.sub(r"\(\s*size_t\s*\)", "", expr)
    expr = re.sub(r"lay\.fmt\s*\[[^\]]*\]\s*\.\s*tail_off", "tail_off", expr)
    expr = expr.replace("lay.native", "native").replace("m.row_sd", "row_sd").replace("nullptr", "None")
    expr = re.sub(r"!(?!=)", " not ", expr)
    return expr.replace("&&", " and ").replace("||", " or ")


def row_sd_argument(args, env):
    """The call's 11th argument (row_sd) evaluated under env; None when the call passes ten arguments."""
    if len(args) < 11:
        return None
    expr = " ".join(args[10].split())
    mark = top_level(expr, "?")
    if mark >= 0:
        colon = top_level(expr[mark + 1:], ":") + mark + 1
        cond = eval(to_python(expr[:mark]), {"__builtins__": {}}, env)
        branch = expr[mark + 1:colon] if cond else expr[colon + 1:]
        return eval(to_python(branch), {"__builtins__": {}}, env)
    return eval(to_python(expr), {"__builtins__": {}}, env)


def combine_peer16(D, slot, w, shared, sg, peer16, rows_local, T, N, row_sd):
    """src/prefill/kernels.cu moe_combine_peer16_kernel, on the CPU."""
    out = []
    for t in range(T):
        for d in range(N):
            s = 0.0
            for k in range(K):
                r = slot[t * K + k]
                if r < rows_local:
                    v = D[r * N + d]
                    if row_sd is not None:
                        v *= row_sd[r]
                    s = s + w[t * K + k] * v
            half = struct.unpack("<e", struct.pack("<H", peer16[t * N + d]))[0]
            out.append((s + half) + shared[t * N + d] / (1.0 + math.exp(-sg[t])))
    return out


def combine_peer(D, slot, w, shared, sg, peer, rows_local, T, N, row_sd):
    """src/prefill/kernels.cu moe_combine_peer_kernel, on the CPU."""
    out = []
    for t in range(T):
        for d in range(N):
            s = 0.0
            for k in range(K):
                r = slot[t * K + k]
                if r < rows_local:
                    v = D[r * N + d]
                    if row_sd is not None:
                        v *= row_sd[r]
                    s = s + w[t * K + k] * v
            out.append((s + peer[t * N + d]) + shared[t * N + d] / (1.0 + math.exp(-sg[t])))
    return out


def f16(x):
    return struct.unpack("<H", struct.pack("<e", x))[0]


def planted(n_row_sd):
    """A planted lockstep chunk: token t's parked sum is row t (grp_slot_id), weight 1 (grp_w_one)."""
    T, N = 2, 4
    D = [0.5, -1.0, 2.0, 0.25, 1.5, 0.75, -0.5, 3.0]
    slot = [OUT_OF_RANGE] * (T * K)
    weight = [0.0] * (T * K)
    for t in range(T):
        slot[t * K] = t
        weight[t * K] = 1.0
    shared = [0.5, 0.25, 0.125, 0.0, -0.5, 0.75, 0.25, 0.5]
    sg = [0.5, -1.0]
    peer = [0.5, -0.25, 0.125, 1.0, 2.0, -1.5, 0.75, 0.25]
    sd = [0.5, 2.0] + [1.0] * (n_row_sd - 2)
    return dict(T=T, N=N, D=D, slot=slot, weight=weight, shared=shared, sg=sg, peer=peer,
                peer16=[f16(v) for v in peer], sd=sd)


def scaled(D, sd, N):
    return [v * sd[i // N] for i, v in enumerate(D)]


def close(got, want):
    return len(got) == len(want) and all(abs(a - b) <= 1e-5 * max(1.0, abs(b)) for a, b in zip(got, want))


def main():
    if not SOURCE.is_file():
        fail(f"{SOURCE} is missing")
    source = SOURCE.read_text(errors="replace")
    start = source.find("if (mac_layer && ls != nullptr)")
    end = source.find("} else if (mac_layer && pf_mac_y16())", start)
    if start < 0 or end < 0:
        fail("prefill.cpp: the lockstep Mac-lane combine block was not found")
    block = source[start:end]
    calls16 = call_arguments(block, "moe_combine_peer16")
    calls32 = call_arguments(block, "moe_combine_peer")
    if len(calls16) != 1 or len(calls32) != 1:
        fail(f"prefill.cpp: expected one lockstep combine per sum width, found {len(calls16)} fp16 and {len(calls32)} fp32")
    for name, args in (("moe_combine_peer16", calls16[0]), ("moe_combine_peer", calls32[0])):
        if len(args) not in (10, 11):
            fail(f"prefill.cpp: {name} in the lockstep branch has {len(args)} arguments, not 10 or 11")

    nvfp4_mmq = {"use_mmq": True, "fused_l": False, "native": True, "tail_off": True}
    fp16_layer = {"use_mmq": False, "fused_l": False, "native": True, "tail_off": False}
    exl3_layer = {"use_mmq": False, "fused_l": False, "native": False, "tail_off": False}

    p = planted(K)
    T, N, D, sd = p["T"], p["N"], p["D"], p["sd"]
    env = dict(nvfp4_mmq, row_sd=sd)
    env_fp16 = dict(fp16_layer, row_sd=sd)
    env_exl3 = dict(exl3_layer, row_sd=sd)
    results = []

    def case(name, ok, detail=""):
        results.append(bool(ok))
        print(("PASS " if ok else "FAIL ") + name + ("" if ok else f": {detail}"))

    # 1, 2: the planted NVFP4 MMQ layer, fp16 and fp32 sums: the s_down must reach the combine
    want16 = combine_peer16(scaled(D, sd, N), p["slot"], p["weight"], p["shared"], p["sg"], p["peer16"], T, T, N, None)
    got16 = combine_peer16(D, p["slot"], p["weight"], p["shared"], p["sg"], p["peer16"], T, T, N,
                           row_sd_argument(calls16[0], env))
    case("an NVFP4 MMQ layer's lockstep fp16 combine applies the row s_down", close(got16, want16),
         f"got {got16[0]:.4f}, want {want16[0]:.4f} (row 0 scaled by {sd[0]})")

    want32 = combine_peer(scaled(D, sd, N), p["slot"], p["weight"], p["shared"], p["sg"], p["peer"], T, T, N, None)
    got32 = combine_peer(D, p["slot"], p["weight"], p["shared"], p["sg"], p["peer"], T, T, N,
                         row_sd_argument(calls32[0], env))
    case("an NVFP4 MMQ layer's lockstep fp32 combine applies the row s_down", close(got32, want32),
         f"got {got32[0]:.4f}, want {want32[0]:.4f} (row 0 scaled by {sd[0]})")

    # 3, 4: ordinary layers (no NVFP4 MMQ): the same combine, with and without weight scaling
    plain16 = combine_peer16(D, p["slot"], p["weight"], p["shared"], p["sg"], p["peer16"], T, T, N, None)
    got_plain16 = combine_peer16(D, p["slot"], p["weight"], p["shared"], p["sg"], p["peer16"], T, T, N,
                                 row_sd_argument(calls16[0], env_fp16))
    case("an ordinary fp16 layer's lockstep combine is unchanged", close(got_plain16, plain16),
         f"got {got_plain16[0]:.4f}, want {plain16[0]:.4f}")

    plain32 = combine_peer(D, p["slot"], p["weight"], p["shared"], p["sg"], p["peer"], T, T, N, None)
    got_plain32 = combine_peer(D, p["slot"], p["weight"], p["shared"], p["sg"], p["peer"], T, T, N,
                               row_sd_argument(calls32[0], env_exl3))
    case("an EXL3 layer's lockstep fp32 combine is unchanged", close(got_plain32, plain32),
         f"got {got_plain32[0]:.4f}, want {plain32[0]:.4f}")

    # 5: the non-lockstep Mac-lane calls (6595-6599) are the contract: same condition, s_down on an NVFP4 MMQ layer
    after = source[end:]
    stop = after.find("else if (peer_now")
    if stop < 0:
        fail("prefill.cpp: the non-lockstep Mac-lane combine block was not found")
    next16, next32 = call_arguments(after[:stop], "moe_combine_peer16"), call_arguments(after[:stop], "moe_combine_peer")
    if not next16 or not next32:
        fail("prefill.cpp: the non-lockstep Mac-lane combines were not found")
    case("the non-lockstep combines select the same s_down for the planted layer",
         row_sd_argument(next16[0], env) is sd and row_sd_argument(next32[0], env) is sd,
         (row_sd_argument(next16[0], env), row_sd_argument(next32[0], env)))

    failed = results.count(False)
    print(f"{len(results) - failed} of {len(results)} cases pass")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
