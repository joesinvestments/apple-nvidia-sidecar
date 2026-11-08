"""A service with no family named stops on end_token_ids(tokenizer)'s ids, not on the pieces of Qwen's markers.

serve/server.py:4916-4917 sets Service.stop_ids to tokenizer.encode(IM_END) + tokenizer.encode("<|endoftext|>")
whatever the tokenizer makes of them, overwriting the end_token_ids(tokenizer) two lines above.  A tokenizer without
Qwen's markers as one token each (DeepSeek-V4.1 splits them into '<', '|', 'im', '_end', '|', '>': #4576) then stops
a reply in front of every "|", ">" or "im" it writes, and its own end token is not a stop id at all.

  python3 tests/test_stop_ids_no_split_pieces.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import IM_END, ByteTokenizer, Service, end_token_ids  # noqa: E402

EOF_TEXT = "<|endoftext|>"
fails = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global fails
    print(f"{'PASS' if ok else 'FAIL'} {name}" + ("" if ok else f": {detail}"))
    if not ok:
        fails += 1


class Engine:
    """A stub engine: Service.__init__ reads the tokenizer's end tokens, no model and no hardware."""

    def alive(self) -> bool:
        return True


class PieceTokenizer:
    """DeepSeek-V4.1's shape: no single token for Qwen's markers, which come out as several ordinary pieces."""

    special_ids = {"eos": 1}
    PIECES = ["_end", "text", "end", "of", "im", "er", "az", "N", "<", "|", ">"]
    IDS = {p: 300 + i for i, p in enumerate(PIECES)}

    def encode(self, text, parse_special=False, plain=()):
        out, i = [], 0
        while i < len(text):
            for piece in self.PIECES:
                if text.startswith(piece, i):
                    out.append(self.IDS[piece])
                    i += len(piece)
                    break
            else:
                out.append(ord(text[i]))
                i += 1
        return out


class HalfTokenizer(PieceTokenizer):
    """A vocabulary that has the end marker as one token and still splits the other one."""

    def encode(self, text, parse_special=False, plain=()):
        return [9] if text == EOF_TEXT else super().encode(text, parse_special, plain)


class BareTokenizer(PieceTokenizer):
    """No single marker token and no end token named: nothing ends a reply of its own."""

    special_ids: dict = {}


def service(tok, template):
    return Service(Engine(), tok, template)


def main() -> int:
    if not (ROOT / "serve" / "server.py").is_file():
        print(f"FAIL the tree root is not {ROOT}: run this from the repository root")
        return 1
    template = ChatTemplate(ROOT / "serve" / "chat_template.jinja")

    # the markers as one token each (a Qwen vocabulary): the ids end a reply, before and after the fix
    qwen = ByteTokenizer()
    want = set(qwen.encode(IM_END, parse_special=True)) | set(qwen.encode(EOF_TEXT, parse_special=True))
    qwen_service = service(qwen, template).stop_ids
    check("a tokenizer whose markers are one token each gets both stop ids", end_token_ids(qwen) == want and qwen_service == want,
          f"end_token_ids {sorted(end_token_ids(qwen))}, Service {sorted(qwen_service)}, want {sorted(want)}")

    # the pieces: end_token_ids leaves them out and names the tokenizer's own end token
    pieces = PieceTokenizer()
    piece_ids = set(pieces.encode(IM_END, parse_special=True)) | set(pieces.encode(EOF_TEXT, parse_special=True))
    check("end_token_ids leaves out the pieces of markers a tokenizer splits", end_token_ids(pieces) == {1},
          f"got {sorted(end_token_ids(pieces))}, want the end token 1")

    # the service takes those ids, and not the pieces the overwrite put there
    got = service(pieces, template).stop_ids
    check("a service with no family named stops on the end token only", got == {1}, f"got {sorted(got)}, want [1]")
    check("a service with no family named does not stop on a piece of the markers", not got & piece_ids,
          f"stop ids {sorted(got)} include piece ids {sorted(got & piece_ids)}")
    nerazim = pieces.encode("Nerazim")
    check("a reply ending in Nerazim is not cut in front of its im", nerazim[-1] not in got,
          f"the im piece id {nerazim[-1]} is a stop id ({sorted(got)})")

    half = service(HalfTokenizer(), template).stop_ids
    check("a marker that is one token stays a stop id while the other one splits", half == {9}, f"got {sorted(half)}, want [9]")
    bare = service(BareTokenizer(), template).stop_ids
    check("a tokenizer with neither marker nor end token gets no stop id", bare == set(), f"got {sorted(bare)}")

    print(f"{fails} case(s) failed" if fails else "all cases passed")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
