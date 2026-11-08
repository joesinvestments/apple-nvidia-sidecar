"""tests/test_fix_0108_0.py - the x86compat xmmintrin shim compiles on every preprocessor it guards (fix-0108#0).

    python3 tests/test_fix_0108_0.py

sidecar-shim/x86compat/xmmintrin.h forwards to the toolchain's own header on an x86_64 host. It used

    #include_next <xmmintrin.h>

for both GCC/Clang and MSVC, but the guard also accepts MSVC's _M_X64, and MSVC has no #include_next: it stops with
error C1021: invalid preprocessor command 'include_next'. So an MSVC x64 build that reaches the shim fails there.

This machine has no MSVC toolchain, so the cases run the real header through a small preprocessor that follows the two
directive sets the header has to satisfy: GCC/Clang, where #include_next continues after the directory that held the
shim, and MSVC, where #include_next is the C1021 error. Each case plants a system header carrying a marker, processes
the header from this tree, and reads which marker came out. The GCC x64, arm64 and MSVC arm64 cases return the same
marker before and after the fix; only the MSVC x64 case must change from an error to the system header. No server,
model, hardware or network is touched; temp folders only.
"""
from __future__ import annotations

import re
import sys
import tempfile
from pathlib import Path

ROOT = Path.cwd() if (Path.cwd() / "sidecar-shim" / "x86compat" / "xmmintrin.h").is_file() else Path(__file__).resolve().parents[1]
INC = ROOT / "sidecar-shim" / "x86compat"
results: list[bool] = []


class PreprocessorError(Exception):
    pass


def logical_lines(text):
    """Yield source lines with backslash continuations joined."""
    buf = ""
    for raw in text.splitlines():
        if raw.endswith("\\"):
            buf += raw[:-1]
            continue
        yield buf + raw
        buf = ""
    if buf:
        yield buf


class MiniPreprocessor:
    """The directive subset the shim needs, with the MSVC rule that #include_next does not exist."""

    def __init__(self, macros, include_dirs, msvc):
        self.macros = dict(macros)
        self.include_dirs = [Path(p).resolve() for p in include_dirs]
        self.msvc = msvc
        self.once: set[Path] = set()
        self.out: list[str] = []
        self.conds: list[dict] = []
        self.depth = 0

    def run(self, path):
        self._file(Path(path).resolve(), -1)
        return "\n".join(self.out)

    def _active(self):
        return all(c["active"] for c in self.conds)

    def _index_of(self, directory):
        directory = Path(directory).resolve()
        for i, d in enumerate(self.include_dirs):
            if d == directory:
                return i
        return -1

    def _find(self, name, quoted, here, found_at, is_next):
        if is_next:
            start = found_at + 1 if found_at >= 0 else 0
            for i in range(start, len(self.include_dirs)):
                p = self.include_dirs[i] / name
                if p.is_file():
                    return p, i
            return None, -1
        if quoted:
            p = here / name
            if p.is_file():
                return p, self._index_of(here)
        for i, d in enumerate(self.include_dirs):
            p = d / name
            if p.is_file():
                return p, i
        return None, -1

    def _file(self, path, found_at):
        if path in self.once:
            return
        self.depth += 1
        if self.depth > 64:
            raise PreprocessorError("fatal error: #include nested too deeply")
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            raise PreprocessorError(f"fatal error: cannot open {path.name}")
        for line in logical_lines(text):
            line = line.strip()
            if not line.startswith("#"):
                if self._active():
                    self.out.append(line)
                continue
            m = re.match(r"#\s*([A-Za-z_]\w*)\s*(.*)$", line)
            if not m:
                continue
            self._directive(path, found_at, m.group(1), m.group(2).strip())
        self.depth -= 1

    def _directive(self, path, found_at, name, rest):
        if name in ("if", "ifdef", "ifndef"):
            parent = self._active()
            if name == "if":
                value = parent and self._eval(rest)
            else:
                word = rest.split()[0] if rest else ""
                value = parent and ((word in self.macros) if name == "ifdef" else (word not in self.macros))
            self.conds.append({"parent": parent, "active": value, "taken": value})
            return
        if name == "elif":
            if not self.conds:
                raise PreprocessorError("fatal error: #elif without #if")
            c = self.conds[-1]
            if c["taken"]:
                c["active"] = False
            else:
                value = c["parent"] and self._eval(rest)
                c["active"] = value
                c["taken"] = value
            return
        if name == "else":
            if not self.conds:
                raise PreprocessorError("fatal error: #else without #if")
            c = self.conds[-1]
            c["active"] = c["parent"] and not c["taken"]
            c["taken"] = True
            return
        if name == "endif":
            if not self.conds:
                raise PreprocessorError("fatal error: #endif without #if")
            self.conds.pop()
            return
        if not self._active():
            return
        if name == "pragma":
            if rest.split()[:1] == ["once"]:
                self.once.add(path.resolve())
            return
        if name == "define":
            m = re.match(r"([A-Za-z_]\w*)(.*)$", rest)
            if m:
                self.macros[m.group(1)] = m.group(2).strip()
            return
        if name == "undef":
            if rest:
                self.macros.pop(rest.split()[0], None)
            return
        if name == "error":
            raise PreprocessorError(f"fatal error: #error {rest}")
        if name in ("include", "include_next"):
            if name == "include_next" and self.msvc:
                raise PreprocessorError("error C1021: invalid preprocessor command 'include_next'")
            self._include(path, found_at, rest, name == "include_next")
            return
        # #line, #warning and other directives do not decide which header is reached

    def _include(self, path, found_at, rest, is_next):
        m = re.match(r'([<"])([^>"]+)[>"]', rest)
        if not m:
            raise PreprocessorError(f"fatal error: bad #include {rest!r}")
        quoted, name = m.group(1) == '"', m.group(2)
        if Path(name).is_absolute():
            target, index = Path(name), -1
        else:
            target, index = self._find(name, quoted, path.parent, found_at, is_next)
        if target is None or not target.is_file():
            raise PreprocessorError(f"fatal error C1083: cannot open include file '{name}'")
        self._file(target.resolve(), index)

    def _eval(self, expr):
        tokens = re.findall(r"defined|[A-Za-z_]\w*|\d+|&&|\|\||[!()]", expr)
        self._tok, self._pos = tokens, 0

        def peek():
            return self._tok[self._pos] if self._pos < len(self._tok) else ""

        def take():
            t = peek()
            self._pos += 1
            return t

        def primary():
            t = take()
            if t == "defined":
                if peek() == "(":
                    take()
                    word = take()
                    if peek() == ")":
                        take()
                else:
                    word = take()
                return word in self.macros
            if t == "(":
                value = or_expr()
                if peek() == ")":
                    take()
                return value
            if t.isdigit():
                return int(t) != 0
            if not t:
                return False
            value = self.macros.get(t, "0")
            return int(value) != 0 if re.fullmatch(r"-?\d+", value or "0") else bool(value)

        def unary():
            if peek() == "!":
                take()
                return not unary()
            return primary()

        def and_expr():
            value = unary()
            while peek() == "&&":
                take()
                value = unary() and value
            return value

        def or_expr():
            value = and_expr()
            while peek() == "||":
                take()
                value = and_expr() or value
            return value

        return or_expr()


def case(name, ok, detail=""):
    results.append(bool(ok))
    print(("PASS " if ok else "FAIL ") + name + (f": {detail}" if detail else ""))


def write(path, text=""):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def process(source, tmp, tag, macros, include_dirs, msvc):
    src = Path(tmp) / f"{tag}.c"
    src.write_text(source + "\n", encoding="utf-8")
    try:
        return MiniPreprocessor(macros, include_dirs, msvc).run(src), ""
    except PreprocessorError as e:
        return "", str(e)


def expect(name, source, tmp, tag, macros, include_dirs, msvc, present, absent=()):
    out, err = process(source, tmp, tag, macros, include_dirs, msvc)
    missing = [m for m in present if m not in out]
    leaked = [m for m in absent if m in out]
    if err:
        detail = err
    elif missing or leaked:
        detail = "missing " + ",".join(missing) if missing else "reached " + ",".join(leaked)
    else:
        detail = ""
    case(name, not err and not missing and not leaked, detail)


def main():
    shim = INC / "xmmintrin.h"
    if not shim.is_file():
        case("the xmmintrin shim is present", False, str(shim))
        return 1
    with tempfile.TemporaryDirectory(prefix="fix-0108-0-") as tmp:
        tmp = Path(tmp)
        sys_msvc = tmp / "sys-msvc"
        sys_gcc = tmp / "sys-gcc"
        sys_common = tmp / "sys-common"
        write(sys_msvc / "xmmintrin.h", "/* Microsoft x64 xmmintrin.h */\nMSVC_SYSTEM_XMMINTRIN\n")
        write(sys_gcc / "xmmintrin.h", "/* GCC x86_64 xmmintrin.h */\nGCC_SYSTEM_XMMINTRIN\n")
        for h in ("avx512", "fma", "f16c", "clmul"):
            write(sys_common / "simde" / "x86" / f"{h}.h", f"SIMDE_{h.upper()}_REACHED\n")
        write(sys_common / "stdint.h", "/* planted stdint.h */\n")
        forced = f'#include "{shim}"'

        # 1: the harm. An MSVC x64 translation unit that force-includes the shim must reach Microsoft's header;
        # today the shim selects #include_next, which MSVC rejects with C1021.
        expect("an MSVC x64 include reaches the system xmmintrin header", forced, tmp, "msvc-x64",
               {"_MSC_VER": "1938", "_M_X64": "100"}, [sys_msvc], True, ["MSVC_SYSTEM_XMMINTRIN"])

        # 2: GCC/Clang on x86_64 keeps forwarding past the shim to the compiler's header (same before and after).
        expect("a GCC x86_64 include forwards past the shim", "#include <xmmintrin.h>", tmp, "gcc-x64",
               {"__x86_64__": "1"}, [INC, sys_gcc], False, ["GCC_SYSTEM_XMMINTRIN"])

        # 3: on arm64 the shim keeps taking the SIMDe fallback (same before and after).
        expect("an arm64 include takes the SIMDe fallback", "#include <xmmintrin.h>", tmp, "arm64",
               {"__aarch64__": "1"}, [INC, sys_common], False, ["SIMDE_AVX512_REACHED"])

        # 4: MSVC on arm64 has no xmmintrin.h to forward to, so it must stay on the SIMDe fallback. This fails if the
        # MSVC guard is put before the x86_64 guard instead of inside it.
        expect("an MSVC arm64 include stays on the SIMDe fallback", forced, tmp, "msvc-arm64",
               {"_MSC_VER": "1938", "__aarch64__": "1"}, [sys_msvc, sys_common], True,
               ["SIMDE_AVX512_REACHED"], ["MSVC_SYSTEM_XMMINTRIN"])
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
