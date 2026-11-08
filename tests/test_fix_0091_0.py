#!/usr/bin/env python3
"""tests/test_fix_0091_0.py - the libc module the shim loads must load on every card the engine builds for.

sidecar-shim/tgcu_libc.ptx:4 says ".target sm_86". That module is what a CUDA driver normally links into cubins: device
printf (vprintf) and assert (__assertfail), which tgcu_api.py:42 puts into ctx.externs. A PTX module names the oldest
card it was written for and the driver compiles it for the card it is handed, so a module that names a newer
architecture than the card is refused: on an RTX 20 or a T4 (sm_75), an A100 (sm_80) or any card below 8.6 neither
function exists and the shim cannot start. The engine builds for compute capability 7.5 and newer (CMakeLists.txt
refuses pre-sm_75, and CUDA 13, the toolkit tools/build-sidecar.sh assembles with, dropped sm_60 and sm_70), so the
module has to name sm_75 or lower.

There is no GPU and no CUDA toolkit on the machine that runs this, so the text of the artifact is handed to a stub
driver that follows the two rules the real one does (a target above the card is refused, and the module must define the
functions the shim resolves) and to a stub ptxas that refuses a target CUDA 13 cannot assemble. The same stubs are shown
to refuse a planted sm_120 module, a planted module with no functions and a planted sm_60 module, so a passing card case
blames the file and not the stub.

Prints one PASS or FAIL line per case and exits nonzero on any FAIL. Run: python3 tests/test_fix_0091_0.py
"""
import os
import re
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.getcwd() if os.path.isfile(os.path.join(os.getcwd(), 'sidecar-shim', 'tgcu_libc.ptx')) else os.path.dirname(HERE)
PTX = os.path.join(ROOT, 'sidecar-shim', 'tgcu_libc.ptx')

PTXAS_FLOOR = 75          # ptxas 13.4 (tools/build-sidecar.sh:42) assembles sm_75 and newer; CMakeLists.txt refuses less
SHIM_EXTERNS = ('vprintf', '__assertfail')   # tgcu_api.py:42, the symbols the shim resolves out of this module
CARDS = (
    ('Tesla T4 / RTX 20 (sm_75)', 75),
    ('A100 (sm_80)', 80),
    ('RTX 30 (sm_86)', 86),
    ('RTX 40 (sm_89)', 89),
    ('RTX 50 (sm_120)', 120),
)

VERSION = re.compile(r'^\s*\.version\s+(\d+)\.(\d+)\s*$', re.M)
TARGET = re.compile(r'^\s*\.target\s+sm_(\d+)([a-z]?)\s*$', re.M)
FUNC = re.compile(r'^\s*\.visible\s+\.func\b.*?(\w+)\s*\(', re.M)


def parse_ptx(text):
    """What a driver reads before it compiles a module: the ISA version, the oldest card it names, its functions."""
    version, target = VERSION.search(text), TARGET.search(text)
    if not version or not target:
        return None
    return dict(version=(int(version.group(1)), int(version.group(2))), target=int(target.group(1)),
                target_name='sm_' + target.group(1) + target.group(2), funcs=FUNC.findall(text))


def assemble(module):
    """Stub ptxas 13.4: the build assembles this module with the CUDA 13 toolkit, which dropped Pascal and Volta."""
    if module['target'] < PTXAS_FLOOR:
        return False, module['target_name'] + ' is below the sm_%d floor of the CUDA 13 toolkit the shim builds with' % PTXAS_FLOOR
    return True, module['target_name']


def load(module, card):
    """Stub driver: assemble the module for this card and hand its functions to the shim. Returns (ok, detail)."""
    ok, why = assemble(module)
    if not ok:
        return False, why
    if module['target'] > card:
        return False, ('a %s module is refused on an sm_%d card, so the shim gets no %s'
                       % (module['target_name'], card, ' and no '.join(SHIM_EXTERNS)))
    missing = [f for f in SHIM_EXTERNS if f not in module['funcs']]
    if missing:
        return False, 'the module defines no ' + ' and no '.join(missing)
    return True, 'loads with ' + ' and '.join(SHIM_EXTERNS)


def case(name, ok, detail=''):
    print(('PASS ' if ok else 'FAIL ') + name + (': ' + detail if detail else ''))
    return 0 if ok else 1


def planted(tmp, name, text):
    path = os.path.join(tmp, name)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)
    return parse_ptx(text)


def main():
    if not os.path.isfile(PTX):
        print('FAIL sidecar-shim/tgcu_libc.ptx is not in ' + ROOT + ' (run this test from the copy root)')
        return 1
    with open(PTX, encoding='utf-8') as f:
        text = f.read()
    module = parse_ptx(text)
    if module is None:
        print('FAIL sidecar-shim/tgcu_libc.ptx has no .version and .target the driver can read')
        return 1
    print('# sidecar-shim/tgcu_libc.ptx: .version %d.%d, .target %s, functions %s'
          % (module['version'][0], module['version'][1], module['target_name'], ', '.join(module['funcs']) or 'none'))
    fails = 0
    for name, card in CARDS:
        ok, why = load(module, card)
        fails += case(name, ok, why)

    tmp = tempfile.mkdtemp(prefix='fix-0091-')
    try:
        ok, why = load(planted(tmp, 'planted-sm120.ptx', TARGET.sub('.target sm_120', text, count=1)), 86)
        fails += case('a planted sm_120 module is refused on an sm_86 card', not ok, why)
        bare = '.version %d.%d\n.target %s\n' % (module['version'] + (module['target_name'],))
        ok, why = load(planted(tmp, 'planted-bare.ptx', bare), 86)
        fails += case('a planted module with no functions is refused on an sm_86 card', not ok, why)
        ok, why = load(planted(tmp, 'planted-sm60.ptx', TARGET.sub('.target sm_60', text, count=1)), 60)
        fails += case('a planted sm_60 module is refused by the CUDA 13 assembler', not ok, why)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print('%d case(s) failed' % fails if fails else 'all cases passed')
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())
