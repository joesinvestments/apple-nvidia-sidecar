#!/bin/sh
set -eu
# Public FP4-Local checkpoint to a local pack; no model files are redistributed.
rev=b9ce18a6887e18c089fdea8509a33e63429a3300
repo=nvidia/Qwen3.8-Flash-Next-FP4-Local
if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
  echo 'usage: prepare-weights.sh SOURCE_ROOT OUTPUT_DIR [REFERENCE_PACK]' >&2; exit 2
fi
# The original pack used these fixed GGUF names; checkpoint folder names are not stable.
if [ "${1:-}" = --normalize-names ]; then
  [ "$#" -eq 2 ] || { echo 'usage: prepare-weights.sh --normalize-names PREPARED_DIR' >&2; exit 2; }
  python3 - "$2" <<'PYNAMES'
import pathlib, struct, sys
root = pathlib.Path(sys.argv[1])
for file, key, wanted in (
    ('dense.gguf', b'general.name', b'Nvidia Qwen38Fn Fp4 Local'),
    ('ple-nvfp4.gguf', b'strata.ple.source', b'nvidia-qwen38fn-fp4-local'),
):
    path = root / file
    with path.open('r+b') as out:
        header = out.read(4096)
        if header[:4] != b'GGUF' or struct.unpack_from('<I', header, 4)[0] != 3:
            sys.exit(f'unsupported GGUF header: {path}')
        marker = struct.pack('<Q', len(key)) + key + struct.pack('<I', 8)
        if header.count(marker) != 1:
            sys.exit(f'missing or duplicate string metadata {key!r}: {path}')
        offset = header.index(marker) + len(marker)
        length = struct.unpack_from('<Q', header, offset)[0]
        if length != len(wanted):
            sys.exit(f'{path}: {key.decode()} length {length}, expected {len(wanted)}; refusing layout change')
        offset += 8
        old = header[offset:offset + length]
        if len(old) != length:
            sys.exit(f'truncated metadata in {path}')
        out.seek(offset)
        out.write(wanted)
    print(f'{file}: {key.decode()} {old.decode("utf-8", "replace")!r} -> {wanted.decode()!r}', flush=True)
PYNAMES
  exit 0
fi
src=$1; out=$2; ref=${3:-}
if [ ! -f "$src/tools/nvfp4_local_pack.py" ] || [ ! -f "$src/requirements-nvfp4-local.txt" ]; then
  echo 'missing converter or requirements in engine source' >&2; exit 2
fi
if [ -e "$out/checkpoint" ] || [ -e "$out/prepared" ]; then
  echo 'refusing existing output; use a fresh directory' >&2; exit 2
fi
python3 - "$out" <<'PY'
import pathlib,shutil,sys
p=pathlib.Path(sys.argv[1]).expanduser().resolve()
p.parent.mkdir(parents=True,exist_ok=True)
free=shutil.disk_usage(p.parent).free
need=102252102010+107*10**9+50*1024**3
if free<need: sys.exit(f'insufficient space: {free} bytes free, need {need} for source, pack and reserve')
print(f'free={free} required={need}')
PY
python3 -m venv "$out/venv"
py="$out/venv/bin/python"
"$py" -m pip install huggingface_hub
mkdir -p "$out/checkpoint" "$out/prepared"
"$py" - "$repo" "$rev" "$out/checkpoint" <<'PY'
import sys
from huggingface_hub import snapshot_download
snapshot_download(repo_id=sys.argv[1],revision=sys.argv[2],local_dir=sys.argv[3])
PY
"$py" - "$repo" "$rev" "$out/checkpoint" <<'PY'
import hashlib,pathlib,sys
from huggingface_hub import HfApi
repo,rev,root=sys.argv[1],sys.argv[2],pathlib.Path(sys.argv[3])
info=HfApi().model_info(repo,revision=rev,files_metadata=True)
if info.sha!=rev: sys.exit('wrong source revision')
checked=0
for f in info.siblings:
    if not f.lfs: continue
    p=root/f.rfilename
    if not p.is_file() or p.stat().st_size!=f.lfs.size: sys.exit(f'missing or wrong-size: {f.rfilename}')
    h=hashlib.sha256()
    with p.open('rb') as inp:
        for block in iter(lambda:inp.read(8*1024*1024),b''): h.update(block)
    if h.hexdigest()!=f.lfs.sha256: sys.exit(f'hash mismatch: {f.rfilename}')
    checked+=1
if checked<21: sys.exit(f'incomplete source: {checked} hashed files')
print(f'source hashes verified: {checked}')
PY
"$py" -m pip install -r "$src/requirements-nvfp4-local.txt"
"$py" "$src/tools/nvfp4_local_pack.py" prepare --model "$out/checkpoint" --out "$out/prepared"
# Source directory names become GGUF metadata; regenerate both with the selected names.
ln -s checkpoint "$out/Nvidia Qwen38Fn Fp4 Local"
ln -s checkpoint "$out/nvidia-qwen38fn-fp4-local"
"$py" "$src/tools/nvfp4_local_pack.py" ple --model "$out/nvidia-qwen38fn-fp4-local" --out "$out/prepared/ple-nvfp4-named.gguf"
"$py" "$src/tools/nvfp4_local_pack.py" convert --dense-only --model "$out/Nvidia Qwen38Fn Fp4 Local" --outfile "$out/prepared/dense-named.gguf"
mv "$out/prepared/ple-nvfp4-named.gguf" "$out/prepared/ple-nvfp4.gguf"
mv "$out/prepared/dense-named.gguf" "$out/prepared/dense.gguf"
# Normalize only equal-length metadata fields; never rewrite tensors or change the selected pack.
"$0" --normalize-names "$out/prepared"
# Compare generated outputs and tensor index against an independently existing pack before asserting identity.

if [ -n "$ref" ]; then
  "$py" - "$out/prepared" "$ref" <<'PYCOMPARE'
import hashlib,pathlib,sys
new,old=map(pathlib.Path,sys.argv[1:])
files=('dense.gguf','ple-nvfp4.gguf','pack/dense.bin','pack/experts.bin',
       'pack/index.txt','pack/native_experts.txt','mtp/rt/dense.bin','mtp/rt/experts.bin')
for name in files:
    a,b=new/name,old/name
    if not a.is_file() or not b.is_file(): sys.exit(f'missing output: {name}')
    def digest(p):
        h=hashlib.sha256()
        with p.open('rb') as f:
            for chunk in iter(lambda:f.read(16*1024*1024),b''): h.update(chunk)
        return h.hexdigest()
    ah,bh=digest(a),digest(b)
    print(f'{name}: {ah} reference={bh} same={ah==bh}',flush=True)
    if ah!=bh: sys.exit(f'output differs: {name}')
entries=lambda root: {line.split()[0] for line in (root/'pack/index.txt').read_text().splitlines()
                      if line and not line.startswith('#')}
if entries(new)!=entries(old): sys.exit('tensor names differ')
print(f'pack tensor names match: {len(entries(new))}')
PYCOMPARE
fi
