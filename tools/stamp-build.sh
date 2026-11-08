#!/bin/bash
# Stamp a build with the commit it came from: REPO OUTDIR writes OUTDIR/BUILD_ID holding the 12-character commit of
# REPO, plus "-dirty" when git status --porcelain shows changes. serve/server.py reads it once at start and puts it in
# /health and /api/health as "build".
set -euo pipefail
REPO=${1:?usage: tools/stamp-build.sh REPO OUTDIR}
OUTDIR=${2:?usage: tools/stamp-build.sh REPO OUTDIR}
mkdir -p "$OUTDIR"
OUTDIR=$(cd "$OUTDIR" && pwd)
id=$(cd "$REPO" && git rev-parse --short=12 HEAD)
tree=$(cd "$REPO" && git rev-parse HEAD^{tree} | cut -c1-12)
if [ -n "$(cd "$REPO" && git status --porcelain)" ]; then
    id="$id-dirty"
    tree="$tree-dirty"
fi
printf '%s\n' "$id" > "$OUTDIR/BUILD_ID"
printf '%s\n' "$tree" > "$OUTDIR/BUILD_TREE"
