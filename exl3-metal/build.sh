#!/bin/bash
# Build libexl3metal.dylib. Needs a Mac with the Metal toolchain (Xcode or the Metal Developer Tools): the Command Line Tools on the
# mini have no `metal` compiler and cannot link ObjC, so build on another Apple-silicon Mac and copy the dylib to $HOME/exl3/metal/.
# Usage: build.sh [OUT_DIR]   (default: this directory; OUT_DIR gets exl3_metal.air/.metallib, exl3_metallib.h and libexl3metal.dylib)
set -e
cd "$(dirname "$0")"
OUT=${1:-$PWD}; mkdir -p "$OUT"; OUT=$(cd "$OUT" && pwd)
# Xcode 27.1 on a Mac: `xcrun metal` reports a missing Metal Toolchain although the component is installed; naming it works
if ! xcrun -sdk macosx metal --version > /dev/null 2>&1; then
  tc=$(xcodebuild -showComponent MetalToolchain 2> /dev/null | awk '/Toolchain Identifier/ {print $3}')
  [ -n "$tc" ] && export TOOLCHAINS=$tc
fi
# metal4.0: the pf_tensor_ops GEMMs need mpp::tensor_ops::matmul2d and metal::tensor (Metal 3.2 has neither); the kernels below compile unchanged
xcrun -sdk macosx metal -std=metal4.0 -O2 -fno-fast-math -ffp-contract=off -mmacosx-version-min=14.0 -c exl3_metal.metal -o "$OUT/exl3_metal.air"
xcrun -sdk macosx metallib "$OUT/exl3_metal.air" -o "$OUT/exl3_metal.metallib"
(cd "$OUT" && xxd -i exl3_metal.metallib | sed 's/exl3_metal_metallib/exl3_metallib/g; s/^unsigned char/static const unsigned char/; s/^unsigned int/static const unsigned int/' > exl3_metallib.h)
clang++ -O2 -std=c++17 -fobjc-arc -fPIC -arch arm64 -mmacosx-version-min=14.0 -dynamiclib -install_name @rpath/libexl3metal.dylib \
  -I"$OUT" -framework Foundation -framework Metal -o "$OUT/libexl3metal.dylib" exl3_metal.mm
echo built "$OUT/libexl3metal.dylib"
shasum -a 256 "$OUT/libexl3metal.dylib"
