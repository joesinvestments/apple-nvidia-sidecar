#!/bin/bash
# Build the v0.2.11 Side Car engine, its CUDA device code, Mac shim, and Metal lane.
# Prerequisites: arm64 macOS, Colima/Docker, Xcode Metal tools, clang, ninja,
# Python 3.12 headers, CUDA 12.8 headers, and SIMDe headers.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
ROOT="$REPO/.sidecar-work"
DOCKER=${SIDECAR_DOCKER:-docker}
COLIMA=${SIDECAR_COLIMA:-colima}
PYTHON=${SIDECAR_PYTHON:-python3.12}
CUDA_HOST=${SIDECAR_CUDA_HOST:?set SIDECAR_CUDA_HOST to the installed CUDA 12.8 header tree}
SIMDE=${SIDECAR_SIMDE:?set SIDECAR_SIMDE to the installed SIMDe header tree}
PY_PREFIX=${SIDECAR_PY_PREFIX:-$("$PYTHON" -c 'import sys; print(sys.prefix)')}
SDK=${SIDECAR_SDK:-/Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk}
for path in "$CUDA_HOST/include/cuda_runtime.h" "$SIMDE/simde/x86/avx2.h" "$PY_PREFIX/include/python3.12/Python.h" "$SDK"; do
    test -e "$path" || { echo "Missing prerequisite: $path" >&2; exit 1; }
done
mkdir -p "$ROOT" "$REPO/build/sidecar"
exec > >(tee "$ROOT/build.log") 2>&1
VM_STARTED=0
CONTAINER=""
cleanup() {
    if [ -n "$CONTAINER" ]; then "$DOCKER" rm -f "$CONTAINER" >/dev/null 2>&1 || true; fi
    if [ "$VM_STARTED" = 1 ]; then "$COLIMA" stop >/dev/null 2>&1 || true; fi
}
trap cleanup EXIT
if command -v "$COLIMA" >/dev/null 2>&1; then
    if "$COLIMA" status >/dev/null 2>&1; then
        printf 'build VM: already running, left running\n'
    else
        "$COLIMA" start
        VM_STARTED=1
        printf 'build VM: started by this build, stopped on exit\n'
    fi
elif "$DOCKER" info >/dev/null 2>&1; then
    printf 'build VM: already running, left running\n'
else
    printf 'build VM: Docker is not running and Colima is not installed\n' >&2
    exit 1
fi
printf 'Building Side Car at %s\n' "$REPO"
"$DOCKER" build --platform linux/arm64 --target engine-configure -f "$REPO/sidecar-build/Dockerfile" -t sidecar-configure-v0211 "$REPO"
CONTAINER=$($DOCKER create --platform linux/arm64 --name "sidecar-device-$$" sidecar-configure-v0211 sleep 3600)
"$DOCKER" start "$CONTAINER" >/dev/null
"$DOCKER" exec "$CONTAINER" bash -euc '
  sed -i "s/-arch=sm_86 -no-compress/-gencode arch=compute_89,code=sm_89 -no-compress/" /work/strata-mac/device.ninja
  sed -i "s/-gencode arch=compute_89,code=sm_89 -no-compress/-gencode arch=compute_89,code=sm_89 -gencode arch=compute_120,code=sm_120 -no-compress/" /work/strata-mac/device.ninja
  grep -q "arch=compute_120,code=sm_120" /work/strata-mac/device.ninja
  t=$(grep -o "^build [^:]*mmq_nvfp4_w4a4.cu.o.fatbin" /work/strata-mac/device.ninja | cut -c7-)
  # The block-scaled MMA unit requires sm_120a; every other kernel is dual
  # sm_89/sm_120. Never compile this unit as plain sm_120 first.
  mapfile -t ordinary < <(grep -o "^build [^:]*\.fatbin: nvcc" /work/strata-mac/device.ninja | cut -d: -f1 | cut -c7- | grep -v "mmq_nvfp4_w4a4.cu.o.fatbin")
  ninja -C /work/strata-mac -f device.ninja -j 6 "${ordinary[@]}"
  if [ -n "$t" ]; then
    mkdir -p "$(dirname "$t")"
    c=$(ninja -C /work/strata-mac -f device.ninja -t commands "$t" | tail -1)
    c=${c/arch=compute_120,code=sm_120/arch=compute_120a,code=sm_120a}
    (cd /work/strata-mac && eval "$c")
  fi
  cd /work/shim
  /work/cuda-13.4/bin/nvcc -std=c++17 -O3 --fatbin -gencode arch=compute_89,code=sm_89 -gencode arch=compute_120,code=sm_120 -no-compress -x cu tgcu_builtin.cu -o tgcu_builtin.fatbin
  /work/cuda-13.4/bin/ptxas -arch=sm_89 -c tgcu_libc.ptx -o tgcu_libc.cubin
  /work/cuda-13.4/bin/ptxas -arch=sm_120 -c tgcu_libc.ptx -o tgcu_libc_sm120.cubin
  /work/cuda-13.4/bin/nvcc -std=c++17 -O3 --fatbin -arch=sm_120 -no-compress -I/work/strata/src -I/work/strata/include -x cu /work/strata/src/prefill/prefill_plan_dev.cu -o prefill_plan_dev.sm120.fatbin
'
"$DOCKER" cp "$CONTAINER:/work/strata-linux" "$ROOT/strata-linux"
"$DOCKER" cp "$CONTAINER:/work/strata-mac" "$ROOT/strata-mac-dev"
"$DOCKER" cp "$CONTAINER:/work/shim/tgcu_builtin.fatbin" "$ROOT/tgcu_builtin.fatbin"
"$DOCKER" cp "$CONTAINER:/work/shim/tgcu_libc.cubin" "$ROOT/tgcu_libc.cubin"
"$DOCKER" cp "$CONTAINER:/work/shim/tgcu_libc_sm120.cubin" "$ROOT/tgcu_libc_sm120.cubin"
"$DOCKER" cp "$CONTAINER:/work/shim/prefill_plan_dev.sm120.fatbin" "$ROOT/prefill_plan_dev.sm120.fatbin"
ln -sfn "$CUDA_HOST" "$ROOT/cuda-12.8"
ln -sfn "$SIMDE" "$ROOT/simde"
mkdir -p "$ROOT/cuda-13.4" "$ROOT/strata-mac"
ln -sfn "$REPO" "$ROOT/strata"
mkdir -p "$ROOT/shim"
rsync -a "$REPO/sidecar-shim/" "$ROOT/shim/"
cp "$ROOT"/tgcu_builtin.fatbin "$ROOT"/tgcu_libc.cubin "$ROOT"/tgcu_libc_sm120.cubin "$ROOT/shim/"
# The configure commands refer to the container's /work; translate only that root.
"$PYTHON" - "$ROOT/strata-linux/strata_cmds.txt" "$ROOT" <<'PY'
import pathlib,sys
p=pathlib.Path(sys.argv[1]);s=p.read_text();p.write_text(s.replace('/work/',sys.argv[2]+'/'))
PY
rsync -a --include='*/' --include='*.fatbin' --exclude='*' "$ROOT/strata-mac-dev/" "$ROOT/strata-mac/"
(cd "$ROOT" && HOME="$ROOT" SDK="$SDK" "$PYTHON" shim/macbuild.py --lin "$ROOT/strata-linux")
PY="$PY_PREFIX" CUDA_HOST="$CUDA_HOST" SDK="$SDK" zsh "$ROOT/shim/build-libcudart.sh" "$ROOT/shim"
ninja -C "$ROOT/strata-mac" -f host.ninja -j 8
bash "$REPO/exl3-metal/build.sh" "$REPO/build/sidecar/metal"
cp "$ROOT/strata-mac/strata" "$REPO/build/sidecar/strata"
cp "$ROOT/shim/libcudart.dylib" "$REPO/build/sidecar/"
cp "$ROOT/prefill_plan_dev.sm120.fatbin" "$REPO/build/sidecar/"
cp "$ROOT/shim/geom.py" "$REPO/build/sidecar/"
cp "$ROOT/shim/tgcu.py" "$ROOT/shim/tgcu_api.py" "$ROOT/shim/tgcu_builtin.fatbin" "$ROOT/shim/tgcu_libc.cubin" "$ROOT/shim/tgcu_libc_sm120.cubin" "$REPO/build/sidecar/"
bash "$REPO/tools/stamp-build.sh" "$REPO" "$REPO/build/sidecar"
DYLD_LIBRARY_PATH="$REPO/build/sidecar" "$REPO/build/sidecar/strata" --version
printf 'Build ready in %s\n' "$REPO/build/sidecar"
