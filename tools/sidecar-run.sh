#!/bin/zsh
# Side Car's Mac/card engine process. Paths are relative to this checkout.
set -e
ROOT=${0:A:h:h}
: ${SIDECAR_TGD_LIB:?set SIDECAR_TGD_LIB to the installed dock driver library}
: ${SIDECAR_TINYGRAD:?set SIDECAR_TINYGRAD to the installed tinygrad source root}
: ${SIDECAR_PY_SITE:?set SIDECAR_PY_SITE to the Python environment site-packages}
export TGD_LIB=$SIDECAR_TGD_LIB
export DYLD_LIBRARY_PATH=$ROOT/build/sidecar${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}
export PYTHONPATH=$ROOT/build/sidecar:$SIDECAR_TINYGRAD:$SIDECAR_PY_SITE${PYTHONPATH:+:$PYTHONPATH}
export PYTHONNOUSERSITE=1
export HOME=$ROOT/.sidecar-work
export TGCU_MAPPED_MIB=${TGCU_MAPPED_MIB:-16} TGCU_CODE_MIB=${TGCU_CODE_MIB:-32}
cd $ROOT
exec ${D7_BIN:-./build/sidecar/strata} "$@"
