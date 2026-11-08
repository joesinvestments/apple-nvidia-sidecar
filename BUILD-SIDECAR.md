# Side Car v0.2.11 source build

The base is Strata v0.2.11. The Mac/card split is on top; the build replaces no running service and no model files.

Prerequisites: Apple-silicon Mac with Xcode Metal tools and the macOS 26.5 SDK, Homebrew Python 3.12, Ninja, Docker/Colima, an installed CUDA 12.8 header tree, SIMDe headers, and Python 3.12 headers. The build downloads SHA-256-checked CUDA 13.4 device tools into an isolated Linux arm64 image and fetches upstream's pinned llama.cpp dependency. Docker must be running. The CUDA headers and SIMDe are development prerequisites, not model or rig artifacts.

From a fresh checkout, one command builds the CUDA fatbins, Mac host engine, CUDA-to-dock shim, standalone prefill planner and Metal exchange lane:

    SIDECAR_CUDA_HOST=/installed/cuda-12.8 SIDECAR_SIMDE=/installed/simde SIDECAR_PY_PREFIX=/installed/python3.12 ./tools/build-sidecar.sh

The command fails on a missing prerequisite, compiler error or link error. It leaves its logs and intermediate objects in `.sidecar-work/` and its runtime artifacts in `build/sidecar/`; neither is tracked. Build on the Mac that has the card and the Metal toolchain.

`config/sidecar.json` holds the selected model, cache and exchange policy; its `reasons` section explains the settings. The only policy adjustment from the selected engine is 19340 Metal slots: the newer engine had fewer card cache slots and 19300 left 35 RAM-tier entries without Metal copies. Default-off experimental environment switches and a Metal parameter the lane refused are omitted; the three zero-valued command options remain explicit because upstream otherwise enables those paths. The model weights are not bundled; link a prepared compatible model at `weights` (with `weights/pack/tokenizer` and the referenced dense, PLE and MTP files). The ranked expert profile is tracked in `data/`.

The dock driver and its tinygrad-backed runtime are installed prerequisites; see the README. Set `SIDECAR_TGD_LIB` to that driver's dylib, `SIDECAR_TINYGRAD` to the installed tinygrad source root, and `SIDECAR_PY_SITE` to the runtime Python environment's site-packages. Start from the checkout root with that Python environment:

    python3.12 serve/server.py --engine strata --config config/sidecar.json --port 8090

The server binds locally by default. A usable installation answers `/health` and completes a full long request.
