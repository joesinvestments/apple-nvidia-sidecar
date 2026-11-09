# Side Car engine (preview)

A Mac and NVIDIA RTX 5090 split inference engine built from [Strata v0.2.11](https://github.com/Niko1221/Strata/tree/v0.2.11). It runs a locally supplied Qwen3.8-Flash-Next FP4 pack through a Mac host, a card-side CUDA shim and a Metal lane. This source-only preview distributes no model weights or driver binaries.

## Hardware and prerequisites

Tested on two identical setups: a Mac Studio (M5 Max, 64 GB), an NVIDIA GeForce RTX 5090 and a MINISFORUM DEG2 Thunderbolt 5 dock. An arm64 macOS host, the dock driver and patched tinygrad built by [`driver/README.md`](driver/README.md), Python 3.12, Xcode/Metal tools, clang, ninja, Colima/Docker with a CUDA build image, CUDA 12.8 host headers and SIMDe headers are required. The official signed graphics-driver installation path has not been tested. Other machines and configurations have not been proved.

## Model weights

No weights are in this repository. The tested weights come from the public, ungated Hugging Face repository `nvidia/Qwen3.8-Flash-Next-FP4-Local` at revision `b9ce18a6887e18c089fdea8509a33e63429a3300`, under the NVIDIA Open Model agreement named on its model card; read that license before downloading. `tools/prepare-weights.sh SOURCE_ROOT OUTPUT_DIR` downloads that exact revision, checks every file's hash and writes the layout below. It needs about 245 GiB free. Place the result in `weights/`: `pack/`, `dense.gguf`, `ple-nvfp4.gguf`, `mtp/rt/` and `pack/tokenizer/`. Running the script from a fresh download on our machines reproduced all eight output files byte for byte.

## The driver is the hard part

The tested card path uses a locally modified TinyGPU DriverKit driver, ad-hoc signed without a developer team, on a Mac with System Integrity Protection turned off. No compiled driver is included, for the reasons in [`driver/README.md`](driver/README.md). That guide has the exact steps we used, with our driver changes, the `libtgd` card library and our tinygrad changes as source. The public signed TinyGPU has not been made to work with this engine, and reading the code says it will need changes. Making it work is the first thing we would like help with.

## Memory settings

The engine holds about 56 GB of a 64 GB machine in locked memory. Our boot sets these first, and they need root: `sudo sysctl vm.global_user_wire_limit=63350767616 vm.user_wire_limit=63350767616 iogpu.wired_limit_mb=60416`. Run nothing large beside it, and stop any idle Docker or Colima VM before starting: going past physical memory has panicked our machine.

## Build and run

Clone this repository and install prerequisites on the Mac and card. Set `SIDECAR_CUDA_HOST` to your CUDA header tree and `SIDECAR_SIMDE` to the SIMDe root. Put the prepared weights in `weights/` as described above.

Run `tools/build-sidecar.sh` with `SIDECAR_DOCKER`, `SIDECAR_COLIMA`, `SIDECAR_PYTHON` and `SIDECAR_SDK` pointing to local prerequisites when defaults differ. See `config/sidecar.json` for the example engine configuration; set the installed driver paths (`SIDECAR_TGD_LIB`, `SIDECAR_TINYGRAD`, `SIDECAR_PY_SITE`) to your own locations, resolve all model paths against your checkout, then run the local server with `python3 serve/server.py --engine strata --config config/sidecar.json --port 8090`. Read `tools/sidecar-run.sh` and the configuration before exposing the endpoint to a network. The example config is not an installer or a substitute for the proprietary device driver.

## Limits and evidence

The selected deployment serves one conversation at a time. Installation was proved only on our two machines; this preview makes no claim about speed on other hardware. The signed graphics-driver path remains untested, and no compiled driver is included (see above). This repository contains source, not model weights. No performance numbers are claimed here; test and build receipts stay with the machines on which they were measured.

## License and credits

The upstream source and our changes are under the root `LICENSE`; original copyright and notices are retained. Third-party component licenses and origins are listed in `THIRD_PARTY.md`, including the Mac CUDA bridge's tinygrad attribution, ggml-tq, libstrix, and the Metal lane. [Strata](https://github.com/Niko1221/Strata), [tinygrad](https://github.com/tinygrad/tinygrad), [ggml](https://github.com/ggml-org/ggml) and the other authors named in their notices retain credit for their work. No model weights are redistributed; check the license for your own exact model revision before downloading or running it.
