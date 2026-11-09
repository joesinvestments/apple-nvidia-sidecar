# Notices for this folder

`tinygpu-changes.patch` changes tinygrad's TinyGPU installer at commit `d33d97ca7eb117e911d4706d51ffe123e3472ff4`. `tinygrad-changes.patch` changes [tinygrad](https://github.com/tinygrad/tinygrad) at commit `e765a2abf75c`. tinygrad is under the MIT license; its text is in [`../sidecar-shim/LICENSE.tinygrad`](../sidecar-shim/LICENSE.tinygrad).

`tinygrad-changes.patch` includes `0001-tg-strata-nv-usb-fixes.patch` by 0xSero, from [sybil-solutions/qwen38-flash-next-3090-usb-offload](https://github.com/sybil-solutions/qwen38-flash-next-3090-usb-offload) at commit `d134c011bbb98bbe69941ca907d82f998320c6e4`. MIT: [`../sidecar-shim/LICENSE.0xSero`](../sidecar-shim/LICENSE.0xSero).

The way this patch treats the 5090's firmware at exit, and the opt-in link settings for older cards, follow findings published in [deacix/tinygpu](https://github.com/deacix/tinygpu) (MIT).

`libtgd.c`, `libtgd.h`, `tgdprobe.c`, `test_tgd.c`, `Makefile` and `apl_direct.py` (inside the patch) are this project's own, under the root [`LICENSE`](../LICENSE).
