# The dock driver: what we run and how we built it

The engine reaches the card through three local pieces. `config/sidecar.json` turns this path on with `APL_DIRECT=1`, and `tools/sidecar-run.sh` refuses to start without the first two paths below.

| Piece | Setting | Where it comes from |
| --- | --- | --- |
| `libtgd.dylib`, a small library that opens the driver inside the engine process | `SIDECAR_TGD_LIB` | Source in this folder |
| A patched copy of tinygrad, which boots the card | `SIDECAR_TINYGRAD` | Public commit plus `tinygrad-changes.patch` |
| A patched TinyGPU DriverKit driver | installed in `/Applications` | Public commit plus `tinygpu-changes.patch` |

## Why no ready-made driver is included

Source is here; a compiled driver is not. Our driver is ad-hoc signed with no developer team, so it only loads on a Mac with System Integrity Protection (SIP) turned off. Handing strangers a binary that asks them to lower their Mac's security would be a bad trade when they can build the same thing from about 200 lines of reviewed changes. Distributing a driver that loads with SIP on needs a Developer ID, Apple's DriverKit entitlements and notarization, which we do not have.

The public signed TinyGPU should not be expected to work with this engine as it stands. Reading the source, `libtgd` uses three things only our build has: a call that frees one DMA buffer (selector 4), write-combined card memory maps (map types 8 to 13), and a signing entitlement that lets a process other than the TinyGPU app open the driver. We installed the signed build as far as its approval prompt and went no further, so that is a reading of the code, not a test. Making the signed path work is the most useful open problem here.

## Before you start

Turning SIP off lowers the security of the whole Mac. Do this on a machine dedicated to the job, and read `tinygpu-changes.patch` first. This route has only ever been run on our two Macs, so expect to find something we did not.

Versions we used: macOS 27.0.1 (26A434), Xcode 27.0 (27A266a) with the DriverKit 27.0 SDK, Python 3.12.

## 1. Allow a locally signed driver

Restart into Recovery (hold the power button), open Terminal and run:

```bash
csrutil disable
```

Restart, then:

```bash
systemextensionsctl developer on
csrutil status
```

`csrutil status` should report disabled.

## 2. Build and install the driver

Our source is tinygrad's installer at commit `d33d97ca7eb117e911d4706d51ffe123e3472ff4`.

```bash
git clone https://github.com/tinygrad/tinygrad.git
git -C tinygrad checkout d33d97ca7eb117e911d4706d51ffe123e3472ff4
git -C tinygrad apply --directory=extra/usbgpu/tbgpu/installer "$PWD/apple-nvidia-sidecar/driver/tinygpu-changes.patch"
tinygrad/extra/usbgpu/tbgpu/installer/install_nosip.sh
```

The script builds without signing, ad-hoc signs with the `NoSIP` entitlements, copies the app to `/Applications` and asks macOS to activate the driver. If a signed TinyGPU is already installed, run `/Applications/TinyGPU.app/Contents/MacOS/TinyGPU uninstall` first and reboot if macOS says it is waiting to uninstall. The two share a bundle ID, so never leave both active.

Approve it in System Settings > General > Login Items & Extensions > Driver Extensions. Then check:

```bash
systemextensionsctl list | grep tinygpu
```

You want `[activated enabled]` with team `-`.

## 3. Build libtgd and probe the card

```bash
make -C apple-nvidia-sidecar/driver
apple-nvidia-sidecar/driver/test_tgd
apple-nvidia-sidecar/driver/tgdprobe
```

`test_tgd` needs no driver. `tgdprobe` should exit 0 and print `pci 10de:2b85` for a 5090. Exit 2 means the driver service was not found. Exit 3 means the driver refused to open, which usually means the signed build is the one loaded.

## 4. Build the patched tinygrad

The engine does not use tinygrad's current code: upstream removed this Mac client path. Use commit `e765a2abf75c` plus our patch. It includes 0xSero's fixes for running over a dock, and it also applies to `tinygrad/` alone if you keep a smaller copy.

```bash
git -C tinygrad checkout -f e765a2abf75c
git -C tinygrad apply "$PWD/apple-nvidia-sidecar/driver/tinygrad-changes.patch"
python3.12 -m venv sidecar-venv
sidecar-venv/bin/pip install -r apple-nvidia-sidecar/requirements.txt
```

Point the engine at all of it:

```bash
export SIDECAR_TGD_LIB="$PWD/apple-nvidia-sidecar/driver/libtgd.dylib"
export SIDECAR_TINYGRAD="$PWD/tinygrad"
export SIDECAR_PY_SITE="$PWD/sidecar-venv/lib/python3.12/site-packages"
```

Check that tinygrad can compute on the card through this path:

```bash
APL_DIRECT=1 DEV=NV TGD_LIB="$SIDECAR_TGD_LIB" PYTHONPATH="$SIDECAR_TINYGRAD:$SIDECAR_PY_SITE" sidecar-venv/bin/python -c 'from tinygrad import Tensor; print((Tensor([1.,2.,3.])+1).tolist())'
```

Expect `[2.0, 3.0, 4.0]`. Then go back to `BUILD-SIDECAR.md` for the engine.

## If the card stops answering

Power-cycle the dock, wait 10 seconds, and check step 3 again. On this card family, shutting the card's firmware down cleanly breaks the next start, so the patch deliberately leaves it running when the engine exits. Do not set `NV_GSP_RESIDENT=0` on a 5090.

## Undo

```bash
/Applications/TinyGPU.app/Contents/MacOS/TinyGPU uninstall
systemextensionsctl list
```

Reboot if macOS says it is waiting to uninstall, remove `/Applications/TinyGPU.app`, run `systemextensionsctl developer off`, then run `csrutil enable` from Recovery.

## What each patch changes

**`tinygpu-changes.patch`**: logs each card memory region at start; adds write-combined maps of card memory; keeps each DMA buffer's descriptor alive with its command and adds a call to free one buffer; completes DMA on a failed map; refuses undersized or misaligned DMA map requests; removes per-call logging from the hot path.

**`tinygrad-changes.patch`**: adds `apl_direct.py`, the in-process card device; handles a large compiled kernel arriving in several reads; handles the call relocations 5090 kernels use; keeps engine allocations out of the top of card memory, where the firmware lives; re-reads the card's BIOS until it reads stable; leaves the firmware running at exit on this card family; lets the card register existing Mac memory for DMA.

Credits and licenses for this folder are in `NOTICE.md`.
