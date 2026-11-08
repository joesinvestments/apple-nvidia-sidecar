#!/usr/bin/env python3
"""The shim reports the installed card's geometry, with config/sidecar.json as its fallback.

sidecar-shim/tgcudart.cpp:346-374 answered cudaGetDeviceProperties and cudaDeviceGetAttribute from the
TGCU_DEV_* settings alone: SM 120, 170 SMs, 32 GiB, 96 MiB L2, 512-bit bus and the RTX 5090 name that
config/sidecar.json:100-105 carries.  The engine's planner sizes against those literals on every machine,
and the config's reason for the name (line 146) calls the literal "the installed GPU as seen by the
selected driver".  This test runs the shim's geometry resolver (sidecar-shim/geom.py, which tgcu_api hands
to the C++ side) against a small fake device, no GPU and no model: a card that is not the config's must be
reported as itself, a field the backend cannot report must keep the config's value as a documented
fallback, and a machine with no device answer must give today's values.

    python3 tests/test_fix_0004_0.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def find_root():
    """The tree under test: the copy prove_test runs us in, else the one this file sits in."""
    rel = Path("sidecar-shim") / "tgcu_api.py"
    for base in (Path.cwd(), Path(__file__).resolve().parent.parent):
        if (base / rel).is_file():
            return base
    return Path.cwd()


ROOT = find_root()
GEOM = ROOT / "sidecar-shim" / "geom.py"
CONFIG = ROOT / "config" / "sidecar.json"

fails = 0


def check(name, ok, detail=""):
    global fails
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else ": " + detail))
    if not ok:
        fails += 1


def join(values):
    return ", ".join("%s=%s" % (k, values[k]) for k in sorted(values))


class FakeDevice:
    """The one fact tinygrad's device carries that the shim needs: its compute capability."""

    def __init__(self, arch):
        self.arch = arch


class FakePool:
    def __init__(self, size):
        self.size = size


class FakeMM:
    def __init__(self, size):
        self.pa_allocator = FakePool(size)


class FakeImpl:
    def __init__(self, size):
        self.mm = FakeMM(size)


class FakeIface:
    def __init__(self, size):
        self.dev_impl = FakeImpl(size)


class FakeContext:
    """tgcu's Context answers the SM count (its TPC mask read) and the physical pool it sized for the card."""

    def __init__(self, sms, mem_bytes):
        self._sms = sms
        self.iface = FakeIface(mem_bytes)

    def num_sms(self):
        return self._sms


def load_geom():
    if not GEOM.is_file():
        return None
    spec = importlib.util.spec_from_file_location("sidecar_geom", str(GEOM))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def config_env():
    """The selected config's engine environment: the real production sample this fix falls back to."""
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    return {str(k): str(v) for k, v in (cfg.get("env") or {}).items()}


def main():
    env = config_env()
    geom = load_geom()
    if geom is None:
        for name in (
            "the installed card's compute capability is read from the device",
            "the installed card's SM count is read from the device",
            "the installed card's memory is read from the device",
            "L2, bus width and name stay the config's documented fallback",
            "the C++ side's indexed access returns the device's value",
            "with no device answer the config's values still stand",
            "with no config and no device answer today's 3090 values still stand",
        ):
            check(name, False,
                  "sidecar-shim/geom.py is missing, so device properties can only answer with the config's literals")
        return

    card = FakeContext(80, 16 << 30)          # a 4080 Super: arch 8.9, 80 SMs, 16 GiB
    dev = FakeDevice("8.9")
    got = geom.dev_geom(env=env, dev=dev, ctx=card)
    check("the installed card's compute capability is read from the device", got["sm"] == 89,
          "the config says %s, the card is 89; the shim reports %s" % (env.get("TGCU_DEV_SM"), got["sm"]))
    check("the installed card's SM count is read from the device", got["sms"] == 80,
          "the config says %s, the card has 80; the shim reports %s" % (env.get("TGCU_DEV_SMS"), got["sms"]))
    check("the installed card's memory is read from the device", got["mem_gib"] == 16,
          "the config says %s GiB, the card has 16 GiB; the shim reports %s" % (env.get("TGCU_DEV_MEM_GIB"), got["mem_gib"]))
    check("L2, bus width and name stay the config's documented fallback",
          (got["l2_mib"], got["busw"], got["name"]) == (96, 512, env.get("TGCU_DEV_NAME")),
          "want 96, 512, %r; got %s" % (env.get("TGCU_DEV_NAME"), join(got)))

    if hasattr(geom, "dev_geom_int") and hasattr(geom, "dev_geom_name"):
        sm = geom.dev_geom_int(geom.SM, env=env, dev=dev, ctx=card)
        name = geom.dev_geom_name(env=env, dev=dev, ctx=card)
        check("the C++ side's indexed access returns the device's value", sm == 89 and name == env.get("TGCU_DEV_NAME"),
              "want 89 and %r, got %s and %r" % (env.get("TGCU_DEV_NAME"), sm, name))
    else:
        check("the C++ side's indexed access returns the device's value", False,
              "sidecar-shim/geom.py has no dev_geom_int / dev_geom_name for tgcu_api to expose")

    plain = geom.dev_geom(env=env)
    check("with no device answer the config's values still stand",
          (plain["sm"], plain["sms"], plain["mem_gib"], plain["l2_mib"], plain["busw"], plain["name"]) ==
          (120, 170, 32, 96, 512, env.get("TGCU_DEV_NAME")),
          "want the config's 120, 170, 32, 96, 512, %r; got %s" % (env.get("TGCU_DEV_NAME"), join(plain)))

    empty = geom.dev_geom(env={})
    check("with no config and no device answer today's 3090 values still stand",
          (empty["sm"], empty["sms"], empty["mem_gib"], empty["l2_mib"], empty["busw"]) == (86, 82, 24, 6, 384),
          "want today's defaults 86, 82, 24, 6, 384; got %s" % join(empty))


if __name__ == "__main__":
    main()
    sys.exit(1 if fails else 0)
