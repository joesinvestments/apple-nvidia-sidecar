# sidecar-shim/geom.py - the installed card's geometry, with the selected config as the fallback.
#
# cudaGetDeviceProperties used to answer with the TGCU_DEV_* settings alone: the RTX 5090 literals that
# config/sidecar.json carries, or the 3090 defaults, on every machine.  So the engine's planner sized
# against a card that was not installed, and the config's reason for the name called that literal "the
# installed GPU as seen by the selected driver".  The shim now asks the device first through tgcu (its
# compute capability, its SM count, the physical pool tinygrad sized for it) and keeps the TGCU_DEV_*
# setting for what the backend cannot report (L2, bus width, clock rate, a name): the setting is a
# fallback, not a detection.  The clock rate used to be the shim's own 1.695 GHz literal (fix-0098#0), so
# the layer split's SMs x clock model priced a layer against a card that was not installed either.
# This module imports nothing heavy, so the test can pass a small fake device.
from __future__ import annotations

import os

# index order shared with tgcudart.cpp (dev_field) and tgcu_api.dev_geom_int
SM, SMS, MEM_GIB, L2_MIB, BUSW, CLOCK_MHZ = 0, 1, 2, 3, 4, 5
KEYS = ("sm", "sms", "mem_gib", "l2_mib", "busw", "clock_mhz")

# the shim's long-standing answers when the config names nothing (Sero's 3090)
DEFAULTS = (86, 82, 24, 6, 384, 1695)
DEFAULT_NAME = "NVIDIA GeForce RTX 3090 (tgcu USB)"
ENV = ("TGCU_DEV_SM", "TGCU_DEV_SMS", "TGCU_DEV_MEM_GIB", "TGCU_DEV_L2_MIB", "TGCU_DEV_BUSW", "TGCU_DEV_CLOCK_MHZ")


def _positive(value) -> int:
    try:
        n = int(str(value).strip())
    except (TypeError, ValueError):
        return 0
    return n if n > 0 else 0


def _arch_sm(dev) -> int:
    """The compute capability as card_sm() reads it: the digits of dev.arch ("8.9" -> 89, "12.0" -> 120)."""
    arch = getattr(dev, "arch", None)
    if arch is None:
        return 0
    return _positive("".join(ch for ch in str(arch) if ch.isdigit()))


def _sms(ctx) -> int:
    """The card's SM count from tgcu's TPC mask read (Context.num_sms)."""
    f = getattr(ctx, "num_sms", None)
    if not callable(f):
        return 0
    try:
        return _positive(f())
    except Exception:
        return 0


def _mem_gib(ctx) -> int:
    """The physical pool tinygrad sized for this card (ctx.iface.dev_impl.mm.pa_allocator.size), rounded up:
    the pool leaves the driver's reserved pages out, so 23.9 GiB is a 24 GiB card and 31.9 GiB a 32 GiB one."""
    try:
        size = _positive(ctx.iface.dev_impl.mm.pa_allocator.size)
    except (AttributeError, TypeError):
        return 0
    return (size + (1 << 30) - 1) >> 30 if size else 0


def dev_geom(env=None, dev=None, ctx=None) -> dict:
    """sm, sms, mem_gib, l2_mib, busw, clock_mhz and name: the device's own values, else the TGCU_DEV_* fallback."""
    env = os.environ if env is None else env
    out = {}
    for i, (key, name) in enumerate(zip(KEYS, ENV)):
        out[key] = _positive(env.get(name)) or DEFAULTS[i]
    if dev is not None:
        out["sm"] = _arch_sm(dev) or out["sm"]
    if ctx is not None:
        out["sms"] = _sms(ctx) or out["sms"]
        out["mem_gib"] = _mem_gib(ctx) or out["mem_gib"]
    out["name"] = str(env.get("TGCU_DEV_NAME") or "").strip() or DEFAULT_NAME
    return out


def dev_geom_int(idx: int, env=None, dev=None, ctx=None) -> int:
    """One geometry field by index, for tgcu_api.dev_geom_int and the C++ shim."""
    return dev_geom(env=env, dev=dev, ctx=ctx)[KEYS[idx]]


def dev_geom_name(env=None, dev=None, ctx=None) -> str:
    return dev_geom(env=env, dev=dev, ctx=ctx)["name"]
