"""Measured hardware ceilings, shared by every adapter.

Every local generator on this machine runs under one number that is not RAM and
is not VRAM: the Metal working-set recommendation reported by
``mx.device_info()['max_recommended_working_set_size']``. On an M1 Pro with
32 GB of unified memory that is **21.34 GiB**, not 32 GiB. Two thirds of the
number on the box, and the difference is what let a 72-frame H3 job "succeed"
while writing noise.

This module exists because that ceiling is the one fact two very different
models share. H3 spends it as DiT tokens; Flint spends it as weight bytes.
Neither can derive the other's number, but both are bounded by the same wall,
and both fail the same way when they cross it: without an error. H3 exits rc=0
and writes a mosaic. MLX raises late, or starts swapping and never recovers the
throughput.

So the rule the whole pipeline is built on, stated once here:

    The budget is measured, the check happens before the run, and an
    over-budget request is *refused* rather than silently degraded.

Adapters keep their own cost functions — the H3 token arithmetic is empirical
and lives in ``models/h3``; Flint's byte arithmetic is derived from its config
and lives in ``models/flint``. This module only answers "how much is there".
"""

from __future__ import annotations

import shutil
from functools import lru_cache
from pathlib import Path

#: Working-set ceiling measured on this machine, in bytes. 21.34 GiB.
#:
#: Used when MLX cannot be imported. Keeping the literal is deliberate: the
#: pipeline must stay runnable for H3 without pulling in MLX, and a silently
#: absent budget would turn every capacity check into a no-op — the exact bug
#: class this module exists to prevent.
MEASURED_WORKING_SET_BYTES = 22_906_503_168

#: Where the literal came from, so it can be re-measured rather than trusted.
MEASURED_ON = "M1 Pro, 32 GB unified memory, macOS 15.3, mlx 0.32.4"


@lru_cache(maxsize=1)
def _probe() -> tuple[int, bool]:
    """``(ceiling_bytes, came_from_device)``."""
    try:
        import mlx.core as mx  # noqa: PLC0415  (optional, probe-time only)
        value = int(mx.device_info()["max_recommended_working_set_size"])
    except Exception:
        return MEASURED_WORKING_SET_BYTES, False
    if value <= 0:
        return MEASURED_WORKING_SET_BYTES, False
    return value, True


def gpu_working_set_bytes() -> int:
    """The Metal working-set ceiling, probed when MLX is importable."""
    return _probe()[0]


def is_probed() -> bool:
    """True when the ceiling came from the device rather than the literal."""
    return _probe()[1]


def gpu_working_set_gib() -> float:
    return gpu_working_set_bytes() / 2**30


def free_disk_bytes(path: str | Path) -> int:
    return shutil.disk_usage(str(path)).free


def describe() -> str:
    source = "probed from this device" if is_probed() else f"recorded on {MEASURED_ON}"
    return f"{gpu_working_set_gib():.2f} GiB working set ({source})"
