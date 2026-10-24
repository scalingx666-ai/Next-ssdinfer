"""The seam between engine-agnostic machinery and model-specific knowledge.

An ``EngineAdapter`` answers exactly the questions the core layer cannot:

* how a preset becomes a command stream for this renderer;
* what a shot's *capacity* is, and when it exceeds what this hardware can do;
* how to tell a good clip from a bad one;
* where the renderer must run from, and what it needs on disk.

The capacity question is the one worth designing carefully. Every local
generator has a hard ceiling, and it is almost never the one you assume. On
M1 Pro it is not VRAM — the oversized job exits rc=0 and writes garbage. So
``CapacityModel`` is a first-class object, not a comment: it converts a
requested shape into a token count, compares it against a measured budget,
and reports *why* a request is rejected rather than silently degrading it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


@dataclass
class Preset:
    """A named configuration the renderer understands."""

    name: str
    steps: int
    layers: int
    reuse: int
    frames: int
    render_w: int
    render_h: int
    width: int
    height: int
    token_reduction: bool = False
    ssd_streaming: bool = False
    note: str = ""


@dataclass
class Capacity:
    """The outcome of checking a request against the hardware budget."""

    tokens: int
    budget: int
    ok: bool
    reason: str = ""

    @property
    def utilisation(self) -> float:
        return self.tokens / max(self.budget, 1)


@dataclass
class Verdict:
    """A quality judgement for one clip."""

    label: str                       # OK / MOSAIC / MUSH / DARK / ...
    detail: str = ""
    stats: dict[str, float] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.label == "OK"


class CapacityModel(Protocol):
    """How much work this renderer can be asked for on this hardware."""

    budget: int

    def tokens_for(self, render_w: int, render_h: int, frames: int) -> int: ...

    def check(self, preset: Preset, shot: Any = None) -> Capacity: ...


class EngineAdapter(Protocol):
    """Everything the core layer needs to know about one model."""

    name: str
    presets: dict[str, Preset]
    capacity: CapacityModel
    binary: Path
    model_dir: Path
    cwd: Path

    def extra_args(self) -> list[str]:
        """CLI flags this adapter always passes to the renderer."""
        ...

    def header(self, preset: Preset, out_dir: Path) -> list[str]:
        """Command-stream lines that establish global settings.

        Takes the preset and output directory as arguments rather than reading
        them off instance state: the header is emitted before any shot is
        processed, so there is no natural earlier point at which a driver could
        have injected them, and stashing them as side effects of ``shot_lines``
        would make the adapter's behaviour depend on call order.
        """
        ...

    def shot_lines(self, shot: Any, index: int, preset: Preset) -> list[str]:
        """Command-stream lines for one shot, overrides reverted."""
        ...

    def footer(self) -> list[str]:
        """Command-stream lines that end the session."""
        ...

    def is_complete(self, path: Path, shot: Any) -> bool:
        """True only if the output is a usable clip, not a truncated mux."""
        ...

    def judge(self, path: Path, shot: Any = None) -> Verdict:
        """Quality verdict against this model's calibrated thresholds."""
        ...
