"""Idempotent re-entry after a crash.

The failure this guards against is specific and was actually hit: the driver
renames sequential outputs to shot names only after the renderer exits
cleanly. A power loss therefore leaves finished clips as ``video-0001.mp4``
while the shot names never appear, so a naive ``--resume`` that checks for
shot names re-renders work that already exists — an 8-hour loss per occurrence.

Two rules make resume safe:

1. **Adopt before deciding.** Orphaned sequential outputs are mapped onto shot
   names in request order *before* anything is skipped.
2. **Existing is not done.** A file may be a truncated mux from a kill during
   the final write. It is parked aside and regenerated rather than trusted.

Both rules require the caller to supply a completeness check, because only the
model layer knows what "complete" means for its output.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol


class CompletenessCheck(Protocol):
    def __call__(self, path: Path, shot: object) -> bool: ...


@dataclass
class ResumeState:
    """What resume decided, and why."""

    done: list = field(default_factory=list)
    pending: list = field(default_factory=list)
    adopted: list = field(default_factory=list)
    parked: list = field(default_factory=list)

    @property
    def summary(self) -> str:
        parts = [f"{len(self.done)} done", f"{len(self.pending)} to go"]
        if self.adopted:
            parts.append(f"{len(self.adopted)} adopted")
        if self.parked:
            parts.append(f"{len(self.parked)} parked")
        return ", ".join(parts)


def adopt_stray_outputs(out_dir: Path, shots: list,
                        on_adopt: Callable[[Path, Path], None] | None = None,
                        pattern: str = "video-*.mp4") -> list[Path]:
    """Map orphaned sequential outputs onto shot names, in request order.

    Returns the adopted paths. Only fills a name that does not already exist,
    so re-running resume is itself idempotent.

    ``pattern`` comes from the adapter: only a renderer that assigns its own
    sequential names can leave strays, and only the adapter knows that scheme.
    An empty pattern disables adoption, which is correct for a model that
    writes each output where the script said.
    """
    adopted: list[Path] = []
    if not pattern or not out_dir.is_dir():
        return adopted
    strays = sorted(out_dir.glob(pattern))
    for index, stray in enumerate(strays):
        if index >= len(shots):
            break
        target = out_dir / shots[index].out
        if target.exists():
            continue
        stray.rename(target)
        adopted.append(target)
        if on_adopt:
            on_adopt(stray, target)
    return adopted


def partition_shots(out_dir: Path, shots: list,
                    is_complete: CompletenessCheck,
                    on_park: Callable[[Path, Path], None] | None = None
                    ) -> ResumeState:
    """Split shots into done vs pending, parking anything untrustworthy."""
    state = ResumeState()
    for shot in shots:
        path = out_dir / shot.out
        if path.exists() and not is_complete(path, shot):
            parked = path.with_suffix(".partial")
            path.rename(parked)
            state.parked.append(parked)
            if on_park:
                on_park(path, parked)
        if path.exists():
            state.done.append(shot)
        else:
            state.pending.append(shot)
    return state
