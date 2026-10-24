"""Shot records and script loading.

A shot is the unit of work. It is intentionally a plain dataclass over a JSON
record so a script file stays diffable and hand-editable: the production
shots are maintained as data, not code.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator


@dataclass
class Shot:
    """One generation request.

    ``frames`` and ``steps`` may be set per shot to override the preset. That
    is not a convenience feature — it is how a single-variable A/B is run: the
    preset stays fixed and exactly one field changes.
    """

    id: str
    prompt: str
    out: str
    frames: int | None = None
    steps: int | None = None
    seed: int | None = None
    first: str | None = None
    last: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_record(cls, rec: dict[str, Any], index: int) -> "Shot":
        if "prompt" not in rec:
            raise ValueError(f"shot #{index + 1} has no prompt")
        ident = rec.get("id") or f"shot_{index + 1:02d}"
        return cls(
            id=ident,
            prompt=" ".join(str(rec["prompt"]).split()),
            out=rec.get("out") or f"{ident}.mp4",
            frames=rec.get("frames"),
            steps=rec.get("steps"),
            seed=rec.get("seed"),
            first=rec.get("first"),
            last=rec.get("last"),
            meta={k: v for k, v in rec.items()
                  if k not in {"id", "prompt", "out", "frames", "steps",
                               "seed", "first", "last"}},
        )

    def to_record(self) -> dict[str, Any]:
        rec: dict[str, Any] = {"id": self.id, "prompt": self.prompt,
                               "out": self.out}
        for key in ("frames", "steps", "seed", "first", "last"):
            value = getattr(self, key)
            if value is not None:
                rec[key] = value
        rec.update(self.meta)
        return rec


def load_shots(path: str | Path) -> list[Shot]:
    """Read a JSONL shot script. One JSON object per line."""
    shots: list[Shot] = []
    for index, line in enumerate(Path(path).read_text().splitlines()):
        if not line.strip():
            continue
        shots.append(Shot.from_record(json.loads(line), index))
    if not shots:
        raise ValueError(f"{path}: no shots found")
    return shots


def filter_shots(shots: list[Shot], only: list[str] | None) -> list[Shot]:
    """Keep only the requested ids. Applied before resume bookkeeping so a
    single-shot probe never adopts or skips work it did not do itself."""
    if not only:
        return shots
    wanted = set(only)
    kept = [s for s in shots if s.id in wanted]
    if not kept:
        raise ValueError(f"no shots matched --only {sorted(wanted)}")
    return kept


def iter_pending(shots: Iterator[Shot]) -> Iterator[Shot]:  # pragma: no cover
    yield from shots
