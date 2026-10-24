"""Command stream assembly and the renderer session.

Two ideas, both engine-neutral:

**Command stream.** A batch driver is not a loop of one-shot invocations. A
long-lived renderer keeps its weights and conditioning in memory, so the
driver emits one command stream and the renderer executes it in order. The
adapter owns the syntax; this module owns the *shape* of the stream — a
header that establishes global settings, then one body per shot with its
overrides applied and reverted.

The revert is the subtle part. A per-shot override that is not reverted leaks
into every later shot, which produces clips that silently differ in shape.
That defeats caching and makes a cost model meaningless.

**Session.** The renderer is a subprocess fed on stdin. It must run from its
own directory when it resolves sibling resources relative to cwd (h3 does:
h3_gpu.m reads the shader source from the working directory). It names
outputs sequentially in request order, so the driver maps them back onto shot
names afterwards — and must do so even when the process is killed, or a crash
loses finished work.
"""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol


class CommandStream(Protocol):
    """What an adapter must be able to produce."""

    def header(self, preset, out_dir: Path) -> list[str]: ...
    def shot_lines(self, shot: object, index: int, preset) -> list[str]: ...
    def footer(self) -> list[str]: ...


def build_stream(adapter: CommandStream, shots: list, preset,
                 out_dir: Path) -> list[str]:
    """Assemble the full command stream for a run."""
    lines: list[str] = list(adapter.header(preset, out_dir))
    for index, shot in enumerate(shots):
        lines.extend(adapter.shot_lines(shot, index, preset))
    lines.extend(adapter.footer())
    return lines


@dataclass
class SessionResult:
    returncode: int
    elapsed: float
    stdout_path: Path


class Session:
    """A renderer subprocess driven by a command stream on stdin."""

    def __init__(self, binary: Path, args: list[str], cwd: Path,
                 log_path: Path):
        self.binary = binary
        self.args = args
        self.cwd = cwd
        self.log_path = log_path

    def run(self, stdin_text: str) -> SessionResult:
        started = time.time()
        with self.log_path.open("w") as log:
            proc = subprocess.run(
                [str(self.binary), *self.args],
                input=stdin_text, text=True, cwd=str(self.cwd),
                stdout=log, stderr=subprocess.STDOUT,
            )
        return SessionResult(proc.returncode, time.time() - started,
                             self.log_path)


def default_numbered_outputs(out_dir: Path,
                             exclude: set[str]) -> list[Path]:
    """Outputs the renderer created during this run, in request order.

    ``exclude`` holds names that already existed before the run, so a
    ``--resume`` invocation does not re-adopt clips it skipped.
    """
    return sorted(p for p in out_dir.glob("video-*.mp4")
                  if p.name not in exclude)


def map_back(out_dir: Path, numbered: list[Path], shots: list,
             on_rename: Callable[[Path, Path], None] | None = None) -> int:
    """Rename sequential outputs onto shot names.

    Returns the number of files renamed. Called both after a clean run and
    from the resume path, because a hard kill never reaches the normal
    rename step and finished clips would otherwise be orphaned.
    """
    renamed = 0
    for index, path in enumerate(numbered):
        if index >= len(shots):
            break
        shot = shots[index]
        target = out_dir / shot.out
        if path == target:
            continue
        if target.exists():
            target.unlink()
        path.rename(target)
        renamed += 1
        if on_rename:
            on_rename(path, target)
    return renamed
