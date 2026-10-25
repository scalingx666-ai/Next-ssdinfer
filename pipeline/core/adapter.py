"""The seam between engine-agnostic machinery and model-specific knowledge.

An ``EngineAdapter`` answers exactly the questions the core layer cannot:

* how a preset becomes a command stream for this renderer, or — when the
  renderer has no interactive mode — how one request becomes an argv;
* what a request *costs*, and when it exceeds what this hardware can do;
* how to tell a good artifact from a bad one, if that is even a question this
  model raises;
* where the renderer must run from, and what it needs on disk.

The capacity question is the one worth designing carefully. Every local
generator has a hard ceiling, and it is almost never the one you assume. On
M1 Pro it is not VRAM — the oversized job exits rc=0 and writes garbage. So
``CapacityModel`` is a first-class object, not a comment: it converts a
request into a cost, compares it against a measured budget, and reports *why*
a request is rejected rather than silently degrading it.

Two splits matter, and both came from trying to write a second adapter:

**Preset is two layers.** ``Preset`` holds what every generator has — the
number of refinement steps and a human label. ``MediaPreset`` adds the
diffusion canvas: frame count, latent size, output size. A text model has no
canvas, and forcing one into its preset would have made the "generic" interface
a lie. An LLM preset subclasses ``Preset`` and adds its own sampling knobs, the
same way ``MediaPreset`` adds its own.

**``judge`` is optional.** H3 needs a calibrated clip QC because its failure
mode is a plausible-looking file — over budget it still exits 0 and writes a
playable mp4 full of 16px mosaic. A text model's failure mode is visible in the
text, so ``judge is None`` there. Making the method required would have forced
a fake implementation; making it optional is the honest statement that quality
inspection is a property of *some* engines.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol


@dataclass
class Preset:
    """What every generator has: a name, a step count, a note.

    Subclasses add the knobs their modality needs. Nothing in the core layer
    reads past ``name`` — the driver only needs it for lookup and reporting,
    and the adapter knows how to render the rest.
    """

    name: str
    steps: int = 0
    note: str = ""

    def describe(self) -> str:
        """One line for the driver's startup banner."""
        return f"{self.name}: {self.steps} steps" + (f" — {self.note}" if self.note else "")


@dataclass
class MediaPreset(Preset):
    """A diffusion canvas: what to render, at what size, for how long."""

    layers: int = 50
    reuse: int = 1
    frames: int = 37
    render_w: int = 512
    render_h: int = 896
    width: int = 768
    height: int = 1344
    token_reduction: bool = False
    ssd_streaming: bool = True

    def describe(self) -> str:
        return (f"{self.render_w}x{self.render_h} -> {self.width}x{self.height}, "
                f"{self.steps} steps, {self.layers} layers, "
                f"token-reduction {'on' if self.token_reduction else 'off'}, "
                f"{self.frames} frames")


@dataclass
class Capacity:
    """The outcome of checking a request against the hardware budget.

    ``cost`` and ``budget`` are in ``unit``, which is whatever the adapter
    spends: DiT tokens for a video diffusion model, bytes for a weight-bound
    LLM. Comparing those numbers across models is meaningless — the point of
    the type is that a single model can be asked "will this fit" before it
    runs, and can answer with a reason.
    """

    cost: int
    budget: int
    ok: bool
    reason: str = ""
    unit: str = "tokens"

    @property
    def utilisation(self) -> float:
        return self.cost / max(self.budget, 1)

    def describe(self) -> str:
        if self.unit == "bytes":
            return (f"{self.cost / 2**30:.2f} GiB / {self.budget / 2**30:.2f} GiB "
                    f"budget ({self.utilisation:.0%} of the measured working set)")
        return (f"{self.cost}/{self.budget} {self.unit} "
                f"({self.utilisation:.0%} of the measured budget)")


@dataclass
class Verdict:
    """A quality judgement for one artifact."""

    label: str                       # OK / MOSAIC / MUSH / DARK / ...
    detail: str = ""
    stats: dict[str, float] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.label == "OK"


Judge = Callable[[Path, Any], Verdict]


class CapacityModel(Protocol):
    """How much work this renderer can be asked for on this hardware.

    ``check`` is the whole seam. Deliberately *not* ``tokens_for(w, h, frames)``:
    that signature is the video case generalised by accident, and it cannot
    describe a model whose cost is bytes.
    """

    budget: int

    def check(self, preset: Preset, shot: Any = None) -> Capacity: ...


class EngineAdapter(Protocol):
    """Everything the core layer needs to know about one model."""

    name: str
    presets: dict[str, Preset]
    capacity: CapacityModel
    binary: Path
    model_dir: Path
    cwd: Path

    #: ``"stream"`` feeds one long-lived process a line-oriented command
    #: stream (weights stay resident). ``"oneshot"`` runs the binary once per
    #: request because that is the only interface the model offers.
    mode: str

    #: Glob matching outputs whose names are assigned by the renderer rather
    #: than by the driver, in request order. ``""`` when the adapter writes
    #: each output itself and no post-run renaming is needed.
    output_glob: str

    #: True when a request's artifact is its stdout rather than a file the
    #: renderer names itself. Only meaningful in ``oneshot`` mode.
    stdout_is_output: bool

    #: Clip quality inspection, or ``None`` when the output needs none.
    judge: Judge | None

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

    def shot_argv(self, shot: Any, index: int, preset: Preset,
                  out_dir: Path) -> list[str]:
        """Full argv for one request. Required only in ``oneshot`` mode."""
        ...

    def footer(self) -> list[str]:
        """Command-stream lines that end the session."""
        ...

    def is_complete(self, path: Path, shot: Any) -> bool:
        """True only if the output is usable, not a truncated write."""
        ...
