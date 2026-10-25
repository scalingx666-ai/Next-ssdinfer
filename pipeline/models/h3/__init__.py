"""H3 (MiniMax FL2VA) adapter for antirez/h3.c on Apple Silicon.

This module holds every H3-specific fact the framework needs: the preset
ladder, the token arithmetic, the capacity budget, the command syntax, and the
QC thresholds. Each of those was measured on real hardware rather than read
off a datasheet — see docs/PERFORMANCE.md for the experiments behind them.

The single most important number here is ``TOKEN_BUDGET = 4480``. It is not a
memory limit: an over-budget job still exits rc=0 and writes a complete,
playable mp4 full of 16px mosaic noise, because the diffusion never produced a
valid latent. Anything that checks "did it finish" instead of "did it work"
will ship that file.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.adapter import Capacity, MediaPreset, Preset, Verdict
from core.qc import frame_stats, grab_frame, probe_video

# ---------------------------------------------------------------------------
# Measured constants
# ---------------------------------------------------------------------------

#: DiT tokens this hardware can denoise before the latent stops carrying
#: signal. Measured, not derived: 4480 works, 7560 and 8064 do not.
#: The naive "max" preset sits at 8064 and produces mosaic noise.
TOKEN_BUDGET = 4480

#: The largest token count *observed to fail*, for reporting headroom.
TOKEN_KNOWN_FAILURE = 7560

#: Spatial VAE stride: one latent pixel becomes this many output pixels. This
#: is also why an undenoised latent shows up as a block of exactly this size in
#: a decoded frame, which is what the QC path keys on.
VAE_SPATIAL_RATIO = 16

#: Video latent temporal stride (frames per latent frame).
LATENT_TIME = 4

#: DiT patch size over the latent grid. Tokens per latent frame are
#: (render/VAE_SPATIAL_RATIO)/DIT_PATCH on each axis.
DIT_PATCH = 2

#: h3 requires the internal canvas to share the output aspect ratio and both
#: sides to be multiples of 32. These are the legal pairs for the 9:16 ladder.
CANVAS_LADDER = {
    "256x448":  (384, 672),
    "384x672":  (512, 896),
    "512x896":  (768, 1344),
}


@dataclass
class H3Preset(MediaPreset):
    """A media preset. H3 adds no fields of its own.

    It exists as a named type so the preset ladder reads as H3's rather than
    as a generic canvas, and so a future H3-only knob has somewhere to live
    without touching the shared media shape.
    """

    pass


PRESETS: dict[str, Preset] = {
    # `smoke` proves the pipeline end to end at the cheapest real shape.
    "smoke": H3Preset(
        "smoke", steps=4, layers=50, reuse=1, frames=22,
        render_w=256, render_h=448, width=384, height=672,
        token_reduction=False, ssd_streaming=True,
        note="256x448 internal, 4 steps, 22 frames (~23 min)"),
    # The delivered-quality configuration. 512x896 internal is the largest
    # canvas that stays inside the token budget at 37 frames, and it writes
    # 768x1344 natively — no upscaling, which is where the old tier lost
    # detail.
    "max": H3Preset(
        "max", steps=20, layers=50, reuse=1, frames=37,
        render_w=512, render_h=896, width=768, height=1344,
        token_reduction=False, ssd_streaming=True,
        note="PRODUCTION: 512x896 internal, 20 steps, 37 frames, 768x1344 out"),
    # A/B vehicle: identical to `max` but 30 denoise steps. Same canvas, same
    # frames, same seed — the only variable is step count.
    "max30": H3Preset(
        "max30", steps=30, layers=50, reuse=1, frames=37,
        render_w=512, render_h=896, width=768, height=1344,
        token_reduction=False, ssd_streaming=True,
        note="A/B: as max but 30 denoise steps (~67 min/shot)"),
    # Cheaper tier for when duration matters more than native resolution.
    "mid": H3Preset(
        "mid", steps=20, layers=50, reuse=1, frames=61,
        render_w=384, render_h=672, width=512, height=896,
        token_reduction=False, ssd_streaming=True,
        note="384x672 internal, 20 steps, 61 frames, 512x896 out"),
    # Historical tiers kept for comparison; `target` is what produced the
    # low-quality delivery that started this investigation.
    "target": H3Preset(
        "target", steps=4, layers=45, reuse=1, frames=120,
        render_w=256, render_h=448, width=384, height=672,
        token_reduction=True, ssd_streaming=True,
        note="OLD low-quality tier: 4 steps, token reduction on, upscaled"),
    "steps20_same": H3Preset(
        "steps20_same", steps=20, layers=50, reuse=1, frames=120,
        render_w=256, render_h=448, width=384, height=672,
        token_reduction=False, ssd_streaming=True,
        note="same canvas as target but 20 steps and no token reduction"),
}


class H3Capacity:
    """Token accounting for H3 on this hardware.

    Tokens = latent_frames x spatial_tokens, where latent_frames is
    ``(frames - 1) // 4 + 1`` and spatial_tokens is the render canvas divided
    by the DiT patch size of 2. Canvas and duration are interchangeable: the
    product is what matters, so a bigger canvas buys resolution by spending
    the same budget that frames would have used.
    """

    budget = TOKEN_BUDGET

    @staticmethod
    def latent_frames(frames: int) -> int:
        return (frames - 1) // LATENT_TIME + 1

    @classmethod
    def spatial_tokens(cls, render_w: int, render_h: int) -> int:
        """Tokens per latent frame.

        Two downsampling steps compose: the VAE divides the canvas by 16, then
        the DiT patches the latent by 2. Skipping the patch step is an easy
        mistake that inflates the count by 4x and makes every request look
        over budget.
        """
        latent_w = render_w // VAE_SPATIAL_RATIO
        latent_h = render_h // VAE_SPATIAL_RATIO
        return (latent_w // DIT_PATCH) * (latent_h // DIT_PATCH)

    @classmethod
    def tokens_for(cls, render_w: int, render_h: int, frames: int) -> int:
        return cls.latent_frames(frames) * cls.spatial_tokens(
            render_w, render_h)

    def check(self, preset: Preset, shot: Any = None) -> Capacity:
        frames = getattr(shot, "frames", None) or preset.frames
        tokens = self.tokens_for(preset.render_w, preset.render_h, frames)
        if tokens > TOKEN_BUDGET:
            return Capacity(
                tokens, TOKEN_BUDGET, False,
                f"{tokens} tokens exceeds the {TOKEN_BUDGET} budget measured "
                f"on this hardware. The renderer would still exit rc=0 and "
                f"write a 16px-mosaic clip, so this is refused rather than "
                f"degraded. Reduce frames to "
                f"{self.max_frames(preset.render_w, preset.render_h)} or drop "
                f"to a smaller canvas.")
        if tokens > TOKEN_BUDGET * 0.9:
            return Capacity(tokens, TOKEN_BUDGET, True,
                            "within 10% of the measured budget")
        return Capacity(tokens, TOKEN_BUDGET, True)

    @classmethod
    def max_frames(cls, render_w: int, render_h: int) -> int:
        """Largest frame count that stays inside the budget on a canvas."""
        per_frame = cls.spatial_tokens(render_w, render_h)
        return (TOKEN_BUDGET // per_frame - 1) * LATENT_TIME + 1


# ---------------------------------------------------------------------------
# QC thresholds, calibrated against known-good and known-bad clips
# ---------------------------------------------------------------------------

#: block16 above which a frame looks like an undenoised latent, *and* the std
#: below which it has no real content. Both are required: block_ratio was
#: tried first and is not a discriminator — a good 37-frame clip scores 7.66
#: while a bad 72-frame clip scores only 7.26, because a *noisy* mosaic raises
#: adjacent-pixel difference at the same time as block16.
QC_BLOCK16_MOSAIC = 0.105
QC_STD_MOSAIC = 0.14

#: std below which a clip is under-denoised mush.
QC_STD_MUSH = 0.09

#: mean below which the subject is probably missing.
QC_MEAN_DARK = 0.12

#: Thresholds are calibrated on clips whose short side is at least this long.
#: Every calibration sample is 768x1344 (the `max` preset's native output), so
#: the floor is 768 — the production short side itself. A clip smaller than
#: this is a downscaled or non-production render, and its statistics are not
#: comparable.
QC_MIN_SHORT_SIDE = 768


class H3Adapter:
    """Drives antirez/h3.c in its interactive (pipe) mode."""

    name = "h3"
    presets = PRESETS
    capacity = H3Capacity()
    # One long-lived process fed a line-oriented stream: weights load once and
    # every clip amortises that cost, which is why this is not one-shot.
    mode = "stream"
    # h3 names its clips video-NNNN.mp4 in request order; the driver maps them
    # back onto shot names after the run, and resume adopts strays.
    output_glob = "video-*.mp4"
    # stdout is the renderer's log; the artifacts are the mp4 files it writes.
    stdout_is_output = False

    def __init__(self, repo_root: Path | None = None):
        repo = repo_root or Path(__file__).resolve().parents[2]
        self.repo = repo
        self.cwd = repo / "engine"
        self.binary = self.cwd / "h3"
        self.model_dir = Path(
            os.environ.get(
                "H3_MODEL",
                repo / "models" / "ms-h3-cache" / "models" /
                "MiniMax--MiniMax-H3" / "snapshots" / "master"))

    # -- command stream ---------------------------------------------------

    def extra_args(self) -> list[str]:
        return ["-d", str(self.model_dir)]

    def header(self, preset: Preset, out_dir: Path) -> list[str]:
        """Global settings, in the order the interactive session expects.

        The order reproduces the pre-refactor driver exactly; it is verified
        byte-for-byte against that driver rather than argued from first
        principles, because nothing here is load-bearing enough to justify
        changing it. What *is* load-bearing is that ``!frames`` is sent at all:
        h3 defaults to 56, so a clip that never receives the line silently
        becomes a different shape, which defeats the cache and the cost model.

        ``!cache`` sits after the shape settings because it retains the DiT and
        conditioning that those settings have just configured.

        ``!output`` comes first so everything the session writes lands in the
        right directory, including anything emitted before the first
        generation.
        """
        return [
            f"!output {out_dir}",
            *self._preset_lines(preset),
            "!cache",
        ]

    def _preset_lines(self, preset: Preset) -> list[str]:
        return [
            f"!size {preset.width}x{preset.height}",
            f"!render-size {preset.render_w}x{preset.render_h}",
            f"!steps {preset.steps}",
            f"!layers {preset.layers}",
            f"!reuse {preset.reuse}",
            f"!token-reduction {'on' if preset.token_reduction else 'off'}",
            "!ssd-streaming on" if preset.ssd_streaming else "!ssd-streaming off",
            # The preset's frame count is the source duration per clip. h3
            # defaults to 56, so omitting this line silently changes every
            # clip's shape.
            f"!frames {preset.frames}",
        ]

    def shot_lines(self, shot: Any, index: int, preset: Preset) -> list[str]:
        """Lines for one shot, with every override reverted afterwards.

        An unreverted override leaks into every later shot, silently changing
        their shape. That defeats the cache and makes the cost model
        meaningless, so each override is paired with its restore.
        """
        lines: list[str] = []
        overrides: list[str] = []

        if shot.frames and shot.frames != preset.frames:
            lines.append(f"!frames {shot.frames}")
            overrides.append(f"!frames {preset.frames}")
        if shot.steps and shot.steps != preset.steps:
            lines.append(f"!steps {shot.steps}")
            overrides.append(f"!steps {preset.steps}")
        if shot.first:
            lines.append(f"!first {shot.first}")
        if shot.last:
            lines.append(f"!last {shot.last}")
        if shot.seed is not None:
            lines.append(f"!seed {shot.seed}")

        # A bare line is a prompt and triggers generation. stdin is split on
        # newlines, so a multi-line prompt would fire several generations;
        # H3 treats newlines as section separators, so collapsing is
        # equivalent.
        lines.append(shot.prompt)

        lines.extend(overrides)
        if shot.first:
            lines.append("!first clear")
        if shot.last:
            lines.append("!last clear")
        return lines

    def footer(self) -> list[str]:
        return ["!quit"]

    # -- output validation ------------------------------------------------

    def is_complete(self, path: Path, shot: Any) -> bool:
        """A usable clip, not a truncated mux.

        h3 writes frames+2 (37 -> 39), so requiring the requested frame count
        is the right floor. A kill during the final mux leaves a file that
        ffprobe cannot read at all, which is the common case.
        """
        try:
            info = probe_video(path)
        except (subprocess.SubprocessError, OSError, ValueError):
            return False
        if info.frames <= 0:
            return False
        want = getattr(shot, "frames", None)
        if want and info.frames < int(want):
            return False
        return True

    # -- quality ----------------------------------------------------------

    def judge(self, path: Path, shot: Any = None) -> Verdict:
        info = probe_video(path)
        at = max(info.duration * 0.5, 0.05)
        stats = frame_stats(grab_frame(path, at))

        if info.short_side < QC_MIN_SHORT_SIDE:
            # Thresholds are calibrated for native-resolution clips; a
            # downscaled probe gets statistics but no verdict.
            return Verdict("UNVERIFIED",
                           f"short side {info.short_side}px < "
                           f"{QC_MIN_SHORT_SIDE}px, thresholds not calibrated",
                           stats)

        block16 = stats.get(f"block{VAE_SPATIAL_RATIO}", 0.0)
        if block16 > QC_BLOCK16_MOSAIC and stats["std"] < QC_STD_MOSAIC:
            return Verdict(
                "MOSAIC",
                f"block16={block16:.4f} > {QC_BLOCK16_MOSAIC} with "
                f"std={stats['std']:.3f} < {QC_STD_MOSAIC}: the latent never "
                f"denoised, so the VAE decoded noise. This still exits rc=0.",
                stats)
        if stats["std"] < QC_STD_MUSH:
            return Verdict(
                "MUSH",
                f"std={stats['std']:.3f} < {QC_STD_MUSH}: under-denoised, "
                f"no subject", stats)
        if stats["mean"] < QC_MEAN_DARK:
            return Verdict(
                "DARK",
                f"mean={stats['mean']:.3f} < {QC_MEAN_DARK}: subject likely "
                f"missing", stats)
        return Verdict("OK", "", stats)


def load_adapter(repo_root: Path | None = None) -> H3Adapter:
    """Entry point the CLI uses."""
    return H3Adapter(repo_root)
