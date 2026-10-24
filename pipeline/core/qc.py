"""Frame statistics and container probing.

Generic measurements only. Whether a given number means "good" is a model
question answered by ``models/<name>/qc.py`` against calibrated thresholds —
this module must not contain thresholds, or the numbers stop being portable
and start being folklore.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Probe:
    """What the container says about a clip."""

    width: int
    height: int
    frames: int
    fps: float
    duration: float

    @property
    def short_side(self) -> int:
        return min(self.width, self.height)


def probe_video(path: str | Path) -> Probe:
    """Read container metadata by field name, never by column position.

    ``nb_frames`` is optional in MP4: when the container does not carry a frame
    count, ffprobe simply omits the column, and every later field shifts left.
    Positional unpacking then reads ``r_frame_rate`` as the frame count and
    ``24/1`` fails to parse as an int. Asking for one field at a time costs a
    few extra exec calls and cannot mis-parse, which matters because
    ``is_complete`` trusts the frame count to decide whether a clip is
    truncated.
    """
    def field(name: str) -> str:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", f"stream={name}", "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, check=True).stdout.strip()
        lines = [ln for ln in out.splitlines() if ln.strip()]
        return lines[0].strip() if lines else ""

    rate = field("r_frame_rate")
    try:
        num, _, den = rate.partition("/")
        fps = float(num) / float(den or 1)
    except (ValueError, ZeroDivisionError):
        fps = 0.0

    try:
        frames = int(field("nb_frames"))
    except ValueError:
        # No frame count in the container. Count packets instead: slower, but
        # it is the only honest answer, and a missing count is exactly when a
        # truncated mux is most likely.
        counted = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-count_packets", "-show_entries", "stream=nb_read_packets",
             "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, check=True).stdout.strip()
        try:
            frames = int(counted.splitlines()[0].strip())
        except (ValueError, IndexError):
            frames = 0

    try:
        duration = float(field("duration"))
    except ValueError:
        duration = 0.0

    return Probe(int(field("width") or 0), int(field("height") or 0),
                 frames, fps, duration)


def grab_frame(path: str | Path, at: float):
    """Decode one frame as a float32 grayscale array in [0, 1].

    numpy/PIL are imported here rather than at module scope so that
    ``--dry-run`` and ``--check-only`` work on a bare Python with no imaging
    stack installed. Those paths never touch a frame.
    """
    import io

    import numpy as np
    from PIL import Image

    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", f"{at}", "-i", str(path),
         "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "-"],
        capture_output=True).stdout
    if not raw:
        raise SystemExit(f"cannot decode a frame from {path}")
    return np.asarray(Image.open(io.BytesIO(raw)).convert("L"),
                      dtype=np.float32) / 255.0


def frame_stats(a, offsets: tuple[int, ...] = (16,)) -> dict:
    """Spatial statistics of one frame.

    ``adjdiff`` measures neighbour-to-neighbour change. ``blockN`` measures
    change at a given offset, which is the scale at which a latent that never
    denoised becomes visible after an Nx VAE decode. Reporting both matters:
    either one alone is misleading.

    numpy is imported here rather than at module scope for the same reason as
    in ``grab_frame``: importing this module must not require an imaging stack.
    """
    import numpy as np

    horizontal = np.abs(np.diff(a, axis=1)).mean()
    vertical = np.abs(np.diff(a, axis=0)).mean()
    adjdiff = float((horizontal + vertical) / 2)
    stats: dict = {
        "mean": float(a.mean()),
        "std": float(a.std()),
        "adjdiff": adjdiff,
    }
    for n in offsets:
        block = float(np.abs(a[n:, n:] - a[:-n, :-n]).mean())
        stats[f"block{n}"] = block
        stats[f"block{n}_ratio"] = block / max(adjdiff, 1e-6)
    return stats
