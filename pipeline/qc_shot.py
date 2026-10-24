#!/usr/bin/env python3
"""Quality-control check for H3 clips.

Extracts frames and prints statistics that separate real content from the two
failure modes actually observed on this machine, then applies the calibrated
verdict.

This is now a thin CLI over ``models/h3``: the statistics live in
``core/qc.py`` and the thresholds in ``models/h3/__init__.py``. Keeping a
second copy of either would let the two drift apart, and a QC threshold that
disagrees with the one used during a batch run is worse than no QC at all.

Usage:
    python qc_shot.py outputs/h3_probe/p512_37f.mp4 [more.mp4 ...]
    python qc_shot.py --strip out.png a.mp4 b.mp4 c.mp4

Note: needs numpy and Pillow, so it runs under ``../ai-drama/.venv`` rather
than the bare system Python.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from models.h3 import load_adapter  # noqa: E402


def probe(path: Path) -> str:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,nb_frames,r_frame_rate",
         "-of", "csv=p=0", str(path)],
        capture_output=True, text=True).stdout.strip()
    return out


def duration_of(path: Path) -> float:
    try:
        return float(subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(path)],
            capture_output=True, text=True).stdout.strip())
    except ValueError:
        return 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("clips", nargs="+")
    ap.add_argument("--at", type=float, default=None,
                    help="timestamp to sample (default: middle of clip)")
    ap.add_argument("--strip", default=None, help="write side-by-side strip")
    ap.add_argument("--height", type=int, default=896)
    args = ap.parse_args()

    adapter = load_adapter()

    for name in args.clips:
        path = Path(name)
        if not path.exists():
            print(f"{path}: MISSING")
            continue
        meta = probe(path)
        at = args.at if args.at is not None else max(duration_of(path) * 0.5,
                                                     0.05)
        verdict = adapter.judge(path)
        s = verdict.stats
        print(f"{path.name}  [{meta}]  sampled at {at:.2f}s")
        print(f"   mean={s['mean']:.3f} std={s['std']:.3f} "
              f"adjdiff={s['adjdiff']:.4f} block16={s['block16']:.4f} "
              f"block_ratio={s['block16_ratio']:.2f}")
        print(f"   verdict: {verdict.label}"
              + (f" — {verdict.detail}" if verdict.detail else ""))

    if args.strip:
        write_strip(args.clips, args.strip, args.height)
    return 0


def write_strip(clips, out_path: str, height: int) -> None:
    """Side-by-side strip of one frame from each clip."""
    import numpy as np
    from PIL import Image

    from core.qc import grab_frame

    tiles = []
    for name in clips:
        path = Path(name)
        if not path.exists():
            continue
        a = grab_frame(path, max(duration_of(path) * 0.5, 0.05))
        im = Image.fromarray((a * 255).astype(np.uint8))
        w = max(1, int(im.width * height / im.height))
        tiles.append(np.asarray(im.resize((w, height))))
    if tiles:
        Image.fromarray(np.hstack(tiles)).save(out_path)
        print(f"strip -> {out_path}")


if __name__ == "__main__":
    sys.exit(main())
