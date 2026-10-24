#!/usr/bin/env python3
"""Quality-control check for H3 clips.

Extracts frames, prints image statistics that separate real content from the
two failure modes we have actually observed on this machine, and optionally
writes a side-by-side strip.

Failure modes this is meant to catch (see h3c/PERFORMANCE.md section 9):
  * token-count blowup  -> latent stays noise -> 16px mosaic, adjdiff high,
    low-frequency share collapses.
  * under-denoised      -> dark mush, mean/std low, no subject.

Usage:
    python qc_shot.py outputs/h3_probe/p512_22f.mp4 [more.mp4 ...]
    python qc_shot.py --strip out.png a.mp4 b.mp4 c.mp4
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image


def probe(path: Path) -> str:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,nb_frames,r_frame_rate",
         "-of", "csv=p=0", str(path)],
        capture_output=True, text=True).stdout.strip()
    return out


def grab(path: Path, at: float) -> np.ndarray:
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", f"{at}", "-i", str(path),
         "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "-"],
        capture_output=True).stdout
    if not raw:
        raise SystemExit(f"cannot decode a frame from {path}")
    return np.asarray(Image.open(__import__("io").BytesIO(raw)).convert("L"),
                      dtype=np.float32) / 255.0


def stats(a: np.ndarray) -> dict:
    """Statistics that distinguish content from noise.

    adjdiff alone is misleading: isolated grain spikes inflate it while a
    mosaic of 16px blocks keeps it low-ish, because the blocks are internally
    smooth. So we also measure the block-scale jump, which is what a latent
    that never denoised actually looks like after the 16x VAE decode.
    """
    horizontal = np.abs(np.diff(a, axis=1)).mean()
    vertical = np.abs(np.diff(a, axis=0)).mean()
    block = np.abs(a[16:, 16:] - a[:-16, :-16]).mean()
    return {
        "mean": float(a.mean()),
        "std": float(a.std()),
        "adjdiff": float((horizontal + vertical) / 2),
        "block16": float(block),
        "block_ratio": float(block / max((horizontal + vertical) / 2, 1e-6)),
    }


def verdict(s: dict) -> str:
    """Verdict from *absolute* block16, not block_ratio.

    Recalibrated on 2026-10-24 against native-resolution (768x1344) clips:
        512x896/22f  good    block16 0.0689  std 0.175  block_ratio 4.85
        512x896/37f  good    block16 0.0783  std 0.171  block_ratio 7.66
        512x896/72f  mosaic  block16 0.1226  std 0.114  block_ratio 7.26
    block_ratio is NOT a discriminator: the 37f good clip scores higher than
    the 72f mosaic clip, because a *noisy* mosaic block lattice raises adjdiff
    at the same time as block16. Only the absolute 16px-offset difference plus
    a collapsed std separates them, so both conditions are required.
    Thresholds below apply to clips of >=1200px on the short side.
    """
    if s["block16"] > 0.105 and s["std"] < 0.14:
        return "MOSAIC (latent never denoised)"
    if s["std"] < 0.09:
        return "MUSH (under-denoised / no subject)"
    if s["mean"] < 0.12:
        return "DARK (subject likely missing)"
    return "OK"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("clips", nargs="+")
    ap.add_argument("--at", type=float, default=None,
                    help="timestamp to sample (default: middle of clip)")
    ap.add_argument("--strip", default=None, help="write side-by-side strip")
    ap.add_argument("--height", type=int, default=896)
    args = ap.parse_args()

    frames = []
    for name in args.clips:
        path = Path(name)
        if not path.exists():
            print(f"{path}: MISSING")
            continue
        meta = probe(path)
        duration = 0.0
        try:
            duration = float(subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "csv=p=0", str(path)],
                capture_output=True, text=True).stdout.strip())
        except ValueError:
            pass
        at = args.at if args.at is not None else max(duration * 0.5, 0.05)
        a = grab(path, at)
        s = stats(a)
        print(f"{path.name}  [{meta}]  sampled at {at:.2f}s")
        print(f"   mean={s['mean']:.3f} std={s['std']:.3f} "
              f"adjdiff={s['adjdiff']:.4f} block16={s['block16']:.4f} "
              f"block_ratio={s['block_ratio']:.2f}")
        print(f"   verdict: {verdict(s)}")
        frames.append((path.stem, a))

    if args.strip and frames:
        tiles = []
        for _, a in frames:
            im = Image.fromarray((a * 255).astype(np.uint8))
            w = max(1, int(im.width * args.height / im.height))
            tiles.append(np.asarray(im.resize((w, args.height))))
        Image.fromarray(np.hstack(tiles)).save(args.strip)
        print(f"strip -> {args.strip}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
