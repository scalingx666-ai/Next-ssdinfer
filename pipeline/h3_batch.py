#!/usr/bin/env python3
"""Drive h3's interactive mode as a batch generator.

Why this exists: `h3 -p PROMPT -o OUT.mp4` pays a fixed 20 s DiT weight load
plus a ~10 s text-encoder pass on every invocation. On M1 Pro that overhead is
comparable to the denoise itself at small shapes, so generating 20 clips as 20
separate processes wastes 8-10 minutes before any denoising happens.

h3's interactive mode calls h3_cache_set_enabled(ctx, 1) (h3_cli.c:697), which
retains the prepared DiT and its conditioning between generations when the
requested shape is unchanged. Feeding commands on stdin therefore amortises
the load across every clip in the drama.

Usage:
    python h3_batch.py --script shots.jsonl [--dry-run] [--log PATH]

shots.jsonl holds one JSON object per line:
    {"prompt": "...", "frames": 120, "out": "shot_01.mp4"}
Optional keys: "first" / "last" (anchor image paths), "seed" (int).
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
H3C_DIR = REPO / "engine"
H3_BIN = H3C_DIR / "h3"
# The FL2VA weights are ~134 GB and are deliberately not in this repository.
# Point H3_MODEL at your local checkout, or run `scripts/fetch_model.sh`.
MODEL = Path(os.environ.get("H3_MODEL") or
             ROOT / "models" / "ms-h3-cache" / "models" / "MiniMax--MiniMax-H3" /
             "snapshots" / "master")


def build_commands(shots: list[dict], out_dir: Path, preset: dict) -> list[str]:
    """Turn shot records into h3 interactive-mode command lines.

    `!save` is deliberately not used: in the non-TTY pipe mode it is never
    consumed (verified empirically — the session wrote video-000N.mp4 into the
    `!output` directory and silently ignored every `!save` line). The driver
    renames those numbered files to the shot names afterwards instead.
    """
    lines: list[str] = [
        f"!output {out_dir}",
        f"!size {preset['width']}x{preset['height']}",
        f"!render-size {preset['render_w']}x{preset['render_h']}",
        f"!steps {preset['steps']}",
        f"!layers {preset['layers']}",
        f"!reuse {preset['reuse']}",
        f"!token-reduction {'on' if preset.get('token_reduction') else 'off'}",
        f"!ssd-streaming {'on' if preset.get('ssd_streaming', True) else 'off'}",
        # The preset's frame count is the source duration per clip. It must be
        # sent explicitly: h3 defaults to 56 frames, which silently makes every
        # clip a different shape and defeats the cache and the cost model.
        f"!frames {preset['frames']}",
        "!cache",
    ]
    for index, shot in enumerate(shots, start=1):
        name = shot.get("out", f"shot_{index:02d}.mp4")
        # Per-shot frame counts override the preset, then restore it.
        frames = shot.get("frames")
        if frames and frames != preset["frames"]:
            lines.append(f"!frames {frames}")
        # Per-shot step count overrides the preset too, so an A/B can change
        # exactly one variable and keep frames/canvas/seed identical.
        steps = shot.get("steps")
        if steps and steps != preset["steps"]:
            lines.append(f"!steps {steps}")
        # Anchors and seed are applied to the generation that follows them.
        if shot.get("first"):
            lines.append(f"!first {shot['first']}")
        if shot.get("last"):
            lines.append(f"!last {shot['last']}")
        if "seed" in shot:
            lines.append(f"!seed {shot['seed']}")
        # A bare line is a prompt and triggers generation. linenoiseNoTTY()
        # splits stdin on '\n' (linenoise.c:1620), so a multi-line prompt is
        # delivered as several separate "lines" and each fragment before the
        # last would fire its own generation, corrupting the output ordering.
        # The H3 prompt is one raw string where newlines are only section
        # separators, so collapsing them to spaces is semantically equivalent.
        lines.append(" ".join(shot["prompt"].split()))
        if frames and frames != preset["frames"]:
            lines.append(f"!frames {preset['frames']}")
        if steps and steps != preset["steps"]:
            lines.append(f"!steps {preset['steps']}")
        if shot.get("first"):
            lines.append("!first clear")
        if shot.get("last"):
            lines.append("!last clear")
    lines.append("!quit")
    return lines


def clip_complete(path: Path, expect_frames: int | None) -> bool:
    """True only if the file decodes as a video with >= expect_frames.

    A power cut or SIGKILL during the final mux leaves a truncated mp4 whose
    moov may be missing; `exists()` alone would then keep a broken shot
    forever. h3 emits frames+2 (37 -> 39), so require at least `frames`.
    """
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=nb_frames", "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, timeout=120).stdout.strip()
        got = int(out.splitlines()[0].split(",")[0]) if out else 0
    except (subprocess.SubprocessError, ValueError, OSError):
        return False
    if expect_frames and got and got < int(expect_frames):
        return False
    return got > 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--script", required=True, help="JSONL of shot prompts")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="print the command stream instead of running it")
    ap.add_argument("--log", default=None)
    ap.add_argument("--preset", default="target",
                    help="preset name from h3_drama.py")
    ap.add_argument("--resume", action="store_true",
                    help="skip shots whose output file already exists")
    # Escape hatch for h3 experiments that need a CLI flag (e.g.
    # --use-reference-rope) without forking this driver. Repeatable.
    ap.add_argument("--only", nargs="*", default=None,
                    help="render only these shot ids (e.g. --only 01_hairpin)")
    ap.add_argument("--h3-arg", action="append", default=[],
                    help="extra argument passed through to the h3 binary")
    args = ap.parse_args()

    shots = [json.loads(line) for line in
             Path(args.script).read_text().splitlines() if line.strip()]
    if not shots:
        print("no shots in script", file=sys.stderr)
        return 1

    # --only filters before the resume bookkeeping, so a single-shot probe on a
    # full script does not adopt or skip anything it did not render.
    if args.only:
        wanted = set(args.only)
        shots = [s for s in shots if s.get("id") in wanted]
        if not shots:
            print(f"no shots matched --only {sorted(wanted)}", file=sys.stderr)
            return 1
        print(f"only: {len(shots)} shot(s): {', '.join(s['id'] for s in shots)}")

    # Resume support: the session numbers outputs in request order, so a
    # partial run can only be resumed by dropping the leading shots that
    # already exist and keeping the tail.
    out_dir = (Path(args.out_dir) if args.out_dir else
               ROOT / "outputs" / f"batch_{args.preset}").resolve()
    if args.resume:
        # A hard kill (power loss, SIGKILL) never reaches the rename step at the
        # bottom of this function, so finished shots stay as video-NNNN.mp4 and
        # the per-shot names never appear. Adopt them in order before deciding
        # what is left to do. Session numbering follows request order, which is
        # the same order `shots` is in.
        if out_dir.is_dir():
            for index, stray in enumerate(sorted(out_dir.glob("video-*.mp4"))):
                if index >= len(shots):
                    break
                target = out_dir / shots[index].get(
                    "out", f"shot_{index + 1:02d}.mp4")
                if not target.exists():
                    stray.rename(target)
                    print(f"resume: adopted {stray.name} -> {target.name}",
                          flush=True)
        kept, skipped = [], []
        for index, shot in enumerate(shots):
            name = shot.get("out", f"shot_{index + 1:02d}.mp4")
            path = out_dir / name
            if path.exists() and not clip_complete(path, shot.get("frames")):
                # A power cut or kill during the final mux leaves a truncated
                # mp4. Treating "exists" as "done" would silently keep a broken
                # shot forever, so park it and regenerate.
                parked = path.with_suffix(".partial")
                path.rename(parked)
                print(f"resume: {name} incomplete -> parked as {parked.name}",
                      flush=True)
            if path.exists():
                skipped.append(name)
            else:
                kept.append(shot)
        if not kept:
            print("resume: all shots already present")
            return 0
        print(f"resume: {len(skipped)} done, {len(kept)} to go "
              f"({', '.join(skipped)})")
        shots = kept

    from h3_drama import PRESETS   # single source of truth for the shapes
    preset = PRESETS[args.preset].__dict__

    # h3 runs with cwd=h3c/ (h3_gpu.m:353 needs h3_shaders.metal from the
    # working directory), so `!output` must receive an absolute path or the
    # session fails with "cannot use output directory".
    out_dir.mkdir(parents=True, exist_ok=True)

    commands = build_commands(shots, out_dir, preset)
    if args.dry_run:
        print("\n".join(commands))
        return 0

    # h3_gpu.m:353 resolves h3_shaders.metal from the current working
    # directory, so the process must start inside h3c/.
    stdin_text = "\n".join(commands) + "\n"
    log_path = Path(args.log) if args.log else out_dir / "batch.log"

    print(f"batch: {len(shots)} shots, preset {args.preset}")
    print(f"  shape  {preset['width']}x{preset['height']} "
          f"-> {preset['render_w']}x{preset['render_h']}")
    print(f"  steps {preset['steps']}  layers {preset['layers']}  "
          f"reuse {preset['reuse']}  "
          f"token-reduction {preset.get('token_reduction')}")
    print(f"  out    {out_dir}")
    print(f"  log    {log_path}")

    started = time.time()
    before = {p.name for p in out_dir.glob("*.mp4")}
    with log_path.open("w") as log:
        proc = subprocess.run(
            [str(H3_BIN), "-d", str(MODEL), *args.h3_arg],
            input=stdin_text, text=True, cwd=str(H3C_DIR),
            stdout=log, stderr=subprocess.STDOUT,
        )
    elapsed = time.time() - started
    # The session names outputs video-NNNN.mp4 in request order, so map them
    # back onto the shot names. Anything already present before this run is
    # left alone, which keeps --resume behaviour usable.
    numbered = sorted(p for p in out_dir.glob("video-*.mp4")
                      if p.name not in before)
    renamed = 0
    for index, path in enumerate(numbered):
        if index >= len(shots):
            break
        target = out_dir / shots[index].get(
            "out", f"shot_{index + 1:02d}.mp4")
        if target != path:
            target.unlink(missing_ok=True)
            path.rename(target)
            renamed += 1
        print(f"  {target.name}  {target.stat().st_size / 1e6:.1f} MB")
    produced = [out_dir / s.get("out", f"shot_{i + 1:02d}.mp4")
                for i, s in enumerate(shots)]
    ok = sum(1 for p in produced if p.exists())
    print(f"batch rc={proc.returncode} in {elapsed:.0f}s, "
          f"{ok}/{len(shots)} clips written ({renamed} renamed)")
    return 0 if ok == len(shots) else 2


if __name__ == "__main__":
    raise SystemExit(main())
