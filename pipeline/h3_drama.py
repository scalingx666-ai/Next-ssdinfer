#!/usr/bin/env python3
"""H3 long-video production driver.

Generates a full-length vertical drama from drama/script.json by invoking the
native antirez/h3.c binary once per shot and stitching the results with FFmpeg.

Design notes
------------
H3 emits 24 fps and aligns frame counts to `5 + 17*n`, with 362 frames
(15.08 s) as the practical single-clip maximum. A 5-minute drama therefore
needs ~20 clips. Every clip is generated independently and concatenated, so a
failure only costs one shot instead of the whole run.

Continuity strategy
-------------------
`--first-frame` conditions on the previous clip's last frame, which is the
cheapest way to keep subject identity across a cut. It is enabled per shot via
`script.json`'s `chain` flag and requires the previous clip to exist.

Usage
-----
    python h3_drama.py --list
    python h3_drama.py --only s01,s02 --probe
    python h3_drama.py --preset fast
    python h3_drama.py --preset balanced --workers 1
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRIPT = ROOT / "drama" / "script.json"
SHOTS = ROOT / "outputs" / "h3_shots"
WORK = ROOT / "outputs" / "h3_work"
OUT = ROOT / "outputs" / "h3_final"
LOG = ROOT / "outputs" / "h3_drama.log"
H3_BIN = ROOT.parent / "h3c" / "h3"
MODEL_DIR = ROOT / "models" / "ms-h3-cache" / "models" / "MiniMax--MiniMax-H3" / "snapshots" / "master"

FPS = 24
W, H = 432, 768          # 9:16, multiple of 32, product 331,776 <= 768*1344
STYLE = (
    "电影感光影，古装剧照质感，暖色调，浅景深，高清细节，胶片颗粒，国风美学，"
    "vertical composition, cinematic lighting, film grain"
)


@dataclass
class Preset:
    name: str
    steps: int
    layers: int
    reuse: int
    token_reduction: bool = False
    render_w: int | None = None
    render_h: int | None = None
    ssd_streaming: bool = False
    frames: int = 107          # 4.458 s, the mid-range validated shape
    width: int = 384           # output canvas, must match render aspect
    height: int = 672
    note: str = ""


PRESETS: dict[str, Preset] = {
    # Presets below are derived from h3_budget.py's cost model, not from the
    # M5 Max README numbers. `smoke` is the smallest shape that proves the
    # pipeline end to end; `target` is the configuration h3_budget.py selected
    # for the 10-minute -> 5-minute goal; `quality` is the reference path.
    #
    # h3.c:496-511 rejects an internal canvas whose aspect differs from the
    # output (render_w * height != render_h * width), and both sides must be
    # multiples of 32. The pairs below are the smallest such canvases:
    #   432x768  -> 288x512  (44% of output pixels)
    #   384x672  -> 256x448  (44% of output pixels)
    # h3.c:496-511 rejects an internal canvas whose aspect differs from the
    # output (render_w * height != render_h * width), and both sides must be
    # multiples of 32. 432 is NOT a multiple of 32 (432 = 13.5*32), so the
    # earlier 432x768 output canvas was itself illegal. The legal ladder for
    # the 0.5714 vertical aspect (same as the released 768x1344 limit) is:
    #   output 384x672   -> internal 256x448  (44% of output pixels)
    #   output 768x1344  -> internal 512x896  (44% of output pixels)
    "smoke": Preset("smoke", steps=4, layers=50, reuse=1,
                    render_w=256, render_h=448, frames=22, width=384, height=672,
                    note="256x448 internal 4-step 22-frame smoke test"),
    # `target` is the budget-fitting configuration. The key modelling point
    # (see h3c/PERFORMANCE.md section 3.3): lowering the delivered frame rate
    # does not shorten each clip, it only reduces the clip count. The real
    # saving is a short source duration combined with a large slow-mo factor,
    # so each clip is 120 source frames (5 s) stretched to 15 s of the drama.
    "target": Preset("target", steps=4, layers=45, reuse=1, token_reduction=True,
                     render_w=256, render_h=448, frames=120, ssd_streaming=True,
                     width=384, height=672,
                     note="256x448 internal 4-step 120-frame source, 3x slow-mo"),
    "balanced": Preset("balanced", steps=8, layers=45, reuse=1,
                       render_w=256, render_h=448, frames=180,
                       width=384, height=672,
                       note="256x448 internal 8-step 180-frame source, 2x slow-mo"),
    "max": Preset("max", steps=20, layers=50, reuse=1,
                  render_w=512, render_h=896, frames=120, ssd_streaming=True,
                  width=768, height=1344,
                  note="TRUE max: 512x896 internal 20-step 50-layer, no token reduction, 768x1344 out"),
    "steps20_same": Preset("steps20_same", steps=20, layers=50, reuse=1,
                   render_w=256, render_h=448, frames=120, ssd_streaming=True,
                   width=384, height=672,
                   note="SAME canvas as target (256x448) but 20 steps, 50 layers, no token reduction"),
    "mid": Preset("mid", steps=20, layers=50, reuse=1,
                   render_w=384, render_h=672, frames=120, ssd_streaming=True,
                   width=512, height=896,
                   note="384x672 internal 20-step 50-layer, no token reduction"),
    "quality": Preset("quality", steps=20, layers=50, reuse=1,
                      render_w=512, render_h=896, frames=362,
                      width=768, height=1344,
                      note="512x896 internal 20-step 362-frame full-rate source"),
}


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with LOG.open("a") as fh:
        fh.write(line + "\n")


def load_script() -> dict:
    return json.loads(SCRIPT.read_text())


def iter_shots(script: dict):
    for ep in script["episodes"]:
        for sh in ep["shots"]:
            yield ep, sh


def probe_duration(path: Path) -> float:
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "json", str(path)]
    )
    return float(json.loads(out)["format"]["duration"])


def last_frame(path: Path, dst: Path) -> bool:
    """Extract the final frame of a clip as PNG for `--first-frame` chaining."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-sseof", "-0.05",
         "-i", str(path), "-frames:v", "1", "-q:v", "2", str(dst)],
        capture_output=True,
    )
    return r.returncode == 0 and dst.exists() and dst.stat().st_size > 0


def build_prompt(shot: dict, style: str) -> str:
    camera = shot.get("camera", "")
    scene = shot.get("scene", "")
    prompt = shot.get("prompt", scene)
    audio = shot.get("audio", "环境音，无配乐")
    return (
        f"Scene: {scene}\n"
        f"Action: {prompt}\n"
        f"Camera: {camera}\n"
        f"Look: {style}\n"
        f"Audio: {audio}"
    )


def check_canvas(width: int, height: int, preset: Preset) -> None:
    """Fail fast on the canvas rules h3.c enforces at h3.c:496-511.

    Both sides must be multiples of 32, the internal canvas must not exceed
    the output, and the aspect ratios must match exactly
    (render_w * height == render_h * width). Getting this wrong produces a
    hard h3 error after the model has already been loaded.
    """
    if not preset.render_w:
        return
    rw, rh = preset.render_w, preset.render_h
    if width % 32 or height % 32:
        raise SystemExit(f"output {width}x{height} must be multiples of 32")
    if rw % 32 or rh % 32:
        raise SystemExit(f"render {rw}x{rh} must be multiples of 32")
    if rw > width or rh > height:
        raise SystemExit(f"render {rw}x{rh} exceeds output {width}x{height}")
    if rw * height != rh * width:
        raise SystemExit(
            f"render {rw}x{rh} aspect {rw / rh:.5f} != output "
            f"{width}x{height} aspect {width / height:.5f}"
        )


def generate_shot(shot: dict, preset: Preset, style: str, chain: bool,
                  force: bool, dry: bool, width: int = 384, height: int = 672) -> Path | None:
    check_canvas(width, height, preset)
    sid = shot["id"]
    dst = SHOTS / f"{sid}.mp4"
    if dst.exists() and not force:
        log(f"[skip] {sid} exists ({probe_duration(dst):.2f}s)")
        return dst

    cmd = [
        str(H3_BIN),
        "-d", str(MODEL_DIR),
        "-p", build_prompt(shot, style),
        "-o", str(dst),
        "--width", str(width),
        "--height", str(height),
        "--frames", str(preset.frames),
        "--steps", str(preset.steps),
        "--layers", str(preset.layers),
        "--reuse", str(preset.reuse),
        "--seed", str(hash(sid) % 100000),
    ]
    if preset.render_w:
        cmd += ["--render-width", str(preset.render_w),
                "--render-height", str(preset.render_h)]
    if preset.token_reduction:
        cmd.append("--token-reduction")
    if preset.ssd_streaming:
        cmd.append("--ssd-streaming")

    if chain:
        # shot ids are s01..s16; the previous shot is the previous episode's
        # last shot when the current one is an episode opener.
        num = int(sid[1:])
        prev_id = f"s{num - 1:02d}"
        prev = SHOTS / f"{prev_id}.mp4"
        if not prev.exists() and num > 1:
            for ep, cand in iter_shots(script):
                if cand["id"] == prev_id:
                    break
        anchor = WORK / "chain" / f"{sid}.png"
        if prev.exists() and last_frame(prev, anchor):
            cmd += ["--first-frame", str(anchor)]
            log(f"[chain] {sid} <- {prev.name}")
        else:
            log(f"[chain] {sid} has no previous frame, generating standalone")

    log(f"[run] {sid} {preset.name} steps={preset.steps} layers={preset.layers} "
        f"frames={preset.frames} render={preset.render_w}x{preset.render_h}")
    if dry:
        log(f"       {' '.join(cmd)}")
        return None

    t0 = time.time()
    SHOTS.mkdir(parents=True, exist_ok=True)
    logdir = WORK / "logs"
    logdir.mkdir(parents=True, exist_ok=True)
    # h3_gpu.m:353 resolves "h3_shaders.metal" against the current working
    # directory, so the process must run from the h3.c source tree.
    with open(logdir / f"{sid}.log", "w") as logfh:
        r = subprocess.run(cmd, stdout=logfh, stderr=subprocess.STDOUT,
                           cwd=str(H3_BIN.parent))
    dt = time.time() - t0
    if r.returncode != 0 or not dst.exists():
        log(f"[FAIL] {sid} rc={r.returncode} after {dt:.1f}s "
            f"-> {WORK / 'logs' / f'{sid}.log'}")
        return None
    dur = probe_duration(dst)
    log(f"[ok] {sid} {dur:.2f}s video in {dt:.1f}s wall "
        f"({dur / dt:.3f}x realtime)")
    return dst


def cmd_list(script: dict) -> None:
    total = 0.0
    for ep, sh in iter_shots(script):
        total += sh["seconds"]
        print(f"{sh['id']}  {sh['seconds']:>4.1f}s  {ep['title']}  {sh['camera']}")
    print(f"\n{len(script['episodes'])} episodes, "
          f"{sum(len(e['shots']) for e in script['episodes'])} shots, "
          f"{total:.1f}s of scripted content")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="fast", choices=sorted(PRESETS))
    ap.add_argument("--only", default="", help="comma-separated shot ids")
    ap.add_argument("--width", type=int, default=384)
    ap.add_argument("--height", type=int, default=672)
    ap.add_argument("--chain", action="store_true",
                    help="condition each shot on the previous clip's last frame")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    script = load_script()
    if args.list:
        cmd_list(script)
        return

    W, H = args.width, args.height
    preset = PRESETS[args.preset]
    # The preset carries its own canvas because the internal render canvas is
    # only legal at a matching aspect ratio (h3.c:506). An explicit
    # --width/--height wins, but then the preset's render canvas must be
    # recomputed or h3 will reject the pair.
    default_w, default_h = 384, 672
    if args.width != default_w or args.height != default_h:
        W, H = args.width, args.height
        if preset.render_w:
            scale = W / preset.width
            rw, rh = int(preset.render_w * scale), int(preset.render_h * scale)
            preset.render_w = max(32, rw - rw % 32)
            preset.render_h = max(32, rh - rh % 32)
            log(f"[canvas] recomputed render {preset.render_w}x{preset.render_h} "
                f"for output {W}x{H}")
    else:
        W, H = preset.width, preset.height
    style = script.get("global_style_prompt", STYLE)
    only = {x.strip() for x in args.only.split(",") if x.strip()}

    log(f"=== H3 drama: preset={preset.name} ({preset.note}) "
        f"{W}x{H} chain={args.chain} ===")

    if not H3_BIN.exists():
        log(f"FATAL: h3 binary missing at {H3_BIN}")
        sys.exit(1)
    if not MODEL_DIR.exists():
        log(f"FATAL: model dir missing at {MODEL_DIR}")
        sys.exit(1)

    done, failed = [], []
    t_start = time.time()
    for ep, sh in iter_shots(script):
        if only and sh["id"] not in only:
            continue
        out = generate_shot(sh, preset, style, args.chain, args.force, args.dry,
                            W, H)
        (done if out else failed).append(sh["id"])

    dt = time.time() - t_start
    log(f"=== {len(done)} ok, {len(failed)} failed in {dt / 60:.1f} min "
        f"({', '.join(failed)}) ===")


if __name__ == "__main__":
    main()
