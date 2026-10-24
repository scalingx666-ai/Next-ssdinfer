#!/usr/bin/env python3
"""Assemble the 12-shot 《嫁金钗》 episode from H3 clips.

Reuses make_h3_drama.py's per-shot assembler (which does the 3x slow-mo, the
1080x1920 upscale, the unsharp/grain grade and the subtitle overlay) but reads
the shot list from story/shots.json, which is the source the H3
prompts were generated from. The older drama/script.json describes a different
16-shot Wan-era episode, so it is not used here.

Usage:
    python assemble_jiajinchai.py [--slowmo 3.0] [--only 01_hairpin ...]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from make_h3_drama import (OUT, WORK, card_clip, make_bgm,
                           make_subtitle_png, run, shot_clip, title_card)
from PIL import ImageFont

ROOT = Path(__file__).resolve().parent
SHOTS_JSON = ROOT.parent / "story" / "shots.json"
FONT = "/System/Library/Fonts/Hiragino Sans GB.ttc"
CLIPS = ROOT / "outputs" / "h3_drama"
SUBDIR = WORK / "subs"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slowmo", type=float, default=3.0)
    ap.add_argument("--seconds", type=float, default=15.0,
                    help="delivered duration per shot after slow-mo; set this "
                         "to frames/24*slowmo or the tail gets frozen by tpad")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--clips", default=None,
                    help="directory holding generated shots (default outputs/h3_drama)")
    ap.add_argument("--work", default=None,
                    help="intermediate directory, keep separate per quality tier")
    ap.add_argument("--out", default="嫁金钗_双生_H3.mp4",
                    help="final filename inside outputs/h3_final")
    args = ap.parse_args()

    # Rebind the module-level paths before anything touches them, so different
    # quality tiers never overwrite each other's intermediates.
    global CLIPS, WORK, SUBDIR
    if args.clips:
        CLIPS = Path(args.clips).expanduser().resolve()
    if args.work:
        WORK = Path(args.work).expanduser().resolve()
        import make_h3_drama as _m
        _m.WORK = WORK
        SUBDIR = WORK / "subs"

    data = json.loads(SHOTS_JSON.read_text())
    WORK.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    SUBDIR.mkdir(parents=True, exist_ok=True)

    clips = []
    total = 0.0

    title_png = title_card([
        ("嫁 金 钗", ImageFont.truetype(FONT, 132, index=2), (232, 205, 150)),
        ("AI 短剧 · 双生", ImageFont.truetype(FONT, 46, index=0), (200, 190, 180)),
        ("替嫁 · 真假千金 · 金钗信物",
         ImageFont.truetype(FONT, 38, index=0), (170, 160, 150)),
    ], WORK / "title.png")
    clips.append(card_clip(title_png, 0, "title", 3.5))
    total += 3.5

    index = 0
    missing = []
    for shot in data["shots"]:
        if args.only and shot["id"] not in args.only:
            continue
        index += 1
        # Tiers name their outputs with a frame-count suffix, so accept the
        # plain id first and fall back to any "<id>*.mp4" we generated.
        vid = CLIPS / f"{shot['id']}.mp4"
        if not vid.exists():
            hits = sorted(CLIPS.glob(f"{shot['id']}*.mp4"))
            if hits:
                vid = hits[0]
        if not vid.exists():
            missing.append(shot["id"])
            print(f"[miss] {CLIPS}/{shot['id']}*.mp4", flush=True)
            continue
        sub_png = SUBDIR / f"sub_{shot['id']}.png"
        make_subtitle_png(shot["sub"], sub_png)
        clip, dur = shot_clip(shot["id"], vid, sub_png, index,
                              args.seconds, args.slowmo)
        clips.append((clip, dur))
        total += dur

    end_png = title_card([
        ("金 钗 现 世", ImageFont.truetype(FONT, 104, index=2), (232, 205, 150)),
        ("全 剧 终", ImageFont.truetype(FONT, 44, index=0), (200, 190, 180)),
    ], WORK / "end.png")
    index += 1
    clips.append(card_clip(end_png, index, "end", 4.5))
    total += 4.5

    print(f"[concat] {len(clips)} clips, total {total:.1f}s "
          f"({len(missing)} missing)", flush=True)
    lst = WORK / "concat_jjc.txt"
    lst.write_text("".join(f"file '{c}'\n" for c, _ in clips))
    joined = WORK / "joined_jjc.mp4"
    run(["ffmpeg", "-y", "-loglevel", "error",
         "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", joined])

    bgm = WORK / "bgm_jjc.wav"
    make_bgm(bgm, total)
    fc = (
        f"[1:a]atrim=0:{total:.3f},volume=0.22,"
        f"afade=t=in:st=0:d=3.0,afade=t=out:st={max(0, total - 4.5):.3f}:d=4.5[bgm];"
        f"[0:a][bgm]amix=inputs=2:duration=first:normalize=0,"
        f"alimiter=limit=0.95,afade=t=out:st={max(0, total - 1.2):.3f}:d=1.2[a]"
    )
    final = OUT / args.out
    run(["ffmpeg", "-y", "-loglevel", "error",
         "-i", joined, "-stream_loop", "-1", "-i", bgm,
         "-filter_complex", fc, "-map", "0:v", "-map", "[a]",
         "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
         "-ar", "48000", "-ac", "2", "-movflags", "+faststart", final])
    print("DONE", final, f"{total:.1f}s", flush=True)
    if missing:
        print("missing shots:", ", ".join(missing), flush=True)


if __name__ == "__main__":
    main()
