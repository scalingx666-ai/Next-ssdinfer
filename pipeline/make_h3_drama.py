#!/usr/bin/env python3
"""Assemble the H3-rendered 《嫁金钗》 vertical short drama.

Differences from the Wan-era `make_drama.py`:
  * H3 emits 24 fps with native 32 kHz stereo audio, so no slow-mo/retime and
    no BGM-only mix is required; the generated soundtrack is kept and a
    procedural guzheng bed is mixed underneath it.
  * Source clips are 432x768 (or whatever `h3_drama.py --width/--height` used),
    upscaled to 1080x1920 with lanczos + a light sharpen, since H3's 768p-class
    canvas is the quality ceiling of this hardware.
  * Subtitle cards, title/end cards and episode break cards are reused.

Usage:
    python make_h3_drama.py --list
    python make_h3_drama.py --preset fast
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
SCRIPT = ROOT / "drama" / "script.json"


def find_shots_dir() -> Path:
    """Locate the generated per-shot clips.

    h3_batch.py writes into outputs/h3_drama/, while the older h3_drama.py
    driver used outputs/h3_shots/. Prefer whichever directory actually holds
    the clips so both pipelines can feed this assembler.
    """
    for candidate in (ROOT / "outputs" / "h3_drama",
                      ROOT / "outputs" / "h3_shots"):
        if candidate.is_dir() and any(candidate.glob("*.mp4")):
            return candidate
    return ROOT / "outputs" / "h3_drama"


SHOTS = find_shots_dir()
WORK = ROOT / "outputs" / "h3_work"
OUT = ROOT / "outputs" / "h3_final"
FONT = "/System/Library/Fonts/Hiragino Sans GB.ttc"
SRC_FPS = 24
OUT_FPS = 24
W, H = 1080, 1920


def run(cmd, **kw):
    print(" ".join(str(c) for c in cmd)[:220], flush=True)
    subprocess.run([str(c) for c in cmd], check=True, **kw)


def probe_duration(path: Path) -> float:
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "json", str(path)]
    )
    return float(json.loads(out)["format"]["duration"])


def has_audio(path: Path) -> bool:
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", "-select_streams", "a",
         "-show_entries", "stream=codec_type", "-of", "json", str(path)]
    )
    return bool(json.loads(out).get("streams"))


def make_subtitle_png(text: str, dst: Path) -> Path:
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    font = ImageFont.truetype(FONT, 46, index=2)
    lines, cur = [], ""
    for ch in text:
        cur += ch
        if len(cur) >= 16:
            lines.append(cur)
            cur = ""
    if cur:
        lines.append(cur)
    line_h = 66
    block_h = line_h * len(lines)
    y0 = H - 340 - block_h // 2
    widths = [d.textlength(l, font=font) for l in lines]
    tw = max(widths)
    pad_x, pad_y = 44, 26
    d.rounded_rectangle(
        [(W - tw) / 2 - pad_x, y0 - pad_y, (W + tw) / 2 + pad_x, y0 + block_h + pad_y],
        radius=22, fill=(0, 0, 0, 132)
    )
    for i, l in enumerate(lines):
        x = (W - widths[i]) / 2
        d.text((x, y0 + i * line_h), l, font=font, fill=(0, 0, 0, 200),
               stroke_width=3, stroke_fill=(0, 0, 0, 200))
        d.text((x, y0 + i * line_h), l, font=font, fill=(255, 251, 240, 255))
    img.save(dst)
    return dst


def title_card(lines, dst: Path) -> Path:
    img = Image.new("RGB", (W, H), (12, 8, 10))
    arr = np.zeros((H, W, 3), dtype=np.uint8)
    for y in range(H):
        t = y / H
        arr[y, :, 0] = int(26 + 20 * t)
        arr[y, :, 1] = int(13 + 11 * t)
        arr[y, :, 2] = int(15 + 15 * t)
    d = ImageDraw.Draw(Image.fromarray(arr))
    y = 780
    for line, fnt, col in lines:
        w = d.textlength(line, font=fnt)
        d.text(((W - w) / 2, y), line, font=fnt, fill=col)
        y += int(fnt.size * 1.7)
    img = Image.fromarray(arr)
    d = ImageDraw.Draw(img)
    y = 780
    for line, fnt, col in lines:
        w = d.textlength(line, font=fnt)
        d.text(((W - w) / 2, y), line, font=fnt, fill=col)
        y += int(fnt.size * 1.7)
    img.save(dst)
    return dst


def shot_clip(shot_id: str, vid: Path, sub_png: Path, idx: int, target: float,
              slowmo: float = 1.0):
    """Assemble one shot into a 1080x1920 clip of `target` seconds.

    `slowmo` stretches the generated source: a 5 s source with slowmo=3.0
    becomes 15 s. This is the lever that makes the 10-minute budget reachable
    (see h3c/PERFORMANCE.md section 3.3) — generating 120 frames and slowing
    3x costs a third of the DiT work of generating 360 frames.
    """
    vd = probe_duration(vid)
    stretched = vd * slowmo
    # If the slow-mo result is still short of the script duration, hold the
    # last frame; if it is long, trim it.
    hold = max(0.0, target - stretched)
    dst = WORK / f"{idx:02d}_{shot_id}.mp4"
    tail = f",tpad=stop_mode=clone:stop_duration={hold:.3f}" if hold > 0.02 else ""
    vf = (
        f"setpts={slowmo:.5f}*PTS,"
        f"scale={W}:{H}:flags=lanczos,"
        "unsharp=5:5:0.45:5:5:0.0,"
        "eq=contrast=1.05:saturation=1.07:gamma=0.99,"
        "colorbalance=rs=0.02:bs=-0.02,"
        "vignette=angle=PI/5,noise=alls=3:allf=t+u"
        f"{tail},"
        f"fade=t=in:st=0:d=0.30,fade=t=out:st={target - 0.30:.3f}:d=0.30"
    )
    fc = (
        f"[1:v]format=rgba,fade=t=in:st=0.35:d=0.35:alpha=1,"
        f"fade=t=out:st={target - 0.6:.3f}:d=0.4:alpha=1[sub];"
        f"[0:v]{vf}[base];[base][sub]overlay=0:0:format=auto[v]"
    )
    run(["ffmpeg", "-y", "-loglevel", "error",
         "-i", vid, "-loop", "1", "-i", sub_png,
         "-filter_complex", fc, "-map", "[v]",
         "-t", f"{target:.3f}",
         "-r", str(OUT_FPS), "-c:v", "libx264", "-preset", "medium", "-crf", "18",
         "-pix_fmt", "yuv420p", dst])
    return dst, target


def card_clip(png: Path, idx: int, name: str, dur: float):
    dst = WORK / f"{idx:02d}_{name}.mp4"
    vf = (
        "zoompan=z='min(zoom+0.0005,1.05)':d=1:s=1080x1920:fps=24,"
        f"fade=t=in:st=0:d=0.5,fade=t=out:st={dur - 0.6:.3f}:d=0.6,"
        "eq=contrast=1.03:saturation=1.02"
    )
    run(["ffmpeg", "-y", "-loglevel", "error",
         "-loop", "1", "-t", f"{dur}", "-i", png,
         "-f", "lavfi", "-t", f"{dur}", "-i", "anullsrc=r=48000:cl=stereo",
         "-vf", vf, "-map", "0:v", "-map", "1:a",
         "-r", str(OUT_FPS), "-c:v", "libx264", "-preset", "medium", "-crf", "18",
         "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
         "-ar", "48000", "-ac", "2", dst])
    return dst, dur


def make_bgm(dst: Path, total: float, sr: int = 48000):
    n = int(total * sr)
    t = np.arange(n) / sr
    out = np.zeros(n, dtype=np.float32)
    base = 146.83
    ratios = [1.0, 1.122, 1.335, 1.498, 1.782]
    notes = [base * r for r in ratios] + [base * 2 * r for r in ratios]
    rng = np.random.default_rng(11)
    beat = 0.5
    beat_idx = 0
    while beat_idx * beat < total:
        st = int(beat_idx * beat * sr)
        if st >= n:
            break
        deg = notes[int(rng.integers(0, len(notes)))]
        dur = min(1.3, total - st / sr)
        tt = np.arange(int(dur * sr)) / sr
        env = np.exp(-tt * 4.6)
        sig = np.zeros_like(tt)
        for h, a in [(1, 1.0), (2, 0.40), (3, 0.16), (4.2, 0.08)]:
            sig += a * np.sin(2 * np.pi * deg * h * tt + h)
        sig *= env
        sig += rng.normal(0, 0.003, tt.size) * np.exp(-tt * 55)
        out[st:st + tt.size] += sig.astype(np.float32)
        if rng.random() < 0.3:
            st2 = st + int(0.13 * sr)
            if st2 + tt.size < n:
                out[st2:st2 + tt.size] += (sig * 0.55).astype(np.float32)
        beat_idx += 1
    for f, a in [(base / 2, 0.045), (base * 1.5, 0.028)]:
        out += a * np.sin(2 * np.pi * f * t + 0.6 * np.sin(2 * np.pi * 0.11 * t))
    fi, fo = int(3.0 * sr), int(4.5 * sr)
    out[:fi] *= np.linspace(0, 1, fi)
    out[-fo:] *= np.linspace(1, 0, fo)
    out /= max(1e-6, np.abs(out).max())
    out *= 0.42
    stereo = np.stack([out, out], axis=1)
    pcm = (stereo * 32767).astype(np.int16)
    import wave
    with wave.open(str(dst), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return dst


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--only", default="")
    ap.add_argument("--slowmo", type=float, default=1.0,
                    help="stretch generated sources by this factor (3.0 for the "
                         "120-frame 8fps budget route)")
    args = ap.parse_args()

    script = json.loads(SCRIPT.read_text())
    WORK.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    subdir = WORK / "subs"
    subdir.mkdir(exist_ok=True)
    only = {x.strip() for x in args.only.split(",") if x.strip()}

    if args.list:
        for ep in script["episodes"]:
            for sh in ep["shots"]:
                print(f"{sh['id']}  {sh['seconds']:>4.1f}s  {ep['title']}")
        return

    clips = []
    t_png = title_card([
        ("嫁 金 钗", ImageFont.truetype(FONT, 132, index=2), (232, 205, 150)),
        ("AI 短剧 · 第一季", ImageFont.truetype(FONT, 46, index=0), (200, 190, 180)),
        ("假千金 · 替嫁 · 鸠占鹊巢", ImageFont.truetype(FONT, 38, index=0), (170, 160, 150)),
    ], WORK / "title.png")
    clips.append(card_clip(t_png, 0, "title", 3.5))
    total = 3.5

    idx = 0
    for ep in script["episodes"]:
        for sh in ep["shots"]:
            idx += 1
            sid = sh["id"]
            if only and sid not in only:
                continue
            vid = SHOTS / f"{sid}.mp4"
            if not vid.exists():
                print(f"[miss] {vid}", flush=True)
                continue
            sub_png = subdir / f"sub_{sid}.png"
            make_subtitle_png(sh["subtitle"], sub_png)
            clip, dur = shot_clip(sid, vid, sub_png, idx, sh["seconds"],
                                  args.slowmo)
            clips.append((clip, dur))
            total += dur
            if sh is ep["shots"][-1] and ep is not script["episodes"][-1]:
                idx += 1
                c_png = title_card([
                    (ep["title"].split(" · ")[-1],
                     ImageFont.truetype(FONT, 96, index=2), (232, 205, 150)),
                ], WORK / f"card_{ep['id']}.png")
                clips.append(card_clip(c_png, idx, f"card_{ep['id']}", 1.6))
                total += 1.6

    e_png = title_card([
        ("金 钗 现 世", ImageFont.truetype(FONT, 104, index=2), (232, 205, 150)),
        ("全 剧 终", ImageFont.truetype(FONT, 44, index=0), (200, 190, 180)),
    ], WORK / "end.png")
    idx += 1
    clips.append(card_clip(e_png, idx, "end", 4.5))
    total += 4.5

    print(f"[concat] {len(clips)} clips, total {total:.1f}s", flush=True)
    lst = WORK / "concat.txt"
    lst.write_text("".join(f"file '{c}'\n" for c, _ in clips))
    joined = WORK / "joined.mp4"
    run(["ffmpeg", "-y", "-loglevel", "error",
         "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", joined])

    bgm = WORK / "bgm.wav"
    make_bgm(bgm, total)
    fc = (
        f"[1:a]atrim=0:{total:.3f},volume=0.22,"
        f"afade=t=in:st=0:d=3.0,afade=t=out:st={max(0, total - 4.5):.3f}:d=4.5[bgm];"
        f"[0:a][bgm]amix=inputs=2:duration=first:normalize=0,"
        f"alimiter=limit=0.95,afade=t=out:st={max(0, total - 1.2):.3f}:d=1.2[a]"
    )
    final = OUT / "嫁金钗.mp4"
    run(["ffmpeg", "-y", "-loglevel", "error",
         "-i", joined, "-stream_loop", "-1", "-i", bgm,
         "-filter_complex", fc, "-map", "0:v", "-map", "[a]",
         "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
         "-ar", "48000", "-ac", "2", "-movflags", "+faststart", final])
    print("DONE", final, f"{total:.1f}s", flush=True)


if __name__ == "__main__":
    main()
