#!/usr/bin/env python3
"""Batch renderer driver.

Thin CLI over ``core/`` + ``models/<name>/``. All model knowledge lives in the
adapter; this file only parses arguments, applies resume bookkeeping, and runs
the session.

Why a driver at all: a long-lived renderer keeps its weights and conditioning
in memory, so feeding one command stream amortises a fixed weight-load cost
across every clip. On M1 Pro that overhead is comparable to the denoise itself
at small shapes, so one process per clip wastes most of the budget.

Usage:
    python render.py --script shots.jsonl [--model h3] [--preset max]
                     [--only 01_hairpin] [--resume] [--dry-run] [--check-only]

Exit codes:
    0  all requested clips present (or dry-run / check passed)
    1  bad arguments, unknown adapter, or a request over capacity
    2  the run finished but some clips are missing
"""

from __future__ import annotations

import argparse
import shlex
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.session import (Session, build_stream, build_argv_list,  # noqa: E402
                          default_numbered_outputs, map_back, run_oneshot)
from core.shots import filter_shots, load_shots  # noqa: E402
from models.registry import load  # noqa: E402


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--script", required=True, help="JSONL of shot records")
    ap.add_argument("--model", default="h3", help="adapter name (default h3)")
    ap.add_argument("--preset", default=None,
                    help="preset name from the adapter (default: its first)")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--log", default=None)
    ap.add_argument("--only", nargs="*", default=None,
                    help="render only these shot ids")
    ap.add_argument("--resume", action="store_true",
                    help="skip clips already complete, adopt orphaned ones")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the command stream instead of running it")
    ap.add_argument("--check-only", action="store_true",
                    help="only report capacity; render nothing")
    ap.add_argument("--arg", action="append", default=[],
                    help="extra argument passed through to the renderer")
    ap.add_argument("--qc", action="store_true",
                    help="judge each finished clip with the adapter")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    adapter = load(args.model, REPO)
    preset_name = args.preset or next(iter(adapter.presets))
    if preset_name not in adapter.presets:
        print(f"unknown preset {preset_name!r}. available: "
              f"{', '.join(adapter.presets)}", file=sys.stderr)
        return 1
    preset = adapter.presets[preset_name]

    shots = filter_shots(load_shots(args.script), args.only)

    # Capacity is checked before anything runs. An over-budget request is
    # refused rather than degraded, because the failure mode is silent: the
    # renderer exits 0 and writes a playable clip full of noise, or a model
    # load that crosses the working set raises somewhere unhelpful later.
    worst = None
    for shot in shots:
        cap = adapter.capacity.check(preset, shot)
        worst = cap if worst is None or cap.cost > worst.cost else worst
        if not cap.ok:
            print(f"capacity: {shot.id}: {cap.reason}", file=sys.stderr)
            return 1
    print(f"capacity: {worst.describe()}"
          + (f" — {worst.reason}" if worst.reason else ""))

    if args.check_only:
        return 0

    out_dir = (Path(args.out_dir) if args.out_dir else
               ROOT / "outputs" / f"run_{preset_name}").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.resume:
        from core.resume import adopt_stray_outputs, partition_shots

        adopted = adopt_stray_outputs(
            out_dir, shots,
            on_adopt=lambda s, t: print(f"resume: adopted {s.name} -> {t.name}",
                                        flush=True),
            pattern=adapter.output_glob)
        state = partition_shots(
            out_dir, shots, adapter.is_complete,
            on_park=lambda p, q: print(f"resume: {p.name} incomplete -> parked "
                                       f"as {q.name}", flush=True))
        print(f"resume: {state.summary}")
        shots = state.pending
        if not shots:
            print("nothing to do")
            return 0

    oneshot = adapter.mode == "oneshot"

    if oneshot:
        # One argv per request. Dry run prints exactly what would be executed,
        # from the same call, so a dry run cannot drift from the run.
        try:
            argvs = build_argv_list(adapter, shots, preset, out_dir)
        except RuntimeError as exc:
            print(f"preflight: {exc}", file=sys.stderr)
            return 1
        if args.dry_run:
            print("\n".join(shlex.join(a) for a in argvs))
            return 0
    else:
        commands = build_stream(adapter, shots, preset, out_dir)
        if args.dry_run:
            print("\n".join(commands))
            return 0

    log_path = Path(args.log) if args.log else out_dir / "batch.log"
    print(f"batch: {len(shots)} shot(s), preset {preset_name}, model {args.model}")
    print(f"  preset {preset.describe()}")
    print(f"  mode   {adapter.mode}")
    print(f"  model  {adapter.model_dir}")
    print(f"  out    {out_dir}")
    print(f"  log    {log_path}")

    before = ({p.name for p in out_dir.glob(adapter.output_glob)}
              if adapter.output_glob else set())
    started = time.time()
    if oneshot:
        # When stdout is the artifact, name it before the process starts: that
        # is what makes resume idempotent without a post-run rename pass.
        captures = ([out_dir / s.out for s in shots]
                    if getattr(adapter, "stdout_is_output", False) else None)
        result = run_oneshot(argvs, adapter.cwd, log_path, captures)
    else:
        session = Session(adapter.binary, adapter.extra_args(), adapter.cwd,
                          log_path)
        result = session.run("\n".join(commands) + "\n")
    elapsed = result.elapsed

    numbered = default_numbered_outputs(out_dir, before, adapter.output_glob)
    renamed = map_back(out_dir, numbered, shots)
    for shot in shots:
        path = out_dir / shot.out
        if path.exists():
            print(f"  {path.name}  {path.stat().st_size / 1e6:.1f} MB")

    ok = sum(1 for s in shots if (out_dir / s.out).exists())
    unit = "clip(s)" if not oneshot else "response(s)"
    print(f"batch rc={result.returncode} in {elapsed:.0f}s, "
          f"{ok}/{len(shots)} {unit} written ({renamed} renamed)")

    if args.qc:
        if adapter.judge is None:
            print(f"qc: {adapter.name} has no calibrated quality gate; its "
                  f"failure modes are visible in the output", file=sys.stderr)
        else:
            bad = 0
            for shot in shots:
                path = out_dir / shot.out
                if not path.exists():
                    continue
                verdict = adapter.judge(path, shot)
                print(f"qc: {shot.id}: {verdict.label}"
                      + (f" — {verdict.detail}" if verdict.detail else ""))
                if not verdict.ok:
                    bad += 1
            if bad:
                print(f"qc: {bad} clip(s) failed", file=sys.stderr)
                return 2
    return 0 if ok == len(shots) else 2


if __name__ == "__main__":
    raise SystemExit(main())
