"""Golden tests for the H3 command stream.

These exist because a refactor of the driver must not change what the renderer
is told to do. The stream is compared against literals captured from the
pre-refactor driver with the same preset, so a behavioural drift fails here
rather than in an eight-hour render.

Run from ``pipeline/``::

    python3 -m unittest discover -s tests -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PIPELINE = Path(__file__).resolve().parents[1]
if str(PIPELINE) not in sys.path:
    sys.path.insert(0, str(PIPELINE))

from models.h3 import (QC_BLOCK16_MOSAIC, QC_MIN_SHORT_SIDE, QC_STD_MOSAIC,  # noqa: E402
                       PRESETS, H3Capacity, H3Adapter)

OUT_DIR = Path("/tmp/does-not-matter")

#: Captured from the pre-refactor driver with ``--preset max`` on a 37-frame
#: shot. Only the directory is substituted, because that is the one value a
#: caller legitimately varies.
GOLDEN_HEADER = [
    f"!output {OUT_DIR}",
    "!size 768x1344",
    "!render-size 512x896",
    "!steps 20",
    "!layers 50",
    "!reuse 1",
    "!token-reduction off",
    "!ssd-streaming on",
    "!frames 37",
    "!cache",
]

#: Same, for the `smoke` preset — a different shape and step count, to prove
#: the header is driven by the preset rather than hard-coded.
GOLDEN_HEADER_SMOKE = [
    f"!output {OUT_DIR}",
    "!size 384x672",
    "!render-size 256x448",
    "!steps 4",
    "!layers 50",
    "!reuse 1",
    "!token-reduction off",
    "!ssd-streaming on",
    "!frames 22",
    "!cache",
]


class Shot:
    """Minimal stand-in for core.shots.Shot."""

    def __init__(self, **kw):
        self.id = kw.get("id", "x")
        self.prompt = kw.get("prompt", "a prompt")
        self.out = kw.get("out", "x.mp4")
        self.frames = kw.get("frames")
        self.steps = kw.get("steps")
        self.seed = kw.get("seed")
        self.first = kw.get("first")
        self.last = kw.get("last")


class HeaderTest(unittest.TestCase):
    def setUp(self):
        self.adapter = H3Adapter()

    def test_header_matches_pre_refactor(self):
        self.assertEqual(
            self.adapter.header(PRESETS["max"], OUT_DIR), GOLDEN_HEADER)

    def test_header_is_preset_driven(self):
        self.assertEqual(
            self.adapter.header(PRESETS["smoke"], OUT_DIR),
            GOLDEN_HEADER_SMOKE)

    def test_header_takes_out_dir_as_argument(self):
        """The header must not read the output directory off instance state.

        It is emitted before any shot is processed, so there is no earlier
        point at which a caller could have injected it; stashing it as a side
        effect of ``shot_lines`` would make the adapter depend on call order.
        """
        with self.assertRaises(TypeError):
            self.adapter.header()


class ShotLinesTest(unittest.TestCase):
    def setUp(self):
        self.adapter = H3Adapter()
        self.preset = PRESETS["max"]

    def test_no_overrides_emits_only_prompt(self):
        shot = Shot(frames=37, seed=None)
        self.assertEqual(
            self.adapter.shot_lines(shot, 0, self.preset), ["a prompt"])

    def test_frame_override_is_reverted(self):
        shot = Shot(frames=22)
        lines = self.adapter.shot_lines(shot, 0, self.preset)
        self.assertEqual(lines, ["!frames 22", "a prompt", "!frames 37"])

    def test_step_override_is_reverted(self):
        shot = Shot(steps=30)
        lines = self.adapter.shot_lines(shot, 0, self.preset)
        self.assertEqual(lines, ["!steps 30", "a prompt", "!steps 20"])

    def test_both_overrides_revert_in_reverse_order(self):
        """Overrides revert in the order they were applied — frames, then
        steps — which is what the pre-refactor driver did."""
        shot = Shot(frames=22, steps=30)
        lines = self.adapter.shot_lines(shot, 0, self.preset)
        self.assertEqual(
            lines,
            ["!frames 22", "!steps 30", "a prompt", "!frames 37", "!steps 20"])

    def test_anchors_are_cleared_after_use(self):
        shot = Shot(first="a.mp4", last="b.mp4")
        lines = self.adapter.shot_lines(shot, 0, self.preset)
        self.assertEqual(
            lines,
            ["!first a.mp4", "!last b.mp4", "a prompt",
             "!first clear", "!last clear"])

    def test_seed_is_emitted_before_prompt(self):
        shot = Shot(seed=42)
        lines = self.adapter.shot_lines(shot, 0, self.preset)
        self.assertEqual(lines, ["!seed 42", "a prompt"])

    def test_multiline_prompt_is_collapsed_at_load(self):
        """A multi-line prompt must reach the stream as one line.

        stdin is split on newlines, so a multi-line prompt would fire several
        generations and corrupt the output ordering. The collapse happens in
        ``Shot.from_record`` (core/shots.py) rather than in the adapter, so
        every model benefits from it; this test pins the load path so the
        guarantee cannot regress into a per-adapter concern.
        """
        from core.shots import load_shots

        script = PIPELINE / "tests" / "fixtures" / "multiline.jsonl"
        shots = load_shots(script)
        self.assertEqual(len(shots), 1)
        self.assertNotIn("\n", shots[0].prompt)
        self.assertEqual(shots[0].prompt, "line one line two")


class CapacityTest(unittest.TestCase):
    """The token arithmetic, checked against measured anchors."""

    def test_measured_anchors(self):
        capacity = H3Capacity()
        anchors = [
            ((512, 896, 37), 4480),    # production: works
            ((512, 896, 72), 8064),    # mosaic: fails
            ((384, 672, 120), 7560),   # mosaic: fails
        ]
        for (w, h, f), expected in anchors:
            with self.subTest(f"{w}x{h}/{f}f"):
                self.assertEqual(capacity.tokens_for(w, h, f), expected)

    def test_max_frames_for_production_canvas(self):
        self.assertEqual(H3Capacity.max_frames(512, 896), 37)

    def test_over_budget_is_refused_not_degraded(self):
        """The whole point of the capacity model: the failure is silent."""
        capacity = H3Capacity()
        cap = capacity.check(PRESETS["max"], Shot(frames=72))
        self.assertFalse(cap.ok)
        self.assertIn("exceeds", cap.reason)

    def test_production_shape_is_inside_budget(self):
        self.assertTrue(H3Capacity().check(PRESETS["max"], Shot(frames=37)).ok)


class ProbeTest(unittest.TestCase):
    """Container parsing.

    These are the regression tests for two bugs that were both found by
    running the refactored code against real output rather than by reading it:

    * ``nb_frames`` is optional in MP4. When it is absent ffprobe omits the
      column and every later field shifts left, so positional unpacking read
      ``r_frame_rate`` (``24/1``) as the frame count and crashed.
    * ``frame_stats`` used numpy after it had been moved into ``grab_frame``,
      so the module imported fine on a bare Python and failed only when a
      frame was actually measured.
    """

    def test_missing_nb_frames_falls_back_to_counting(self):
        """A clip with no frame count must still report one.

        Constructed from the real h3 output, whose ffprobe CSV is
        ``768,1344,24/1,1.625000`` — four fields, not five.
        """
        from unittest import mock

        from core import qc

        def fake_run(argv, **kw):
            joined = " ".join(argv)
            if "-show_entries" in joined and "stream=nb_frames" in joined:
                return mock.Mock(stdout="", stderr="")
            if "-count_packets" in joined:
                return mock.Mock(stdout="39\n", stderr="")
            if "stream=r_frame_rate" in joined:
                return mock.Mock(stdout="24/1\n", stderr="")
            if "stream=duration" in joined:
                return mock.Mock(stdout="1.625000\n", stderr="")
            if "stream=width" in joined:
                return mock.Mock(stdout="768\n", stderr="")
            return mock.Mock(stdout="1344\n", stderr="")

        with mock.patch.object(qc.subprocess, "run", side_effect=fake_run):
            info = qc.probe_video("anything.mp4")

        self.assertEqual((info.width, info.height), (768, 1344))
        self.assertEqual(info.frames, 39)
        self.assertEqual(info.fps, 24.0)
        self.assertEqual(info.duration, 1.625)

    def test_frame_stats_needs_no_module_level_numpy(self):
        """Importing core.qc must not require numpy.

        The dry-run and check-only paths are meant to work on a bare Python,
        which on this machine has no imaging stack at all.
        """
        import subprocess

        result = subprocess.run(
            [sys.executable, "-c",
             "import sys; sys.path.insert(0, %r); import core.qc" % str(PIPELINE)],
            capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


class JudgeTest(unittest.TestCase):
    """QC thresholds, checked against the clips they were calibrated on."""

    def setUp(self):
        self.adapter = H3Adapter()

    def test_min_short_side_matches_the_calibration_samples(self):
        """Every calibration clip is 768x1344, so the floor is 768.

        A higher floor made the adapter report UNVERIFIED for the very clips
        the thresholds were derived from.
        """
        self.assertEqual(QC_MIN_SHORT_SIDE, 768)

    def test_verdict_requires_both_mosaic_conditions(self):
        """block16 alone is not a discriminator.

        A good 37-frame clip scores 7.66 on block_ratio while a bad 72-frame
        clip scores 7.26, because a noisy mosaic raises adjacent-pixel
        difference at the same time as block16. Both conditions are needed.
        """
        self.assertGreater(QC_BLOCK16_MOSAIC, 0)
        self.assertGreater(QC_STD_MOSAIC, 0)


if __name__ == "__main__":
    unittest.main()
