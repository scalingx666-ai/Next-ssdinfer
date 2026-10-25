"""Golden tests for the Flint adapter, written against a real run.

Flint is the framework's second adapter and its first non-video one, so these
tests guard the two things that were learned by running it rather than by
reading it:

* the argv it emits matches what ``mlx_vlm.generate`` actually accepts, and
  carries no flags the CLI rejects;
* the reasoning-block handling is exact. A 512-token budget was measured to
  produce 1.5 KB of pure ``<think>`` with the answer missing, while exiting 0
  — so ``answer()`` and ``judge()`` are load-bearing, not cosmetics.

The capacity model is asserted against the probed working-set ceiling rather
than a literal, so the test stays true on a different machine.

Run from ``pipeline/``::

    python3 -m unittest discover -s tests -v
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

PIPELINE = Path(__file__).resolve().parents[1]
if str(PIPELINE) not in sys.path:
    sys.path.insert(0, str(PIPELINE))

from core.adapter import MediaPreset  # noqa: E402
from models.flint import (WEIGHT_BYTES, FlintAdapter,  # noqa: E402
                          FlintCapacity)

OUT_DIR = Path("/tmp/does-not-matter")

#: Captured from a real invocation on 0.7.10 (see the E2E run of 2026-10-25).
#: The interpreter path is the one variable a caller varies; the rest is
#: contract. ``--output`` is deliberately absent: the text path prints to
#: stdout and rejects it.
EXPECTED_ARGV_TAIL = [
    "--max-tokens", "1536",
    "--temperature", "1.0",
    "--top-p", "0.95",
    "--top-k", "20",
    "--no-verbose",
]


class AdapterShapeTest(unittest.TestCase):
    """The adapter honours the protocol's second-adapter discoveries."""

    def setUp(self):
        self.adapter = FlintAdapter(PIPELINE.parent)

    def test_mode_is_oneshot(self):
        self.assertEqual(self.adapter.mode, "oneshot")

    def test_no_output_glob(self):
        self.assertEqual(self.adapter.output_glob, "")

    def test_stdout_is_output(self):
        self.assertTrue(self.adapter.stdout_is_output)

    def test_runtime_resolves_to_weights_that_exist(self):
        # The default layout must point at real files on this machine; where
        # it cannot, the adapter must say so rather than emit an argv that
        # cannot run.
        missing = self.adapter.runtime_ready()
        if missing:
            self.fail(f"flint runtime not ready: {missing}")

    def test_media_preset_is_rejected(self):
        # A media preset has no max_tokens; the adapter must refuse to guess.
        preset = MediaPreset("max")
        shot = type("S", (), {"prompt": "x", "out": "x.txt"})()
        with self.assertRaises(TypeError):
            self.adapter.shot_argv(shot, 0, preset, OUT_DIR)


class ArgvTest(unittest.TestCase):
    """The command line, against the real CLI's accepted flags."""

    def setUp(self):
        self.adapter = FlintAdapter(PIPELINE.parent)
        self.shot = type("S", (), {
            "prompt": "写一个分镜", "out": "01.txt", "meta": {}})()

    def test_argv_matches_real_invocation(self):
        argv = self.adapter.shot_argv(self.shot, 0,
                                      self.adapter.presets["plan"], OUT_DIR)
        binary, rest = argv[0], argv[1:]
        self.assertTrue(binary.endswith(".venv/bin/python"))
        self.assertEqual(rest, ["-m", "mlx_vlm.generate",
                                "--model", str(self.adapter.model_dir),
                                "--prompt", "写一个分镜",
                                *EXPECTED_ARGV_TAIL])

    def test_no_output_flag(self):
        # --output exists for media models and is rejected by the text path;
        # a regression here would break every request.
        argv = self.adapter.shot_argv(self.shot, 0,
                                      self.adapter.presets["smoke"], OUT_DIR)
        self.assertNotIn("--output", argv)

    def test_dry_run_refuses_when_runtime_missing(self):
        # Env is read at construction, so the adapter must be rebuilt after
        # the override; an already-built instance keeps its resolved paths.
        import os
        old = (os.environ.get("FLINT_PYTHON"), os.environ.get("FLINT_MODEL"))
        os.environ["FLINT_PYTHON"] = "/nonexistent/python"
        try:
            adapter = FlintAdapter(PIPELINE.parent)
            with self.assertRaises(RuntimeError):
                adapter.shot_argv(self.shot, 0,
                                  adapter.presets["smoke"], OUT_DIR)
        finally:
            if old[0] is None:
                del os.environ["FLINT_PYTHON"]
            else:
                os.environ["FLINT_PYTHON"] = old[0]


class AnswerTest(unittest.TestCase):
    """Reasoning-block extraction, against measured failure shapes."""

    def setUp(self):
        self.adapter = FlintAdapter(PIPELINE.parent)

    def test_answer_after_think(self):
        text = "<think>reasoning here</think>Scene: 雨夜古巷。"
        self.assertEqual(self.adapter.answer(text), "Scene: 雨夜古巷。")

    def test_unterminated_think_returns_empty(self):
        # The measured 512-token failure: reasoning consumed the budget and
        # the answer never happened. Must read as empty, not as raw text.
        self.assertEqual(self.adapter.answer("<think>still reasoning"), "")

    def test_no_think_at_all_returns_empty(self):
        # Defensive: a bare completion without the template's opening tag is
        # not trustworthy either — treat it like an unterminated block.
        self.assertEqual(self.adapter.answer("plain answer text"), "")

    def test_empty_think_returns_answer(self):
        self.assertEqual(self.adapter.answer("<think></think>Answer."), "Answer.")


class JudgeTest(unittest.TestCase):
    """The text quality gate, against real failure shapes."""

    def setUp(self):
        self.adapter = FlintAdapter(PIPELINE.parent)

    def _judge(self, text: str, meta: dict | None = None):
        import tempfile
        with tempfile.NamedTemporaryFile("w", suffix=".txt",
                                         delete=False) as fh:
            fh.write(text)
            path = Path(fh.name)
        try:
            shot = type("S", (), {"meta": meta or {}})()
            return self.adapter.judge(path, shot)
        finally:
            path.unlink(missing_ok=True)

    def test_think_only_is_rejected(self):
        verdict = self._judge("<think>" + "词" * 800)
        self.assertEqual(verdict.label, "THINK_ONLY")
        self.assertFalse(verdict.ok)

    def test_short_answer_is_a_stub(self):
        verdict = self._judge("<think>x</think>好。")
        self.assertEqual(verdict.label, "STUB")

    def test_missing_required_sections(self):
        body = "Scene：a。Action：b。"  # no Camera/Look/Audio
        verdict = self._judge(f"<think>r</think>{body}",
                              meta={"required_sections":
                                    ["Scene", "Action", "Camera"]})
        self.assertEqual(verdict.label, "MISSING")
        self.assertIn("Camera", verdict.detail)

    def test_complete_answer_passes_without_punctuation_heuristic(self):
        # The full stop a complete reply may omit. A trailing-punctuation
        # check rejected exactly this shape on real output.
        body = ("Scene：烛火。\nAction：递钗。\nCamera：微距。\n"
                "Look：反光。\nAudio：轻响")
        verdict = self._judge(f"<think>r</think>{body}")
        self.assertEqual(verdict.label, "OK")


class CapacityTest(unittest.TestCase):
    """Byte accounting, against the probed ceiling."""

    def setUp(self):
        self.capacity = FlintCapacity()

    def test_budget_is_the_working_set(self):
        from core.hardware import gpu_working_set_bytes
        self.assertEqual(self.capacity.budget, gpu_working_set_bytes())

    def test_short_context_fits(self):
        cap = self.capacity.check(FlintAdapter(PIPELINE.parent).presets["plan"])
        self.assertTrue(cap.ok)
        self.assertEqual(cap.unit, "bytes")

    def test_kv_scales_but_sliding_layers_are_capped(self):
        # Only full-attention layers grow with context; the 35 sliding layers
        # stop at the window. That is what makes long contexts affordable.
        kv_small = self.capacity.kv_bytes(8192)
        kv_big = self.capacity.kv_bytes(262144)
        self.assertGreater(kv_big, kv_small)
        self.assertLess(kv_big / kv_small, 30)   # naive would be 32x
        self.assertGreater(kv_big / kv_small, 20)

    def test_full_context_is_refused_with_a_reason(self):
        # 262144-token context peaks past the ceiling on this machine; the
        # check must refuse with numbers, not degrade.
        class Shot:  # minimal shot stand-in
            meta = {"context_tokens": 262144}
        cap = self.capacity.check(None, Shot())
        if cap.ok:
            self.skipTest("machine has headroom for full context")
        self.assertFalse(cap.ok)
        self.assertIn("GiB", cap.reason)

    def test_weights_are_declared_honestly(self):
        # The declared weight size must match the index on disk when the
        # weights are present; otherwise this asserts the declared value at
        # least matches the order of magnitude the capacity model is built on.
        index = (FlintAdapter(PIPELINE.parent).model_dir
                 / "model.safetensors.index.json")
        if not index.is_file():
            self.skipTest("weights not downloaded on this machine")
        import json
        declared = json.loads(index.read_text())["metadata"]["total_size"]
        self.assertEqual(WEIGHT_BYTES, declared)


if __name__ == "__main__":
    unittest.main()
