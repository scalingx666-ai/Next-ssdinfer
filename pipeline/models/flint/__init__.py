"""Flint (Recursive Flint, 39.8B MoE) adapter for mlx-vlm on Apple Silicon.

This is the second adapter in this repository, and it exists to answer one
question honestly: **does the framework abstraction hold for a model that is
not a video diffusion model?**

It holds, with two qualifications the H3 adapter could not have surfaced on its
own.

First, the capacity model is a *different quantity*. H3 spends the working set
as DiT tokens against an empirically measured budget of 4480. Flint spends it
as weight bytes against the probed Metal working-set ceiling, and its cost is
*derived from the config* rather than measured — which is exactly what makes
it trustworthy here: the weights on disk are 16.61 GiB against a 21.34 GiB
ceiling, and KV is cheap because only 14 of 49 layers attend over the full
context while the other 35 slide on a 513-token window at 8-bit K/V.

Second, the output needs no calibrated QC. H3's failure mode is a file that
*looks* finished — over budget it still exits rc=0 and writes a playable mp4
of 16px mosaic. Flint's failure mode is visible in the text it emits, so this
adapter ships ``judge = None`` rather than a fake threshold, and the driver
skips QC.

The run mode is ``oneshot``: ``mlx_vlm.generate`` takes a complete argv per
invocation. Weight reload per request is the throughput cost, and for the
smoke-test shape that is the honest price. The sampling parameters are
Recursive's own recommendations, not tuning.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.adapter import Capacity, Preset, Verdict
from core.hardware import gpu_working_set_bytes

# ---------------------------------------------------------------------------
# Config-derived constants. Derived, not measured — see the capacity model.
# ---------------------------------------------------------------------------

#: Byte-for-byte weight size on disk (from model.safetensors.index.json).
WEIGHT_BYTES = 17_837_530_688

#: Parameter count, for the record: 39.785B total, 2.5B activated per token.
PARAMS_TOTAL_B = 39.785
PARAMS_ACTIVE_B = 2.5

#: Sliding-window size for the layers that use one. With only 14 full-attention
#: layers this is what keeps KV cache bounded — the reason a 262144-token
#: context still fits a 21.34 GiB working set.
SLIDING_WINDOW = 513

FULL_ATTN_LAYERS = 14
SLIDING_LAYERS = 35

#: KV cache bytes per token, per layer, at the exported 8-bit K/V:
#: 2 (K and V) x num_key_value_heads (4) x head_dim (128) = 1024 elements,
#: 1 byte each.
KV_BYTES_PER_TOKEN_PER_LAYER = 2 * 4 * 128

#: Sampling parameters from Recursive's own recommendations. Not tuning: they
#: are what the model card asks for, and the adapter's job is to run the
#: model the way its authors expect.
TEMPERATURE = 1.0
TOP_P = 0.95
TOP_K = 20

#: The reasoning block Flint's template puts in front of every answer.
#:
#: Not a toggle: ``--thinking-mode disabled`` is accepted by the CLI and
#: ignored by this model, because the template emits the opening tag
#: unconditionally. Measured on this machine — reasoning costs 400-1500 tokens
#: on a Chinese prompt, which is why the token budgets below are not small.
THINK_OPEN = "<" + "think" + ">"
THINK_CLOSE = "<" + "/think" + ">"

# A trailing-punctuation heuristic was tried here and removed: calibrated
# against real output it rejected a complete five-section plan that simply
# omitted its final full stop. Whether a reply is finished belongs to the
# request (``shot.meta.required_sections``), not to the model layer.

#: Activation + Metal scratch allowance, in bytes. The community 3-bit export
#: reported a peak near 19.5 GB decimal on a 64 GB machine; this is the
#: explicit reservation we gate against rather than a magic constant.
ACTIVATION_RESERVE_BYTES = 1_500_000_000


@dataclass
class LLMPreset(Preset):
    """Sampling knobs. No canvas: a text model has none, which is the point."""

    max_tokens: int = 512
    temperature: float = TEMPERATURE
    top_p: float = TOP_P
    top_k: int = TOP_K

    def describe(self) -> str:
        return (f"{self.name}: max_tokens {self.max_tokens}, "
                f"temp {self.temperature}, top_p {self.top_p}, top_k {self.top_k}"
                + (f" — {self.note}" if self.note else ""))


PRESETS: dict[str, Preset] = {
    # Proves the adapter end to end on one prompt. 512 tokens is enough for a
    # short function or a shot-plan paragraph; more would only spend KV and
    # latency on a first run whose real question is "does it run at all".
    "smoke": LLMPreset(
        "smoke", steps=1,
        max_tokens=1536,
        note="one prompt, 1536 max tokens — room for reasoning plus an answer"),
    # Longer generations for shot planning: a drama shot block with Scene /
    # Action / Camera / Look / Audio sections lands around 200-400 tokens.
    # A 512-token budget was measured to be insufficient: the reasoning block
    # alone consumed it on a Chinese shot-plan prompt. 1536 leaves roughly
    # 400-1000 tokens of headroom for the answer, which is a full five-section
    # plan.
    "plan": LLMPreset(
        "plan", steps=1,
        max_tokens=1536,
        note="one prompt, 1536 max tokens, for shot-plan drafts"),
}


class FlintCapacity:
    """Byte accounting for Flint on this hardware.

    The budget is the probed Metal working-set ceiling. The cost is the weights
    on disk, plus the KV cache for the requested context, plus a fixed
    activation and scratch reserve. Everything except the reserve is derived
    from ``config.json`` and the index file, which is why this model can state
    its numbers rather than discover them by crashing.

    KV is the part worth explaining. A naive reader assumes a 262144-token
    context makes the cache enormous. It does not here, because only 14 of 49
    layers attend over the full window; the other 35 slide on 513 tokens, and
    the exported K/V is 8-bit rather than bf16. Full-context KV is 3.47 GiB,
    and an 8k generation spends 0.13 GiB — so the ceiling is the weights, not
    the context.
    """

    budget = gpu_working_set_bytes()

    @staticmethod
    def kv_bytes(context_tokens: int) -> int:
        """KV cache size for a context length, in bytes.

        Sliding layers are capped at :data:`SLIDING_WINDOW` regardless of the
        request, which is the mechanism that makes long contexts affordable.
        """
        full = FULL_ATTN_LAYERS * KV_BYTES_PER_TOKEN_PER_LAYER * context_tokens
        sliding = (SLIDING_LAYERS * KV_BYTES_PER_TOKEN_PER_LAYER
                   * min(context_tokens, SLIDING_WINDOW))
        return full + sliding

    @classmethod
    def peak_bytes(cls, context_tokens: int = 8192) -> int:
        return WEIGHT_BYTES + cls.kv_bytes(context_tokens) + ACTIVATION_RESERVE_BYTES

    def check(self, preset: Preset, shot: Any = None) -> Capacity:
        """Will this request fit the working set, and if not, by how much.

        The context length comes from the shot's ``meta`` when present, so a
        script can request a longer generation and get an honest refusal
        rather than a late OOM. 8192 is the default assumption for a short
        prompt-and-answer shape.
        """
        context = 8192
        if shot is not None:
            meta = getattr(shot, "meta", None) or {}
            context = int(meta.get("context_tokens", context))

        peak = self.peak_bytes(context)
        budget = self.budget
        if peak > budget:
            over = (peak - budget) / 2**30
            return Capacity(
                peak, budget, False,
                f"{peak / 2**30:.2f} GiB peak exceeds the "
                f"{budget / 2**30:.2f} GiB working set by {over:.2f} GiB at "
                f"{context}-token context. MLX would fail late or swap rather "
                f"than error early, so this is refused. Drop context_tokens "
                f"or use a smaller quantisation.",
                unit="bytes")
        margin = (budget - peak) / 2**30
        detail = (f"{margin:.2f} GiB headroom at {context}-token context "
                  f"(weights {WEIGHT_BYTES / 2**30:.2f} GiB, KV "
                  f"{self.kv_bytes(context) / 2**30:.3f} GiB)")
        if peak > budget * 0.95:
            return Capacity(peak, budget, True, detail, unit="bytes")
        return Capacity(peak, budget, True, "", unit="bytes")


class FlintAdapter:
    """Drives ``mlx_vlm.generate``, one process per prompt."""

    name = "flint"
    presets = PRESETS
    capacity = FlintCapacity()
    mode = "oneshot"
    output_glob = ""          # each request writes its own file; no renaming
    # The completion arrives on stdout, so the driver captures it straight into
    # shot.out. This is the whole reason ``output_glob`` is empty: there is no
    # post-run rename to do because the artifact is named before it exists.
    stdout_is_output = True
    # Filled rather than None: a truncated reasoning block is a real failure
    # that reads as fluent output. See FlintAdapter.judge.

    def __init__(self, repo_root: Path | None = None):
        repo = repo_root or Path(__file__).resolve().parents[2]
        self.repo = repo
        # The weights and the MLX venv do not live in this repository: they
        # are sibling project material, resolved here so the adapter has a
        # default that works on the machine it was written on. Both are
        # overridable, because a layout that only works when the two
        # repositories are siblings is a convention, not a contract.
        self.cwd = self._resolve_root(repo)
        self.binary = Path(os.environ.get(
            "FLINT_PYTHON", str(self.cwd / ".venv" / "bin" / "python")))
        self.model_dir = Path(os.environ.get(
            "FLINT_MODEL", str(self.cwd / "models" / "flint-mixed-3-4bit")))

    @staticmethod
    def _resolve_root(repo: Path) -> Path:
        """Where the Flint runtime lives, checked not assumed.

        ``repo/ai-drama`` first, then the sibling layout used on this
        machine. Falling back to ``repo`` itself is the wrong default — the
        capacity check would pass against a budget the files never load —
        so the missing-runtime case is surfaced by :meth:`check` instead of
        being papered over with a plausible path.
        """
        for candidate in (repo / "ai-drama", repo.parent / "ai-drama"):
            if candidate.is_dir() and (candidate / ".venv").is_dir():
                return candidate
        return repo

    # -- capacity ---------------------------------------------------------

    def runtime_ready(self) -> str:
        """A description of what is missing, or ``""`` when nothing is.

        Deliberately not folded into ``CapacityModel``: the working-set
        question and the "are the files here" question fail differently, and
        mixing them would make one message lie about the other.
        """
        if not self.binary.is_file():
            return f"python interpreter not found: {self.binary} (set FLINT_PYTHON)"
        if not self.model_dir.is_dir():
            return (f"weights not found: {self.model_dir} (set FLINT_MODEL, "
                    f"see ai-drama/models/flint-mixed-3-4bit)")
        if not (self.model_dir / "config.json").is_file():
            return f"weights present but incomplete: {self.model_dir}"
        return ""

    # -- command stream ---------------------------------------------------

    def extra_args(self) -> list[str]:
        # No global flags: every knob is per request in oneshot mode.
        return []

    def header(self, preset: Preset, out_dir: Path) -> list[str]:
        # Nothing global to establish: there is no resident session to
        # configure, and the output path is per request.
        return []

    def shot_argv(self, shot: Any, index: int, preset: Preset,
                  out_dir: Path) -> list[str]:
        """Full argv for one ``mlx_vlm.generate`` invocation.

        Flags verified against ``mlx_vlm.generate --help`` on 0.7.10 and
        against a real run. Three natural guesses the CLI rejects:
        ``--extra-eos-token`` (it is ``--eos-tokens``, nargs+), ``--verbose
        False`` (it is a flag pair; ``--no-verbose`` is the off switch), and
        ``--out`` (it is ``--output``).

        ``--output`` is not passed: on the text path ``mlx_vlm.generate``
        prints the completion to stdout and leaves ``--output`` to the media
        models. So the artifact is captured from stdout instead, which is what
        ``stdout_is_output = True`` tells the driver.
        """
        if not hasattr(preset, "max_tokens"):
            raise TypeError(
                f"{preset.name!r} is not an LLM preset; Flint cannot render "
                f"a canvas. Use one of {', '.join(self.presets)}.")
        if missing := self.runtime_ready():
            raise RuntimeError(
                f"flint runtime is not ready: {missing}. A dry run refuses to "
                f"print a command that cannot be executed.")
        return [
            str(self.binary), "-m", "mlx_vlm.generate",
            "--model", str(self.model_dir),
            "--prompt", shot.prompt,
            "--max-tokens", str(preset.max_tokens),
            "--temperature", str(preset.temperature),
            "--top-p", str(preset.top_p),
            "--top-k", str(preset.top_k),
            "--no-verbose",
        ]

    def shot_lines(self, shot: Any, index: int, preset: Preset) -> list[str]:
        # Not used in oneshot mode; required by the protocol.
        return [shot.prompt]

    def footer(self) -> list[str]:
        return []

    # -- output validation ------------------------------------------------

    @staticmethod
    def answer(text: str) -> str:
        """Everything after the reasoning block — the part anyone wants.

        Flint's chat template opens every assistant turn with ``<think>`` and
        offers no switch to suppress it: ``--thinking-mode disabled`` is
        accepted by the CLI and has no effect here, because the template
        hardcodes the opening token rather than consulting a flag. So the
        reasoning is not an option to disable but a prefix to strip, and
        stripping it is model knowledge, which is why it lives in this file
        and not in the core layer.

        An unterminated block returns ``""`` deliberately. That case means
        the token budget was spent entirely on reasoning and no answer was
        ever produced — the text-model analogue of an undenoised latent, and
        the reason a byte-size check is not enough.
        """
        if THINK_CLOSE not in text:
            return ""
        return text.split(THINK_CLOSE, 1)[1].strip()

    def is_complete(self, path: Path, shot: Any) -> bool:
        """True only when a real answer followed the reasoning block.

        Measured, not assumed: a 512-token budget on a Chinese shot-plan
        prompt produced 1921 bytes of pure ``<think>`` and no answer at all,
        while exiting 0. A size check passes that file. Requiring text after
        ``</think>`` does not.
        """
        try:
            text = path.read_text(errors="replace")
        except OSError:
            return False
        return len(self.answer(text)) >= 16

    def judge(self, path: Path, shot: Any = None) -> Verdict:
        """Text quality gate: did an answer survive the reasoning budget.

        ``judge`` was made optional in the protocol because a text model's
        failures looked self-evident. Flint disproved that gently: its most
        common failure is a file that reads as a wall of fluent reasoning with
        the actual answer missing, which is not self-evident at all when the
        output is 2 KB of plausible Chinese. So this adapter fills the field
        rather than leaving it ``None`` — the option was right, the assumption
        behind it was not.

        What is deliberately *not* here: a punctuation heuristic for
        "did the reply finish". Calibrated against real output, it is a false
        positive generator — a complete five-section plan that omits the
        trailing full stop gets rejected, and a truncated one that happens to
        end on a full stop passes. Whether a reply is finished is a property
        of the request, so it lives on the shot as ``meta.required_sections``
        and is checked below; the model layer only answers what it can know.
        """
        try:
            text = path.read_text(errors="replace")
        except OSError as exc:
            return Verdict("UNREADABLE", str(exc))
        body = self.answer(text)
        stats = {"chars_total": float(len(text)), "chars_answer": float(len(body))}
        if not body:
            return Verdict(
                "THINK_ONLY",
                f"{len(text)} chars of reasoning and no answer: max_tokens ran "
                f"out inside the think block, which still exits 0. Raise "
                f"max_tokens or shorten the prompt.",
                stats)
        if len(body) < 16:
            return Verdict("STUB", f"answer is only {len(body)} chars", stats)
        required = (getattr(shot, "meta", None) or {}).get("required_sections")
        if required:
            missing = [s for s in required if s not in body]
            if missing:
                return Verdict(
                    "MISSING",
                    f"answer omits required section(s): {', '.join(missing)}. "
                    f"The request asked for them; the reply did not deliver.",
                    stats)
        return Verdict("OK", "", stats)


def load_adapter(repo_root: Path | None = None) -> FlintAdapter:
    """Entry point the registry uses."""
    return FlintAdapter(repo_root)


def read_config(model_dir: Path | None = None) -> dict[str, Any]:
    """Flint's ``config.json``, for tests that want to assert against the
    source of the derived constants rather than the constants themselves."""
    directory = model_dir or Path(
        os.environ.get("FLINT_MODEL", "models/flint-mixed-3-4bit"))
    return json.loads((directory / "config.json").read_text())
