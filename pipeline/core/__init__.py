"""Engine-agnostic primitives for the batch render pipeline.

Nothing in this package knows about H3, Metal, video diffusion, or tokenisers.
It holds the things that are true for *any* local generator driven by a batch
script:

  * Shot            one generation request (prompt + shape + identity)
  * Preset          the universal knobs; MediaPreset adds a diffusion canvas,
                    an LLM preset adds sampling parameters
  * Capacity        "will this fit" asked before the run, with a reason
  * Session         a long-lived renderer process driven over stdin/stdout,
                    or one process per request when that is all the model
                    offers
  * resume          idempotent re-entry after a crash, without redoing work
  * hardware        the measured working-set ceiling every adapter shares
  * QC statistics   frame-level measurements that separate content from noise

Model-specific knowledge — canvas ladders, token budgets, byte budgets,
failure thresholds, streaming semantics, sampling defaults — lives in
``models/<name>/`` behind the ``EngineAdapter`` protocol. Keeping it there is
deliberate: the measured limits are the most valuable thing in this repository,
and burying them in a "generic model framework" would destroy that value.

The split was tested by writing the second adapter, not by asserting it. Flint
is a 39.8B MoE text model: no canvas, no clips, and a cost measured in bytes
rather than tokens. Two things had to move out of the way — ``Preset`` grew a
``MediaPreset`` subclass, and ``judge`` became optional rather than required.
Both are recorded in ``core/adapter.py``.
"""

from .adapter import (Capacity, EngineAdapter, Judge, MediaPreset, Preset,
                      Verdict)
from .hardware import gpu_working_set_bytes, gpu_working_set_gib
from .shots import Shot, load_shots, filter_shots
from .session import (Session, CommandStream, build_argv_list, run_oneshot)
from .resume import ResumeState, adopt_stray_outputs, partition_shots
from .qc import frame_stats, Probe, probe_video

__all__ = [
    "Capacity", "EngineAdapter", "Judge", "MediaPreset", "Preset", "Verdict",
    "gpu_working_set_bytes", "gpu_working_set_gib",
    "Shot", "load_shots", "filter_shots",
    "Session", "CommandStream", "build_argv_list", "run_oneshot",
    "ResumeState", "adopt_stray_outputs", "partition_shots",
    "frame_stats", "Probe", "probe_video",
]
