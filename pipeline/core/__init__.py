"""Engine-agnostic primitives for the batch render pipeline.

Nothing in this package knows about H3, Metal, or video diffusion. It holds
the four things that are true for *any* local generator driven by a batch
script:

  * Shot            one generation request (prompt + shape + identity)
  * Session         a long-lived renderer process driven over stdin/stdout
  * resume          idempotent re-entry after a crash, without redoing work
  * QC statistics   frame-level measurements that separate content from noise

Model-specific knowledge — canvas ladders, token budgets, failure
thresholds, streaming semantics — lives in ``models/<name>/`` behind the
``EngineAdapter`` protocol. Keeping it there is deliberate: the measured
limits are the most valuable thing in this repository, and burying them in a
"generic video framework" would destroy that value.
"""

from .shots import Shot, load_shots, filter_shots
from .session import Session, CommandStream
from .resume import ResumeState, adopt_stray_outputs, partition_shots
from .qc import frame_stats, Probe, probe_video

__all__ = [
    "Shot", "load_shots", "filter_shots",
    "Session", "CommandStream",
    "ResumeState", "adopt_stray_outputs", "partition_shots",
    "frame_stats", "Probe", "probe_video",
]
