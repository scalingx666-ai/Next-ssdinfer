"""Adapter registry.

The CLI takes ``--model <name>`` and resolves it here. Registration is
deliberately a plain dict rather than entry points: this project has two
models, one of which exists to prove the abstraction, and a plugin system
would still be speculation about a third.
"""

from __future__ import annotations

import sys
from pathlib import Path

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.adapter import EngineAdapter  # noqa: E402


def load(name: str, repo_root: Path | None = None) -> EngineAdapter:
    """Instantiate an adapter by name."""
    if name == "h3":
        from .h3 import load_adapter
        return load_adapter(repo_root)
    if name == "flint":
        from .flint import load_adapter
        return load_adapter(repo_root)
    raise KeyError(
        f"unknown model adapter {name!r}. available: {', '.join(available())}")


def available() -> list[str]:
    return ["h3", "flint"]
