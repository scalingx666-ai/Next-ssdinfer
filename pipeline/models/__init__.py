"""Model adapters.

Each subdirectory is one model family. An adapter is the only place that knows
model-specific facts: preset ladders, token arithmetic, command syntax,
capacity budgets, and calibrated QC thresholds.

To add a model, create ``models/<name>/__init__.py`` exposing ``load_adapter()``
and register it in ``registry.py``. Nothing in ``core/`` should change.
"""
