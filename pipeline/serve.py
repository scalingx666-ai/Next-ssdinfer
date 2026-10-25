#!/usr/bin/env python3
"""Run a model adapter's OpenAI-compatible server.

Serving is an adapter capability, not a driver one: only the adapter knows
which binary speaks HTTP, which flags select the weights, and what it calls
itself in ``/v1/models``. This file only decides the two things a driver
should decide — where it listens, and whether it demands a key.

Two defaults are deliberate and opinionated:

**Loopback.** ``--host`` defaults to ``127.0.0.1``. A local model server on
``0.0.0.0`` is an unauthenticated LLM on whatever network the laptop joins.

**A key always exists.** With no key supplied, one is generated and printed
once rather than the server starting open. An empty key would be a silent
downgrade to no auth, and nobody reads the absence of a prompt as a decision.

Usage:
    python3 serve.py --model flint [--host H] [--port P] [--api-key K]

Clients then use:
    base_url = http://127.0.0.1:8080/v1
    api_key  = <printed key>
    model    = the id printed at startup  (NOT a short alias)
"""

from __future__ import annotations

import argparse
import os
import secrets
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.registry import load  # noqa: E402


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="adapter name, e.g. flint")
    ap.add_argument("--host", default="127.0.0.1",
                    help="bind address (default 127.0.0.1, loopback only)")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--api-key", default=None,
                    help="bearer token clients must send; generated if omitted")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the resolved argv and the model id, then exit")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    try:
        adapter = load(args.model, ROOT.parent)
    except KeyError as exc:
        print(f"serve.py: {exc.args[0]}", file=sys.stderr)
        return 1

    if not hasattr(adapter, "serve_argv"):
        print(f"serve.py: adapter {adapter.name!r} has no OpenAI-compatible "
              f"server. Adapters that do: flint.", file=sys.stderr)
        return 1

    key = args.api_key or os.environ.get("FLINT_API_KEY") or ""
    generated = ""
    if not key:
        key = "sk-" + secrets.token_urlsafe(24)
        generated = key

    try:
        argv = adapter.serve_argv(host=args.host, port=args.port, api_key=key)
    except RuntimeError as exc:
        print(f"preflight: {exc}", file=sys.stderr)
        return 1

    # The model id is what /v1/models will report, and OpenAI clients must
    # send it back verbatim. Printing the resolved path here saves a round
    # trip through curl to find out.
    model_id = str(adapter.model_dir)
    print(f"adapter    {adapter.name}")
    print(f"model id   {model_id}")
    print(f"base_url   http://{args.host}:{args.port}/v1")
    print(f"api_key    {'(generated) ' if generated else ''}{key}")
    print(f"argv       {' '.join(argv)}")
    if generated:
        print("\n上面这把 key 只在本进程生命周期内有效，重启会重新生成；"
              "要固定就设 FLINT_API_KEY 或传 --api-key。")
    sys.stdout.flush()

    if args.dry_run:
        return 0

    # exec, not subprocess: signals (Ctrl-C) and exit status belong to the
    # server, and a wrapper that swallows them makes a stuck load unkillable.
    os.execv(argv[0], argv)
    return 0   # unreachable


if __name__ == "__main__":
    raise SystemExit(main())
