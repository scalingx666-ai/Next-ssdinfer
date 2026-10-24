#!/bin/sh
# Fetch the MiniMax H3 FL2VA weights (~134 GB).
#
# They are intentionally NOT in this repository. ModelScope is used because
# huggingface.co and github.com are unreachable from mainland China without a
# proxy; ModelScope must be reached with all proxy variables cleared.
#
# Usage:
#   H3_MODEL_DIR=/data/models ./scripts/fetch_model.sh
#
# After it finishes, point the pipeline at the result:
#   export H3_MODEL="$H3_MODEL_DIR/MiniMax-H3/snapshots/master"

set -eu

MODEL_DIR="${H3_MODEL_DIR:-$PWD/models}"
REPO="MiniMax/MiniMax-H3"
SNAP="$MODEL_DIR/MiniMax-H3/snapshots/master"

if [ -d "$SNAP" ] && [ -n "$(ls -A "$SNAP" 2>/dev/null)" ]; then
    echo "already present: $SNAP"
    exit 0
fi

command -v modelscope >/dev/null 2>&1 || {
    echo "modelscope CLI not found. Install with: pip install modelscope" >&2
    exit 1
}

# Proxy variables must be cleared: a stale local proxy makes ModelScope
# unreachable, and a live one is unnecessary for it.
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY

mkdir -p "$MODEL_DIR"
echo "downloading $REPO -> $SNAP (this is ~134 GB, expect hours)"
modelscope download --model "$REPO" --local_dir "$MODEL_DIR/MiniMax-H3"

echo
echo "done. now run:"
echo "  export H3_MODEL=\"$SNAP\""
