#!/usr/bin/env bash
# Run dataset curation agent, looping through classes.json.
# Usage:
#   bash .mcp_version/run.sh                  # run all categories
#   bash .mcp_version/run.sh anti_aesthetics  # run one category

set -e

DIR="$(cd "$(dirname "$0")" && pwd)"
CONDA_ENV="neg"

echo "[run.sh] Starting agent SDK runner..."
conda run -n "$CONDA_ENV" --no-capture-output python "$DIR/agent_sdk_runner.py" "$@"
