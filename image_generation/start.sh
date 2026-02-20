#!/bin/bash
# Start both Flux and MCP servers in their respective conda environments.
# Usage: bash image_generation/start.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

conda run -n py311 --no-capture-output python "$SCRIPT_DIR/flux_server.py" &
FLUX_PID=$!
conda run -n neg --no-capture-output python "$SCRIPT_DIR/server.py" 
MCP_PID=$!



kill $FLUX_PID $MCP_PID 2>/dev/null
wait
