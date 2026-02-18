#!/bin/bash
# Start both Flux and MCP servers in their respective conda environments.
# Usage: bash image_generation/start.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "Starting Flux server (conda: py311)..."
conda run -n py311 --no-capture-output python "$SCRIPT_DIR/flux_server.py" &
FLUX_PID=$!

echo "Waiting for Flux server to be ready..."
for i in $(seq 1 60); do
    if curl -s http://127.0.0.1:5001/health > /dev/null 2>&1; then
        echo "Flux server ready (PID: $FLUX_PID)"
        break
    fi
    sleep 2
done

echo "Starting MCP server (conda: neg)..."
conda run -n neg --no-capture-output uv run "$SCRIPT_DIR/server.py" &
MCP_PID=$!
echo "MCP server started (PID: $MCP_PID)"

echo ""
echo "Both servers running:"
echo "  Flux: http://127.0.0.1:5001 (PID: $FLUX_PID)"
echo "  MCP:  http://127.0.0.1:8000 (PID: $MCP_PID)"
echo ""
echo "Press Ctrl+C to stop both servers."

trap "echo 'Stopping servers...'; kill $FLUX_PID $MCP_PID 2>/dev/null; exit" INT TERM
wait
