#!/usr/bin/env bash
# Start the MCP server in the background, then run the client.
# Both use the 'neg' conda environment.
# Usage: bash mcp_version/run.sh

set -e

DIR="$(cd "$(dirname "$0")" && pwd)"
CONDA_ENV="neg"

echo "[run.sh] Starting server..."
conda run -n "$CONDA_ENV" --no-capture-output python "$DIR/server.py" &
SERVER_PID=$!

# Wait for server to be ready
for i in $(seq 1 60); do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "[run.sh] Server exited unexpectedly."
        exit 1
    fi
    if curl -s -o /dev/null http://localhost:8765/sse 2>/dev/null; then
        echo "[run.sh] Server ready (pid=$SERVER_PID)."
        break
    fi
    sleep 0.5
done

# Run client; kill server on exit
trap "echo '[run.sh] Stopping server...'; kill $SERVER_PID 2>/dev/null; wait $SERVER_PID 2>/dev/null" EXIT

echo "[run.sh] Starting client..."
conda run -n "$CONDA_ENV" --no-capture-output python "$DIR/client.py"
