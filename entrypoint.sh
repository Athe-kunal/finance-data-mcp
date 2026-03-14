#!/usr/bin/env bash
set -euo pipefail

VLLM_PORT="${PORT:-8000}"
VLLM_HEALTH="http://localhost:${VLLM_PORT}/health"
POLL_INTERVAL=10

echo "Starting vllm server on port ${VLLM_PORT}..."
make vllm-olmocr-serve &
VLLM_PID=$!

echo "Waiting for vllm to become ready at ${VLLM_HEALTH}..."
until curl -sf "${VLLM_HEALTH}" > /dev/null 2>&1; do
    if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
        echo "ERROR: vllm process exited unexpectedly" >&2
        exit 1
    fi
    echo "  vllm not ready yet, retrying in ${POLL_INTERVAL}s..."
    sleep "${POLL_INTERVAL}"
done

echo "vllm is ready. Starting API server..."
exec make start-server
