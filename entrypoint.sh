#!/usr/bin/env bash
set -euo pipefail

GPU_DEVICE="${GPU_DEVICE:-0}"

VLLM_PORT="${PORT:-8000}"
EMBD_PORT="${EMBD_PORT:-8002}"

VLLM_HEALTH="http://localhost:${VLLM_PORT}/health"
EMBD_HEALTH="http://localhost:${EMBD_PORT}/health"
POLL_INTERVAL=10

echo "Starting olmOCR vLLM server on port ${VLLM_PORT} (GPU ${GPU_DEVICE})..."
make vllm-olmocr-serve GPU_DEVICE="${GPU_DEVICE}" &
VLLM_PID=$!

echo "Starting embeddings vLLM server on port ${EMBD_PORT} (GPU ${GPU_DEVICE})..."
make vllm-embd-serve GPU_DEVICE="${GPU_DEVICE}" &
EMBD_PID=$!

wait_for_health() {
    local name="$1"
    local url="$2"
    local pid="$3"

    echo "Waiting for ${name} to become ready at ${url}..."
    until curl -sf "${url}" > /dev/null 2>&1; do
        if ! kill -0 "${pid}" 2>/dev/null; then
            echo "ERROR: ${name} process exited unexpectedly" >&2
            exit 1
        fi
        echo "  ${name} not ready yet, retrying in ${POLL_INTERVAL}s..."
        sleep "${POLL_INTERVAL}"
    done
    echo "${name} is ready."
}

wait_for_health "olmOCR vLLM" "${VLLM_HEALTH}" "${VLLM_PID}"
wait_for_health "embeddings vLLM" "${EMBD_HEALTH}" "${EMBD_PID}"

echo "All servers ready. Starting API server..."
exec make start-server
