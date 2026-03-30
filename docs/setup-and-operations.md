# Setup and Operations Guide

This guide complements `README.md` and focuses on setup and runtime commands.

## 1) Read README.md first

Start with the root `README.md` for project context, environment variables, and architecture notes.

## 2) Installation

Use `uv` for dependency management.

```bash
uv sync
```

For OCR + embeddings workflows:

```bash
uv sync --group ocr-md
```

For MCP workflows:

```bash
uv sync --group ocr-md --group mcp
```

## 3) Docker setup

Use Makefile targets:

```bash
make docker-build
make docker-run
```

Or run both:

```bash
make docker-start
```

Stop and clean containers on the API port:

```bash
make docker-stop
make docker-remove
```

Example with overrides:

```bash
make docker-start GPU_DEVICE=0 API_PORT=8081 IMAGE_NAME=sec-filings-md
```

## 4) Start model servers

### olmOCR server

```bash
make vllm-olmocr-serve
```

### Embedding server (Qwen pooling)

```bash
make vllm-embd-serve
```

> Requirement: Use a GPU with **at least 12 GB VRAM** for local olmOCR + Qwen embedding serving.

Optional benchmark:

```bash
make guidellm-benchmark
```

## 5) Start API server

```bash
make start-server
```

This runs Uvicorn for `server:app` on `0.0.0.0` using `API_PORT`.

## 6) MCP server and ngrok

Install MCP dependencies:

```bash
uv sync --group ocr-md --group mcp
```

Start MCP server:

```bash
uv run --group ocr-md --group mcp python mcp_server.py
```

Expose with ngrok (replace port if needed):

```bash
ngrok http 8069
```

Then:

1. Copy the public ngrok hostname.
2. Add it to `MCP_NGROK_ALLOWED_HOSTS` in `.env` as a JSON array.
3. Restart `mcp_server.py`.
4. Connect MCP client to `https://<ngrok-hostname>/mcp`.

Example:

```bash
MCP_NGROK_ALLOWED_HOSTS='["example-name.ngrok-free.app"]'
```

## 7) Makefile command reference

- `make vllm-olmocr-serve`: Start olmOCR model server.
- `make vllm-embd-serve`: Start embedding model server.
- `make start-server`: Start FastAPI service.
- `make guidellm-benchmark`: Run throughput benchmark.
- `make docker-build`: Build Docker image.
- `make docker-run`: Run Docker container.
- `make docker-start`: Build and run Docker.
- `make docker-stop`: Stop container exposing `API_PORT`.
- `make docker-remove`: Remove container exposing `API_PORT`.

## 8) Makefile arguments explained

### Shared Makefile variables

| Variable | Used by | Meaning |
|---|---|---|
| `MODEL` | `vllm-olmocr-serve` | olmOCR model id served by vLLM. |
| `EMBD_MODEL` | `vllm-embd-serve` | Embedding model id for pooling runner. |
| `GPU_DEVICE` | vLLM + Docker targets | GPU index (via `CUDA_VISIBLE_DEVICES` or env passthrough). |
| `SERVER` | vLLM targets | Host for vLLM server bind. |
| `PORT` | `vllm-olmocr-serve`, benchmark | olmOCR vLLM port. |
| `EMBD_PORT` | `vllm-embd-serve` | Embedding vLLM port. |
| `API_PORT` | API + Docker lifecycle | FastAPI host port and Docker publish filter. |
| `IMAGE_NAME` | Docker targets | Docker image name/tag. |

### `make vllm-olmocr-serve` flags

| vLLM argument | Makefile source | Meaning |
|---|---|---|
| `--gpu-memory-utilization` | `GPU_MEMORY_UTILIZATION` | Fraction of visible GPU memory reserved by vLLM. |
| `--max-model-len` | `MAX_MODEL_LEN` | Maximum context length accepted by the server. |
| `--tensor-parallel-size` | `TENSOR_PARALLEL_SIZE` | Number of GPUs for tensor parallelism. |
| `--data-parallel-size` | `DATA_PARALLEL_SIZE` | Number of data-parallel replicas. |
| `--max-num-batched-tokens 65536` | fixed in Makefile | Max total batched tokens per scheduling step. |
| `--mm-encoder-tp-mode "data"` | fixed in Makefile | Multi-modal encoder parallel mode. |
| `--limit-mm-per-prompt '{"video": 0}'` | fixed in Makefile | Disables video inputs for prompts. |
| `--max-num-seqs 8192` | fixed in Makefile | Max sequences concurrently tracked by scheduler. |
| `--port` / `--host` | `PORT` / `SERVER` | vLLM bind address and port. |

### `make vllm-embd-serve` flags

| vLLM argument | Makefile source | Meaning |
|---|---|---|
| `--gpu-memory-utilization` | `EMBD_GPU_MEMORY_UTILIZATION` | Fraction of GPU memory for embedding model server. |
| `--runner pooling` | fixed in Makefile | Uses pooling runner for embedding outputs. |
| `--max-model-len 8192` | fixed in Makefile | Context length for embedding requests. |
| `--port` / `--host` | `EMBD_PORT` / `SERVER` | Embedding server bind address and port. |

### `make start-server`

| Command argument | Makefile source | Meaning |
|---|---|---|
| `uvicorn server:app` | fixed in Makefile | Runs FastAPI app object from `server.py`. |
| `--host 0.0.0.0` | fixed in Makefile | Exposes API on all interfaces. |
| `--reload` | fixed in Makefile | Auto-reloads on code changes (dev mode). |
| `--port` | `API_PORT` | API listen port. |

### `make docker-run`

| Docker argument | Makefile source | Meaning |
|---|---|---|
| `--gpus all` | fixed in Makefile | Exposes all GPUs to container runtime. |
| `-e GPU_DEVICE` | `GPU_DEVICE` | Passes selected GPU index to container env. |
| `-e SEC_API_ORGANIZATION` / `-e SEC_API_EMAIL` | same vars | SEC API User-Agent identity fields. |
| `-v ./sec_data:/app/sec_data` | fixed in Makefile | Persists downloaded SEC files. |
| `-v ./localworkspace:/app/localworkspace` | fixed in Makefile | Persists OCR workspace outputs. |
| `-p $(API_PORT):8081` | `API_PORT` | Publishes container API port to host. |
