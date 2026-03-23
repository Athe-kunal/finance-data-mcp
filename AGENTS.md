## Dev environment tips
- This repository is Python-first and uses `uv` for environment and dependency management.
- Run `uv sync` for the default development environment.
- Run `uv sync --group ocr-md` when working on OCR/embedding pipelines.
- Run `uv sync --group ocr-md --group mcp` when working on the MCP server.
- Use `uv run <command>` so commands execute inside the managed project environment.
- Keep runtime configuration in environment variables or `.env` (see `settings.py` and `README.md`).
- Prefer `Makefile` targets for model serving, API serving, benchmarking, and Docker workflows.

## Testing instructions
- Run lightweight validation before committing:
  - `uv run python -m compileall .`
- If you modify API/server logic, verify imports:
  - `uv run python -c "import server"`
  - `uv run python -c "import finance_data.api"`
- If you modify SEC download or OCR pipelines, verify module imports:
  - `uv run python -c "import filings.sec_data, ocr.olmocr_pipeline"`
- If you modify MCP workflows, verify the MCP server imports:
  - `uv run --group ocr-md --group mcp python -c "import mcp_server"`
- Add or update tests when behavior changes.

## PR instructions
- Title format: `[sec-filings-data] <Title>`.
- Include a short validation section listing exact commands run.
- Keep this file aligned with the repository's current Python/`uv` workflow.

## Makefile commands
- `make vllm-olmocr-serve`: Starts the vLLM server for olmOCR.
- `make vllm-embd-serve`: Starts the embedding vLLM server (pooling runner).
- `make start-server`: Runs the FastAPI app with Uvicorn on `0.0.0.0` using `API_PORT`.
- `make guidellm-benchmark`: Runs throughput benchmarking against the local vLLM endpoint and writes `benchmark.yaml`.
- `make docker-build`: Builds the Docker image using `IMAGE_NAME`.
- `make docker-run`: Runs the Docker image with GPU, SEC API environment variables, mounted data/workspace volumes, and port mapping.
- `make docker-start`: Builds then runs the Docker container.
- `make docker-stop`: Stops container(s) publishing `API_PORT`.
- `make docker-remove`: Removes container(s) publishing `API_PORT`.
