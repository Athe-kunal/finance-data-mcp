## Repository layout
- **`finance_data/`** — library package:
  - `filings/`: SEC download + helpers.
  - `ocr/`: olmOCR pipeline.
  - `dataloader/`: chunking, Chroma indexing, semantic + BM25 retrieval.
  - `earnings_transcripts/`: transcript fetch + persistence.
  - `server_api/`: FastAPI models + batch helpers.
  - `finance_data_api/`: packaged CLI entrypoint.
- **Repo root** — `server.py` (FastAPI app), `mcp_server.py` (MCP entrypoint).
- Shared settings live in `finance_data/settings.py`.

## Dev environment tips
- Use `uv` for dependency and environment management.
- `uv sync` for default development dependencies.
- `uv sync --group ocr-md` for OCR/embedding/BM25 retrieval workflows.
- `uv sync --group ocr-md --group mcp` for MCP workflows.
- Prefer `uv run <command>` for project commands.
- Keep runtime config in environment variables or `.env`.

## Testing instructions
- Run lightweight validation before commit:
  - `uv run python -m compileall .`
- If you modify API/server logic:
  - `uv run python -c "import server"`
  - `uv run python -c "import finance_data.server_api.models"`
- If you modify filings/OCR/vector retrieval logic (including BM25):
  - `uv run python -c "import finance_data.filings.sec_data, finance_data.ocr.olmocr_pipeline, finance_data.dataloader.vector_store"`
- If you modify transcript retrieval/search logic:
  - `uv run python -c "import finance_data.earnings_transcripts.transcripts"`
- If you modify MCP workflows:
  - `uv run --group ocr-md --group mcp python -c "import mcp_server"`
- Add or update tests whenever behavior changes.

## PR instructions
- Title format: `[sec-filings-data] <Title>`.
- Include a short validation section with exact commands run.
- Keep this file aligned with the current Python/`uv` workflow and supported retrieval features (semantic + BM25).

## Makefile commands
- `make vllm-olmocr-serve`: Start the vLLM server for olmOCR.
- `make vllm-embd-serve`: Start the embedding vLLM server (pooling runner).
- `make start-server`: Run FastAPI with Uvicorn using `API_PORT`.
- `make guidellm-benchmark`: Run throughput benchmarking and write `benchmark.yaml`.
- `make docker-build`: Build the Docker image using `IMAGE_NAME`.
- `make docker-run`: Run the Docker image with GPU, SEC API env vars, volumes, and port mapping.
- `make docker-start`: Build then run the Docker container.
- `make docker-stop`: Stop container(s) publishing `API_PORT`.
- `make docker-remove`: Remove container(s) publishing `API_PORT`.
