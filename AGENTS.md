## Dev environment tips
- This repository is Python-first and uses `uv` for environment and dependency management.
- Run `uv sync` to install project dependencies into the managed virtual environment.
- Use `uv run <command>` for all local commands so they run with the project environment.
- Keep runtime configuration in environment variables or `.env` (see `settings.py` and `README.md`).
- Prefer the provided `Makefile` targets for common workflows (model serving, API serving, benchmarking, and Docker tasks).

## Testing instructions
- Run lightweight validation before committing:
  - `uv run python -m compileall .`
- If you modify API/server logic, verify the app imports correctly:
  - `uv run python -c "import server"`
- If you modify SEC download/OCR pipelines, at minimum run module import checks:
  - `uv run python -c "import filings.sec_data, ocr.olmocr_pipeline"`
- Add or update tests when you change behavior.

## PR instructions
- Title format: `[sec-filings-data] <Title>`
- Include a short validation section listing exact commands you ran.
- Ensure `AGENTS.md` instructions remain aligned with this repository’s Python/uv workflow.

## Makefile commands
- `make vllm-olmocr-serve`: Starts the vLLM server for the olmOCR model using configurable model/runtime variables.
- `make start-server`: Runs the FastAPI app with Uvicorn on `0.0.0.0` using `API_PORT`.
- `make guidellm-benchmark`: Executes a throughput benchmark against the local vLLM endpoint and writes `benchmark.yaml`.
- `make docker-build`: Builds the Docker image using `IMAGE_NAME`.
- `make docker-run`: Runs the Docker image with GPU, SEC API environment variables, mounted data/workspace volumes, and port mapping.
- `make docker-start`: Convenience target that builds then runs the Docker container.
- `make docker-stop`: Stops container(s) publishing `API_PORT`.
- `make docker-remove`: Removes container(s) publishing `API_PORT`.
