# SEC-filings-Markdown

## Configuration

Settings are loaded via Pydantic Settings from environment variables or a `.env` file:

| Variable | Description | Default |
|----------|-------------|---------|
| `SEC_API_ORGANIZATION` | Organization name for SEC API User-Agent | `Your-Organization` |
| `SEC_API_EMAIL` | Contact email for SEC API User-Agent | `your-email@example.com` |
| `OLMOCR_SERVER` | vLLM server URL for olmOCR | `http://localhost:8000/v1` |
| `OLMOCR_MODEL` | Model name for olmOCR | `allenai/olmOCR-2-7B-1025-FP8` |
| `OLMOCR_WORKSPACE` | Workspace directory for OCR output | `./localworkspace` |
| `EARNINGS_TRANSCRIPTS_DIR` | Directory for fetched transcript JSONL files | `earnings_transcripts_data` |
| `EMBEDDING_SERVER` | OpenAI-compatible embedding API (e.g. vLLM pooling) | `http://127.0.0.1:8888/v1` |
| `EMBEDDING_MODEL` | Model id passed to the embedding server | `Qwen/Qwen3-Embedding-0.6B` |
| `CHROMA_PERSIST_DIR` | ChromaDB persistence directory | `./chroma_db` |


## MCP server

This repository includes an MCP server at `mcp_server.py` that exposes the same operational functions as `server.py` (SEC fetch, OCR, embedding, and search), plus file exploration tools for PDFs, JSONL, markdown, and other artifacts under configured data roots.

Run it with the MCP dependency group:

```bash
uv run --group mcp python mcp_server.py
```

Key exploration tools exposed to MCP clients:

- `list_data_roots_tool`: shows root directories available for browsing.
- `list_data_files_tool`: glob file listing (for example `**/*.pdf`, `**/*.jsonl`).
- `read_data_file_tool`: reads text-based files directly and provides metadata/preview for binary files.

## Docker

### Build

```bash
docker build -t sec-filings-md .
```

The image now defaults to a smaller footprint by using the CUDA runtime base while still preinstalling Playwright Chromium for scraping.
If you want to skip Playwright browser installation (to reduce image size further), build with:

```bash
docker build --build-arg INSTALL_PLAYWRIGHT_BROWSER=0 -t sec-filings-md .
```

Or via Makefile:

```bash
make docker-build
```

### Run

```bash
GPU_DEVICE=${GPU_DEVICE:-3}
docker run --gpus device=${GPU_DEVICE} \
  -e SEC_API_ORGANIZATION="Your-Organization" \
  -e SEC_API_EMAIL="your-email@example.com" \
  -v ./sec_data:/app/sec_data \
  -v ./localworkspace:/app/localworkspace \
  -p 8081:8081 \
  sec-filings-md
```

Or via Makefile (build + run in one step):

```bash
make docker-start
```

Makefile overrides:

| Variable | Description | Default |
|----------|-------------|---------|
| `IMAGE_NAME` | Docker image name | `sec-filings-md` |
| `GPU_DEVICE` | GPU device index | `0` |
| `API_PORT` | Host port for API | `8081` |
| `SEC_API_ORGANIZATION` | SEC API User-Agent org | `Your-Organization` |
| `SEC_API_EMAIL` | SEC API contact email | `your-email@example.com` |

Example with overrides:

```bash
make docker-start GPU_DEVICE=3 SEC_API_EMAIL="you@example.com"
```

The two volumes persist data across container restarts:

| Volume | Container path | Purpose |
|--------|---------------|---------|
| `sec_data` | `/app/sec_data` | Downloaded SEC filing PDFs |
| `localworkspace` | `/app/localworkspace` | OCR workspace and output markdown |

Override the workspace path at runtime with `-e OLMOCR_WORKSPACE=/custom/path`.

## Installation

```bash
uv sync
playwright install chromium
```

## Usage

Start vLLM server:
```bash
make vllm-olmocr-serve
```

Benchmark vLLM with guidellm (start the vLLM server first, then in another terminal):
```bash
make guidellm-benchmark
```

Fetch SEC filings:
```bash
uv run python -m filings.sec_data --ticker AMZN --year 2025
```

Run OCR pipeline:
```bash
uv run python ocr/olmocr_pipeline.py --pdf-dir sec_data/AMZN-2025
```

## Earnings call transcripts

Transcripts are scraped from [discountingcashflows.com](https://discountingcashflows.com) (Playwright + Chromium). Each quarter is saved as one JSONL file under `{EARNINGS_TRANSCRIPTS_DIR}/{TICKER}/{year}/Q{n}.jsonl`.

### 1. Fetch transcripts

**CLI** (writes files under `earnings_transcripts_data` by default):

```bash
uv run python -m earnings_transcripts.transcripts AMZN 2025
```

Optional: `--max-concurrency` (default `4`) to limit parallel quarter fetches.

**HTTP** (same fetch + persist, with the API running):

```bash
curl -s -X POST "http://127.0.0.1:8081/earnings_transcripts/for_year" \
  -H "Content-Type: application/json" \
  -d '{"ticker":"AMZN","year":2025}'
```

Response body is a JSON array of transcript objects (`ticker`, `year`, `quarter_num`, `date`, `speaker_texts`, …).

### 2. Start embedding server and API

Transcript chunks are embedded with the same OpenAI-compatible embedding endpoint as SEC filings (`EMBEDDING_SERVER` / `EMBEDDING_MODEL`). In one terminal:

```bash
make vllm-embd-serve
```

In another:

```bash
make start-server
```

(Adjust `API_PORT` / `EMBD_PORT` in the `Makefile` or your environment if needed.)

### 3. Index transcripts in Chroma

```bash
curl -s -X POST "http://127.0.0.1:8081/vector_store/embed_transcripts" \
  -H "Content-Type: application/json" \
  -d '{"ticker":"AMZN","year":"2025","force":false}'
```

Use `"force": true` to replace existing vectors for those quarters. Filing types in the index appear as `Q1`–`Q4`.

### 4. Search across indexed quarters

Search merges hits from all transcript quarters present for that ticker/year:

```bash
curl -s -X POST "http://127.0.0.1:8081/vector_store/search_transcripts" \
  -H "Content-Type: application/json" \
  -d '{"ticker":"AMZN","year":"2025","query":"AWS revenue growth","top_k":5}'
```

Each result includes `filing_type` (`Q1`, …) so you can see which call the chunk came from.
