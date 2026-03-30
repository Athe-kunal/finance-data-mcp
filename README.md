# SEC-filings-Markdown

## Documentation

Detailed operational docs are available in `docs/`:

- `docs/README.md`
- `docs/setup-and-operations.md`

## Core library functions

These functions are good entry points when using the package directly:

| Function | Location | What it does |
|---|---|---|
| `sec_main(ticker, year, filing_type)` | `finance_data/filings/sec_data.py` | Fetches SEC filing metadata, downloads the filing PDF, and returns `(SecResults, pdf_path)`. |
| `sec_main_to_markdown(ticker, year, filing_type)` | `finance_data/filings/sec_data.py` | Ensures the filing PDF exists, runs olmOCR when needed, and returns markdown text + file paths. |
| `prepare_sec_filing_envs(ticker, year, filing_type)` | `finance_data/dataloader/pipeline.py` | Ensures PDFs exist, runs OCR, and builds REPL-friendly markdown environments for each filing. |
| `sec_main_to_markdown_and_embed(ticker, year, filing_type, force)` | `finance_data/dataloader/pipeline.py` | Runs SEC fetch/OCR flow and stores vectors in ChromaDB for semantic search. |
| `get_transcript_for_quarter_async(ticker, year, quarter_num)` | `finance_data/earnings_transcripts/transcripts.py` | Fetches a quarter earnings transcript asynchronously and returns parsed transcript data. |
| `save_transcript_markdown(transcript)` | `finance_data/earnings_transcripts/transcripts.py` | Persists one transcript into a standardized markdown file path under `EARNINGS_TRANSCRIPTS_DIR`. |

## Configuration

Settings are loaded via Pydantic Settings from environment variables or a `.env` file:

| Variable | Description | Default |
|----------|-------------|---------|
| `SEC_API_ORGANIZATION` | Organization name for SEC API User-Agent | `Your-Organization` |
| `SEC_API_EMAIL` | Contact email for SEC API User-Agent | `your-email@example.com` |
| `OLMOCR_SERVER` | vLLM server URL for olmOCR | `http://localhost:8000/v1` |
| `OLMOCR_MODEL` | Model name for olmOCR | `allenai/olmOCR-2-7B-1025-FP8` |
| `OLMOCR_WORKSPACE` | Workspace directory for OCR output | `./localworkspace` |
| `EARNINGS_TRANSCRIPTS_DIR` | Directory for fetched transcript Markdown files | `earnings_transcripts_data` |
| `EMBEDDING_SERVER` | OpenAI-compatible embedding API (e.g. vLLM pooling) | `http://127.0.0.1:8888/v1` |
| `EMBEDDING_MODEL` | Model id passed to the embedding server | `Qwen/Qwen3-Embedding-0.6B` |
| `CHROMA_PERSIST_DIR` | ChromaDB persistence directory | `./chroma_db` |
| `MCP_HOST` | Bind address for the MCP HTTP server | `127.0.0.1` |
| `MCP_PORT` | Listen port for the MCP HTTP server | `8069` |
| `MCP_NGROK_ALLOWED_HOSTS` | JSON list of extra `Host` values allowed through the tunnel (see MCP section) | (see `finance_data/settings.py`) |


## MCP server

`mcp_server.py` exposes SEC filing and earnings-transcript workflows over MCP (fetch/OCR, embed, semantic search) using the same backends as the REST API: **olmOCR** and an **OpenAI-compatible embedding** endpoint backed by vLLM.

### 1. Start the vLLM backends

The MCP tools need both servers running before you start the MCP process.

**Terminal A — olmOCR (vision / markdown pipeline)** — must match `OLMOCR_SERVER` (default `http://localhost:8000/v1`):

```bash
make vllm-olmocr-serve
```

**Terminal B — embeddings (pooling runner)** — must match `EMBEDDING_SERVER` (default `http://127.0.0.1:8888/v1`):

```bash
make vllm-embd-serve
```

If you change `PORT` / `EMBD_PORT` in the `Makefile` or your environment, set `OLMOCR_SERVER` and `EMBEDDING_SERVER` in `.env` so they point at the same hosts and ports.

### 2. Install dependencies and run the MCP server

Chroma, OpenAI client, and OCR-related imports require the `ocr-md` group in addition to `mcp`:

```bash
uv sync --group ocr-md --group mcp
uv run --group ocr-md --group mcp python mcp_server.py
```

The server listens on `MCP_HOST` / `MCP_PORT` (defaults `127.0.0.1:8069`) using the **streamable HTTP** transport. The HTTP endpoint path is **`/mcp`** (FastMCP default), so locally that is `http://127.0.0.1:8069/mcp`.

### 3. Expose with ngrok and connect a client

To use the MCP server from another machine or from a hosted MCP client, tunnel the MCP port with [ngrok](https://ngrok.com/) (or a similar HTTPS reverse proxy).

1. Install and log in to ngrok (`ngrok config add-authtoken …`).
2. With `mcp_server.py` still running, forward the MCP port (replace `8069` if you changed `MCP_PORT`):

   ```bash
   ngrok http 8069
   ```

3. Note the **public HTTPS hostname** ngrok assigns (for example `https://random-name.ngrok-free.app` or `*.ngrok-free.dev`).
4. Add that hostname to **`MCP_NGROK_ALLOWED_HOSTS`** so DNS rebinding protection accepts the tunnel’s `Host` header. In `.env`, use a JSON array, for example:

   ```bash
   MCP_NGROK_ALLOWED_HOSTS='["random-name.ngrok-free.app"]'
   ```

   Restart `mcp_server.py` after changing this.

5. Point your MCP client at the tunneled URL **including `/mcp`**, for example:

   `https://random-name.ngrok-free.app/mcp`

Use your client’s documented configuration for **Streamable HTTP** / URL-based MCP servers. If the tunnel hostname changes each time you run ngrok, update `MCP_NGROK_ALLOWED_HOSTS` and restart the MCP process.

### Tools and resources

**Tools** (representative):

- `company_name_to_ticker_tool`, `list_resources_tool`
- `sec_main_to_markdown_and_embed_tool`, `earnings_transcript_for_quarter_tool`
- `search_sec_filings_tool`, `search_transcripts_tool`

For an interactive walkthrough of how to use the MCP, [open this ChatGPT chats](https://chatgpt.com/share/69c0bf65-54a8-8010-bd40-6aa33908a1e6).

**Resources** (URI catalogs under `resource://sec-filings-data/...`): combined SEC + transcript file listings and per-root trees.

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

Install OCR/markdown + embedding stack dependencies when you need those pipelines:

```bash
uv sync --group ocr-md
```

Package install (for publishing/consuming from PyPI):

```bash
pip install finance_data_llm
```

Use package functions directly from Python (no server process required):

```python
import asyncio

from finance_data.filings.sec_data import sec_main
from finance_data.filings.utils import company_to_ticker

ticker = company_to_ticker("Amazon") or "AMZN"
sec_result, pdf_path = asyncio.run(
    sec_main(ticker=ticker, year="2025", filing_type="10-K")
)
```

If you do want to run the API, use the packaged console script:

```bash
finance-data-llm-server
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
uv run python -m finance_data.filings.sec_data --ticker AMZN --year 2025
```

Run OCR pipeline:
```bash
uv run python -m finance_data.ocr.olmocr_pipeline --pdf-dir sec_data/AMZN-2025
```

## Earnings call transcripts

Transcripts are scraped from [discountingcashflows.com](https://discountingcashflows.com) (Playwright + Chromium). Each quarter is saved as one Markdown file under `{EARNINGS_TRANSCRIPTS_DIR}/{TICKER}/{year}/Q{n}_{YYYY-MM-DD}.md` (date may be `unknown-date` when unavailable).

### 1. Fetch transcripts

**CLI** (writes files under `earnings_transcripts_data` by default):

```bash
uv run python -m finance_data.earnings_transcripts.transcripts AMZN 2025
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
