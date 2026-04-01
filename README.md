# sec-filings-data

A Python-first toolkit for SEC filing ingestion, OCR-to-Markdown conversion, transcript collection, and retrieval across **semantic** and **BM25 lexical** search.

## What this project does

- Downloads SEC filings and stores filing metadata.
- Converts filing PDFs to Markdown via olmOCR.
- Chunks and indexes filings/transcripts in Chroma.
- Supports both:
  - **Semantic search** (embedding similarity).
  - **BM25 search** (keyword/lexical ranking).
- Exposes workflows through:
  - FastAPI (`server.py`).
  - MCP server (`mcp_server.py`).

## Repository layout

- `finance_data/filings/`: SEC download + helpers.
- `finance_data/ocr/`: olmOCR pipeline.
- `finance_data/dataloader/`: chunking, Chroma indexing, semantic + BM25 retrieval.
- `finance_data/earnings_transcripts/`: transcript fetch + persistence.
- `finance_data/server_api/`: API request/response models + batch helpers.
- `server.py`: FastAPI app.
- `mcp_server.py`: MCP entrypoint.
- `docs/`: setup and operations docs.

## Quick start

### 1) Install dependencies

```bash
uv sync
```

For OCR/embedding flows:

```bash
uv sync --group ocr-md
```

For MCP workflows:

```bash
uv sync --group ocr-md --group mcp
```

### 2) Configure environment

Use `.env` or environment variables. Common settings:

- `SEC_API_ORGANIZATION`, `SEC_API_EMAIL`
- `OLMOCR_SERVER`, `OLMOCR_MODEL`, `OLMOCR_WORKSPACE`
- `EMBEDDING_SERVER`, `EMBEDDING_MODEL`
- `CHROMA_PERSIST_DIR`
- `MCP_HOST`, `MCP_PORT`, `MCP_NGROK_ALLOWED_HOSTS`

See `finance_data/settings.py` for defaults.

### 3) Run services

Start model servers:

```bash
make vllm-olmocr-serve
make vllm-embd-serve
```

Start API:

```bash
make start-server
```

Start MCP:

```bash
uv run --group ocr-md --group mcp python mcp_server.py
```

## Search capabilities

### SEC filings API

- Semantic: `POST /vector_store/search_sec_filings`
- BM25: `POST /vector_store/search_sec_filings_bm25`

### Transcript API

- Semantic: `POST /vector_store/search_transcripts`
- BM25: `POST /vector_store/search_transcripts_bm25`

### MCP tools

- Semantic: `search_sec_filings_tool`, `search_transcripts_tool`
- BM25: `search_sec_filings_bm25_tool`, `search_transcripts_bm25_tool`

## Core workflows

### SEC filing → Markdown

```bash
uv run python -m finance_data.filings.sec_data --ticker AMZN --year 2025
uv run python -m finance_data.ocr.olmocr_pipeline --pdf-dir sec_data/AMZN-2025
```

### Embed and search filings (API)

```bash
curl -s -X POST "http://127.0.0.1:8081/vector_store/embed_sec_filings" \
  -H "Content-Type: application/json" \
  -d '{"ticker":"AMZN","year":"2025","filing_type":"10-K","force":false}'

curl -s -X POST "http://127.0.0.1:8081/vector_store/search_sec_filings_bm25" \
  -H "Content-Type: application/json" \
  -d '{"ticker":"AMZN","year":"2025","filing_type":"10-K","query":"operating income margin","top_k":5}'
```

### Earnings transcripts

Fetch quarterly transcripts:

```bash
uv run python -m finance_data.earnings_transcripts.transcripts AMZN 2025
```

Embed + BM25 search transcripts:

```bash
curl -s -X POST "http://127.0.0.1:8081/vector_store/embed_transcripts" \
  -H "Content-Type: application/json" \
  -d '{"ticker":"AMZN","year":"2025","force":false}'

curl -s -X POST "http://127.0.0.1:8081/vector_store/search_transcripts_bm25" \
  -H "Content-Type: application/json" \
  -d '{"ticker":"AMZN","year":"2025","query":"AWS revenue growth","top_k":5}'
```

## Docker

Use Makefile wrappers:

```bash
make docker-build
make docker-start
```

Stop/remove by API port:

```bash
make docker-stop
make docker-remove
```

## Documentation

- `docs/README.md`
- `docs/setup-and-operations.md`
