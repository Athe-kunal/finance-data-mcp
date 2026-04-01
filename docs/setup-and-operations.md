# Setup and Operations

This guide focuses on setup and runtime commands for API and MCP workflows, including hybrid retrieval and reranking.

## 1) Install dependencies

```bash
uv sync
```

For OCR + embedding pipelines:

```bash
uv sync --group ocr-md
```

For MCP server usage:

```bash
uv sync --group ocr-md --group mcp
```

## 2) Configure environment

Set values in `.env` or shell environment:

- SEC access: `SEC_API_ORGANIZATION`, `SEC_API_EMAIL`
- OCR serving: `OLMOCR_SERVER`, `OLMOCR_MODEL`, `OLMOCR_WORKSPACE`
- Embeddings: `EMBEDDING_SERVER`, `EMBEDDING_MODEL`
- Vector persistence: `CHROMA_PERSIST_DIR`
- MCP network: `MCP_HOST`, `MCP_PORT`, `MCP_NGROK_ALLOWED_HOSTS`

Defaults are defined in `finance_data/settings.py`.

## 3) Start model servers

```bash
make vllm-olmocr-serve
make vllm-embd-serve
make vllm-reranker-serve
```

Optional benchmark:

```bash
make guidellm-benchmark
```

## 4) Start application servers

### FastAPI

```bash
make start-server
```

### MCP

```bash
uv run --group ocr-md --group mcp python mcp_server.py
```

## 5) SEC filing workflow

Fetch filing + OCR markdown (CLI):

```bash
uv run python -m finance_data.filings.sec_data --ticker AMZN --year 2025
uv run python -m finance_data.ocr.olmocr_pipeline --pdf-dir sec_data/AMZN-2025
```

Embed for retrieval (API):

```bash
curl -s -X POST "http://127.0.0.1:8081/vector_store/embed_sec_filings" \
  -H "Content-Type: application/json" \
  -d '{"ticker":"AMZN","year":"2025","filing_type":"10-K","force":false}'
```

Search indexed filings:

- Hybrid (dense + BM25 + reranker): `POST /vector_store/search_sec_filings`

Hybrid example:

```bash
curl -s -X POST "http://127.0.0.1:8081/vector_store/search_sec_filings" \
  -H "Content-Type: application/json" \
  -d '{"ticker":"AMZN","year":"2025","filing_type":"10-K","query":"operating margin guidance","top_k":5}'
```

## 6) Earnings transcript workflow

Fetch transcripts:

```bash
uv run python -m finance_data.earnings_transcripts.transcripts AMZN 2025
```

Embed transcript chunks:

```bash
curl -s -X POST "http://127.0.0.1:8081/vector_store/embed_transcripts" \
  -H "Content-Type: application/json" \
  -d '{"ticker":"AMZN","year":"2025","force":false}'
```

Search transcript chunks:

- Hybrid (dense + BM25 + reranker): `POST /vector_store/search_transcripts`

Hybrid example:

```bash
curl -s -X POST "http://127.0.0.1:8081/vector_store/search_transcripts" \
  -H "Content-Type: application/json" \
  -d '{"ticker":"AMZN","year":"2025","query":"AWS demand trends","top_k":5}'
```

## 7) MCP operations + remote access

When exposing MCP with ngrok:

```bash
ngrok http 8069
```

Then:

1. Copy ngrok hostname.
2. Add hostname to `MCP_NGROK_ALLOWED_HOSTS` as a JSON array.
3. Restart `mcp_server.py`.
4. Connect clients to `https://<hostname>/mcp`.

MCP search tools use the hybrid retrieval pipeline:

- `search_sec_filings_tool`
- `search_transcripts_tool`

## 8) Docker commands

```bash
make docker-build
make docker-run
make docker-start
make docker-stop
make docker-remove
```
