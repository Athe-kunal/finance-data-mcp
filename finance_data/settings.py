from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class SECSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # SEC API (required for filings; SEC requires User-Agent with org + email)
    sec_api_organization: str = "Your-Organization"
    sec_api_email: str = "your-email@example.com"

    # olmOCR pipeline
    olmocr_server: str = "http://localhost:8000/v1"
    olmocr_model: str = "allenai/olmOCR-2-7B-1025-FP8"
    olmocr_workspace: str = "./localworkspace"

    # Downloaded SEC PDFs: {sec_data_dir}/{ticker}-{year}/
    sec_data_dir: str = "sec_data"

    earnings_transcripts_dir: str = "earnings_transcripts_data"
    # Embedding server (vLLM pooling runner)
    embedding_server: str = "http://127.0.0.1:8002/v1"
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"

    # Reranker server (vLLM score/rerank runner)
    reranker_server: str = "http://127.0.0.1:8003"
    reranker_model: str = "Qwen/Qwen3-Reranker-0.6B"

    # ChromaDB vector persistence
    chroma_persist_dir: str = "./chroma_db"
    chroma_collection_name: str = "sec_filings"
    chroma_bm25_collection_name: str = "sec_filings_bm25"

    # Main FastAPI app (uvicorn server:app); Makefile uses API_PORT for the same default.
    api_host: str = "127.0.0.1"
    api_port: int = 8888
    # MCP server (FastMCP streamable-http bind + transport DNS rebinding allowlist)
    mcp_host: str = "127.0.0.1"
    mcp_port: int = 8069
    # Ngrok (or similar) tunnel hostnames allowed by MCP transport security Host checks.
    mcp_ngrok_allowed_hosts: list[str] = Field(
        default_factory=lambda: [
            "shirleen-supercritical-contributively.ngrok-free.dev",
        ],
    )

    # Processed-data index scanning.
    processed_index_max_workers: int = 8
    ignore_ocr: bool = False
    processed_index_cache_file: str = "processed_data_index_cache.orjson"
    # Set to False for short-lived scripts to skip the background watcher thread.
    processed_index_start_watcher: bool = False

    # Batch endpoint backpressure.
    batch_max_concurrent_jobs: int = 8


sec_settings = SECSettings()
