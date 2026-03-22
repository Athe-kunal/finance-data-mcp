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

    # Downloaded SEC PDFs and SEC metadata JSON: {sec_data_dir}/{ticker}-{year}/
    sec_data_dir: str = "sec_data"
    sec_results_filename: str = "sec_results.json"
    earnings_transcripts_dir: str = "earnings_transcripts_data"
    # Embedding server (vLLM pooling runner)
    embedding_server: str = "http://127.0.0.1:8888/v1"
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"

    # ChromaDB vector persistence
    chroma_persist_dir: str = "./chroma_db"
    chroma_collection_name: str = "sec_filings"

    # FastAPI server URL
    server_url: str = "http://127.0.0.1:8888"


sec_settings = SECSettings()
