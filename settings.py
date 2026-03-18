from pydantic_settings import BaseSettings, SettingsConfigDict


class OlmoOCRSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # SEC API (required for filings; SEC requires User-Agent with org + email)
    sec_api_organization: str = "Your-Organization"
    sec_api_email: str = "your-email@example.com"

    # olmOCR pipeline
    olmocr_server: str = "http://localhost:8000/v1"
    olmocr_model: str = "allenai/olmOCR-2-7B-1025-FP8"
    olmocr_workspace: str = "./localworkspace"

    # Embedding server (vLLM pooling runner)
    embedding_server: str = "http://127.0.0.1:8888/v1"
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"

    # FAISS vector index persistence
    faiss_index_dir: str = "./faiss_indexes"
    faiss_use_gpu: bool = True


olmocr_settings = OlmoOCRSettings()
