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


olmocr_settings = OlmoOCRSettings()
