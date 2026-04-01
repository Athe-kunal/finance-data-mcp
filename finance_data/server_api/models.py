"""Pydantic models used by FastAPI endpoints."""

from pydantic import BaseModel, field_validator

from finance_data.earnings_transcripts.transcripts import quarter_label_to_num
from finance_data.filings.models import SecFilingType


class CompanyNameRequest(BaseModel):
    """Request for company-name to ticker lookup."""

    name: str


class SecMainRequest(BaseModel):
    """Request to download one SEC filing PDF."""

    ticker: str
    year: str
    filing_type: SecFilingType | str = SecFilingType.FORM_10_K


class EarningsTranscriptQuarterRequest(BaseModel):
    """Request to download one earnings transcript."""

    ticker: str
    year: int
    quarter: str

    @field_validator("quarter")
    @classmethod
    def validate_quarter_label(cls, value: str) -> str:
        return f"Q{quarter_label_to_num(value)}"


class EarningsTranscriptQuarterEmbedRequest(EarningsTranscriptQuarterRequest):
    """Request to download one earnings transcript and embed it."""

    force: bool = False


class SecMainToMarkdownRequest(BaseModel):
    """Request to download one SEC filing and convert it to markdown."""

    ticker: str
    year: str
    filing_type: SecFilingType | str = SecFilingType.FORM_10_K


class SecMainToMarkdownEmbedRequest(BaseModel):
    """Request to download one SEC filing and embed it."""

    ticker: str
    year: str
    filing_type: SecFilingType | str = SecFilingType.FORM_10_K
    force: bool = False


class BatchSecFilingItem(BaseModel):
    """One SEC batch item with one ticker/year and multiple filing types."""

    ticker: str
    year: str
    filing_types: list[str]
    force: bool = False

    @field_validator("filing_types")
    @classmethod
    def validate_filing_types(cls, values: list[str]) -> list[str]:
        cleaned = [value.strip() for value in values if value and value.strip()]
        if not cleaned:
            raise ValueError("filing_types must contain at least one filing type")
        return cleaned


class BatchSecFilingsRequest(BaseModel):
    """Request body for SEC batch endpoint."""

    requests: list[BatchSecFilingItem]

    @field_validator("requests")
    @classmethod
    def validate_requests(
        cls,
        values: list[BatchSecFilingItem],
    ) -> list[BatchSecFilingItem]:
        if not values:
            raise ValueError("requests must contain at least one batch item")
        return values


class BatchEarningsTranscriptItem(BaseModel):
    """One transcript batch item with one ticker and year/quarter lists."""

    ticker: str
    years: list[int]
    quarters: list[str]

    @field_validator("years")
    @classmethod
    def validate_years(cls, values: list[int]) -> list[int]:
        if not values:
            raise ValueError("years must contain at least one year")
        return values

    @field_validator("quarters")
    @classmethod
    def validate_quarters(cls, values: list[str]) -> list[str]:
        if not values:
            raise ValueError("quarters must contain at least one quarter")
        return [f"Q{quarter_label_to_num(value)}" for value in values]


class BatchEarningsTranscriptsRequest(BaseModel):
    """Request body for transcript batch endpoint."""

    requests: list[BatchEarningsTranscriptItem]

    @field_validator("requests")
    @classmethod
    def validate_requests(
        cls,
        values: list[BatchEarningsTranscriptItem],
    ) -> list[BatchEarningsTranscriptItem]:
        if not values:
            raise ValueError("requests must contain at least one batch item")
        return values


class RunOlmoOcrRequest(BaseModel):
    """Request to run OCR under one PDF directory."""

    pdf_dir: str


class SecFilingsEmbedRequest(BaseModel):
    """Request to build vectors from SEC markdown for ticker/year."""

    ticker: str
    year: str
    force: bool = False


class TranscriptEmbedRequest(BaseModel):
    """Request to build vectors from transcript markdown for ticker/year."""

    ticker: str
    year: str
    force: bool = False


class TranscriptSearchRequest(BaseModel):
    """Request to search transcript vectors."""

    ticker: str
    year: str
    query: str
    top_k: int = 5


class SecFilingsListRequest(BaseModel):
    """Request to list filing types currently in vector store."""

    ticker: str
    year: str


class SecFilingsSearchRequest(BaseModel):
    """Request to search one SEC filing vector index."""

    ticker: str
    year: str
    filing_type: SecFilingType | str
    query: str
    top_k: int = 5


class ChunkResult(BaseModel):
    """Structured search result row from chunk + score."""

    text: str
    chunk_type: str
    page_num: int | None
    section_title: str | None
    chunk_index: int
    score: float
    filing_type: SecFilingType | str | None = None
