import dataclasses
import shutil
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator

from dataloader.text_splitter import Chunk
from dataloader.vector_store import ChromaVectorStore
from earnings_transcripts.transcripts import (
    get_transcript_for_quarter_async,
    quarter_label_to_num,
)
from filings.utils import company_to_ticker
from filings.sec_data import (
    sec_main,
    sec_main_to_markdown,
    sec_main_to_markdown_and_embed,
)
from ocr.olmocr_pipeline import run_olmo_ocr
from settings import sec_settings

vector_index: ChromaVectorStore


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: F811
    global vector_index
    vector_index = ChromaVectorStore()
    yield


app = FastAPI(lifespan=lifespan)


class CompanyNameRequest(BaseModel):
    name: str


@app.post("/company_name_to_ticker")
def company_name_to_ticker(request: CompanyNameRequest):
    """Resolve a company name to its stock ticker symbol."""
    ticker = company_to_ticker(request.name)
    if ticker is None:
        raise HTTPException(status_code=404, detail="No ticker found for company name")
    return {"ticker": ticker}


class SecMainRequest(BaseModel):
    ticker: str
    year: str
    filing_type: str = "10-K"


class EarningsTranscriptQuarterRequest(BaseModel):
    ticker: str
    year: int
    quarter: str

    @field_validator("quarter")
    @classmethod
    def validate_quarter_label(cls, value: str) -> str:
        return f"Q{quarter_label_to_num(value)}"


@app.post("/earnings_transcripts/for_quarter")
async def earnings_transcript_for_quarter(request: EarningsTranscriptQuarterRequest):
    transcript = await get_transcript_for_quarter_async(
        request.ticker, request.year, request.quarter
    )
    if transcript is None:
        raise HTTPException(
            status_code=404,
            detail="Transcript not available for this ticker, year, and quarter",
        )
    return dataclasses.asdict(transcript)


@app.post("/sec_main")
async def sec_main_endpoint(request: SecMainRequest):
    """Fetch SEC filings and save them as PDFs."""
    sec_result, pdf_path = await sec_main(
        ticker=request.ticker,
        year=request.year,
        filing_type=request.filing_type,
    )
    return {
        "sec_result": {
            "dashes_acc_num": sec_result.dashes_acc_num,
            "form_name": sec_result.form_name,
            "filing_date": sec_result.filing_date,
            "report_date": sec_result.report_date,
            "primary_document": sec_result.primary_document,
        },
        "pdf_path": str(pdf_path),
    }


class SecMainToMarkdownRequest(BaseModel):
    ticker: str
    year: str
    filing_type: str = "10-K"


@app.post("/sec_main_to_markdown")
async def sec_main_to_markdown_endpoint(request: SecMainToMarkdownRequest):
    payload = await sec_main_to_markdown(
        ticker=request.ticker,
        year=request.year,
        filing_type=request.filing_type,
    )
    sec_result = payload["sec_result"]
    return {
        "sec_result": {
            "dashes_acc_num": sec_result.dashes_acc_num,
            "form_name": sec_result.form_name,
            "filing_date": sec_result.filing_date,
            "report_date": sec_result.report_date,
            "primary_document": sec_result.primary_document,
        },
        "pdf_path": str(payload["pdf_path"]),
        "markdown_path": str(payload["markdown_path"]),
        "markdown": payload["markdown_text"],
    }


class SecMainToMarkdownEmbedRequest(BaseModel):
    ticker: str
    year: str
    filing_type: str = "10-K"
    force: bool = False


@app.post("/sec_main_to_markdown_and_embed")
async def sec_main_to_markdown_and_embed_endpoint(
    request: SecMainToMarkdownEmbedRequest,
):
    payload = await sec_main_to_markdown_and_embed(
        ticker=request.ticker,
        year=request.year,
        filing_type=request.filing_type,
        force=request.force,
    )
    sec_result = payload["sec_result"]
    return {
        "sec_result": {
            "dashes_acc_num": sec_result.dashes_acc_num,
            "form_name": sec_result.form_name,
            "filing_date": sec_result.filing_date,
            "report_date": sec_result.report_date,
            "primary_document": sec_result.primary_document,
        },
        "pdf_path": str(payload["pdf_path"]),
        "markdown_path": str(payload["markdown_path"]),
        "embedded": payload["embedded"],
    }


class RunOlmoOcrRequest(BaseModel):
    pdf_dir: str


@app.post("/run_olmo_ocr")
async def run_olmo_ocr_endpoint(request: RunOlmoOcrRequest):
    """Run OCR on PDFs in the given folder."""
    await run_olmo_ocr(pdf_dir=request.pdf_dir)
    return {"status": "completed", "pdf_dir": request.pdf_dir}


@app.delete("/worker_locks")
def delete_worker_locks():
    """Delete the configured olmOCR worker lock directory."""
    worker_locks_dir = Path(sec_settings.olmocr_workspace) / "worker_locks"
    existed = worker_locks_dir.exists()

    if existed and not worker_locks_dir.is_dir():
        raise HTTPException(
            status_code=400,
            detail=f"Worker locks path is not a directory: {worker_locks_dir}",
        )

    if existed:
        shutil.rmtree(worker_locks_dir)

    return {
        "status": "deleted" if existed else "not_found",
        "worker_locks_dir": str(worker_locks_dir),
    }


class SecFilingsEmbedRequest(BaseModel):
    """Build ChromaDB vectors from OCR markdown under the workspace.

    Reads all ``*.md`` in
    ``{olmocr_workspace}/markdown/{sec_data_dir}/{ticker}-{year}/``.
    Filing type is each file's stem (e.g. ``10-Q1.md`` → ``"10-Q1"``).
    """

    ticker: str
    year: str
    force: bool = False


class TranscriptEmbedRequest(BaseModel):
    ticker: str
    year: str
    force: bool = False


class TranscriptSearchRequest(BaseModel):
    ticker: str
    year: str
    query: str
    top_k: int = 5


@app.post("/vector_store/embed_sec_filings")
def embed_sec_filings(request: SecFilingsEmbedRequest):
    """Build and persist ChromaDB vectors from workspace SEC markdown for ticker/year.

    Returns the list of index keys that were built or already existed.
    """
    try:
        keys = vector_index.from_markdown_sec_filings(
            ticker=request.ticker,
            year=request.year,
            force=request.force,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return {
        "built": [
            {"ticker": k.ticker, "year": k.year, "filing_type": k.filing_type}
            for k in keys
        ]
    }


@app.post("/vector_store/embed_transcripts")
def embed_transcripts(request: TranscriptEmbedRequest):
    try:
        keys = vector_index.from_earnings_transcript_markdown(
            request.ticker,
            request.year,
            force=request.force,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return {
        "built": [
            {"ticker": k.ticker, "year": k.year, "filing_type": k.filing_type}
            for k in keys
        ]
    }


class SecFilingsListRequest(BaseModel):
    ticker: str
    year: str


@app.post("/vector_store/list_sec_filings")
def list_sec_filings(request: SecFilingsListRequest):
    """List ingested SEC filing types for a ticker and year.

    Returns each filing's type and its SEC submission date.
    """
    return vector_index.list_filings(request.ticker, request.year)


class SecFilingsSearchRequest(BaseModel):
    ticker: str
    year: str
    filing_type: str
    query: str
    top_k: int = 5


class ChunkResult(BaseModel):
    text: str
    chunk_type: str
    page_num: int | None
    section_title: str | None
    chunk_index: int
    score: float
    filing_type: str | None = None


@app.post("/vector_store/search_sec_filings", response_model=list[ChunkResult])
def search_sec_filings(request: SecFilingsSearchRequest):
    """Semantic search over one SEC filing (10-K, 10-Q, …) in Chroma.

    Build the index first via ``/vector_store/embed_sec_filings`` after
    ``/sec_main`` + ``/run_olmo_ocr``.

    Returns the top-k chunks by cosine similarity to the query embedding.
    """
    try:
        results = vector_index.search(
            ticker=request.ticker,
            year=request.year,
            filing_type=request.filing_type,
            query=request.query,
            top_k=request.top_k,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    return [
        ChunkResult(
            text=chunk.text,
            chunk_type=chunk.chunk_type,
            page_num=chunk.page_num,
            section_title=chunk.section_title,
            chunk_index=chunk.index,
            score=score,
        )
        for chunk, score in results
    ]


@app.post("/vector_store/search_transcripts", response_model=list[ChunkResult])
def search_transcripts(request: TranscriptSearchRequest):
    year_s = str(request.year).strip()
    resolved = vector_index.resolve_transcript_quarters(request.ticker, year_s)
    if not resolved:
        raise HTTPException(
            status_code=404,
            detail="No transcript indexes (Q1–Q4) for this ticker/year.",
        )
    ticker_key, quarters = resolved
    merged: list[tuple[Chunk, float, str]] = []
    for ft in quarters:
        try:
            hits = vector_index.search(
                ticker=ticker_key,
                year=year_s,
                filing_type=ft,
                query=request.query,
                top_k=request.top_k,
            )
        except FileNotFoundError:
            continue
        for chunk, score in hits:
            merged.append((chunk, score, ft))
    merged.sort(key=lambda item: -item[1])
    merged = merged[: request.top_k]
    if not merged:
        raise HTTPException(status_code=404, detail="No transcript search hits.")
    return [
        ChunkResult(
            text=chunk.text,
            chunk_type=chunk.chunk_type,
            page_num=chunk.page_num,
            section_title=chunk.section_title,
            chunk_index=chunk.index,
            score=score,
            filing_type=ft,
        )
        for chunk, score, ft in merged
    ]
