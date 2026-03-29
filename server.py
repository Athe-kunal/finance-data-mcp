import asyncio
import dataclasses
import shutil
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException

from finance_data.dataloader.text_splitter import Chunk
from finance_data.dataloader.vector_store import ChromaVectorStore
from finance_data.earnings_transcripts.transcripts import (
    get_transcript_for_quarter_async,
)
from finance_data.dataloader.pipeline import sec_main_to_markdown_and_embed
from finance_data.filings.sec_data import (
    sec_main,
    sec_main_to_markdown,
)
from finance_data.filings.utils import company_to_ticker
from finance_data.ocr.olmocr_pipeline import run_olmo_ocr
from finance_data.server_api.batch_jobs import (
    expand_earnings_batch_jobs,
    expand_sec_batch_jobs,
    run_earnings_transcript_job,
    run_sec_markdown_embed_job,
    serialize_sec_result,
)
from finance_data.server_api.models import (
    BatchEarningsTranscriptsRequest,
    BatchSecFilingsRequest,
    ChunkResult,
    CompanyNameRequest,
    EarningsTranscriptQuarterRequest,
    RunOlmoOcrRequest,
    SecFilingsEmbedRequest,
    SecFilingsListRequest,
    SecFilingsSearchRequest,
    SecMainRequest,
    SecMainToMarkdownEmbedRequest,
    SecMainToMarkdownRequest,
    TranscriptEmbedRequest,
    TranscriptSearchRequest,
)
from finance_data.settings import sec_settings

vector_index: ChromaVectorStore


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: F811
    global vector_index
    vector_index = ChromaVectorStore()
    yield


app = FastAPI(lifespan=lifespan)


@app.post("/company_name_to_ticker")
def company_name_to_ticker(request: CompanyNameRequest):
    """Resolve a company name to its stock ticker symbol."""
    ticker = company_to_ticker(request.name)
    if ticker is None:
        raise HTTPException(status_code=404, detail="No ticker found for company name")
    return {"ticker": ticker}


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
        "sec_result": serialize_sec_result(sec_result),
        "pdf_path": str(pdf_path),
    }


@app.post("/sec_main_to_markdown")
async def sec_main_to_markdown_endpoint(request: SecMainToMarkdownRequest):
    payload = await sec_main_to_markdown(
        ticker=request.ticker,
        year=request.year,
        filing_type=request.filing_type,
    )
    sec_result = payload["sec_result"]
    return {
        "sec_result": serialize_sec_result(sec_result),
        "pdf_path": str(payload["pdf_path"]),
        "markdown_path": str(payload["markdown_path"]),
        "markdown": payload["markdown_text"],
    }


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
        "sec_result": serialize_sec_result(sec_result),
        "pdf_path": str(payload["pdf_path"]),
        "markdown_path": str(payload["markdown_path"]),
        "embedded": payload["embedded"],
    }


@app.post("/sec_main_to_markdown_and_embed/batch")
async def sec_main_to_markdown_and_embed_batch_endpoint(
    request: BatchSecFilingsRequest,
):
    """Run SEC download + markdown + embedding jobs in parallel."""
    jobs = expand_sec_batch_jobs(request.requests)
    tasks = [
        run_sec_markdown_embed_job(
            ticker=ticker,
            year=year,
            filing_type=filing_type,
            force=force,
        )
        for ticker, year, filing_type, force in jobs
    ]
    results = await asyncio.gather(*tasks)
    success_count = sum(1 for item in results if item["status"] == "success")
    error_count = len(results) - success_count
    return {
        "total_jobs": len(results),
        "success_count": success_count,
        "error_count": error_count,
        "results": results,
    }


@app.post("/earnings_transcripts/for_quarter/batch")
async def earnings_transcript_for_quarter_batch(
    request: BatchEarningsTranscriptsRequest,
):
    """Run earnings transcript downloads in parallel for all combinations."""
    jobs = expand_earnings_batch_jobs(request.requests)
    tasks = [
        run_earnings_transcript_job(
            ticker=ticker,
            year=year,
            quarter=quarter,
        )
        for ticker, year, quarter in jobs
    ]
    results = await asyncio.gather(*tasks)
    success_count = sum(1 for item in results if item["status"] == "success")
    not_found_count = sum(1 for item in results if item["status"] == "not_found")
    error_count = len(results) - success_count - not_found_count
    return {
        "total_jobs": len(results),
        "success_count": success_count,
        "not_found_count": not_found_count,
        "error_count": error_count,
        "results": results,
    }


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


@app.post("/vector_store/list_sec_filings")
def list_sec_filings(request: SecFilingsListRequest):
    """List ingested SEC filing types for a ticker and year.

    Returns each filing's type and its SEC submission date.
    """
    return vector_index.list_filings(request.ticker, request.year)


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
