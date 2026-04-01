import dataclasses
import shutil
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Callable

from fastapi import FastAPI, HTTPException

from finance_data.earnings_transcripts.transcripts import (
    get_transcript_for_quarter_async,
)
from finance_data.dataloader.pipeline import (
    earnings_transcripts_main_and_embed,
    sec_main_to_markdown_and_embed,
)
from finance_data.filings.sec_data import (
    sec_main,
    sec_main_to_markdown,
)
from finance_data.filings.utils import company_to_ticker
from finance_data.server_api.batch_jobs import (
    expand_earnings_batch_jobs,
    expand_sec_batch_jobs,
    run_jobs_with_limit,
    run_earnings_transcript_job,
    run_sec_markdown_embed_job,
    serialize_sec_result,
)
from finance_data.server_api.models import (
    BatchEarningsTranscriptsRequest,
    BatchSecFilingsRequest,
    ChunkResult,
    CompanyNameRequest,
    EarningsTranscriptQuarterEmbedRequest,
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

vector_index: Any | None = None


def _load_vector_store_class() -> Any:
    try:
        from finance_data.dataloader.vector_store import ChromaVectorStore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Vector store dependencies are not installed. Install with "
            "`uv sync --group ocr-md` to use vector endpoints."
        ) from exc
    return ChromaVectorStore


def _require_vector_index() -> Any:
    if vector_index is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Vector index is unavailable because optional dependencies are "
                "missing. Install with `uv sync --group ocr-md`."
            ),
        )
    return vector_index


def _load_run_olmo_ocr() -> Any:
    try:
        from finance_data.ocr.olmocr_pipeline import run_olmo_ocr
    except ModuleNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                "OCR dependencies are not installed. Install with "
                "`uv sync --group ocr-md` to use OCR endpoints."
            ),
        ) from exc
    return run_olmo_ocr


def _search_chunks(
    index: Any,
    *,
    ticker: str,
    year: str,
    filing_type: str,
    query: str,
    top_k: int,
) -> list[tuple[Any, float]]:
    return index.hybrid_search(
        ticker=ticker,
        year=year,
        filing_type=filing_type,
        query=query,
        top_k=top_k,
    )


def _search_transcript_chunks(
    index: Any,
    *,
    ticker: str,
    year: str,
    query: str,
    top_k: int,
    quarter: str | None,
    search_fn: Callable[..., list[tuple[Any, float]]],
) -> list[tuple[Any, float, str]]:
    resolved = index.resolve_transcript_quarters(ticker, year)
    if not resolved:
        raise HTTPException(
            status_code=404,
            detail="No transcript indexes (Q1–Q4) for this ticker/year.",
        )

    ticker_key, quarters = resolved
    if quarter is not None:
        q_key = quarter.upper()
        if q_key not in quarters:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"No indexed transcript for {q_key} for this ticker/year "
                    f"(available: {', '.join(quarters)})."
                ),
            )
        quarters = [q_key]

    merged: list[tuple[Any, float, str]] = []
    for filing_type in quarters:
        try:
            hits = search_fn(
                index=index,
                ticker=ticker_key,
                year=year,
                filing_type=filing_type,
                query=query,
                top_k=top_k,
            )
        except FileNotFoundError:
            continue
        for chunk, score in hits:
            merged.append((chunk, score, filing_type))

    merged.sort(key=lambda item: -item[1])
    return merged[:top_k]


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: F811
    global vector_index
    try:
        vector_store_cls = _load_vector_store_class()
    except ModuleNotFoundError:
        vector_index = None
    else:
        vector_index = vector_store_cls()
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


@app.post("/earnings_transcripts/main_and_embed")
async def earnings_transcripts_main_and_embed_endpoint(
    request: EarningsTranscriptQuarterEmbedRequest,
):
    try:
        return await earnings_transcripts_main_and_embed(
            ticker=request.ticker,
            year=request.year,
            quarter=request.quarter,
            force=request.force,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


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
    coroutines = [
        run_sec_markdown_embed_job(
            ticker=ticker,
            year=year,
            filing_type=filing_type,
            force=force,
        )
        for ticker, year, filing_type, force in jobs
    ]
    results = await run_jobs_with_limit(
        coroutines,
        max_concurrent=sec_settings.batch_max_concurrent_jobs,
    )
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
    coroutines = [
        run_earnings_transcript_job(
            ticker=ticker,
            year=year,
            quarter=quarter,
        )
        for ticker, year, quarter in jobs
    ]
    results = await run_jobs_with_limit(
        coroutines,
        max_concurrent=sec_settings.batch_max_concurrent_jobs,
    )
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
    run_olmo_ocr = _load_run_olmo_ocr()
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
    index = _require_vector_index()
    try:
        keys = index.from_markdown_sec_filings(
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
    index = _require_vector_index()
    try:
        keys = index.from_earnings_transcript_markdown(
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
    index = _require_vector_index()
    return index.list_filings(request.ticker, request.year)


@app.post("/vector_store/search_sec_filings", response_model=list[ChunkResult])
def search_sec_filings(request: SecFilingsSearchRequest):
    """Hybrid search over one SEC filing (10-K, 10-Q, …) in Chroma.

    Build the index first via ``/vector_store/embed_sec_filings`` after
    ``/sec_main`` + ``/run_olmo_ocr``.

    Returns the top-k chunks after dense + BM25 fusion and reranking.
    """
    index = _require_vector_index()
    try:
        results = _search_chunks(
            index,
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
    index = _require_vector_index()
    year_s = str(request.year).strip()
    merged = _search_transcript_chunks(
        index,
        ticker=request.ticker,
        year=year_s,
        query=request.query,
        top_k=request.top_k,
        quarter=request.quarter,
        search_fn=_search_chunks,
    )
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
