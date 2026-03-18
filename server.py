import shutil
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from dataloader.vector_store import FaissVectorIndex
from filings.utils import company_to_ticker
from filings.sec_data import sec_main
from ocr.olmocr_pipeline import run_olmo_ocr
from settings import olmocr_settings

vector_index: FaissVectorIndex


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: F811
    global vector_index
    vector_index = FaissVectorIndex()
    for key in vector_index.list_indexes():
        try:
            vector_index._load_filing(key)
        except Exception:
            pass
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
    filing_types: list[str] = ["10-K", "10-Q"]
    include_amends: bool = True


@app.post("/sec_main")
async def sec_main_endpoint(request: SecMainRequest):
    """Fetch SEC filings and save them as PDFs."""
    sec_results, pdf_paths = await sec_main(
        ticker=request.ticker,
        year=request.year,
        filing_types=request.filing_types,
        include_amends=request.include_amends,
    )
    return {
        "sec_results": [
            {
                "dashes_acc_num": r.dashes_acc_num,
                "form_name": r.form_name,
                "filing_date": r.filing_date,
                "report_date": r.report_date,
                "primary_document": r.primary_document,
            }
            for r in sec_results
        ],
        "pdf_paths": [str(p) for p in pdf_paths],
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
    worker_locks_dir = Path(olmocr_settings.olmocr_workspace) / "worker_locks"
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


class VectorEmbedRequest(BaseModel):
    """Build FAISS indexes from already-OCR'd markdown files.

    ``markdown_dir`` should be the folder that contains ``{filing_type}.md``
    files (e.g. ``localworkspace/markdown/sec_data/AMZN-2025``).  All ``.md``
    files found in that directory are indexed; the filing type is derived from
    each file's stem (e.g. ``10-Q1.md`` → ``"10-Q1"``).
    """

    ticker: str
    year: str
    markdown_dir: str
    force: bool = False


@app.post("/vector_store/embed")
def vector_store_embed(request: VectorEmbedRequest):
    """Build and persist FAISS indexes from all markdown files in a directory.

    Discovers every ``*.md`` file in ``markdown_dir`` and calls
    ``from_markdown()``.  The filing type is extracted from each file stem.

    Returns the list of index keys that were built or already existed.
    """
    md_dir = Path(request.markdown_dir)
    if not md_dir.exists():
        raise HTTPException(
            status_code=400,
            detail=f"markdown_dir does not exist: {md_dir}",
        )

    md_paths: list[Path] = sorted(md_dir.glob("*.md"))
    if not md_paths:
        raise HTTPException(
            status_code=400,
            detail=f"No .md files found in {md_dir}",
        )

    try:
        keys = vector_index.from_markdown(
            ticker=request.ticker,
            year=request.year,
            markdown_paths=md_paths,
            force=request.force,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return {
        "built": [
            {"ticker": k.ticker, "year": k.year, "filing_type": k.filing_type}
            for k in keys
        ]
    }


class ListFilingsRequest(BaseModel):
    ticker: str
    year: str


@app.post("/vector_store/list_filings")
def vector_store_list_filings(request: ListFilingsRequest):
    """List all ingested filings for a ticker and year.

    Returns each filing's type and its SEC submission date.
    """
    return vector_index.list_filings(request.ticker, request.year)


class VectorSearchRequest(BaseModel):
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


@app.post("/vector_store/search", response_model=list[ChunkResult])
def vector_store_search(request: VectorSearchRequest):
    """Semantic search over a single filing's FAISS index.

    The index must have been built first via ``/vector_store/embed`` or
    ``/sec_main`` + ``/run_olmo_ocr`` + ``/vector_store/embed``.

    Returns the top-k most relevant chunks with their cosine similarity scores.
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
