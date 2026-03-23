"""Function-level API that can be used without running the HTTP server."""

from __future__ import annotations

import asyncio
import dataclasses
from collections.abc import Sequence


def company_name_to_ticker(name: str) -> str | None:
    """Resolve a company name to ticker symbol."""
    from filings.utils import company_to_ticker

    return company_to_ticker(name)


async def fetch_sec_filings(
    ticker: str,
    year: str,
    filing_types: Sequence[str] = ("10-K", "10-Q"),
    include_amends: bool = True,
) -> dict:
    """Fetch SEC filings and return the same payload shape as the server endpoint."""
    from filings.sec_data import sec_main

    sec_results, pdf_paths = await sec_main(
        ticker=ticker,
        year=year,
        filing_types=list(filing_types),
        include_amends=include_amends,
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


def fetch_sec_filings_sync(
    ticker: str,
    year: str,
    filing_types: Sequence[str] = ("10-K", "10-Q"),
    include_amends: bool = True,
) -> dict:
    """Synchronous wrapper for `fetch_sec_filings`."""
    return asyncio.run(
        fetch_sec_filings(
            ticker=ticker,
            year=year,
            filing_types=filing_types,
            include_amends=include_amends,
        )
    )


async def fetch_earnings_transcript_for_quarter(
    ticker: str, year: int, quarter: str
) -> dict | None:
    """Fetch and serialize one quarterly transcript. Returns None if unavailable.

    ``quarter`` must be a label such as ``Q1``, ``Q2``, ``Q3``, or ``Q4``.
    """
    from earnings_transcripts.transcripts import get_transcript_for_quarter_async

    transcript = await get_transcript_for_quarter_async(ticker, year, quarter)
    if transcript is None:
        return None
    return dataclasses.asdict(transcript)


def fetch_earnings_transcript_for_quarter_sync(
    ticker: str, year: int, quarter: str
) -> dict | None:
    """Synchronous wrapper for `fetch_earnings_transcript_for_quarter`."""
    return asyncio.run(
        fetch_earnings_transcript_for_quarter(ticker, year, quarter)
    )


async def run_olmo_ocr(pdf_dir: str) -> None:
    """Run olmOCR over all PDFs in a directory."""
    from ocr.olmocr_pipeline import run_olmo_ocr as _run_olmo_ocr

    await _run_olmo_ocr(pdf_dir=pdf_dir)


def run_olmo_ocr_sync(pdf_dir: str) -> None:
    """Synchronous wrapper for `run_olmo_ocr`."""
    asyncio.run(run_olmo_ocr(pdf_dir=pdf_dir))
