"""Helpers for batch SEC and transcript processing."""

import dataclasses

from finance_data.earnings_transcripts.transcripts import (
    get_transcript_for_quarter_async,
)
from finance_data.dataloader.pipeline import sec_main_to_markdown_and_embed
from finance_data.filings.models import SecResults
from finance_data.server_api.models import (
    BatchEarningsTranscriptItem,
    BatchSecFilingItem,
)


def serialize_sec_result(sec_result: SecResults) -> dict[str, str]:
    """Convert SEC result dataclass/object into a JSON-friendly dict."""
    return {
        "dashes_acc_num": sec_result.dashes_acc_num,
        "form_name": sec_result.form_name,
        "filing_date": sec_result.filing_date,
        "report_date": sec_result.report_date,
        "primary_document": sec_result.primary_document,
    }


async def run_sec_markdown_embed_job(
    ticker: str,
    year: str,
    filing_type: str,
    force: bool,
) -> dict:
    """Execute one SEC download+markdown+embed job."""
    try:
        payload = await sec_main_to_markdown_and_embed(
            ticker=ticker,
            year=year,
            filing_type=filing_type,
            force=force,
        )
        return {
            "ticker": ticker,
            "year": year,
            "filing_type": filing_type,
            "status": "success",
            "sec_result": serialize_sec_result(payload["sec_result"]),
            "pdf_path": str(payload["pdf_path"]),
            "markdown_path": str(payload["markdown_path"]),
            "embedded": payload["embedded"],
        }
    except Exception as exc:
        return {
            "ticker": ticker,
            "year": year,
            "filing_type": filing_type,
            "status": "error",
            "error": str(exc),
        }


async def run_earnings_transcript_job(
    ticker: str,
    year: int,
    quarter: str,
) -> dict:
    """Execute one transcript download job."""
    transcript = await get_transcript_for_quarter_async(ticker, year, quarter)
    if transcript is None:
        return {
            "ticker": ticker,
            "year": year,
            "quarter": quarter,
            "status": "not_found",
            "error": "Transcript not available for this ticker, year, and quarter",
        }

    return {
        "ticker": ticker,
        "year": year,
        "quarter": quarter,
        "status": "success",
        "transcript": dataclasses.asdict(transcript),
    }


def expand_sec_batch_jobs(
    requests: list[BatchSecFilingItem],
) -> list[tuple[str, str, str, bool]]:
    """Build SEC jobs as ticker/year/filing_type/force tuples."""
    jobs: list[tuple[str, str, str, bool]] = []
    for item in requests:
        for filing_type in item.filing_types:
            jobs.append((item.ticker, item.year, filing_type, item.force))
    return jobs


def expand_earnings_batch_jobs(
    requests: list[BatchEarningsTranscriptItem],
) -> list[tuple[str, int, str]]:
    """Build transcript jobs as ticker/year/quarter tuples."""
    jobs: list[tuple[str, int, str]] = []
    for item in requests:
        for year in item.years:
            for quarter in item.quarters:
                jobs.append((item.ticker, year, quarter))
    return jobs
