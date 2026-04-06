"""Helpers for batch SEC and transcript processing."""

import asyncio
import dataclasses
from collections.abc import Awaitable, Iterable
from typing import NamedTuple, TypeVar

from finance_data.common import processed_data_index
from finance_data.earnings_transcripts.transcripts import (
    get_transcript_for_quarter_async,
)
from finance_data.dataloader.pipeline import sec_main_to_markdown_and_embed
from finance_data.filings.models import SecResults
from finance_data.server_api.models import (
    BatchEarningsTranscriptItem,
    BatchSecFilingItem,
)

_T = TypeVar("_T")


class _SecBatchJob(NamedTuple):
    """Normalized SEC batch job payload."""

    ticker: str
    year: str
    filing_type: str
    force: bool


class _TranscriptBatchJob(NamedTuple):
    """Normalized earnings transcript batch job payload."""

    ticker: str
    year: int
    quarter: str


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
) -> list[_SecBatchJob]:
    """Build SEC jobs as ticker/year/filing_type/force tuples."""
    jobs: list[_SecBatchJob] = []
    for item in requests:
        for filing_type in item.filing_types:
            filing_key = filing_type.strip().upper()
            if processed_data_index.has_sec_filing(
                item.ticker,
                item.year,
                filing_key,
            ):
                continue
            jobs.append(
                _SecBatchJob(
                    ticker=item.ticker,
                    year=item.year,
                    filing_type=filing_type,
                    force=item.force,
                )
            )
    return jobs


def expand_earnings_batch_jobs(
    requests: list[BatchEarningsTranscriptItem],
) -> list[_TranscriptBatchJob]:
    """Build transcript jobs as ticker/year/quarter tuples."""
    jobs: list[_TranscriptBatchJob] = []
    for item in requests:
        for year in item.years:
            for quarter in item.quarters:
                quarter_key = quarter.strip().upper()
                if processed_data_index.has_transcript(
                    item.ticker,
                    str(year),
                    quarter_key,
                ):
                    continue
                jobs.append(
                    _TranscriptBatchJob(
                        ticker=item.ticker,
                        year=year,
                        quarter=quarter,
                    )
                )
    return jobs


async def run_jobs_with_limit(
    coroutines: Iterable[Awaitable[_T]],
    max_concurrent: int,
) -> list[_T]:
    """Execute coroutine jobs with bounded concurrency."""
    if max_concurrent <= 0:
        raise ValueError("max_concurrent must be positive")

    semaphore = asyncio.Semaphore(max_concurrent)
    tasks = [
        asyncio.create_task(_run_one_job(coroutine, semaphore))
        for coroutine in coroutines
    ]
    if not tasks:
        return []
    return await asyncio.gather(*tasks)


async def _run_one_job(coroutine: Awaitable[_T], semaphore: asyncio.Semaphore) -> _T:
    async with semaphore:
        return await coroutine
