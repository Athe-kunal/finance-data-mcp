"""Public API for using this project as a Python package.

This module is intentionally lightweight so consumers can import `finance_data`
without starting (or depending on) the FastAPI server runtime.
"""

from .api import (
    company_name_to_ticker,
    fetch_earnings_transcripts_for_year,
    fetch_earnings_transcripts_for_year_sync,
    fetch_sec_filings,
    fetch_sec_filings_sync,
    run_olmo_ocr,
    run_olmo_ocr_sync,
)
from .app import get_app

__all__ = [
    "company_name_to_ticker",
    "fetch_earnings_transcripts_for_year",
    "fetch_earnings_transcripts_for_year_sync",
    "fetch_sec_filings",
    "fetch_sec_filings_sync",
    "run_olmo_ocr",
    "run_olmo_ocr_sync",
    "get_app",
]
