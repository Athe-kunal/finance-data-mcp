"""Filesystem-backed processed-data index for fast local existence checks."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
import re
import threading
from loguru import logger
from finance_data.settings import sec_settings

_SEC_CASE_DIR_RE = re.compile(r"^(?P<ticker>.+)-(?P<year>\d{4})$")
_TRANSCRIPT_FILENAME_RE = re.compile(
    r"^Q(?P<quarter>[1-4])(?:_.+)?\.md$", re.IGNORECASE
)


@dataclass(frozen=True)
class ProcessedDataSnapshot:
    sec_filings: frozenset[tuple[str, str, str]]
    sec_markdown_filings: frozenset[tuple[str, str, str]]
    transcript_quarters: frozenset[tuple[str, str, str]]


class ProcessedDataIndex:
    """Caches processed SEC filings/transcripts discovered on local storage."""

    def __init__(
        self,
        sec_data_dir: str,
        sec_markdown_dir: str,
        transcripts_dir: str,
        max_workers: int,
        ignore_ocr: bool,
    ) -> None:
        self._sec_data_dir = Path(sec_data_dir)
        self._sec_markdown_dir = Path(sec_markdown_dir)
        self._transcripts_dir = Path(transcripts_dir)
        self._max_workers = max(1, max_workers)
        self._ignore_ocr = ignore_ocr
        self._lock = threading.Lock()
        self._snapshot = ProcessedDataSnapshot(
            sec_filings=frozenset(),
            sec_markdown_filings=frozenset(),
            transcript_quarters=frozenset(),
        )
        self.refresh()

    def refresh(self) -> None:
        """Rebuild both indexes from disk in parallel."""
        with ThreadPoolExecutor(max_workers=2) as pool:
            sec_future = pool.submit(self._scan_sec_filings)
            markdown_future = pool.submit(self._scan_sec_markdown_filings)
            transcript_future = pool.submit(self._scan_transcripts)
            sec_filings = sec_future.result()
            sec_markdown_filings = markdown_future.result()
            transcript_quarters = transcript_future.result()
        snapshot = ProcessedDataSnapshot(
            sec_filings=frozenset(sec_filings),
            sec_markdown_filings=frozenset(sec_markdown_filings),
            transcript_quarters=frozenset(transcript_quarters),
        )
        with self._lock:
            self._snapshot = snapshot

    def has_sec_filing(self, ticker: str, year: str, filing_type: str) -> bool:
        key = _normalized_sec_key(ticker, year, filing_type)
        with self._lock:
            in_snapshot = (
                key in self._snapshot.sec_filings
                or key in self._snapshot.sec_markdown_filings
            )
        if in_snapshot and self._sec_filing_exists_on_disk(*key):
            logger.info(
                f"Processed-data cache hit for SEC filing ticker={key[0]} year={key[1]} filing_type={key[2]}.",
            )
            return True
        if self._sec_filing_exists_on_disk(*key):
            self.mark_sec_filing(*key)
            return True
        return False

    def list_sec_filings(self, ticker: str, year: str) -> list[str]:
        ticker_key = ticker.upper().strip()
        year_key = str(year).strip()
        with self._lock:
            sec_items = [
                filing_type
                for t, y, filing_type in self._snapshot.sec_filings
                if t == ticker_key and y == year_key
            ]
            markdown_items = [
                filing_type
                for t, y, filing_type in self._snapshot.sec_markdown_filings
                if t == ticker_key and y == year_key
            ]
        all_items = sorted(set(sec_items + markdown_items))
        return [
            filing_type
            for filing_type in all_items
            if self._sec_filing_exists_on_disk(ticker_key, year_key, filing_type)
        ]

    def mark_sec_filing(self, ticker: str, year: str, filing_type: str) -> None:
        key = _normalized_sec_key(ticker, year, filing_type)
        with self._lock:
            sec_filings = set(self._snapshot.sec_filings)
            sec_filings.add(key)
            self._snapshot = ProcessedDataSnapshot(
                sec_filings=frozenset(sec_filings),
                sec_markdown_filings=self._snapshot.sec_markdown_filings,
                transcript_quarters=self._snapshot.transcript_quarters,
            )

    def has_transcript(self, ticker: str, year: str, quarter: str) -> bool:
        key = _normalized_transcript_key(ticker, year, quarter)
        with self._lock:
            in_snapshot = key in self._snapshot.transcript_quarters
        if in_snapshot and self._transcript_exists_on_disk(*key):
            logger.info(
                f"Processed-data cache hit for transcript ticker={key[0]} year={key[1]} quarter={key[2]}.",
            )
            return True
        if self._transcript_exists_on_disk(*key):
            self.mark_transcript(*key)
            return True
        return False

    def mark_transcript(self, ticker: str, year: str, quarter: str) -> None:
        key = _normalized_transcript_key(ticker, year, quarter)
        with self._lock:
            transcript_quarters = set(self._snapshot.transcript_quarters)
            transcript_quarters.add(key)
            self._snapshot = ProcessedDataSnapshot(
                sec_filings=self._snapshot.sec_filings,
                sec_markdown_filings=self._snapshot.sec_markdown_filings,
                transcript_quarters=frozenset(transcript_quarters),
            )

    def _scan_sec_filings(self) -> set[tuple[str, str, str]]:
        if not self._sec_data_dir.exists():
            return set()
        paths = list(self._sec_data_dir.glob("*/*.pdf"))
        results: set[tuple[str, str, str]] = set()
        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            for parsed in pool.map(self._parse_sec_filing_path, paths):
                if parsed is not None:
                    results.add(parsed)
        return results

    def _scan_sec_markdown_filings(self) -> set[tuple[str, str, str]]:
        if not self._sec_markdown_dir.exists():
            return set()
        paths = list(self._sec_markdown_dir.glob("*/*.md"))
        results: set[tuple[str, str, str]] = set()
        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            for parsed in pool.map(self._parse_sec_markdown_path, paths):
                if parsed is not None:
                    results.add(parsed)
        return results

    def _scan_transcripts(self) -> set[tuple[str, str, str]]:
        if not self._transcripts_dir.exists():
            return set()
        paths = list(self._transcripts_dir.glob("*/*/Q*.md"))
        results: set[tuple[str, str, str]] = set()
        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            for parsed in pool.map(self._parse_transcript_path, paths):
                if parsed is not None:
                    results.add(parsed)
        return results

    def _parse_sec_filing_path(self, pdf_path: Path) -> tuple[str, str, str] | None:
        case_match = _SEC_CASE_DIR_RE.match(pdf_path.parent.name)
        if case_match is None:
            return None
        ticker = case_match.group("ticker").upper().strip()
        year = case_match.group("year").strip()
        filing_type = pdf_path.stem.upper().strip()
        if not ticker or not year or not filing_type:
            return None
        return ticker, year, filing_type

    def _parse_transcript_path(
        self, transcript_path: Path
    ) -> tuple[str, str, str] | None:
        quarter_match = _TRANSCRIPT_FILENAME_RE.match(transcript_path.name)
        if quarter_match is None:
            return None
        ticker = transcript_path.parent.parent.name.upper().strip()
        year = transcript_path.parent.name.strip()
        quarter = f"Q{quarter_match.group('quarter')}"
        if not ticker or not year:
            return None
        return ticker, year, quarter

    def _parse_sec_markdown_path(
        self,
        markdown_path: Path,
    ) -> tuple[str, str, str] | None:
        case_match = _SEC_CASE_DIR_RE.match(markdown_path.parent.name)
        if case_match is None:
            return None
        ticker = case_match.group("ticker").upper().strip()
        year = case_match.group("year").strip()
        filing_type = markdown_path.stem.upper().strip()
        if not ticker or not year or not filing_type:
            return None
        return ticker, year, filing_type

    def _sec_filing_exists_on_disk(
        self,
        ticker: str,
        year: str,
        filing_type: str,
    ) -> bool:
        case_key = f"{ticker}-{year}"
        pdf_dir = self._sec_data_dir / case_key
        md_dir = self._sec_markdown_dir / case_key
        if self._ignore_ocr:
            return self._sec_pdf_exists(pdf_dir, filing_type)
        return self._sec_markdown_exists(md_dir, filing_type)

    def _sec_pdf_exists(self, pdf_dir: Path, filing_type: str) -> bool:
        if filing_type == "10-K":
            return self._glob_any(pdf_dir, "10-K*.pdf")
        if re.fullmatch(r"10-Q[1-3]", filing_type):
            return self._glob_any(pdf_dir, f"{filing_type}*.pdf")
        return (pdf_dir / f"{filing_type}.pdf").exists()

    def _sec_markdown_exists(self, markdown_dir: Path, filing_type: str) -> bool:
        if filing_type == "10-K":
            return self._glob_any(markdown_dir, "10-K*.md")
        if re.fullmatch(r"10-Q[1-3]", filing_type):
            return self._glob_any(markdown_dir, f"{filing_type}*.md")
        return (markdown_dir / f"{filing_type}.md").exists()

    def _transcript_exists_on_disk(self, ticker: str, year: str, quarter: str) -> bool:
        transcript_dir = self._transcripts_dir / ticker / year
        if not transcript_dir.is_dir():
            return False
        return self._glob_any(transcript_dir, f"{quarter}_*.md")

    def _glob_any(self, root: Path, pattern: str) -> bool:
        if not root.is_dir():
            return False
        return any(root.glob(pattern))


def _normalized_sec_key(
    ticker: str, year: str, filing_type: str
) -> tuple[str, str, str]:
    return ticker.upper().strip(), str(year).strip(), filing_type.upper().strip()


def _normalized_transcript_key(
    ticker: str, year: str, quarter: str
) -> tuple[str, str, str]:
    return ticker.upper().strip(), str(year).strip(), quarter.upper().strip()


processed_data_index = ProcessedDataIndex(
    sec_data_dir=sec_settings.sec_data_dir,
    sec_markdown_dir=(
        Path(sec_settings.olmocr_workspace) / "markdown" / sec_settings.sec_data_dir
    ).as_posix(),
    transcripts_dir=sec_settings.earnings_transcripts_dir,
    max_workers=sec_settings.processed_index_max_workers,
    ignore_ocr=sec_settings.ignore_ocr,
)
