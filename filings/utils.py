import re
import asyncio
from concurrent.futures import ProcessPoolExecutor
import requests
from typing import Final, NamedTuple, Optional, Union
from pathlib import Path
import os
from loguru import logger
from ratelimit import limits, sleep_and_retry
import weasyprint

SEC_ARCHIVE_URL: Final[str] = "https://www.sec.gov/Archives/edgar/data"
SEC_VIEWER_URL: Final[str] = "https://www.sec.gov/ix?doc=/Archives/edgar/data"
SEC_SEARCH_URL: Final[str] = "http://www.sec.gov/cgi-bin/browse-edgar"
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions"


def _drop_dashes(accession_number: Union[str, int]) -> str:
    """Converts the accession number to the no dash representation."""
    accession_number = str(accession_number).replace("-", "")
    return accession_number.zfill(18)


def _add_dashes(accession_number: Union[str, int]) -> str:
    """Adds the dashes back into the accession number"""
    accession_number = str(accession_number).replace("-", "").zfill(18)
    return f"{accession_number[:10]}-{accession_number[10:12]}-{accession_number[12:]}"


def archive_url(cik: Union[str, int], accession_number: Union[str, int]) -> str:
    """Builds the archive URL for the SEC accession number."""
    filename = f"{_add_dashes(accession_number)}.txt"
    accession_number = _drop_dashes(accession_number)
    return f"{SEC_ARCHIVE_URL}/{cik}/{accession_number}/{filename}"


def _get_session(
    company: Optional[str] = "Indiana-University-Bloomington",
    email: Optional[str] = "athecolab@gmail.com",
) -> requests.Session:
    """Creates a requests sessions with the appropriate headers set."""
    if company is None:
        company = os.environ.get("SEC_API_ORGANIZATION")
    if email is None:
        email = os.environ.get("SEC_API_EMAIL")
    assert company
    assert email
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": f"{company} {email}",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
        }
    )
    return session


def _search_url(cik: Union[str, int]) -> str:
    search_string = f"CIK={cik}&Find=Search&owner=exclude&action=getcompany"
    url = f"{SEC_SEARCH_URL}?{search_string}"
    return url


@sleep_and_retry
@limits(calls=2, period=1)
def get_cik_by_ticker(ticker: str) -> str:
    """Gets a CIK number from a stock ticker by running a search on the SEC website."""
    cik_re = re.compile(r".*CIK=(\d{10}).*")
    url = _search_url(ticker)
    company = "Indiana-University-Bloomington"
    email = "athecolab@gmail.com"
    headers = {
        "User-Agent": f"{company} {email}",
        "Content-Type": "text/html",
    }
    response = requests.get(url, stream=True, headers=headers)
    response.raise_for_status()
    results = cik_re.findall(response.text)
    return str(results[0])


def viewer_url(
    cik: Union[str, int],
    accession_number: Union[str, int],
    primary_document: str,
) -> str:
    """Builds the SEC inline XBRL viewer URL for the primary .htm document."""
    acc_no_dashes = _drop_dashes(accession_number)
    return f"{SEC_VIEWER_URL}/{cik}/{acc_no_dashes}/{primary_document}"


def document_url(
    cik: Union[str, int],
    accession_number: Union[str, int],
    primary_document: str,
) -> str:
    """Builds the direct archive URL for the primary .htm document.

    Unlike the XBRL viewer URL, this endpoint is accessible to automated
    clients that supply a valid SEC User-Agent header.
    """
    acc_no_dashes = _drop_dashes(accession_number)
    return f"{SEC_ARCHIVE_URL}/{cik}/{acc_no_dashes}/{primary_document}"


class FilingToSave(NamedTuple):
    """A single filing to render to PDF."""

    cik: Union[str, int]
    accession_number: Union[str, int]
    primary_document: str
    output_path: Union[str, Path]


class DownloadedFiling(NamedTuple):
    """Downloaded HTML content plus the output location for PDF rendering."""

    html_content: str
    base_url: str
    output_path: Path


def _download_filing_html(
    filing: FilingToSave,
    headers: dict[str, str],
) -> DownloadedFiling:
    output_path = Path(filing.output_path)
    url = document_url(filing.cik, filing.accession_number, filing.primary_document)
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return DownloadedFiling(
        html_content=response.text,
        base_url=url,
        output_path=output_path,
    )


async def download_filings_html_contents(
    filings: list[FilingToSave],
    company: str,
    email: str,
    max_concurrent: int = 16,
) -> list[DownloadedFiling]:
    """Download filing HTML documents with bounded concurrency."""
    if not filings:
        return []

    sem = asyncio.Semaphore(max_concurrent)
    headers = dict(_get_session(company, email).headers)

    async def _download_one(filing: FilingToSave) -> Optional[DownloadedFiling]:
        async with sem:
            url = document_url(
                filing.cik,
                filing.accession_number,
                filing.primary_document,
            )
            logger.info(f"Fetching {url}")
            try:
                downloaded_filing = await asyncio.to_thread(
                    _download_filing_html,
                    filing,
                    headers,
                )
                logger.info(f"Downloaded HTML: {url}")
                return downloaded_filing
            except Exception as exc:
                logger.error(f"Failed loading SEC filing HTML from {url}: {exc}")
                return None

    downloaded_filings = await asyncio.gather(
        *[_download_one(filing) for filing in filings]
    )
    return [filing for filing in downloaded_filings if filing is not None]


def _render_pdf(html_content: str, base_url: str, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    weasyprint.HTML(string=html_content, base_url=base_url).write_pdf(str(output_path))
    return output_path


def _render_downloaded_filing(downloaded_filing: DownloadedFiling) -> Path:
    return _render_pdf(
        downloaded_filing.html_content,
        downloaded_filing.base_url,
        downloaded_filing.output_path,
    )


def render_filings_to_pdfs(
    downloaded_filings: list[DownloadedFiling],
    max_workers: Optional[int] = 8,
) -> list[Path]:
    """Render downloaded filing HTML to PDFs using a process pool."""
    if not downloaded_filings:
        return []
    worker_count = max_workers or min(
        len(downloaded_filings), (os.cpu_count() or 1) // 2
    )
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        results = list(executor.map(_render_downloaded_filing, downloaded_filings))

    for pdf_path in results:
        logger.info(f"Saved PDF: {pdf_path}")
    return list(results)


async def save_filings_as_pdfs(
    filings: list[FilingToSave],
    company: str,
    email: str,
    max_concurrent: int = 16,
) -> list[Path]:
    """Render each filing's primary .htm document to PDF via WeasyPrint.

    Args:
        filings: List of FilingToSave named tuples.
        company: Company name for SEC User-Agent header.
        email: Contact e-mail for SEC User-Agent header.
        max_concurrent: Maximum number of simultaneous HTML downloads.

    Returns:
        List of Path objects pointing to the saved PDFs (same order as input).
    """
    downloaded_filings = await download_filings_html_contents(
        filings=filings,
        company=company,
        email=email,
        max_concurrent=max_concurrent,
    )
    return render_filings_to_pdfs(
        downloaded_filings=downloaded_filings,
    )
