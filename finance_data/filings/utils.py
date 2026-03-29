import re
import asyncio
from typing import Final, NamedTuple, Optional, Union
from pathlib import Path
import os
import aiohttp
from loguru import logger
import yfinance as yf
from playwright.async_api import async_playwright, Browser

SEC_ARCHIVE_URL: Final[str] = "https://www.sec.gov/Archives/edgar/data"
SEC_VIEWER_URL: Final[str] = "https://www.sec.gov/ix?doc=/Archives/edgar/data"
SEC_SEARCH_URL: Final[str] = "http://www.sec.gov/cgi-bin/browse-edgar"
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions"


def company_to_ticker(name: str) -> str | None:
    """Resolve a company name to its stock ticker symbol via Yahoo finance_data.

    Args:
        name: The company name to look up (e.g. ``"Apple Inc"``).

    Returns:
        The ticker symbol string (e.g. ``"AAPL"``), or ``None`` if no match
        is found.
    """
    results = yf.Search(name).quotes

    if not results:
        return None

    return results[0]["symbol"]


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


def _sec_request_headers(
    company: Optional[str] = "Indiana-University-Bloomington",
    email: Optional[str] = "athecolab@gmail.com",
) -> dict[str, str]:
    """Build SEC request headers with a valid user agent."""
    if company is None:
        company = os.environ.get("SEC_API_ORGANIZATION")
    if email is None:
        email = os.environ.get("SEC_API_EMAIL")
    assert company
    assert email
    return {
        "User-Agent": f"{company} {email}",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
    }


def _search_url(cik: Union[str, int]) -> str:
    search_string = f"CIK={cik}&Find=Search&owner=exclude&action=getcompany"
    url = f"{SEC_SEARCH_URL}?{search_string}"
    return url


async def get_cik_by_ticker(ticker: str) -> str:
    """Gets a CIK number from a stock ticker by running a search on the SEC website."""
    cik_re = re.compile(r".*CIK=(\d{10}).*")
    url = _search_url(ticker)
    headers = _sec_request_headers()
    timeout = aiohttp.ClientTimeout(total=30)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get(url, headers=headers) as response:
            response.raise_for_status()
            response_text = await response.text()
    results = cik_re.findall(response_text)
    if not results:
        raise ValueError(f"Couldn't find the CIK for {ticker=}")
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


async def _download_filing_html(
    filing: FilingToSave,
    session: aiohttp.ClientSession,
    headers: dict[str, str],
    sem: asyncio.Semaphore,
) -> DownloadedFiling:
    output_path = Path(filing.output_path)
    url = document_url(filing.cik, filing.accession_number, filing.primary_document)
    async with sem:
        async with session.get(url, headers=headers) as response:
            response.raise_for_status()
            response_text = await response.text()
    return DownloadedFiling(
        html_content=response_text,
        base_url=url,
        output_path=output_path,
    )


async def _download_filing_html_with_logging(
    filing: FilingToSave,
    session: aiohttp.ClientSession,
    headers: dict[str, str],
    sem: asyncio.Semaphore,
) -> Optional[DownloadedFiling]:
    """Download one filing with structured logging and error handling."""
    url = document_url(filing.cik, filing.accession_number, filing.primary_document)
    logger.info(f"Fetching {url}")
    try:
        downloaded_filing = await _download_filing_html(
            filing=filing,
            session=session,
            headers=headers,
            sem=sem,
        )
        logger.info(f"Downloaded HTML: {url}")
        return downloaded_filing
    except Exception as exc:
        logger.error(f"Failed loading SEC filing HTML from {url}: {exc}")
        return None


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
    headers = _sec_request_headers(company, email)
    timeout = aiohttp.ClientTimeout(total=60)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        downloaded_filings = await asyncio.gather(
            *[
                _download_filing_html_with_logging(
                    filing=filing,
                    session=session,
                    headers=headers,
                    sem=sem,
                )
                for filing in filings
            ]
        )
    return [filing for filing in downloaded_filings if filing is not None]


async def _render_pdf_with_browser(
    browser: Browser,
    downloaded_filing: DownloadedFiling,
    sem: asyncio.Semaphore,
) -> Optional[Path]:
    """Render a single filing's HTML to PDF using a Playwright browser page."""
    output_path = downloaded_filing.output_path
    if output_path.exists():
        logger.info(f"Skipping existing PDF: {output_path}")
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    async with sem:
        page = await browser.new_page()
        try:
            await page.set_content(
                downloaded_filing.html_content,
                wait_until="networkidle",
            )
            await page.pdf(
                path=str(output_path), format="Letter", print_background=True
            )
            logger.info(f"Saved PDF: {output_path}")
            return output_path
        except Exception as exc:
            logger.error(f"Failed rendering PDF {output_path}: {exc}")
            return None
        finally:
            await page.close()


async def render_filings_to_pdfs(
    downloaded_filings: list[DownloadedFiling],
    max_concurrent: int = 4,
) -> list[Path]:
    """Render downloaded filing HTML to PDFs using headless Chromium via Playwright."""
    if not downloaded_filings:
        return []

    sem = asyncio.Semaphore(max_concurrent)
    async with async_playwright() as pw:
        browser = await pw.chromium.launch()
        tasks = [
            _render_pdf_with_browser(browser, filing, sem)
            for filing in downloaded_filings
        ]
        results = await asyncio.gather(*tasks)
        await browser.close()

    return [path for path in results if path is not None]


async def save_filings_as_pdfs(
    filings: list[FilingToSave],
    company: str,
    email: str,
    max_concurrent: int = 16,
) -> list[Path]:
    """Render each filing's primary .htm document to PDF via headless Chromium.

    Args:
        filings: List of FilingToSave named tuples.
        company: Company name for SEC User-Agent header.
        email: Contact e-mail for SEC User-Agent header.
        max_concurrent: Maximum number of simultaneous HTML downloads.

    Returns:
        List of Path objects pointing to the saved PDFs.
    """
    downloaded_filings = await download_filings_html_contents(
        filings=filings,
        company=company,
        email=email,
        max_concurrent=max_concurrent,
    )
    return await render_filings_to_pdfs(
        downloaded_filings=downloaded_filings,
    )
