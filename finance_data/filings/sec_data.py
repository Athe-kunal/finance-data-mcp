import re
import asyncio
import aiohttp
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

from . import utils
from finance_data.settings import sec_settings
from finance_data.ocr.olmocr_pipeline import get_markdown_path, run_olmo_ocr
from finance_data.dataloader.vector_store import ChromaVectorStore, IndexKey


def sec_data_case_dir(ticker: str, year: str) -> Path:
    """Directory for one ticker/year: ``{sec_data_dir}/{ticker}-{year}/``."""
    return Path(sec_settings.sec_data_dir) / f"{ticker}-{year}"


@dataclass(frozen=True)
class SecResults:
    dashes_acc_num: str
    form_name: str
    filing_date: str
    report_date: str
    primary_document: str


def _parse_filing_type_for_sec_query(
    filing_type: str,
) -> tuple[frozenset[int] | None, frozenset[str]]:
    """Build SEC ``form`` names and an optional 10-Q quarter filter from ``10-Qn``."""
    raw = filing_type.strip()
    u = raw.upper().replace(" ", "")
    m = re.fullmatch(r"10-Q(\d)", u)
    if m:
        q = int(m.group(1))
        if q not in (1, 2, 3):
            if q == 4:
                raise ValueError(
                    "10-Q4 is not a valid filing type; the fourth quarter is filed as 10-K."
                )
            raise ValueError(
                f"Invalid quarterly type {raw!r}; use 10-Q1, 10-Q2, or 10-Q3."
            )
        return frozenset({q}), frozenset({"10-Q"})
    if u == "10-Q":
        raise ValueError(
            "Use a single quarterly filing type (10-Q1, 10-Q2, or 10-Q3), not plain 10-Q."
        )
    return None, frozenset({raw})


async def get_sec_results(
    ticker: str,
    year: str,
    filing_type: str = "10-K",
    company: str | None = None,
    email: str | None = None,
) -> list[SecResults]:
    """Fetch SEC filing metadata for the given ticker and year.

    Pass a single filing selector such as ``10-K`` or ``10-Q1``/``10-Q2``/``10-Q3``
    to restrict by fiscal quarter from ``reportDate``. ``10-Q4`` and plain ``10-Q``
    are invalid in this flow.
    """
    company = company or sec_settings.sec_api_organization
    email = email or sec_settings.sec_api_email
    cik = await utils.get_cik_by_ticker(ticker)
    logger.info(f"For {ticker=} found {cik=}")

    quarter_filter, forms = _parse_filing_type_for_sec_query(filing_type)

    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    headers = {
        "User-Agent": f"{company} {email}",
        "Content-Type": "text/html",
    }
    timeout = aiohttp.ClientTimeout(total=60)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get(url, headers=headers) as response:
            if response.status != 200:
                raise RuntimeError(
                    f"Unable to fetch submissions. Status code: {response.status}"
                )
            json_data = await response.json()

    filings = json_data["filings"]
    recent_filings = filings["recent"]
    sec_form_names: list[str] = []
    form_lists: list[SecResults] = []

    for acc_num, form_name, filing_date, report_date, primary_doc in zip(
        recent_filings["accessionNumber"],
        recent_filings["form"],
        recent_filings["filingDate"],
        recent_filings["reportDate"],
        recent_filings["primaryDocument"],
        strict=True,
    ):
        if form_name in forms and report_date.startswith(str(year)):
            display_name = form_name
            if form_name == "10-Q":
                datetime_obj = datetime.strptime(report_date, "%Y-%m-%d")
                quarter = (datetime_obj.month + 2) // 3
                if quarter_filter is not None and quarter not in quarter_filter:
                    continue
                display_name = f"10-Q{quarter}"
                if display_name in sec_form_names:
                    display_name += "-1"
            no_dashes_acc_num = re.sub("-", "", acc_num)
            form_lists.append(
                SecResults(
                    dashes_acc_num=no_dashes_acc_num,
                    form_name=display_name,
                    filing_date=filing_date,
                    report_date=report_date,
                    primary_document=primary_doc,
                )
            )
            sec_form_names.append(display_name)

    return form_lists


async def save_sec_results_as_pdfs(
    sec_result: SecResults,
    ticker: str,
    year: str,
    company: str | None = None,
    email: str | None = None,
) -> Path:
    """Save one SEC filing as PDF if needed, or reuse existing PDF."""
    company = company or sec_settings.sec_api_organization
    email = email or sec_settings.sec_api_email
    cik = await utils.get_cik_by_ticker(ticker)
    rgld_cik = int(cik.lstrip("0"))
    output_dir = sec_data_case_dir(ticker, year)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{sec_result.form_name}.pdf"
    if output_path.exists():
        logger.info(f"PDF already exists, skipping download: {output_path}")
        return output_path

    filings_to_save = [
        utils.FilingToSave(
            cik=rgld_cik,
            accession_number=sec_result.dashes_acc_num,
            primary_document=sec_result.primary_document,
            output_path=output_path,
        )
    ]

    pdf_paths = await utils.save_filings_as_pdfs(
        filings=filings_to_save,
        company=company,
        email=email,
    )
    if not pdf_paths:
        raise RuntimeError(
            f"Failed to save PDF for {ticker} {year} {sec_result.form_name}"
        )
    logger.info(f"Saved SEC PDF to {output_path}")
    return pdf_paths[0]


async def sec_main(
    ticker: str,
    year: str,
    filing_type: str = "10-K",
) -> tuple[SecResults, Path]:
    """Fetch one SEC filing result and ensure its PDF exists.

    ``filing_type`` should identify a single filing type, for example
    ``10-K`` or one of ``10-Q1``/``10-Q2``/``10-Q3``.
    """
    ticker_name = utils.company_to_ticker(ticker)
    assert ticker_name, f"The {ticker=} that you provided, is not valid"
    sec_results = await get_sec_results(
        ticker=ticker,
        year=year,
        filing_type=filing_type,
    )
    if not sec_results:
        raise FileNotFoundError(
            f"No SEC filing found for ticker={ticker}, year={year}, filing_type={filing_type}."
        )
    sec_result = sec_results[0]
    if len(sec_results) > 1:
        logger.warning(
            f"Multiple filings matched {ticker=} {year=} {filing_type=}; using first: "
            f"{sec_result.form_name} {sec_result.filing_date}"
        )

    pdf_path = await save_sec_results_as_pdfs(
        sec_result=sec_result,
        ticker=ticker,
        year=year,
    )
    return sec_result, pdf_path


def sec_markdown_path_for_pdf(pdf_path: str | Path) -> Path:
    """Resolve olmOCR markdown output path for a filing PDF path."""

    return Path(get_markdown_path(sec_settings.olmocr_workspace, str(pdf_path)))


async def sec_main_to_markdown(
    ticker: str,
    year: str,
    filing_type: str = "10-K",
) -> dict[str, Any]:
    """Ensure one SEC filing is downloaded and OCR markdown exists, then return markdown."""

    sec_result, pdf_path = await sec_main(
        ticker=ticker,
        year=year,
        filing_type=filing_type,
    )
    markdown_path = sec_markdown_path_for_pdf(pdf_path)

    if not markdown_path.exists():
        await run_olmo_ocr(pdf_dir=str(Path(pdf_path).parent))

    if not markdown_path.exists():
        raise FileNotFoundError(f"Markdown output not found after OCR: {markdown_path}")

    markdown_text = markdown_path.read_text(encoding="utf-8")
    return {
        "sec_result": sec_result,
        "pdf_path": Path(pdf_path),
        "markdown_path": markdown_path,
        "markdown_text": markdown_text,
    }


async def sec_main_to_markdown_and_embed(
    ticker: str,
    year: str,
    filing_type: str = "10-K",
    force: bool = False,
) -> dict[str, Any]:
    """Ensure SEC filing markdown exists, then embed it into ChromaDB."""

    payload = await sec_main_to_markdown(
        ticker=ticker,
        year=year,
        filing_type=filing_type,
    )
    vector_store = ChromaVectorStore()
    embedded_keys = vector_store.from_markdown_sec_filing(
        ticker=ticker,
        year=year,
        filing_type=payload["sec_result"].form_name,
        markdown_path=payload["markdown_path"],
        filing_date=payload["sec_result"].filing_date,
        force=force,
    )
    payload["embedded"] = [
        {"ticker": k.ticker, "year": k.year, "filing_type": k.filing_type}
        for k in embedded_keys
    ]
    return payload


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fetch SEC filings and save as PDFs")
    parser.add_argument("--ticker", type=str, required=True, help="Stock ticker symbol")
    parser.add_argument("--year", type=str, required=True, help="Filing year")
    parser.add_argument(
        "--filing-type",
        type=str,
        default="10-K",
        help="SEC form to fetch (e.g. 10-K, 10-Q1, 10-Q2, 10-Q3)",
    )
    args = parser.parse_args()

    asyncio.run(
        sec_main(
            ticker=args.ticker,
            year=args.year,
            filing_type=args.filing_type,
        )
    )
