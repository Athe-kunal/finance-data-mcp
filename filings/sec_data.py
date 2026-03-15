import json
import re
import asyncio
import requests
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

from loguru import logger

from filings import utils
from settings import olmocr_settings


@dataclass(frozen=True)
class SecResults:
    dashes_acc_num: str
    form_name: str
    filing_date: str
    report_date: str
    primary_document: str


def get_sec_results(
    ticker: str,
    year: str,
    filing_types: list[str] = ["10-K", "10-Q"],
    include_amends: bool = True,
    company: str | None = None,
    email: str | None = None,
) -> list[SecResults]:
    """Fetch SEC filing metadata for the given ticker and year."""
    company = company or olmocr_settings.sec_api_organization
    email = email or olmocr_settings.sec_api_email
    cik = utils.get_cik_by_ticker(ticker)
    logger.info(f"For {ticker=} found {cik=}")

    forms = []
    if include_amends:
        for ft in filing_types:
            forms.append(ft)
            forms.append(ft + "/A")
    else:
        forms = list(filing_types)

    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    headers = {
        "User-Agent": f"{company} {email}",
        "Content-Type": "text/html",
    }
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        json_data = response.json()
    else:
        raise RuntimeError(
            f"Unable to fetch submissions. Status code: {response.status_code}"
        )

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
    sec_results: list[SecResults],
    ticker: str,
    year: str,
    company: str | None = None,
    email: str | None = None,
) -> list[Path]:
    """Save SEC results as PDF files and persist metadata to ``sec_results.json``."""
    company = company or olmocr_settings.sec_api_organization
    email = email or olmocr_settings.sec_api_email
    cik = utils.get_cik_by_ticker(ticker)
    rgld_cik = int(cik.lstrip("0"))
    output_dir = Path("sec_data") / f"{ticker}-{year}"
    output_dir.mkdir(parents=True, exist_ok=True)

    filings_to_save = [
        utils.FilingToSave(
            cik=rgld_cik,
            accession_number=sr.dashes_acc_num,
            primary_document=sr.primary_document,
            output_path=output_dir / f"{sr.form_name}.pdf",
        )
        for sr in sec_results
    ]

    pdf_paths = await utils.save_filings_as_pdfs(
        filings=filings_to_save,
        company=company,
        email=email,
    )

    # Persist metadata so filing dates are available without re-hitting SEC API.
    json_path = output_dir / "sec_results.json"
    json_path.write_text(
        json.dumps([asdict(sr) for sr in sec_results], indent=2),
        encoding="utf-8",
    )

    logger.info(f"Saved {len(pdf_paths)} PDFs and metadata to {output_dir}")
    return pdf_paths


def load_sec_results(ticker: str, year: str) -> list[SecResults]:
    """Load previously persisted SEC filing metadata from ``sec_results.json``.

    Returns an empty list if the file does not yet exist (i.e. filings have
    not been downloaded yet for this ticker/year).
    """
    json_path = Path("sec_data") / f"{ticker}-{year}" / "sec_results.json"
    if not json_path.exists():
        return []
    records = json.loads(json_path.read_text(encoding="utf-8"))
    return [SecResults(**r) for r in records]


async def sec_main(
    ticker: str,
    year: str,
    filing_types: list[str] = ["10-K", "10-Q"],
    include_amends: bool = True,
) -> tuple[list[SecResults], list[Path]]:
    """Fetch SEC results and save them as PDFs."""
    sec_results = get_sec_results(
        ticker=ticker,
        year=year,
        filing_types=filing_types,
        include_amends=include_amends,
    )
    pdf_paths = await save_sec_results_as_pdfs(
        sec_results=sec_results,
        ticker=ticker,
        year=year,
    )
    return sec_results, pdf_paths


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fetch SEC filings and save as PDFs")
    parser.add_argument("--ticker", type=str, required=True, help="Stock ticker symbol")
    parser.add_argument("--year", type=str, required=True, help="Filing year")
    parser.add_argument(
        "--filing-types",
        nargs="+",
        default=["10-K", "10-Q"],
        help="Filing types to fetch",
    )
    parser.add_argument(
        "--include-amends", type=bool, default=True, help="Include amended filings"
    )

    args = parser.parse_args()

    asyncio.run(
        sec_main(
            ticker=args.ticker,
            year=args.year,
            filing_types=args.filing_types,
            include_amends=args.include_amends,
        )
    )
