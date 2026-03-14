import re
import asyncio
import requests
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

from loguru import logger

from filings import utils, sec_data


class SecResults(NamedTuple):
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
    company: str = "Indiana University Bloomington",
    email: str = "astmohap@iu.edu",
) -> list[SecResults]:
    """Fetch SEC filing metadata for the given ticker and year."""
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
    company: str = "Indiana University Bloomington",
    email: str = "astmohap@iu.edu",
) -> list[Path]:
    """Save SEC results as PDF files."""
    cik = utils.get_cik_by_ticker(ticker)
    rgld_cik = int(cik.lstrip("0"))
    output_dir = Path("sec_data") / f"{ticker}-{year}"

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

    logger.info(f"Saved {len(pdf_paths)} PDFs to {output_dir}")
    return pdf_paths


async def sec_main(
    ticker: str,
    year: str,
    filing_types: list[str] = ["10-K", "10-Q"],
    include_amends: bool = True,
    company: str = "Indiana University Bloomington",
    email: str = "astmohap@iu.edu",
) -> tuple[list[SecResults], list[Path]]:
    """Fetch SEC results and save them as PDFs."""
    sec_results = get_sec_results(
        ticker=ticker,
        year=year,
        filing_types=filing_types,
        include_amends=include_amends,
        company=company,
        email=email,
    )
    pdf_paths = await save_sec_results_as_pdfs(
        sec_results=sec_results,
        ticker=ticker,
        year=year,
        company=company,
        email=email,
    )
    return sec_results, pdf_paths


if __name__ == "__main__":
    data = asyncio.run(sec_main(ticker="GOOG", year="2025"))
