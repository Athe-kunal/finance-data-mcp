from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from filings.utils import company_to_ticker
from filings.sec_data import sec_main
from ocr.olmocr_pipeline import run_olmo_ocr

app = FastAPI()


class CompanyNameRequest(BaseModel):
    name: str


@app.post("/company_name_to_ticker")
def company_name_to_ticker(request: CompanyNameRequest):
    """Resolve a company name to its stock ticker symbol."""
    ticker = company_to_ticker(request.name)
    if ticker is None:
        raise HTTPException(status_code=404, detail="No ticker found for company name")
    return {"ticker": ticker}


class SecMainRequest(BaseModel):
    ticker: str
    year: str
    filing_types: list[str] = ["10-K", "10-Q"]
    include_amends: bool = True


@app.post("/sec_main")
async def sec_main_endpoint(request: SecMainRequest):
    """Fetch SEC filings and save them as PDFs."""
    sec_results, pdf_paths = await sec_main(
        ticker=request.ticker,
        year=request.year,
        filing_types=request.filing_types,
        include_amends=request.include_amends,
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


class RunOlmoOcrRequest(BaseModel):
    pdf_dir: str


@app.post("/run_olmo_ocr")
async def run_olmo_ocr_endpoint(request: RunOlmoOcrRequest):
    """Run OCR on PDFs in the given folder."""
    await run_olmo_ocr(pdf_dir=request.pdf_dir)
    return {"status": "completed", "pdf_dir": request.pdf_dir}
