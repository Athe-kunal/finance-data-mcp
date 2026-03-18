"""Pipeline to fetch SEC filings, run OCR, and prepare REPL environments."""
from __future__ import annotations

from pathlib import Path

from filings.sec_data import (
    SecResults,
    get_sec_results,
    save_sec_results_as_pdfs,
)
from ocr.olmocr_pipeline import get_markdown_path, run_olmo_ocr
from settings import olmocr_settings

from .repl_env import MarkdownReplEnvironment, markdown_to_repl_env


def _matches_filing_type(sec_result: SecResults, filing_type: str) -> bool:
    """Return True if sec_result matches the requested filing type."""
    if filing_type == "10-K":
        return sec_result.form_name == "10-K"
    if filing_type == "10-Q":
        return sec_result.form_name.startswith("10-Q")
    return sec_result.form_name == filing_type


async def ensure_sec_data(
    ticker: str,
    year: str,
    filing_types: list[str],
    include_amends: bool = True,
) -> tuple[list[SecResults], list[Path]]:
    """
    Ensure SEC filing PDFs exist locally. Download only missing files.

    PDFs are stored in sec_data/{ticker}-{year}/ (per filings.sec_data).

    Returns:
        (sec_results matching filing_types, paths to all PDFs)
    """
    sec_results = get_sec_results(
        ticker=ticker,
        year=year,
        filing_types=filing_types,
        include_amends=include_amends,
    )
    output_dir = Path("sec_data") / f"{ticker}-{year}"

    filtered = [
        sr
        for sr in sec_results
        if any(_matches_filing_type(sr, ft) for ft in filing_types)
    ]
    existing_paths: list[Path] = []
    missing_results: list[SecResults] = []
    for sr in filtered:
        p = output_dir / f"{sr.form_name}.pdf"
        if p.exists():
            existing_paths.append(p)
        else:
            missing_results.append(sr)

    if missing_results:
        new_paths = await save_sec_results_as_pdfs(
            sec_results=missing_results,
            ticker=ticker,
            year=year,
        )
        pdf_paths = existing_paths + new_paths
    else:
        pdf_paths = existing_paths

    return filtered, pdf_paths


async def prepare_sec_filing_envs(
    ticker: str,
    year: str,
    filing_type: str,
    include_amends: bool = True,
    workspace: str | Path | None = None,
) -> list[MarkdownReplEnvironment]:
    """
    Fetch SEC filings (if needed), run OCR, and return REPL environments.

    Args:
        ticker: Stock ticker symbol (e.g. "GOOG", "AAPL").
        year: Filing year (e.g. "2025").
        filing_type: One of "10-K" or "10-Q".
        include_amends: Include amended filings.
        workspace: olmOCR workspace (default from settings). Markdown is written
            to workspace/markdown/sec_data/{ticker}-{year}/...

    Returns:
        List of MarkdownReplEnvironment, one per filing (e.g. 10-K or 10-Q1..10-Q4).
    """
    workspace_str = str(workspace or olmocr_settings.olmocr_workspace)
    pdf_dir_str = f"sec_data/{ticker}-{year}"

    filing_types = [filing_type]
    sec_results, _pdf_paths = await ensure_sec_data(
        ticker=ticker,
        year=year,
        filing_types=filing_types,
        include_amends=include_amends,
    )
    if not sec_results:
        return []

    await run_olmo_ocr(
        pdf_dir=pdf_dir_str,
        workspace=workspace_str,
    )

    envs: list[MarkdownReplEnvironment] = []
    # OCR writes markdown to workspace/markdown/<source_file>.md; source_file
    # matches glob output (e.g. sec_data/GOOG-2025/10-K.pdf)
    rel_pdf_base = f"sec_data/{ticker}-{year}"
    for sr in sec_results:
        source_file = f"{rel_pdf_base}/{sr.form_name}.pdf"
        markdown_path_str = get_markdown_path(workspace_str, source_file)
        markdown_path = Path(markdown_path_str)

        if not markdown_path.exists():
            continue
        env = markdown_to_repl_env(
            markdown_path=markdown_path,
            ticker=ticker,
            year=year,
            sec_result=sr,
        )
        envs.append(env)

    return envs
