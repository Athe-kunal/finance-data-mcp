"""Pipeline to fetch SEC filings, run OCR, and prepare REPL environments."""

from __future__ import annotations
import asyncio

import re
from pathlib import Path

from finance_data.filings.models import SecFilingType, SecResults
from finance_data.filings.sec_data import (
    get_sec_results,
    save_sec_results_as_pdfs,
    sec_main_to_markdown,
)
from finance_data.ocr.olmocr_pipeline import get_markdown_path, run_olmo_ocr
from finance_data.settings import sec_settings

from .vector_store import ChromaVectorStore

from .repl_env import MarkdownReplEnvironment, markdown_to_repl_env


def _matches_filing_type(sec_result: SecResults, filing_type: SecFilingType | str) -> bool:
    """Return True if sec_result matches the requested filing type."""
    ft = filing_type.strip().upper().replace(" ", "")
    if ft == "10-K":
        fn = sec_result.form_name
        return fn == "10-K" or fn.startswith("10-K-")
    q_match = re.fullmatch(r"10-Q([1-3])", ft)
    if q_match:
        base = f"10-Q{q_match.group(1)}"
        name = sec_result.form_name
        return name == base or name.startswith(f"{base}-")
    if ft == "10-Q":
        return sec_result.form_name.startswith("10-Q")
    return sec_result.form_name.upper().replace(" ", "") == ft


async def ensure_sec_data(
    ticker: str,
    year: str,
    filing_type: SecFilingType | str,
) -> tuple[list[SecResults], list[Path]]:
    """
    Ensure SEC filing PDFs exist locally. Download only missing files.

    PDFs are stored in sec_data/{ticker}-{year}/ (per filings.sec_data).

    Returns:
        (sec_results matching filing_type, paths to all PDFs)
    """
    sec_results = await get_sec_results(
        ticker=ticker,
        year=year,
        filing_type=filing_type,
    )
    output_dir = Path("sec_data") / f"{ticker}-{year}"

    filtered = [sr for sr in sec_results if _matches_filing_type(sr, filing_type)]
    existing_paths: list[Path] = []
    missing_results: list[SecResults] = []
    for sr in filtered:
        p = output_dir / f"{sr.form_name}.pdf"
        if p.exists():
            existing_paths.append(p)
        else:
            missing_results.append(sr)

    if missing_results:
        new_paths = await asyncio.gather(
            *[
                save_sec_results_as_pdfs(
                    sec_result=sr,
                    ticker=ticker,
                    year=year,
                )
                for sr in missing_results
            ]
        )
        # Flatten the list of lists (one list per SecResults)
        new_paths = [
            item
            for sublist in new_paths
            for item in (sublist if isinstance(sublist, list) else [sublist])
        ]
        pdf_paths = existing_paths + new_paths
    else:
        pdf_paths = existing_paths

    return filtered, pdf_paths


async def prepare_sec_filing_envs(
    ticker: str,
    year: str,
    filing_type: SecFilingType | str,
    workspace: str | Path | None = None,
) -> list[MarkdownReplEnvironment]:
    """
    Fetch SEC filings (if needed), run OCR, and return REPL environments.

    Args:
        ticker: Stock ticker symbol (e.g. "GOOG", "AAPL").
        year: Filing year (e.g. "2025").
        filing_type: ``10-K``, ``10-Q`` (all quarters), or ``10-Q1`` / ``10-Q2`` / ``10-Q3``.
        workspace: olmOCR workspace (default from settings). Markdown is written
            to workspace/markdown/sec_data/{ticker}-{year}/...

    Returns:
        List of MarkdownReplEnvironment, one per filing (e.g. 10-K or 10-Q1..10-Q4).
    """
    workspace_str = str(workspace or sec_settings.olmocr_workspace)
    pdf_dir_str = f"sec_data/{ticker}-{year}"

    sec_results, _pdf_paths = await ensure_sec_data(
        ticker=ticker,
        year=year,
        filing_type=filing_type,
    )
    if not sec_results:
        return []

    await run_olmo_ocr(
        pdf_dir=pdf_dir_str,
        workspace=workspace_str,
    )

    envs: list[MarkdownReplEnvironment] = []
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


async def sec_main_to_markdown_and_embed(
    ticker: str,
    year: str,
    filing_type: SecFilingType | str = SecFilingType.FORM_10_K,
    force: bool = False,
) -> dict:
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
