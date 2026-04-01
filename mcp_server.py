from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Literal

from mcp.server.fastmcp import FastMCP

from finance_data.earnings_transcripts.transcripts import (
    get_transcript_for_quarter_async,
    save_transcript_markdown,
)
from finance_data.dataloader.text_splitter import Chunk
from finance_data.dataloader.vector_store import ChromaVectorStore
from mcp.server.transport_security import TransportSecuritySettings
from finance_data.dataloader.pipeline import sec_main_to_markdown_and_embed
from finance_data.filings.utils import company_to_ticker
from finance_data.settings import sec_settings

_vector_index: ChromaVectorStore | None = None
_RESOURCE_HINT = (
    "If the exact file is missing, generate and embed it with "
    "`sec_main_to_markdown_and_embed_tool` (SEC filings) or "
    "`earnings_transcript_for_quarter_tool` (transcripts + embeddings)."
)


def _mcp_transport_allowed_hosts() -> list[str]:
    return [
        f"{sec_settings.mcp_host}:*",
        "localhost:*",
        *sec_settings.mcp_ngrok_allowed_hosts,
    ]


mcp = FastMCP(
    "sec-filings-data",
    host=sec_settings.mcp_host,
    port=sec_settings.mcp_port,
    transport_security=TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=_mcp_transport_allowed_hosts(),
    ),
)


def _get_vector_index() -> ChromaVectorStore:
    """Return a lazily initialized Chroma vector index client."""
    global _vector_index
    if _vector_index is None:
        _vector_index = ChromaVectorStore()
    return _vector_index


def _hybrid_search(
    vector_index: ChromaVectorStore,
    *,
    ticker: str,
    year: str,
    filing_type: str,
    query: str,
    top_k: int,
) -> list[tuple[Chunk, float]]:
    return vector_index.hybrid_search(
        ticker=ticker,
        year=year,
        filing_type=filing_type,
        query=query,
        top_k=top_k,
    )


def _search_transcripts_common(
    vector_index: ChromaVectorStore,
    *,
    ticker: str,
    year: str,
    query: str,
    top_k: int,
    quarter: Literal["Q1", "Q2", "Q3", "Q4"] | None,
    search_fn: Callable[..., list[tuple[Chunk, float]]],
) -> list[tuple[Chunk, float, str]]:
    resolved = vector_index.resolve_transcript_quarters(ticker, year)
    if not resolved:
        raise FileNotFoundError("No transcript indexes (Q1–Q4) for this ticker/year.")

    ticker_key, quarters = resolved
    if quarter is not None:
        q_key = quarter.upper()
        if q_key not in quarters:
            available = ", ".join(quarters)
            raise ValueError(
                f"No indexed transcript for {q_key} for this ticker/year "
                f"(available: {available})."
            )
        quarters = [q_key]

    merged: list[tuple[Chunk, float, str]] = []
    for filing_type in quarters:
        try:
            hits = search_fn(
                vector_index=vector_index,
                ticker=ticker_key,
                year=year,
                filing_type=filing_type,
                query=query,
                top_k=top_k,
            )
        except FileNotFoundError:
            continue
        for chunk, score in hits:
            merged.append((chunk, score, filing_type))
    merged.sort(key=lambda item: -item[1])
    return merged[:top_k]


def _sec_pdf_root() -> Path:
    return Path(sec_settings.sec_data_dir)


def _transcript_root() -> Path:
    return Path(sec_settings.earnings_transcripts_dir)


def _list_relative_files(root: Path, pattern: str) -> list[str]:
    if not root.exists():
        return []
    return sorted(
        str(path.relative_to(root)) for path in root.glob(pattern) if path.is_file()
    )


def _directory_tree(root: Path) -> str:
    """Render a human-readable tree where leaves are files under ``root``."""
    if not root.exists():
        return f"{root} (missing)"

    lines = [str(root)]
    entries = sorted(
        root.rglob("*"), key=lambda p: (len(p.relative_to(root).parts), str(p))
    )
    for entry in entries:
        depth = len(entry.relative_to(root).parts)
        indent = "  " * depth
        suffix = "/" if entry.is_dir() else ""
        lines.append(f"{indent}- {entry.name}{suffix}")
    return "\n".join(lines)


def _sec_resources_payload() -> dict[str, object]:
    root = _sec_pdf_root()
    return {
        "resource_kind": "sec_filings",
        "base_dir": str(root),
        "directory_structure_format": (
            "Tree format uses `- <name>/` for directories and `- <name>` for files. "
            "Indentation is two spaces per depth level below `base_dir`."
        ),
        "expected_layout": "{SEC_DATA_DIR}/{TICKER}-{YEAR}/{FILING_TYPE}.pdf",
        "pdf_files": _list_relative_files(root, "**/*.pdf"),
        "directory_structure": _directory_tree(root),
        "hint": _RESOURCE_HINT,
    }


def _transcript_resources_payload() -> dict[str, object]:
    root = _transcript_root()
    return {
        "resource_kind": "earnings_transcripts",
        "base_dir": str(root),
        "directory_structure_format": (
            "Tree format uses `- <name>/` for directories and `- <name>` for files. "
            "Indentation is two spaces per depth level below `base_dir`."
        ),
        "expected_layout": "{EARNINGS_TRANSCRIPTS_DIR}/{TICKER}/{YEAR}/Q{N}_{YYYY-MM-DD}.md",
        "transcript_files": _list_relative_files(root, "**/*.md"),
        "directory_structure": _directory_tree(root),
        "hint": _RESOURCE_HINT,
    }


def _all_resources_payload() -> dict[str, object]:
    return {
        "resources": [
            _sec_resources_payload(),
            _transcript_resources_payload(),
        ]
    }


@mcp.resource(
    "resource://sec-filings-data/resources/all",
    name="All SEC resources",
    description=(
        "Lists SEC filing PDFs and transcript markdown files from configured paths, "
        "including directory structure and generation hints."
    ),
)
def all_resources_catalog() -> str:
    """List all available SEC/transcript resources based on ``finance_data.settings`` paths."""
    return json.dumps(_all_resources_payload(), indent=2)


@mcp.resource(
    "resource://sec-filings-data/resources/sec-filings",
    name="SEC filing resources",
    description=(
        "Lists SEC filing PDF resources and directory structure rooted at "
        "`settings.sec_data_dir`."
    ),
)
def sec_filings_resource_catalog() -> str:
    """List SEC filing PDFs with directory structure from ``settings.sec_data_dir``.

    Directory tree format:
    - ``- <name>/`` denotes a directory.
    - ``- <name>`` denotes a file.
    - Two-space indentation denotes each directory depth.
    """
    return json.dumps(_sec_resources_payload(), indent=2)


@mcp.resource(
    "resource://sec-filings-data/resources/transcripts",
    name="Transcript resources",
    description=(
        "Lists earnings transcript markdown resources and directory structure rooted at "
        "`settings.earnings_transcripts_dir`."
    ),
)
def transcripts_resource_catalog() -> str:
    """List transcript markdown files with directory structure from transcript settings.

    Directory tree format:
    - ``- <name>/`` denotes a directory.
    - ``- <name>`` denotes a file.
    - Two-space indentation denotes each directory depth.
    """
    return json.dumps(_transcript_resources_payload(), indent=2)


@mcp.tool()
def company_name_to_ticker_tool(name: str) -> dict[str, str]:
    """Resolve a company name to its stock ticker symbol.
    Usage guidance:
      - Use this when a user gives only a company name (for example
        "Amazon" or "Alphabet") and downstream tools require a ticker.
      - Skip this when the user already provided a valid ticker symbol.
      - If you already know the ticker from prior knowledge or prior steps, do not call
        this tool.
      - Call this once per company and reuse the returned ticker.
    Args:
        name: Full or partial company name to resolve.
    """
    ticker = company_to_ticker(name)
    if ticker is None:
        raise ValueError(f"No ticker found for company name: {name}")
    return {"ticker": ticker}


@mcp.tool()
async def earnings_transcript_for_quarter_tool(
    ticker: str,
    year: int,
    quarter: Literal["Q1", "Q2", "Q3", "Q4"],
) -> dict[str, object]:
    """Fetch one earnings-call transcript, save markdown, and embed it.

    Args:
        ticker: Equity ticker symbol, for example ``"AMZN"``.
        year: Four-digit fiscal year.
        quarter: Fiscal quarter label ``Q1``, ``Q2``, ``Q3``, or ``Q4``.
    """
    transcript = await get_transcript_for_quarter_async(ticker, year, quarter)
    if transcript is None:
        raise ValueError(
            f"No transcript available for ticker={ticker} year={year} {quarter}"
        )
    markdown_path = save_transcript_markdown(transcript)
    embedded_keys = _get_vector_index().from_earnings_transcript_markdown(
        ticker=ticker, year=str(year), transcript_paths=[markdown_path], force=False
    )
    return {
        "ticker": ticker,
        "year": year,
        "quarter": quarter,
        "markdown_path": str(markdown_path),
        "embedded": [
            {"ticker": key.ticker, "year": key.year, "filing_type": key.filing_type}
            for key in embedded_keys
        ],
    }


@mcp.tool()
def list_resources_tool() -> dict[str, object]:
    """Return all SEC/transcript file resources and directory structures.

    Notes:
    - SEC filings are discovered from ``settings.sec_data_dir`` and reported as PDFs.
    - Earnings transcripts are discovered from ``settings.earnings_transcripts_dir``
      and reported as markdown files.
    - If a file is missing, use ``sec_main_to_markdown_and_embed_tool`` or
      ``earnings_transcript_for_quarter_tool`` to generate + embed data.
    """
    return _all_resources_payload()


@mcp.tool()
async def sec_main_to_markdown_and_embed_tool(
    ticker: str,
    year: str,
    filing_type: str = "10-K",
) -> dict:
    """Download one SEC filing PDF (if needed), OCR to markdown (if needed), and embed markdown.

    Args:
        ticker: Equity ticker symbol, for example ``"GOOG"`` or ``"AMZN"``.
        year: Filing year as a string (four digits), matching the SEC filing period.
        filing_type: Form or period key to fetch, for example ``"10-K"`` or ``"10-Q1"``
            through ``"10-Q3"`` for quarterly reports.
    """
    payload = await sec_main_to_markdown_and_embed(
        ticker=ticker,
        year=year,
        filing_type=filing_type,
        force=False,
    )
    sec_result = payload["sec_result"]
    return {
        "sec_result": {
            "dashes_acc_num": sec_result.dashes_acc_num,
            "form_name": sec_result.form_name,
            "filing_date": sec_result.filing_date,
            "report_date": sec_result.report_date,
            "primary_document": sec_result.primary_document,
        },
        "pdf_path": str(payload["pdf_path"]),
        "markdown_path": str(payload["markdown_path"]),
        "embedded": payload["embedded"],
    }


@mcp.tool()
def search_sec_filings_tool(
    ticker: str,
    year: str,
    filing_type: str,
    query: str,
    top_k: int = 5,
) -> str:
    """Run hybrid search over one indexed SEC filing.

    **Prerequisite — data must be indexed before searching:**
    Before calling this tool, verify the filing has been downloaded, OCR-ed, and
    embedded. If the filing is missing or the search returns no results:
      1. Call ``sec_main_to_markdown_and_embed_tool(ticker, year, filing_type)``
         to download, OCR, and embed the filing.
      2. Retry this search tool with the same arguments.

    Args:
        ticker: Equity ticker symbol, for example ``"AMZN"``.
        year: Filing year.
        filing_type: Indexed filing key, such as ``"10-K"`` or ``"10-Q1"``.
        query: Natural-language search query.
        top_k: Maximum number of chunks to return.
    """
    results = _hybrid_search(
        _get_vector_index(),
        ticker=ticker,
        year=year,
        filing_type=filing_type,
        query=query,
        top_k=top_k,
    )
    return "\n\n".join([chunk.text for chunk, _ in results])


@mcp.tool()
def search_transcripts_tool(
    ticker: str,
    year: str,
    query: str,
    top_k: int = 5,
    quarter: Literal["Q1", "Q2", "Q3", "Q4"] | None = None,
) -> str:
    """Run hybrid search over indexed transcript chunks.

    **Prerequisite — transcripts must be indexed before searching:**
    Before calling this tool, verify that at least one quarterly transcript for the
    given ticker and year has been fetched and embedded. If no transcripts are indexed
    or the search returns no results:
      1. Call ``earnings_transcript_for_quarter_tool(ticker, year, quarter)`` for each
         quarter of interest (Q1 through Q4) to fetch, save, and embed transcripts.
      2. Retry this search tool with the same arguments.

    Args:
        ticker: Equity ticker symbol, for example ``"AMZN"``.
        year: Transcript year.
        query: Natural-language search query.
        top_k: Maximum number of chunks to return after quarter-level merging.
        quarter: If set, search only that quarter (``Q1``–``Q4``). If omitted, merge
            hits across all indexed quarters for the year (same as the HTTP API).
    """
    year_s = str(year).strip()
    vector_index = _get_vector_index()
    merged = _search_transcripts_common(
        vector_index,
        ticker=ticker,
        year=year_s,
        query=query,
        top_k=top_k,
        quarter=quarter,
        search_fn=_hybrid_search,
    )
    return "\n\n".join([chunk.text for chunk, _, _ in merged])


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
