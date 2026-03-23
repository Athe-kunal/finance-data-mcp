from __future__ import annotations

import dataclasses
import mimetypes
from pathlib import Path
from typing import Literal

from mcp.server.fastmcp import FastMCP

from earnings_transcripts.transcripts import get_transcript_for_quarter_async
from mcp.server.transport_security import TransportSecuritySettings
from filings.sec_data import sec_main
from filings.utils import company_to_ticker
from settings import sec_settings

mcp = FastMCP(
    "sec-filings-data",
    host="127.0.0.1",
    port=8000,
    transport_security=TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=[
            "127.0.0.1:*",
            "localhost:*",
            "shirleen-supercritical-contributively.ngrok-free.dev",
        ],
        allowed_origins=[
            "http://localhost:*",
            "https://shirleen-supercritical-contributively.ngrok-free.dev",
        ],
    ),
)


def _resolve_path(path: str) -> Path:
    """Resolve a path string to an absolute path."""
    return Path(path).expanduser().resolve()


def _allowed_roots() -> list[Path]:
    """Return the main data roots this MCP server exposes."""
    return [
        Path(sec_settings.sec_data_dir).resolve(),
        Path(sec_settings.earnings_transcripts_dir).resolve(),
    ]


def _is_under(path: Path, root: Path) -> bool:
    """Return whether ``path`` is within ``root``."""
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _is_allowed_path(path: Path) -> bool:
    """Allow access only to known data roots."""
    return any(_is_under(path, root) for root in _allowed_roots())


@mcp.tool()
def company_name_to_ticker_tool(name: str) -> dict[str, str]:
    """Resolve a company name to its stock ticker symbol.

    Args:
        name: Full or partial company name to resolve.
    """
    ticker = company_to_ticker(name)
    if ticker is None:
        raise ValueError(f"No ticker found for company name: {name}")
    return {"ticker": ticker}


@mcp.tool()
async def earnings_transcript_for_quarter_tool(
    ticker: str, year: int, quarter: Literal["Q1", "Q2", "Q3", "Q4"]
) -> dict:
    """Fetch one earnings-call transcript for a ticker, year, and quarter.

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
    return dataclasses.asdict(transcript)


@mcp.tool()
async def sec_main_tool(
    ticker: str,
    year: str,
    filing_types: list[str] | None = None,
    include_amends: bool = True,
) -> dict:
    """Fetch SEC filings and persist PDFs under the configured SEC data directory.

    Args:
        ticker: Equity ticker symbol, for example ``"AMZN"``.
        year: Filing year, typically a four-digit string.
        filing_types: SEC form types to request, such as ``["10-K", "10-Q"]``.
        include_amends: Whether amended forms (for example ``10-K/A``) are included.
    """
    effective_filing_types = filing_types or ["10-K", "10-Q"]
    sec_results, pdf_paths = await sec_main(
        ticker=ticker,
        year=year,
        filing_types=effective_filing_types,
        include_amends=include_amends,
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


@mcp.tool()
def list_data_roots_tool() -> list[dict[str, str]]:
    """List root directories exposed for file exploration."""
    return [{"path": str(root)} for root in _allowed_roots()]


@mcp.tool()
def list_data_files_tool(
    path: str, pattern: str = "**/*", limit: int = 200
) -> list[dict]:
    """List files under an allowed directory.

    Args:
        path: Absolute or relative directory path to scan.
        pattern: Glob pattern, e.g. ``**/*.pdf`` or ``**/*.jsonl``.
        limit: Maximum number of records returned.
    """
    base = _resolve_path(path)
    if not base.exists() or not base.is_dir():
        raise FileNotFoundError(f"Directory not found: {base}")
    if not _is_allowed_path(base):
        raise ValueError(f"Path is not within allowed data roots: {base}")

    files: list[dict] = []
    for candidate in sorted(base.glob(pattern)):
        if not candidate.exists() or not candidate.is_file():
            continue
        files.append(
            {
                "path": str(candidate),
                "name": candidate.name,
                "suffix": candidate.suffix.lower(),
                "size_bytes": candidate.stat().st_size,
                "mime_type": mimetypes.guess_type(candidate.name)[0]
                or "application/octet-stream",
            }
        )
        if len(files) >= limit:
            break

    return files


@mcp.tool()
def read_data_file_tool(path: str, max_bytes: int = 200_000) -> dict:
    """Read a file under the exposed data roots.

    Text-like files return UTF-8 content (with replacement for invalid bytes).
    Binary files return base metadata and a short hex preview.

    Args:
        path: Absolute or relative file path under an allowed root.
        max_bytes: Maximum number of bytes read from the file.
    """
    file_path = _resolve_path(path)
    if not file_path.exists() or not file_path.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")
    if not _is_allowed_path(file_path):
        raise ValueError(f"Path is not within allowed data roots: {file_path}")

    mime_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
    raw = file_path.read_bytes()[:max_bytes]

    if mime_type.startswith("text/") or file_path.suffix.lower() in {
        ".json",
        ".jsonl",
        ".md",
        ".txt",
        ".csv",
        ".yaml",
        ".yml",
    }:
        return {
            "path": str(file_path),
            "mime_type": mime_type,
            "truncated": file_path.stat().st_size > len(raw),
            "content": raw.decode("utf-8", errors="replace"),
        }

    return {
        "path": str(file_path),
        "mime_type": mime_type,
        "truncated": file_path.stat().st_size > len(raw),
        "hex_preview": raw[:256].hex(),
    }


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
