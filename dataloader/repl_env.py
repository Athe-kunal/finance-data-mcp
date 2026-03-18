from __future__ import annotations
from code import InteractiveConsole
from dataclasses import dataclass
from pathlib import Path
import functools

from filings.sec_data import SecResults


@dataclass
class MarkdownReplEnvironment:
    ticker: str
    year: str
    filing_type: str
    markdown_path: Path
    markdown_text: str
    namespace: dict[str, object]
    console: InteractiveConsole
    sec_result: SecResults


@functools.lru_cache
def markdown_to_repl_env(
    markdown_path: Path,
    ticker: str,
    year: str,
    sec_result: SecResults,
) -> MarkdownReplEnvironment:
    resolved_path = markdown_path.resolve()
    markdown_text = resolved_path.read_text(encoding="utf-8")
    filing_type = resolved_path.stem
    namespace: dict[str, object] = {
        "ticker": ticker,
        "year": year,
        "filing_type": filing_type,
        "markdown_path": resolved_path,
        "markdown_text": markdown_text,
        "sec_result": sec_result,
    }
    console = InteractiveConsole(locals=namespace)

    return MarkdownReplEnvironment(
        ticker=ticker,
        year=year,
        filing_type=filing_type,
        markdown_path=resolved_path,
        markdown_text=markdown_text,
        namespace=namespace,
        console=console,
        sec_result=sec_result,
    )


if __name__ == "__main__":
    import asyncio

    from dataloader.pipeline import prepare_sec_filing_envs

    envs = asyncio.run(
        prepare_sec_filing_envs(
            ticker="AMZN",
            year="2025",
            filing_type="10-K",
            include_amends=True,
        )
    )
    for env in envs:
        print(env.ticker, env.year, env.filing_type, env.markdown_path)
