"""Dataloader for SEC filings: fetch, OCR, embed, and vector search."""

from filings.utils import company_to_ticker

from .pipeline import ensure_sec_data, prepare_sec_filing_envs
from .repl_env import MarkdownReplEnvironment, markdown_to_repl_env
from .vector_store import (
    Chunk,
    FaissVectorIndex,
    chunk_markdown,
    embed_chunks,
)

__all__ = [
    "company_to_ticker",
    "ensure_sec_data",
    "prepare_sec_filing_envs",
    "MarkdownReplEnvironment",
    "markdown_to_repl_env",
    "Chunk",
    "FaissVectorIndex",
    "chunk_markdown",
    "embed_chunks",
]
