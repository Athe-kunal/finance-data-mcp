"""Dataloader for SEC filings: fetch, OCR, embed, and vector search."""

from finance_data.filings.utils import company_to_ticker

from .pipeline import ensure_sec_data, prepare_sec_filing_envs, sec_main_to_markdown_and_embed
from .repl_env import MarkdownReplEnvironment, markdown_to_repl_env
from .text_splitter import Chunk, chunk_markdown
from .vector_store import (
    ChromaVectorStore,
    FaissVectorIndex,
    embed_chunks,
)

__all__ = [
    "ChromaVectorStore",
    "company_to_ticker",
    "ensure_sec_data",
    "prepare_sec_filing_envs",
    "sec_main_to_markdown_and_embed",
    "MarkdownReplEnvironment",
    "markdown_to_repl_env",
    "Chunk",
    "FaissVectorIndex",
    "chunk_markdown",
    "embed_chunks",
]
