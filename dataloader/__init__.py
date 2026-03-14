"""Dataloader for SEC filings: fetch, OCR, and REPL/embedding environments."""

from filings.utils import company_to_ticker

from .embd_env import (
    Chunk,
    EmbeddingEnvironment,
    FaissVectorIndex,
    chunk_markdown,
    embed_chunks,
    markdown_to_embedding_env,
)
from .pipeline import ensure_sec_data, prepare_sec_filing_envs
from .repl_env import MarkdownReplEnvironment, markdown_to_repl_env
from .vector_store import FaissVectorStore

__all__ = [
    "company_to_ticker",
    "ensure_sec_data",
    "prepare_sec_filing_envs",
    "MarkdownReplEnvironment",
    "markdown_to_repl_env",
    "Chunk",
    "EmbeddingEnvironment",
    "FaissVectorIndex",
    "FaissVectorStore",
    "chunk_markdown",
    "embed_chunks",
    "markdown_to_embedding_env",
]
