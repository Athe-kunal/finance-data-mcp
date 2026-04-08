"""Utilities for sparse (BM25) indexing and query tokenization."""

from __future__ import annotations

import re
from typing import Any

from nltk.stem import PorterStemmer

_BM25_K1 = 1.2  # term-frequency saturation
_BM25_B = 0.75  # length normalisation (0 = none, 1 = full)
_STEMMER = PorterStemmer()
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "from",
    "if",
    "in",
    "into",
    "is",
    "it",
    "no",
    "not",
    "of",
    "on",
    "or",
    "such",
    "that",
    "the",
    "their",
    "then",
    "there",
    "these",
    "they",
    "this",
    "to",
    "was",
    "will",
    "with",
}


def tokenize_for_bm25(text: str) -> list[str]:
    """Tokenize text for BM25 with stopword removal and stemming."""
    tokens = re.findall(r"\b\w+\b", text.lower())
    return [_STEMMER.stem(token) for token in tokens if token not in _STOPWORDS]


def build_bm25_index(texts: list[str]) -> Any:
    """Return a BM25Okapi index built from ``texts``."""
    try:
        from rank_bm25 import BM25Okapi
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "rank_bm25 is not installed. Install with `uv sync --group ocr-md`."
        ) from exc

    tokenized = [tokenize_for_bm25(text) for text in texts]
    return BM25Okapi(tokenized, k1=_BM25_K1, b=_BM25_B)
