from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple, Sequence

from chromadb.types import Metadata

import chromadb
import numpy as np
from chromadb.api.models.Collection import Collection
from openai import OpenAI

from finance_data.dataloader.text_splitter import Chunk, chunk_markdown
from finance_data.earnings_transcripts.transcripts import Transcript
from finance_data.settings import sec_settings

_log = logging.getLogger(__name__)
_EMBED_BATCH_SIZE = 2048
_CHROMA_MISSING_PAGE_NUM = -1


class IndexKey(NamedTuple):
    ticker: str
    year: str
    filing_type: str


@dataclass
class _EmbeddedChunks:
    chunks: list[Chunk]
    embeddings: np.ndarray


def embed_chunks(chunks: list[Chunk], client: OpenAI, model: str) -> np.ndarray:
    texts = [c.text for c in chunks]
    all_vectors: list[list[float]] = []

    for start in range(0, len(texts), _EMBED_BATCH_SIZE):
        batch = texts[start : start + _EMBED_BATCH_SIZE]
        response = client.embeddings.create(input=batch, model=model)
        batch_vecs = [
            item.embedding for item in sorted(response.data, key=lambda x: x.index)
        ]
        all_vectors.extend(batch_vecs)

    matrix = np.array(all_vectors, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return matrix / norms


class ChromaVectorStore:
    """Single-collection vector store backed by ChromaDB + vLLM embeddings."""

    def __init__(
        self,
        persist_dir: str | Path | None = None,
        collection_name: str | None = None,
        embedding_server: str | None = None,
        embedding_model: str | None = None,
    ) -> None:
        self._persist_dir = Path(persist_dir or sec_settings.chroma_persist_dir)
        self._collection_name = collection_name or sec_settings.chroma_collection_name
        self._embedding_server = embedding_server or sec_settings.embedding_server
        self._embedding_model = embedding_model or sec_settings.embedding_model

        self._client = chromadb.PersistentClient(path=str(self._persist_dir))
        self._collection: Collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def _make_client(self) -> OpenAI:
        return OpenAI(base_url=self._embedding_server, api_key="not-needed")

    @staticmethod
    def _parse_chunk_metadata(meta: dict) -> Chunk:
        pn = meta.get("page_num")
        page_num = None if pn == _CHROMA_MISSING_PAGE_NUM else pn
        return Chunk(
            text=str(meta.get("text", "")),
            chunk_type=str(meta.get("chunk_type", "text")),
            page_num=page_num,
            section_title=meta.get("section_title"),
            index=int(meta.get("chunk_index", -1)),
        )

    @staticmethod
    def _resolve_transcript_paths(
        ticker: str,
        year: str,
        transcript_paths: Sequence[Path] | None,
    ) -> list[Path]:
        if transcript_paths is not None:
            return [Path(path) for path in transcript_paths]

        base = Path(sec_settings.earnings_transcripts_dir)
        year_s = str(year).strip()
        ticker_l = ticker.strip().lower()

        candidates: list[Path] = []
        direct = base / ticker / year_s
        if direct.is_dir():
            candidates.extend(sorted(direct.glob("Q*.md")))

        if not candidates and base.is_dir():
            for ticker_dir in base.iterdir():
                if not ticker_dir.is_dir() or ticker_dir.name.lower() != ticker_l:
                    continue
                yr = ticker_dir / year_s
                if yr.is_dir():
                    candidates.extend(sorted(yr.glob("Q*.md")))

        if not candidates and base.is_dir():
            candidates.extend(sorted(base.glob(f"**/{year_s}/Q*.md")))

        return candidates

    @staticmethod
    def _resolve_sec_markdown_paths(ticker: str, year: str) -> list[Path]:
        year_s = str(year).strip()
        md_dir = (
            Path(sec_settings.olmocr_workspace)
            / "markdown"
            / Path(sec_settings.sec_data_dir)
            / f"{ticker}-{year_s}"
        )
        if not md_dir.is_dir():
            return []
        return sorted(md_dir.glob("*.md"))

    @staticmethod
    def _chunk_transcript_markdown(
        markdown_text: str,
        *,
        chunk_size: int = 2048,
        overlap: int = 256,
    ) -> list[Chunk]:
        speaker_sections = re.findall(
            r"<speaker-start>[\s\S]*?<speaker-end>",
            markdown_text,
            flags=re.IGNORECASE,
        )
        sections = speaker_sections or [markdown_text]

        chunks: list[Chunk] = []
        index = 0
        for section in sections:
            for chunk in chunk_markdown(
                section,
                chunk_size=chunk_size,
                overlap=overlap,
            ):
                chunks.append(
                    Chunk(
                        text=chunk.text,
                        chunk_type=chunk.chunk_type,
                        page_num=chunk.page_num,
                        section_title=chunk.section_title,
                        index=index,
                    )
                )
                index += 1
        return chunks

    def _embed_for_upsert(self, chunks: list[Chunk]) -> _EmbeddedChunks:
        if not chunks:
            raise ValueError("No chunks produced for embedding.")
        embeddings = embed_chunks(chunks, self._make_client(), self._embedding_model)
        return _EmbeddedChunks(chunks=chunks, embeddings=embeddings)

    def _upsert_document_chunks(
        self,
        *,
        ticker: str,
        year: str,
        filing_type: str,
        filing_date: str | None,
        source_path: str,
        chunks: list[Chunk],
        embeddings: np.ndarray,
        force: bool,
    ) -> None:
        where = {
            "$and": [
                {"ticker": ticker},
                {"year": str(year)},
                {"filing_type": filing_type},
            ]
        }

        existing = self._collection.get(where=where, include=[])
        existing_ids = existing.get("ids", [])
        if existing_ids and not force:
            _log.info(
                "Documents already exist for %s/%s/%s, skipping (force=False).",
                ticker,
                year,
                filing_type,
            )
            return
        if existing_ids:
            self._collection.delete(ids=existing_ids)

        ids: list[str] = []
        metadatas: list[Metadata] = []
        documents: list[str] = []
        for i, chunk in enumerate(chunks):
            chunk_id = f"{ticker}:{year}:{filing_type}:{i}"
            ids.append(chunk_id)
            documents.append(chunk.text)
            metadatas.append(
                {
                    "ticker": ticker,
                    "year": str(year),
                    "filing_type": filing_type,
                    "filing_date": filing_date or "",
                    "source_path": source_path,
                    "chunk_index": chunk.index,
                    "chunk_type": chunk.chunk_type,
                    "section_title": chunk.section_title or "",
                    "page_num": (
                        chunk.page_num
                        if chunk.page_num is not None
                        else _CHROMA_MISSING_PAGE_NUM
                    ),
                    "text": chunk.text,
                }
            )

        self._collection.add(
            ids=ids,
            embeddings=embeddings.tolist(),
            documents=documents,
            metadatas=metadatas,
        )

    def from_markdown_sec_filings(
        self,
        ticker: str,
        year: str,
        force: bool = False,
    ) -> list[IndexKey]:
        """Index every ``*.md`` under workspace markdown for this ticker/year.

        Path: ``{olmocr_workspace}/markdown/{sec_data_dir}/{ticker}-{year}/``
        (same layout as the olmOCR markdown writer).
        """
        resolved_paths = self._resolve_sec_markdown_paths(ticker, year)
        if not resolved_paths:
            raise FileNotFoundError(
                f"No .md files for ticker={ticker}, year={year} under "
                f"{Path(sec_settings.olmocr_workspace) / 'markdown' / sec_settings.sec_data_dir} "
                f"(expected a folder named like {ticker!r}-{year!r})."
            )

        ingested: list[IndexKey] = []

        for raw_path in resolved_paths:
            md_path = Path(raw_path)
            filing_type = md_path.stem
            key = IndexKey(ticker=ticker, year=str(year), filing_type=filing_type)

            markdown_text = md_path.read_text(encoding="utf-8")
            chunks = chunk_markdown(markdown_text)
            embedded = self._embed_for_upsert(chunks)

            self._upsert_document_chunks(
                ticker=ticker,
                year=str(year),
                filing_type=filing_type,
                filing_date=None,
                source_path=str(md_path),
                chunks=embedded.chunks,
                embeddings=embedded.embeddings,
                force=force,
            )
            ingested.append(key)

        return ingested

    def from_markdown_sec_filing(
        self,
        ticker: str,
        year: str,
        filing_type: str,
        markdown_path: str | Path,
        filing_date: str | None = None,
        force: bool = False,
    ) -> list[IndexKey]:
        md_path = Path(markdown_path)
        if not md_path.exists():
            raise FileNotFoundError(f"Markdown file not found: {md_path}")

        markdown_text = md_path.read_text(encoding="utf-8")
        chunks = chunk_markdown(markdown_text)
        embedded = self._embed_for_upsert(chunks)
        key = IndexKey(ticker=ticker, year=str(year), filing_type=filing_type)

        self._upsert_document_chunks(
            ticker=ticker,
            year=str(year),
            filing_type=filing_type,
            filing_date=filing_date,
            source_path=str(md_path),
            chunks=embedded.chunks,
            embeddings=embedded.embeddings,
            force=force,
        )
        return [key]

    def from_earnings_transcript_markdown(
        self,
        ticker: str,
        year: str,
        transcript_paths: Sequence[Path] | None = None,
        *,
        force: bool = False,
        chunk_size: int = 2048,
        overlap: int = 256,
    ) -> list[IndexKey]:
        ingested: list[IndexKey] = []
        resolved_paths = self._resolve_transcript_paths(ticker, year, transcript_paths)
        if not resolved_paths:
            raise FileNotFoundError(
                f"No transcript markdown files found for ticker={ticker}, year={year} "
                f"under {sec_settings.earnings_transcripts_dir!r}."
            )

        for raw_path in resolved_paths:
            tx_path = Path(raw_path)
            transcript = Transcript.from_markdown(tx_path)
            filing_type = f"Q{transcript.quarter_num}"
            key = IndexKey(ticker=ticker, year=str(year), filing_type=filing_type)

            markdown_text = tx_path.read_text(encoding="utf-8")
            chunks = self._chunk_transcript_markdown(
                markdown_text,
                chunk_size=chunk_size,
                overlap=overlap,
            )
            embedded = self._embed_for_upsert(chunks)

            self._upsert_document_chunks(
                ticker=ticker,
                year=str(year),
                filing_type=filing_type,
                filing_date=transcript.date,
                source_path=str(tx_path),
                chunks=embedded.chunks,
                embeddings=embedded.embeddings,
                force=force,
            )
            ingested.append(key)

        return ingested

    def list_filings(self, ticker: str, year: str) -> list[dict[str, str | None]]:
        rows = self._collection.get(
            where={"$and": [{"ticker": ticker}, {"year": str(year)}]},
            include=["metadatas"],
        )
        out: dict[str, str | None] = {}
        for metadata in rows.get("metadatas", []):
            filing_type = str(metadata.get("filing_type", ""))
            if not filing_type:
                continue
            filing_date = str(metadata.get("filing_date", "") or "") or None
            out.setdefault(filing_type, filing_date)

        return [
            {"filing_type": filing_type, "filing_date": filing_date}
            for filing_type, filing_date in sorted(out.items())
        ]

    def resolve_transcript_quarters(
        self, ticker: str, year: str
    ) -> tuple[str, list[str]] | None:
        rows = self._collection.get(
            where={"$and": [{"ticker": ticker}, {"year": str(year)}]},
            include=["metadatas"],
        )
        quarters = {
            str(m.get("filing_type", "")).upper()
            for m in rows.get("metadatas", [])
            if re.fullmatch(r"Q[1-4]", str(m.get("filing_type", "")).upper())
        }
        if quarters:
            return ticker, sorted(quarters, key=lambda s: int(s[1]))

        # fallback for case-mismatched ticker values
        if ticker.upper() != ticker:
            return self.resolve_transcript_quarters(ticker.upper(), year)
        return None

    def list_indexes(self) -> list[IndexKey]:
        rows = self._collection.get(include=["metadatas"])
        keys: set[IndexKey] = set()
        for metadata in rows.get("metadatas", []):
            t = str(metadata.get("ticker", ""))
            y = str(metadata.get("year", ""))
            f = str(metadata.get("filing_type", ""))
            if t and y and f:
                keys.add(IndexKey(ticker=t, year=y, filing_type=f))
        return sorted(keys)

    def search(
        self,
        ticker: str,
        year: str,
        filing_type: str,
        query: str,
        top_k: int = 5,
    ) -> list[tuple[Chunk, float]]:
        q_chunk = Chunk(
            text=query,
            chunk_type="text",
            page_num=None,
            section_title=None,
            index=-1,
        )
        q_vec = embed_chunks([q_chunk], self._make_client(), self._embedding_model)[0]

        where = {
            "$and": [
                {"ticker": ticker},
                {"year": str(year)},
                {"filing_type": filing_type},
            ]
        }

        result = self._collection.query(
            query_embeddings=[q_vec.tolist()],
            where=where,
            n_results=top_k,
            include=["metadatas", "distances"],
        )

        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        if not metadatas:
            raise FileNotFoundError(
                f"No vectors found for ticker={ticker}, year={year}, filing_type={filing_type}."
            )

        hits: list[tuple[Chunk, float]] = []
        for metadata, distance in zip(metadatas, distances):
            chunk = self._parse_chunk_metadata(metadata)
            score = 1.0 - float(distance)
            hits.append((chunk, score))
        return hits

    def __len__(self) -> int:
        return len(self.list_indexes())


FaissVectorIndex = ChromaVectorStore
