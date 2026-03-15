from __future__ import annotations

import logging
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
import faiss
from typing import TYPE_CHECKING
from settings import olmocr_settings
import numpy as np
from openai import OpenAI

from filings.sec_data import load_sec_results
from ocr.olmocr_pipeline import get_markdown_path, run_olmo_ocr
from dataloader.pipeline import ensure_sec_data

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PAGE_TAG_RE = re.compile(r"<PAGE-NUM-(\d+)>")
_TABLE_RE = re.compile(r"<table[\s\S]*?</table>", re.IGNORECASE)
_SECTION_TITLE_RE = re.compile(
    r"^(Item\s+\d+[A-C]?\..*|Part\s+[IV]+.*)", re.MULTILINE | re.IGNORECASE
)
_MIN_CHUNK_CHARS = 200
_EMBED_BATCH_SIZE = 32

# Key type: (ticker, year, filing_type, filing_date)
IndexKey = tuple[str, str, str, str]


# ---------------------------------------------------------------------------
# Chunk
# ---------------------------------------------------------------------------


@dataclass
class Chunk:
    """A single logical unit extracted from a markdown document."""

    text: str
    chunk_type: str  # "table" | "text"
    page_num: int | None
    section_title: str | None
    index: int

    def __repr__(self) -> str:  # pragma: no cover
        preview = self.text[:80].replace("\n", " ")
        return (
            f"Chunk(index={self.index}, type={self.chunk_type!r}, "
            f"page={self.page_num}, section={self.section_title!r}, "
            f"text={preview!r}...)"
        )


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


def chunk_markdown(text: str) -> list[Chunk]:
    """Split markdown *text* into :class:`Chunk` objects.

    Strategy:
    1. Track current page via ``<PAGE-NUM-X>`` tags.
    2. HTML ``<table>…</table>`` blocks are kept as atomic "table" chunks.
    3. Remaining text is split on blank lines; short fragments (< 200 chars)
       are merged into their predecessor.
    4. Section titles (``Item 1.``, ``Part II``, …) are attached as metadata.
    """
    chunks: list[Chunk] = []
    current_page: int | None = None
    current_section: str | None = None
    index = 0

    parts = _TABLE_RE.split(text)
    tables = _TABLE_RE.findall(text)

    for part_idx, non_table_text in enumerate(parts):
        for raw_para in re.split(r"\n{2,}", non_table_text):
            para = raw_para.strip()
            if not para:
                continue

            for m in _PAGE_TAG_RE.finditer(para):
                current_page = int(m.group(1))
            para_clean = _PAGE_TAG_RE.sub("", para).strip()
            if not para_clean:
                continue

            title_match = _SECTION_TITLE_RE.search(para_clean)
            if title_match:
                current_section = title_match.group(0).strip()

            if (
                chunks
                and chunks[-1].chunk_type == "text"
                and len(chunks[-1].text) < _MIN_CHUNK_CHARS
            ):
                chunks[-1].text = chunks[-1].text + "\n\n" + para_clean
                if title_match:
                    chunks[-1].section_title = current_section
                continue

            chunks.append(
                Chunk(
                    text=para_clean,
                    chunk_type="text",
                    page_num=current_page,
                    section_title=current_section,
                    index=index,
                )
            )
            index += 1

        if part_idx < len(tables):
            chunks.append(
                Chunk(
                    text=tables[part_idx].strip(),
                    chunk_type="table",
                    page_num=current_page,
                    section_title=current_section,
                    index=index,
                )
            )
            index += 1

    return chunks


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------


def embed_chunks(chunks: list[Chunk], client: "OpenAI", model: str) -> np.ndarray:
    """Embed *chunks* via an OpenAI-compatible endpoint.

    Returns a ``(len(chunks), dim)`` float32 array with L2-normalised rows.
    """
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


# ---------------------------------------------------------------------------
# GPU helpers (graceful fallback when faiss-gpu-cu12 differs from conda API)
# ---------------------------------------------------------------------------


def _try_move_to_gpu(cpu_index: "faiss.Index", device: int = 0) -> "faiss.Index":
    if faiss.get_num_gpus() == 0:
        _log.warning("faiss.get_num_gpus() == 0; falling back to CPU index.")
        return cpu_index

    if hasattr(faiss, "StandardGpuResources") and hasattr(faiss, "index_cpu_to_gpu"):
        try:
            res = faiss.StandardGpuResources()
            return faiss.index_cpu_to_gpu(res, device, cpu_index)
        except Exception as exc:
            _log.warning("index_cpu_to_gpu failed (%s); falling back to CPU.", exc)
            return cpu_index

    if hasattr(faiss, "index_cpu_to_all_gpus"):
        try:
            return faiss.index_cpu_to_all_gpus(cpu_index)
        except Exception as exc:
            _log.warning("index_cpu_to_all_gpus failed (%s); falling back to CPU.", exc)
            return cpu_index

    _log.warning(
        "No supported GPU transfer function found in faiss; using CPU. "
        "Install faiss via conda for full GPU support."
    )
    return cpu_index


def _index_gpu_to_cpu(gpu_index: "faiss.Index") -> "faiss.Index":

    if hasattr(faiss, "index_gpu_to_cpu"):
        return faiss.index_gpu_to_cpu(gpu_index)
    if hasattr(gpu_index, "quantizer"):
        return gpu_index.quantizer  # type: ignore[attr-defined]
    _log.warning("Cannot convert GPU index to CPU; returning as-is.")
    return gpu_index


# ---------------------------------------------------------------------------
# FaissVectorIndex
# ---------------------------------------------------------------------------


class FaissVectorIndex:
    """An in-memory FAISS index wrapping a list of :class:`Chunk` objects."""

    def __init__(
        self,
        chunks: list[Chunk],
        embeddings: np.ndarray,
        faiss_index: "faiss.Index",
        client: "OpenAI",
        model: str,
    ) -> None:
        self.chunks = chunks
        self.embeddings = embeddings
        self._index = faiss_index
        self._client = client
        self._model = model

    @classmethod
    def from_markdown(
        cls,
        text: str,
        client: "OpenAI",
        model: str,
        use_gpu: bool = False,
    ) -> "FaissVectorIndex":

        chunks = chunk_markdown(text)
        if not chunks:
            raise ValueError("No chunks produced from the provided markdown text.")

        embeddings = embed_chunks(chunks, client, model)
        dim = embeddings.shape[1]

        cpu_index = faiss.IndexFlatIP(dim)
        cpu_index.add(embeddings)
        index = _try_move_to_gpu(cpu_index) if use_gpu else cpu_index

        return cls(
            chunks=chunks,
            embeddings=embeddings,
            faiss_index=index,
            client=client,
            model=model,
        )

    @classmethod
    def load(
        cls,
        path: Path,
        client: "OpenAI",
        model: str,
        use_gpu: bool = False,
        device: int = 0,
    ) -> "FaissVectorIndex":

        path = Path(path)
        cpu_index = faiss.read_index(str(path / "index.faiss"))
        with open(path / "chunks.pkl", "rb") as fh:
            chunks: list[Chunk] = pickle.load(fh)

        index = _try_move_to_gpu(cpu_index, device=device) if use_gpu else cpu_index
        embeddings = np.zeros((cpu_index.ntotal, cpu_index.d), dtype=np.float32)

        return cls(
            chunks=chunks,
            embeddings=embeddings,
            faiss_index=index,
            client=client,
            model=model,
        )

    def save(self, path: Path) -> None:

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        cpu_index = (
            _index_gpu_to_cpu(self._index)
            if hasattr(self._index, "getDevice")
            else self._index
        )
        faiss.write_index(cpu_index, str(path / "index.faiss"))
        with open(path / "chunks.pkl", "wb") as fh:
            pickle.dump(self.chunks, fh)

    def to_gpu(self, device: int = 0) -> None:
        """Move the internal index to GPU in-place."""
        if hasattr(self._index, "getDevice"):
            return
        gpu = _try_move_to_gpu(self._index, device=device)
        if gpu is not self._index:
            self._index = gpu

    def to_cpu(self) -> None:
        """Move the internal index back to CPU in-place."""
        if not hasattr(self._index, "getDevice"):
            return
        self._index = _index_gpu_to_cpu(self._index)

    def search(self, query: str, top_k: int = 5) -> list[tuple[Chunk, float]]:
        """Return top-*k* chunks most similar to *query*."""
        q_chunk = Chunk(
            text=query, chunk_type="text", page_num=None, section_title=None, index=-1
        )
        q_vec = embed_chunks([q_chunk], self._client, self._model)

        k = min(top_k, len(self.chunks))
        scores, indices = self._index.search(q_vec, k)

        return [
            (self.chunks[idx], float(score))
            for score, idx in zip(scores[0], indices[0])
            if idx != -1
        ]

    def __len__(self) -> int:
        return len(self.chunks)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"FaissVectorIndex(chunks={len(self.chunks)}, "
            f"dim={self.embeddings.shape[1]}, model={self._model!r})"
        )


# ---------------------------------------------------------------------------
# FaissVectorStore
# ---------------------------------------------------------------------------


class FaissVectorStore:
    """Persistent registry of FAISS indexes, one per SEC filing.

    Indexes are stored on disk under::

        {index_dir}/{TICKER}/{YEAR}/{FILING_TYPE}/{FILING_DATE}/
            index.faiss
            chunks.pkl

    and looked up by an ``IndexKey = (ticker, year, filing_type, filing_date)``.

    Parameters
    ----------
    index_dir:
        Root directory for persisted indexes.
        Defaults to ``settings.faiss_index_dir`` (``"./faiss_indexes"``).
    embedding_server:
        vLLM embedding endpoint base URL.
        Defaults to ``settings.embedding_server``.
    embedding_model:
        Embedding model name.
        Defaults to ``settings.embedding_model``.
    use_gpu:
        Move FAISS indexes to GPU when loading/building.
        Defaults to ``settings.faiss_use_gpu``.
    """

    def __init__(
        self,
        index_dir: str | Path | None = None,
        embedding_server: str | None = None,
        embedding_model: str | None = None,
        use_gpu: bool | None = None,
    ) -> None:

        self._index_dir = Path(index_dir or olmocr_settings.faiss_index_dir)
        self._embedding_server = embedding_server or olmocr_settings.embedding_server
        self._embedding_model = embedding_model or olmocr_settings.embedding_model
        self._use_gpu = (
            use_gpu if use_gpu is not None else olmocr_settings.faiss_use_gpu
        )
        self._cache: dict[IndexKey, FaissVectorIndex] = {}

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _key_path(self, key: IndexKey) -> Path:
        ticker, year, filing_type, filing_date = key
        return self._index_dir / ticker / year / filing_type / filing_date

    def _make_client(self) -> "OpenAI":

        return OpenAI(base_url=self._embedding_server, api_key="not-needed")

    def _load_cached(self, key: IndexKey) -> FaissVectorIndex:
        if key in self._cache:
            return self._cache[key]
        path = self._key_path(key)
        if not (path / "index.faiss").exists():
            raise FileNotFoundError(
                f"No FAISS index on disk for {key}. " "Call embed() or ingest() first."
            )
        idx = FaissVectorIndex.load(
            path=path,
            client=self._make_client(),
            model=self._embedding_model,
            use_gpu=self._use_gpu,
        )
        self._cache[key] = idx
        return idx

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def list_indexes(self) -> list[IndexKey]:
        """Return all ``(ticker, year, filing_type, filing_date)`` keys on disk."""
        if not self._index_dir.exists():
            return []
        keys: list[IndexKey] = []
        for f in sorted(self._index_dir.glob("*/*/**/index.faiss")):
            parts = f.relative_to(self._index_dir).parts
            if len(parts) == 5:
                ticker, year, filing_type, filing_date, _ = parts
                keys.append((ticker, year, filing_type, filing_date))
        return keys

    @staticmethod
    def _resolve_filing_date(ticker: str, year: str, filing_type: str) -> str:
        """Look up filing_date from ``sec_data/{ticker}-{year}/sec_results.json``.

        Raises ``FileNotFoundError`` if the JSON does not exist, and
        ``KeyError`` if no entry matches *filing_type*.
        """

        results = load_sec_results(ticker, year)
        if not results:
            raise FileNotFoundError(
                f"sec_data/{ticker}-{year}/sec_results.json not found or empty. "
                "Run sec_main() / save_sec_results_as_pdfs() first."
            )
        for sr in results:
            if sr.form_name == filing_type:
                return sr.filing_date
        raise KeyError(
            f"No entry with form_name={filing_type!r} in "
            f"sec_data/{ticker}-{year}/sec_results.json. "
            f"Available: {[r.form_name for r in results]}"
        )

    def embed(
        self,
        ticker: str,
        year: str,
        filing_type: str,
        markdown_path: str | Path,
        filing_date: str | None = None,
        force: bool = False,
    ) -> IndexKey:
        """Chunk, embed, and persist a single markdown file.

        Use this when you already have the markdown on disk (e.g. after OCR)
        and just want to build/update the vector index.

        Parameters
        ----------
        ticker, year, filing_type:
            Filing metadata that forms the lookup key.
        markdown_path:
            Path to the ``.md`` file produced by olmOCR.
        filing_date:
            SEC submission date (e.g. ``"2026-02-06"``).  When omitted the
            value is read automatically from
            ``sec_data/{ticker}-{year}/sec_results.json``.
        force:
            Re-build the index even if it already exists on disk.

        Returns
        -------
        IndexKey
            ``(ticker, year, filing_type, filing_date)``
        """
        if filing_date is None:
            filing_date = self._resolve_filing_date(ticker, year, filing_type)

        key: IndexKey = (ticker, year, filing_type, filing_date)
        dest = self._key_path(key)

        if (dest / "index.faiss").exists() and not force:
            _log.info(
                "Index already exists for %s, skipping. Pass force=True to rebuild.",
                key,
            )
            return key

        markdown_text = Path(markdown_path).read_text(encoding="utf-8")
        _log.info("Building index for %s (%s) …", key, markdown_path)

        idx = FaissVectorIndex.from_markdown(
            text=markdown_text,
            client=self._make_client(),
            model=self._embedding_model,
            use_gpu=self._use_gpu,
        )
        idx.save(dest)
        self._cache[key] = idx
        _log.info("Saved %d chunks to %s", len(idx), dest)
        return key

    async def ingest(
        self,
        ticker: str,
        year: str,
        filing_type: str,
        include_amends: bool = True,
        workspace: str | Path | None = None,
        force: bool = False,
    ) -> list[IndexKey]:
        """Full pipeline: download PDFs → OCR → embed → save.

        Parameters
        ----------
        ticker, year, filing_type:
            Identify the filing(s) to ingest.
        include_amends:
            Include amended filings (e.g. 10-K/A).
        workspace:
            olmOCR workspace directory. Defaults to ``settings.olmocr_workspace``.
        force:
            Re-build indexes even if they already exist on disk.

        Returns
        -------
        list[IndexKey]
            Keys for every index that was built/found.
        """

        workspace_str = str(workspace or olmocr_settings.olmocr_workspace)

        sec_results, _ = await ensure_sec_data(
            ticker=ticker,
            year=year,
            filing_types=[filing_type],
            include_amends=include_amends,
        )
        if not sec_results:
            _log.warning(
                "No SEC results found for %s %s %s.", ticker, year, filing_type
            )
            return []

        await run_olmo_ocr(
            pdf_dir=f"sec_data/{ticker}-{year}",
            workspace=workspace_str,
        )

        ingested: list[IndexKey] = []
        for sr in sec_results:
            md_path = Path(
                get_markdown_path(
                    workspace_str, f"sec_data/{ticker}-{year}/{sr.form_name}.pdf"
                )
            )
            if not md_path.exists():
                _log.warning("Markdown not found: %s, skipping.", md_path)
                continue
            key = self.embed(
                ticker=ticker,
                year=year,
                filing_type=sr.form_name,
                filing_date=sr.filing_date,
                markdown_path=md_path,
                force=force,
            )
            ingested.append(key)

        return ingested

    def search(
        self,
        ticker: str,
        year: str,
        filing_type: str,
        filing_date: str,
        query: str,
        top_k: int = 5,
    ) -> list[tuple[Chunk, float]]:
        """Semantic search over a filing's vector index.

        Loads the index from disk on first access, then caches in memory.

        Parameters
        ----------
        ticker, year, filing_type, filing_date:
            Identify the filing. Use :meth:`list_indexes` for available keys.
        query:
            Natural-language query string.
        top_k:
            Number of results to return.

        Returns
        -------
        list[tuple[Chunk, float]]
            ``(chunk, cosine_score)`` pairs in descending score order.
        """
        key: IndexKey = (ticker, year, filing_type, filing_date)
        return self._load_cached(key).search(query, top_k=top_k)

    def evict(self, ticker: str, year: str, filing_type: str, filing_date: str) -> None:
        """Drop a loaded index from the memory cache to free GPU/CPU RAM."""
        self._cache.pop((ticker, year, filing_type, filing_date), None)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"FaissVectorStore(index_dir={str(self._index_dir)!r}, "
            f"on_disk={len(self.list_indexes())}, "
            f"in_memory={len(self._cache)}, gpu={self._use_gpu})"
        )
