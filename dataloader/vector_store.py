from __future__ import annotations

import json
import logging
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple, Sequence
import faiss
from settings import olmocr_settings
import numpy as np
from openai import OpenAI
import markdownify

from filings.sec_data import load_sec_results
from ocr.olmocr_pipeline import get_markdown_path, run_olmo_ocr
from dataloader.pipeline import ensure_sec_data

_log = logging.getLogger(__name__)
_PAGE_TAG_RE = re.compile(r"<PAGE-NUM-(\d+)>")
_TABLE_RE = re.compile(r"<table[\s\S]*?</table>", re.IGNORECASE)
_SECTION_TITLE_RE = re.compile(
    r"^(Item\s+\d+[A-C]?\..*|Part\s+[IV]+.*)", re.MULTILINE | re.IGNORECASE
)
_MIN_CHUNK_CHARS = 2048
_EMBED_BATCH_SIZE = 2048


class IndexKey(NamedTuple):
    ticker: str
    year: str
    filing_type: str


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


@dataclass
class _FilingData:
    chunks: list[Chunk]
    embeddings: np.ndarray
    faiss_index: "faiss.Index"
    filing_date: str | None


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
                and len(chunks[-1].text.replace(" ", "")) < _MIN_CHUNK_CHARS
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


class FaissVectorIndex:
    """Registry of FAISS indexes, one per SEC filing.

    Indexes are stored on disk under::

        {index_dir}/{ticker}/{year}/{filing_type}/
            index.faiss
            chunks.pkl
            meta.json

    and looked up by an :class:`IndexKey` ``(ticker, year, filing_type)``.

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
        self._cache: dict[IndexKey, _FilingData] = {}

    def _key_path(self, key: IndexKey) -> Path:
        return self._index_dir / key.ticker / key.year / key.filing_type

    def _make_client(self) -> "OpenAI":
        return OpenAI(base_url=self._embedding_server, api_key="not-needed")

    def _build_filing(
        self, key: IndexKey, markdown_text: str, filing_date: str | None
    ) -> _FilingData:
        chunks = chunk_markdown(markdown_text)
        if not chunks:
            raise ValueError(f"No chunks produced from markdown for {key}.")

        embeddings = embed_chunks(chunks, self._make_client(), self._embedding_model)
        dim = embeddings.shape[1]

        cpu_index = faiss.IndexFlatIP(dim)
        cpu_index.add(embeddings)
        index = _try_move_to_gpu(cpu_index) if self._use_gpu else cpu_index

        return _FilingData(
            chunks=chunks,
            embeddings=embeddings,
            faiss_index=index,
            filing_date=filing_date,
        )

    def _save_filing(self, key: IndexKey, data: _FilingData) -> None:
        path = self._key_path(key)
        path.mkdir(parents=True, exist_ok=True)

        cpu_index = (
            _index_gpu_to_cpu(data.faiss_index)
            if hasattr(data.faiss_index, "getDevice")
            else data.faiss_index
        )
        faiss.write_index(cpu_index, str(path / "index.faiss"))
        with open(path / "chunks.pkl", "wb") as fh:
            pickle.dump(data.chunks, fh)
        with open(path / "meta.json", "w") as fh:
            json.dump({"filing_date": data.filing_date}, fh)

    def _load_filing(self, key: IndexKey) -> _FilingData:
        if key in self._cache:
            return self._cache[key]

        path = self._key_path(key)
        if not (path / "index.faiss").exists():
            raise FileNotFoundError(
                f"No FAISS index on disk for {key}. "
                "Call from_markdown() or ingest() first."
            )

        cpu_index = faiss.read_index(str(path / "index.faiss"))
        with open(path / "chunks.pkl", "rb") as fh:
            chunks: list[Chunk] = pickle.load(fh)

        filing_date: str | None = None
        meta_path = path / "meta.json"
        if meta_path.exists():
            with open(meta_path) as fh:
                filing_date = json.load(fh).get("filing_date")

        index = _try_move_to_gpu(cpu_index) if self._use_gpu else cpu_index
        embeddings = np.zeros((cpu_index.ntotal, cpu_index.d), dtype=np.float32)

        data = _FilingData(
            chunks=chunks,
            embeddings=embeddings,
            faiss_index=index,
            filing_date=filing_date,
        )
        self._cache[key] = data
        return data

    @staticmethod
    def _resolve_filing_date(ticker: str, year: str, filing_type: str) -> str | None:
        """Look up filing_date from ``sec_data/{ticker}-{year}/sec_results.json``."""
        results = load_sec_results(ticker, year)
        if not results:
            return None
        for sr in results:
            if sr.form_name == filing_type:
                return sr.filing_date
        return None

    def from_markdown(
        self,
        ticker: str,
        year: str,
        markdown_paths: Sequence[Path],
        force: bool = False,
    ) -> list[IndexKey]:
        """Chunk, embed, and persist a list of markdown files.

        The filing type is extracted from each file's stem (e.g. ``10-Q1.md``
        → ``"10-Q1"``).  The filing date is looked up automatically from
        ``sec_data/{ticker}-{year}/sec_results.json`` when available.

        Parameters
        ----------
        ticker, year:
            Identify the company and reporting period.
        markdown_paths:
            Paths to ``.md`` files produced by olmOCR.
        force:
            Re-build indexes even if they already exist on disk.

        Returns
        -------
        list[IndexKey]
            One key per file that was successfully indexed.
        """
        ingested: list[IndexKey] = []

        for raw_path in markdown_paths:
            md_path = Path(raw_path)
            filing_type = md_path.stem  # "10-Q1" from "10-Q1.md"
            key = IndexKey(ticker, year, filing_type)
            dest = self._key_path(key)

            if (dest / "index.faiss").exists() and not force:
                _log.info(
                    "Index already exists for %s, skipping. Pass force=True to rebuild.",
                    key,
                )
                ingested.append(key)
                continue

            filing_date = self._resolve_filing_date(ticker, year, filing_type)
            markdown_text = md_path.read_text(encoding="utf-8")
            _log.info("Building index for %s (%s) …", key, md_path)

            data = self._build_filing(key, markdown_text, filing_date)
            self._save_filing(key, data)
            self._cache[key] = data
            _log.info("Saved %d chunks to %s", len(data.chunks), dest)
            ingested.append(key)

        return ingested

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

        md_paths: list[Path] = []
        for sr in sec_results:
            md_path = Path(
                get_markdown_path(
                    workspace_str, f"sec_data/{ticker}-{year}/{sr.form_name}.pdf"
                )
            )
            if not md_path.exists():
                _log.warning("Markdown not found: %s, skipping.", md_path)
                continue
            md_paths.append(md_path)

        return self.from_markdown(ticker, year, md_paths, force=force)

    def list_filings(self, ticker: str, year: str) -> list[dict[str, str | None]]:
        """List all ingested filings for a given ticker and year.

        Returns a list of dicts with ``"filing_type"`` and ``"filing_date"`` for
        each filing found on disk under ``{index_dir}/{ticker}/{year}/``.
        """
        base = self._index_dir / ticker / year
        if not base.exists():
            return []

        results: list[dict[str, str | None]] = []
        for index_file in sorted(base.glob("*/index.faiss")):
            filing_type = index_file.parent.name
            filing_date: str | None = None
            meta_path = index_file.parent / "meta.json"
            if meta_path.exists():
                with open(meta_path) as fh:
                    filing_date = json.load(fh).get("filing_date")
            results.append({"filing_type": filing_type, "filing_date": filing_date})

        return results

    def list_indexes(self) -> list[IndexKey]:
        """Return all ``IndexKey``s present on disk (across all tickers and years)."""
        if not self._index_dir.exists():
            return []
        keys: list[IndexKey] = []
        for f in sorted(self._index_dir.glob("*/*/*/index.faiss")):
            parts = f.relative_to(self._index_dir).parts
            if len(parts) == 4:
                ticker, year, filing_type, _ = parts
                keys.append(IndexKey(ticker, year, filing_type))
        return keys

    def search(
        self,
        ticker: str,
        year: str,
        filing_type: str,
        query: str,
        top_k: int = 5,
    ) -> list[tuple[Chunk, float]]:
        """Semantic search over a single filing's vector index.

        Loads the index from disk on first access, then caches in memory.

        Parameters
        ----------
        ticker, year, filing_type:
            Identify the filing. Use :meth:`list_filings` for available keys.
        query:
            Natural-language query string.
        top_k:
            Number of results to return.

        Returns
        -------
        list[tuple[Chunk, float]]
            ``(chunk, cosine_score)`` pairs in descending score order.
        """
        key = IndexKey(ticker, year, filing_type)
        data = self._load_filing(key)

        q_chunk = Chunk(
            text=query, chunk_type="text", page_num=None, section_title=None, index=-1
        )
        q_vec = embed_chunks([q_chunk], self._make_client(), self._embedding_model)

        k = min(top_k, len(data.chunks))
        scores, indices = data.faiss_index.search(q_vec, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            chunk = data.chunks[idx]
            if chunk.chunk_type == "table":
                chunk = Chunk(
                    text=markdownify.markdownify(chunk.text, heading_style="ATX"),
                    chunk_type=chunk.chunk_type,
                    page_num=chunk.page_num,
                    section_title=chunk.section_title,
                    index=chunk.index,
                )
            results.append((chunk, float(score)))
        return results

    def evict(self, ticker: str, year: str, filing_type: str, filing_date: str) -> None:
        """Drop a loaded index from the memory cache to free GPU/CPU RAM."""
        self._cache.pop(IndexKey(ticker, year, filing_type), None)

    def to_gpu(self, ticker: str, year: str, filing_type: str, device: int = 0) -> None:
        """Move a cached filing's FAISS index to GPU in-place."""
        key = IndexKey(ticker, year, filing_type)
        if key not in self._cache:
            return
        data = self._cache[key]
        if hasattr(data.faiss_index, "getDevice"):
            return
        gpu = _try_move_to_gpu(data.faiss_index, device=device)
        if gpu is not data.faiss_index:
            data.faiss_index = gpu

    def to_cpu(self, ticker: str, year: str, filing_type: str) -> None:
        """Move a cached filing's FAISS index back to CPU in-place."""
        key = IndexKey(ticker, year, filing_type)
        if key not in self._cache:
            return
        data = self._cache[key]
        if not hasattr(data.faiss_index, "getDevice"):
            return
        data.faiss_index = _index_gpu_to_cpu(data.faiss_index)

    def __len__(self) -> int:
        return len(self.list_indexes())

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"FaissVectorIndex(index_dir={str(self._index_dir)!r}, "
            f"on_disk={len(self.list_indexes())}, "
            f"in_memory={len(self._cache)}, gpu={self._use_gpu})"
        )
