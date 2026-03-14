from __future__ import annotations

import logging
import pickle
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

_log = logging.getLogger(__name__)

if TYPE_CHECKING:
    import faiss as _faiss
    from openai import OpenAI as _OpenAI

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

_PAGE_TAG_RE = re.compile(r"<PAGE-NUM-(\d+)>")
_TABLE_RE = re.compile(r"<table[\s\S]*?</table>", re.IGNORECASE)
_SECTION_TITLE_RE = re.compile(
    r"^(Item\s+\d+[A-C]?\..*|Part\s+[IV]+.*)", re.MULTILINE | re.IGNORECASE
)

# Minimum character count for a text chunk before merging with the next.
_MIN_CHUNK_CHARS = 200

# Batch size for embedding requests.
_EMBED_BATCH_SIZE = 32


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class Chunk:
    """A single logical unit extracted from the markdown document."""

    text: str
    chunk_type: str  # "table" | "text"
    page_num: int | None
    section_title: str | None
    index: int  # sequential position in the document

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
    """Split *text* into a list of :class:`Chunk` objects.

    Strategy:
    1. Walk through the document tracking the current page number via
       ``<PAGE-NUM-X>`` tags.
    2. HTML ``<table>…</table>`` blocks are kept as atomic "table" chunks.
    3. Remaining text is split on blank lines; short fragments (< 200 chars)
       are merged into their predecessor to avoid noise.
    4. Section titles (``Item 1.``, ``Part II``, etc.) are detected and
       attached as metadata to subsequent chunks.
    """
    chunks: list[Chunk] = []
    current_page: int | None = None
    current_section: str | None = None
    index = 0

    # We split the document into alternating "non-table" and "table" segments.
    # _TABLE_RE.split() returns [pre, table, pre, table, …, pre]
    # _TABLE_RE.findall() returns [table, table, …]
    parts = _TABLE_RE.split(text)
    tables = _TABLE_RE.findall(text)

    for part_idx, non_table_text in enumerate(parts):
        # ----------------------------------------------------------------
        # Process non-table text before the next <table>
        # ----------------------------------------------------------------
        for raw_para in re.split(r"\n{2,}", non_table_text):
            para = raw_para.strip()
            if not para:
                continue

            # Update tracked page number from any page tags in this segment.
            for m in _PAGE_TAG_RE.finditer(para):
                current_page = int(m.group(1))
            # Strip page tags from content.
            para_clean = _PAGE_TAG_RE.sub("", para).strip()
            if not para_clean:
                continue

            # Detect section title.
            title_match = _SECTION_TITLE_RE.search(para_clean)
            if title_match:
                current_section = title_match.group(0).strip()

            # Merge short fragments with the previous text chunk.
            if (
                chunks
                and chunks[-1].chunk_type == "text"
                and len(chunks[-1].text) < _MIN_CHUNK_CHARS
            ):
                chunks[-1].text = chunks[-1].text + "\n\n" + para_clean
                # Update section title if this paragraph carries one.
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

        # ----------------------------------------------------------------
        # Process the <table> that follows this non-table segment (if any)
        # ----------------------------------------------------------------
        if part_idx < len(tables):
            table_text = tables[part_idx].strip()
            chunks.append(
                Chunk(
                    text=table_text,
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


def embed_chunks(
    chunks: list[Chunk],
    client: "_OpenAI",
    model: str,
) -> "np.ndarray":
    """Embed *chunks* using the OpenAI-compatible vLLM endpoint.

    Returns a 2-D float32 array of shape ``(len(chunks), embed_dim)``
    with L2-normalised rows (cosine-similarity-ready).
    """
    import numpy as np  # local to keep import lazy

    texts = [c.text for c in chunks]
    all_vectors: list[list[float]] = []

    for start in range(0, len(texts), _EMBED_BATCH_SIZE):
        batch = texts[start : start + _EMBED_BATCH_SIZE]
        response = client.embeddings.create(input=batch, model=model)
        # openai SDK returns embeddings ordered by index
        batch_vecs = [
            item.embedding for item in sorted(response.data, key=lambda x: x.index)
        ]
        all_vectors.extend(batch_vecs)

    matrix = np.array(all_vectors, dtype=np.float32)
    # L2-normalise so that inner-product == cosine similarity.
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return matrix / norms


# ---------------------------------------------------------------------------
# GPU helpers  (gracefully degrade when faiss-gpu-cu12 lacks the full API)
# ---------------------------------------------------------------------------


def _try_move_to_gpu(cpu_index: "_faiss.Index", device: int = 0) -> "_faiss.Index":
    """Attempt to move *cpu_index* to GPU device *device*.

    Falls back to the original CPU index (with a warning) if:
    - ``faiss.StandardGpuResources`` is not available (PyPI ``faiss-gpu-cu12``
      vs conda ``faiss-gpu`` API difference), or
    - no CUDA-capable GPU is visible to FAISS.
    """
    import faiss

    # PyPI faiss-gpu-cu12 uses a minimal GPU wrapper that exposes
    # index_cpu_to_all_gpus / index_cpu_to_gpus_list but not StandardGpuResources.
    num_gpus = faiss.get_num_gpus()
    if num_gpus == 0:
        _log.warning("faiss.get_num_gpus() == 0; falling back to CPU index.")
        return cpu_index

    # Try the standard conda faiss-gpu API first.
    if hasattr(faiss, "StandardGpuResources") and hasattr(faiss, "index_cpu_to_gpu"):
        try:
            res = faiss.StandardGpuResources()
            return faiss.index_cpu_to_gpu(res, device, cpu_index)
        except Exception as exc:
            _log.warning("index_cpu_to_gpu failed (%s); falling back to CPU.", exc)
            return cpu_index

    # Fall back to index_cpu_to_all_gpus if available.
    if hasattr(faiss, "index_cpu_to_all_gpus"):
        try:
            return faiss.index_cpu_to_all_gpus(cpu_index)
        except Exception as exc:
            _log.warning("index_cpu_to_all_gpus failed (%s); falling back to CPU.", exc)
            return cpu_index

    _log.warning(
        "No supported GPU index transfer function found in faiss; "
        "falling back to CPU index. Install faiss via conda for full GPU support."
    )
    return cpu_index


def _index_gpu_to_cpu(gpu_index: "_faiss.Index") -> "_faiss.Index":
    """Convert a GPU FAISS index back to CPU, handling API differences."""
    import faiss

    if hasattr(faiss, "index_gpu_to_cpu"):
        return faiss.index_gpu_to_cpu(gpu_index)

    # Shards wrapper (used by index_cpu_to_all_gpus) exposes quantizer on CPU.
    if hasattr(gpu_index, "quantizer"):
        return gpu_index.quantizer  # type: ignore[attr-defined]

    _log.warning(
        "Cannot convert GPU index to CPU (faiss.index_gpu_to_cpu not available). "
        "Returning as-is."
    )
    return gpu_index


# ---------------------------------------------------------------------------
# FAISS index
# ---------------------------------------------------------------------------


class FaissVectorIndex:
    """An in-memory FAISS index built from a list of :class:`Chunk` objects.

    Use :meth:`from_markdown` to build directly from a markdown string, or
    construct manually when you already have chunks and embeddings.
    """

    def __init__(
        self,
        chunks: list[Chunk],
        embeddings: "np.ndarray",
        faiss_index: "_faiss.Index",
        client: "_OpenAI",
        model: str,
    ) -> None:
        self.chunks = chunks
        self.embeddings = embeddings
        self._index = faiss_index
        self._client = client
        self._model = model

    # ------------------------------------------------------------------

    @classmethod
    def from_markdown(
        cls,
        text: str,
        client: "_OpenAI",
        model: str,
        use_gpu: bool = False,
    ) -> "FaissVectorIndex":
        """Build a :class:`FaissVectorIndex` from raw markdown text.

        Steps:
        1. Chunk the markdown (tables kept whole, text split by paragraphs).
        2. Embed all chunks in batches via *client*.
        3. Insert L2-normalised vectors into a ``faiss.IndexFlatIP`` index.
        4. Optionally move the index to GPU.
        """
        import faiss  # lazy import

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

    # ------------------------------------------------------------------

    def to_gpu(self, device: int = 0) -> None:
        """Move the internal FAISS index to a GPU device in-place."""
        if hasattr(self._index, "getDevice"):
            return  # already on GPU
        gpu_index = _try_move_to_gpu(self._index, device=device)
        if gpu_index is not self._index:
            self._index = gpu_index

    def to_cpu(self) -> None:
        """Move the internal FAISS index back to CPU in-place."""
        if not hasattr(self._index, "getDevice"):
            return  # already on CPU
        self._index = _index_gpu_to_cpu(self._index)

    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        """Persist the index and chunks to *path*.

        Creates the directory if needed and writes:
        - ``index.faiss`` -- the FAISS index (always serialised as CPU index)
        - ``chunks.pkl``  -- pickled list of :class:`Chunk` objects
        """
        import faiss

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Always write the CPU version so it is portable across machines.
        if hasattr(self._index, "getDevice"):
            cpu_index = _index_gpu_to_cpu(self._index)
        else:
            cpu_index = self._index
        faiss.write_index(cpu_index, str(path / "index.faiss"))

        with open(path / "chunks.pkl", "wb") as fh:
            pickle.dump(self.chunks, fh)

    @classmethod
    def load(
        cls,
        path: Path,
        client: "_OpenAI",
        model: str,
        use_gpu: bool = False,
        device: int = 0,
    ) -> "FaissVectorIndex":
        """Load a previously saved index from *path*.

        Args:
            path: Directory containing ``index.faiss`` and ``chunks.pkl``.
            client: OpenAI-compatible client for query embedding.
            model: Embedding model name.
            use_gpu: Move the index to GPU after loading.
            device: GPU device index (default 0).

        Returns:
            A fully-reconstructed :class:`FaissVectorIndex`.
        """
        import faiss

        path = Path(path)
        cpu_index = faiss.read_index(str(path / "index.faiss"))

        with open(path / "chunks.pkl", "rb") as fh:
            chunks: list[Chunk] = pickle.load(fh)

        index = _try_move_to_gpu(cpu_index, device=device) if use_gpu else cpu_index

        n = cpu_index.ntotal
        dim = cpu_index.d
        embeddings = np.zeros((n, dim), dtype=np.float32)

        return cls(
            chunks=chunks,
            embeddings=embeddings,
            faiss_index=index,
            client=client,
            model=model,
        )

    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[tuple[Chunk, float]]:
        """Return the *top_k* most relevant chunks for *query*.

        Args:
            query: Natural-language query string.
            top_k: Number of results to return.

        Returns:
            List of ``(chunk, score)`` tuples sorted by descending cosine
            similarity score.
        """
        import numpy as np  # lazy

        query_chunk = Chunk(
            text=query, chunk_type="text", page_num=None, section_title=None, index=-1
        )
        q_vec = embed_chunks([query_chunk], self._client, self._model)

        k = min(top_k, len(self.chunks))
        scores, indices = self._index.search(q_vec, k)

        results: list[tuple[Chunk, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append((self.chunks[idx], float(score)))
        return results

    def __len__(self) -> int:
        return len(self.chunks)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"FaissVectorIndex(chunks={len(self.chunks)}, "
            f"dim={self.embeddings.shape[1]}, model={self._model!r})"
        )


# ---------------------------------------------------------------------------
# EmbeddingEnvironment
# ---------------------------------------------------------------------------


@dataclass
class EmbeddingEnvironment:
    """Pairs a :class:`FaissVectorIndex` with SEC filing metadata."""

    ticker: str
    year: str
    filing_type: str
    markdown_path: Path
    markdown_text: str
    vector_index: FaissVectorIndex

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[tuple[Chunk, float]]:
        """Delegate to :meth:`FaissVectorIndex.search`."""
        return self.vector_index.search(query, top_k=top_k)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"EmbeddingEnvironment(ticker={self.ticker!r}, year={self.year!r}, "
            f"filing_type={self.filing_type!r}, "
            f"chunks={len(self.vector_index)})"
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def markdown_to_embedding_env(
    markdown_path: Path,
    ticker: str,
    year: str,
    filing_type: str | None = None,
    embedding_server: str | None = None,
    embedding_model: str | None = None,
    use_gpu: bool | None = None,
) -> EmbeddingEnvironment:
    """Build an :class:`EmbeddingEnvironment` from a markdown file.

    Args:
        markdown_path: Path to the ``.md`` file produced by olmOCR.
        ticker: Stock ticker symbol (e.g. ``"AMZN"``).
        year: Filing year (e.g. ``"2025"``).
        filing_type: Override filing type; inferred from filename if omitted.
        embedding_server: Base URL of the vLLM embedding server
            (default: ``settings.embedding_server``).
        embedding_model: Model name to pass to the embeddings endpoint
            (default: ``settings.embedding_model``).
        use_gpu: Move FAISS index to GPU after building
            (default: ``settings.faiss_use_gpu``).

    Returns:
        A fully-built :class:`EmbeddingEnvironment` ready for querying.
    """
    from openai import OpenAI

    from settings import olmocr_settings

    resolved = Path(markdown_path).resolve()
    markdown_text = resolved.read_text(encoding="utf-8")
    ft = filing_type or resolved.stem

    server = embedding_server or olmocr_settings.embedding_server
    model = embedding_model or olmocr_settings.embedding_model
    gpu = use_gpu if use_gpu is not None else olmocr_settings.faiss_use_gpu

    client = OpenAI(base_url=server, api_key="not-needed")

    vector_index = FaissVectorIndex.from_markdown(
        text=markdown_text,
        client=client,
        model=model,
        use_gpu=gpu,
    )

    return EmbeddingEnvironment(
        ticker=ticker,
        year=year,
        filing_type=ft,
        markdown_path=resolved,
        markdown_text=markdown_text,
        vector_index=vector_index,
    )
