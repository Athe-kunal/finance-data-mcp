from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openai import OpenAI as _OpenAI

    from .embd_env import Chunk, FaissVectorIndex

# Key type: (ticker, year, filing_type, filing_date)
IndexKey = tuple[str, str, str, str]


class FaissVectorStore:
    """Registry of persistent FAISS indexes, one per SEC filing.

    Parameters
    ----------
    index_dir:
        Root directory for persisted indexes.
        Defaults to ``settings.faiss_index_dir``.
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
        from settings import olmocr_settings

        self._index_dir = Path(index_dir or olmocr_settings.faiss_index_dir)
        self._embedding_server = embedding_server or olmocr_settings.embedding_server
        self._embedding_model = embedding_model or olmocr_settings.embedding_model
        self._use_gpu = (
            use_gpu if use_gpu is not None else olmocr_settings.faiss_use_gpu
        )

        # Cache: key -> loaded FaissVectorIndex (lazy)
        self._cache: dict[IndexKey, FaissVectorIndex] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _index_path(self, key: IndexKey) -> Path:
        ticker, year, filing_type, filing_date = key
        return self._index_dir / ticker / year / filing_type / filing_date

    def _make_client(self) -> "_OpenAI":
        from openai import OpenAI

        return OpenAI(base_url=self._embedding_server, api_key="not-needed")

    def _load_index(self, key: IndexKey) -> "FaissVectorIndex":
        """Load (or return cached) the index for *key*."""
        if key in self._cache:
            return self._cache[key]

        from .embd_env import FaissVectorIndex

        path = self._index_path(key)
        if not (path / "index.faiss").exists():
            raise FileNotFoundError(
                f"No FAISS index found at {path}. "
                "Run ingest() first to build and save the index."
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
        """Return all (ticker, year, filing_type, filing_date) keys on disk.

        Scans ``{index_dir}/*/*/*/*/`` directories that contain
        ``index.faiss``.
        """
        keys: list[IndexKey] = []
        if not self._index_dir.exists():
            return keys
        for faiss_file in sorted(self._index_dir.glob("*/*/**/index.faiss")):
            # Structure: {index_dir}/{ticker}/{year}/{filing_type}/{filing_date}/index.faiss
            parts = faiss_file.relative_to(self._index_dir).parts
            if len(parts) == 5:  # ticker/year/filing_type/filing_date/index.faiss
                ticker, year, filing_type, filing_date, _ = parts
                keys.append((ticker, year, filing_type, filing_date))
        return keys

    async def ingest(
        self,
        ticker: str,
        year: str,
        filing_type: str,
        include_amends: bool = True,
        workspace: str | Path | None = None,
        force: bool = False,
    ) -> list[IndexKey]:
        """Download, OCR, embed, and persist indexes for one filing.

        Parameters
        ----------
        ticker:
            Stock ticker symbol (e.g. ``"AMZN"``).
        year:
            Filing year (e.g. ``"2023"``).
        filing_type:
            Filing type (e.g. ``"10-K"`` or ``"10-Q"``).
        include_amends:
            Include amended filings.
        workspace:
            olmOCR workspace directory (default from settings).
        force:
            Re-build and overwrite existing indexes even if they already exist
            on disk.

        Returns
        -------
        list[IndexKey]
            The keys of the indexes that were ingested.
        """
        from settings import olmocr_settings

        from filings.sec_data import get_sec_results
        from ocr.olmocr_pipeline import get_markdown_path, run_olmo_ocr

        from .embd_env import FaissVectorIndex
        from .pipeline import ensure_sec_data

        workspace_str = str(workspace or olmocr_settings.olmocr_workspace)

        # Step 1: Download PDFs
        sec_results, _pdf_paths = await ensure_sec_data(
            ticker=ticker,
            year=year,
            filing_types=[filing_type],
            include_amends=include_amends,
        )
        if not sec_results:
            print(f"[ingest] No SEC results found for {ticker} {year} {filing_type}.")
            return []

        # Step 2: OCR
        pdf_dir_str = f"sec_data/{ticker}-{year}"
        await run_olmo_ocr(pdf_dir=pdf_dir_str, workspace=workspace_str)

        client = self._make_client()
        ingested_keys: list[IndexKey] = []

        # Step 3: Embed and save each filing's markdown
        for sr in sec_results:
            source_file = f"sec_data/{ticker}-{year}/{sr.form_name}.pdf"
            markdown_path_str = get_markdown_path(workspace_str, source_file)
            markdown_path = Path(markdown_path_str)

            if not markdown_path.exists():
                print(f"[ingest] Markdown not found: {markdown_path}, skipping.")
                continue

            key: IndexKey = (ticker, year, sr.form_name, sr.filing_date)
            dest = self._index_path(key)

            if (dest / "index.faiss").exists() and not force:
                print(
                    f"[ingest] Index already exists for {key}, skipping. Pass force=True to rebuild."
                )
                ingested_keys.append(key)
                continue

            print(f"[ingest] Building index for {key} ...")
            markdown_text = markdown_path.read_text(encoding="utf-8")
            idx = FaissVectorIndex.from_markdown(
                text=markdown_text,
                client=client,
                model=self._embedding_model,
                use_gpu=self._use_gpu,
            )
            idx.save(dest)
            self._cache[key] = idx
            ingested_keys.append(key)
            print(f"[ingest] Saved {len(idx)} chunks to {dest}")

        return ingested_keys

    def query(
        self,
        ticker: str,
        year: str,
        filing_type: str,
        filing_date: str,
        query: str,
        top_k: int = 5,
    ) -> list[tuple["Chunk", float]]:
        """Search the FAISS index for *query*.

        Loads the index from disk on first access (then caches in memory).

        Parameters
        ----------
        ticker, year, filing_type, filing_date:
            Identify the filing. Use :meth:`list_indexes` to see available keys.
        query:
            Natural-language query string.
        top_k:
            Number of results to return.

        Returns
        -------
        list[tuple[Chunk, float]]
            ``(chunk, score)`` pairs sorted by descending cosine similarity.
        """
        key: IndexKey = (ticker, year, filing_type, filing_date)
        idx = self._load_index(key)
        return idx.search(query, top_k=top_k)

    def evict(self, ticker: str, year: str, filing_type: str, filing_date: str) -> None:
        """Remove a loaded index from the in-memory cache (frees GPU/CPU memory)."""
        key: IndexKey = (ticker, year, filing_type, filing_date)
        self._cache.pop(key, None)

    def __repr__(self) -> str:  # pragma: no cover
        n_disk = len(self.list_indexes())
        n_mem = len(self._cache)
        return (
            f"FaissVectorStore(index_dir={str(self._index_dir)!r}, "
            f"on_disk={n_disk}, in_memory={n_mem}, gpu={self._use_gpu})"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m dataloader.vector_store",
        description="Manage persistent FAISS indexes for SEC filings.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # -- ingest ----------------------------------------------------------------
    ingest_p = sub.add_parser("ingest", help="Download, OCR, embed, and save index.")
    ingest_p.add_argument("--ticker", required=True, help="Stock ticker (e.g. AMZN)")
    ingest_p.add_argument("--year", required=True, help="Filing year (e.g. 2023)")
    ingest_p.add_argument(
        "--filing-type",
        required=True,
        dest="filing_type",
        help="Filing type (e.g. 10-K, 10-Q)",
    )
    ingest_p.add_argument(
        "--no-amends",
        action="store_false",
        dest="include_amends",
        help="Exclude amended filings",
    )
    ingest_p.add_argument(
        "--force",
        action="store_true",
        help="Rebuild and overwrite existing indexes",
    )
    ingest_p.add_argument(
        "--index-dir",
        dest="index_dir",
        default=None,
        help="Override faiss_index_dir from settings",
    )
    ingest_p.add_argument("--no-gpu", action="store_true", help="Disable GPU for FAISS")

    # -- query -----------------------------------------------------------------
    query_p = sub.add_parser("query", help="Query an existing index.")
    query_p.add_argument("--ticker", required=True)
    query_p.add_argument("--year", required=True)
    query_p.add_argument("--filing-type", required=True, dest="filing_type")
    query_p.add_argument(
        "--filing-date",
        required=True,
        dest="filing_date",
        help="Exact filing date, e.g. 2024-02-02",
    )
    query_p.add_argument("--query", required=True, help="Search query string")
    query_p.add_argument("--top-k", type=int, default=5, dest="top_k")
    query_p.add_argument("--index-dir", dest="index_dir", default=None)
    query_p.add_argument("--no-gpu", action="store_true")

    # -- list ------------------------------------------------------------------
    list_p = sub.add_parser("list", help="List all available indexes on disk.")
    list_p.add_argument("--index-dir", dest="index_dir", default=None)
    list_p.add_argument(
        "--json", action="store_true", dest="as_json", help="Output as JSON array"
    )

    return parser


def _main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    use_gpu = not getattr(args, "no_gpu", False) if hasattr(args, "no_gpu") else None
    store = FaissVectorStore(
        index_dir=getattr(args, "index_dir", None),
        use_gpu=use_gpu,
    )

    if args.command == "ingest":
        keys = asyncio.run(
            store.ingest(
                ticker=args.ticker,
                year=args.year,
                filing_type=args.filing_type,
                include_amends=args.include_amends,
                force=args.force,
            )
        )
        print(f"\nIngested {len(keys)} index(es):")
        for k in keys:
            print(f"  {k}")

    elif args.command == "query":
        results = store.query(
            ticker=args.ticker,
            year=args.year,
            filing_type=args.filing_type,
            filing_date=args.filing_date,
            query=args.query,
            top_k=args.top_k,
        )
        print(f"\nQuery: {args.query!r}")
        print(f"Top {len(results)} results:\n")
        for rank, (chunk, score) in enumerate(results, 1):
            preview = chunk.text[:300].replace("\n", " ")
            print(
                f"  [{rank}] score={score:.4f}  page={chunk.page_num}  "
                f"section={chunk.section_title!r}  type={chunk.chunk_type}"
            )
            print(f"       {preview!r}...")
            print()

    elif args.command == "list":
        keys = store.list_indexes()
        if getattr(args, "as_json", False):
            print(json.dumps([list(k) for k in keys], indent=2))
        else:
            if not keys:
                print("No indexes found.")
            else:
                print(f"{'Ticker':<10} {'Year':<6} {'Filing':<8} {'Filing Date'}")
                print("-" * 46)
                for ticker, year, filing_type, filing_date in keys:
                    print(f"{ticker:<10} {year:<6} {filing_type:<8} {filing_date}")


if __name__ == "__main__":
    _main()
