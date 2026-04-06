"""Filesystem-backed processed-data index for fast local existence checks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import threading
from typing import NamedTuple

from loguru import logger
import orjson
from watchdog.events import (
    FileCreatedEvent,
    FileDeletedEvent,
    FileModifiedEvent,
    FileMovedEvent,
    FileSystemEventHandler,
)
from watchdog.observers import Observer

from finance_data.settings import sec_settings

_SEC_CASE_DIR_RE = re.compile(r"^(?P<ticker>.+)-(?P<year>\d{4})$")
_TRANSCRIPT_FILENAME_RE = re.compile(
    r"^Q(?P<quarter>[1-4])(?:_.+)?\.md$", re.IGNORECASE
)


class ProcessedDataKey(NamedTuple):
    ticker: str
    year: str
    item: str


@dataclass(frozen=True)
class CacheEntry:
    exists: bool
    mtime: int
    entry_type: str


@dataclass(frozen=True)
class ProcessedDataSnapshot:
    sec_filings: frozenset[str]
    sec_markdown_filings: frozenset[str]
    transcript_quarters: frozenset[str]


class _ProcessedDataEventHandler(FileSystemEventHandler):
    """Dispatches file system changes to ``ProcessedDataIndex``."""

    def __init__(self, index: ProcessedDataIndex) -> None:
        self._index = index

    def on_created(self, event: FileCreatedEvent) -> None:
        self._index.handle_filesystem_event(Path(event.src_path))

    def on_modified(self, event: FileModifiedEvent) -> None:
        self._index.handle_filesystem_event(Path(event.src_path))

    def on_deleted(self, event: FileDeletedEvent) -> None:
        self._index.handle_filesystem_event(Path(event.src_path), is_deleted=True)

    def on_moved(self, event: FileMovedEvent) -> None:
        self._index.handle_filesystem_event(Path(event.src_path), is_deleted=True)
        self._index.handle_filesystem_event(Path(event.dest_path))


class ProcessedDataIndex:
    """Caches processed SEC filings/transcripts discovered on local storage."""

    def __init__(
        self,
        sec_data_dir: str,
        sec_markdown_dir: str,
        transcripts_dir: str,
        max_workers: int,
        ignore_ocr: bool,
        cache_file: str,
    ) -> None:
        self._sec_data_dir = Path(sec_data_dir)
        self._sec_markdown_dir = Path(sec_markdown_dir)
        self._transcripts_dir = Path(transcripts_dir)
        self._max_workers = max(1, max_workers)
        self._ignore_ocr = ignore_ocr
        self._cache_file = Path(cache_file)
        self._lock = threading.Lock()
        self._observer: Observer | None = None
        self._cache_entries: dict[str, CacheEntry] = {}
        self._snapshot = ProcessedDataSnapshot(
            sec_filings=frozenset(),
            sec_markdown_filings=frozenset(),
            transcript_quarters=frozenset(),
        )
        self._load_or_refresh_cache()
        self._start_watcher()

    def _load_or_refresh_cache(self) -> None:
        cache_loaded = self._load_cache_from_file()
        logger.info(f"{cache_loaded=}")
        if cache_loaded:
            return
        self.refresh()

    def _load_cache_from_file(self) -> bool:
        if not self._cache_file.exists():
            logger.info(f"{self._cache_file=}")
            return False
        try:
            payload = orjson.loads(self._cache_file.read_bytes())
        except orjson.JSONDecodeError:
            logger.warning(f"Invalid cache file. {self._cache_file=}")
            return False
        if not isinstance(payload, dict):
            logger.warning(f"Cache payload must be object. {type(payload)=}")
            return False
        entries = self._deserialize_cache(payload)
        snapshot = self._snapshot_from_entries(entries)
        with self._lock:
            self._cache_entries = entries
            self._snapshot = snapshot
        return True

    def refresh(self) -> None:
        """Rebuild indexes from disk and persist on-disk cache."""
        entries = self._build_entries_from_disk()
        snapshot = self._snapshot_from_entries(entries)
        with self._lock:
            self._cache_entries = entries
            self._snapshot = snapshot
        self._save_cache_to_file()

    def has_sec_filing(self, ticker: str, year: str, filing_type: str) -> bool:
        key = _sec_cache_key(ticker, year, filing_type)
        with self._lock:
            in_snapshot = (
                key in self._snapshot.sec_filings
                or key in self._snapshot.sec_markdown_filings
            )
        if in_snapshot:
            logger.info(
                f"Processed-data cache hit for SEC filing {key=}.",
            )
            return True
        return False

    def list_sec_filings(self, ticker: str, year: str) -> list[str]:
        ticker_key = ticker.upper().strip()
        year_key = str(year).strip()
        with self._lock:
            sec_items = [
                filing_type
                for key in self._snapshot.sec_filings
                if _has_ticker_year_match(key, ticker_key, year_key)
                for filing_type in [_parse_key(key).item]
            ]
            markdown_items = [
                filing_type
                for key in self._snapshot.sec_markdown_filings
                if _has_ticker_year_match(key, ticker_key, year_key)
                for filing_type in [_parse_key(key).item]
            ]
        return sorted(set(sec_items + markdown_items))

    def mark_sec_filing(self, ticker: str, year: str, filing_type: str) -> None:
        key = _sec_cache_key(ticker, year, filing_type)
        self._upsert_cache_entry(key=key, entry_type="sec_filings")

    def has_transcript(self, ticker: str, year: str, quarter: str) -> bool:
        key = _transcript_cache_key(ticker, year, quarter)
        with self._lock:
            in_snapshot = key in self._snapshot.transcript_quarters
        if in_snapshot:
            logger.info(
                f"Processed-data cache hit for transcript {key=}.",
            )
            return True
        return False

    def mark_transcript(self, ticker: str, year: str, quarter: str) -> None:
        key = _transcript_cache_key(ticker, year, quarter)
        self._upsert_cache_entry(key=key, entry_type="earnings_data")

    def _build_entries_from_disk(self) -> dict[str, CacheEntry]:
        entries: dict[str, CacheEntry] = {}
        self._scan_sec_files_to_entries(entries)
        self._scan_sec_markdown_to_entries(entries)
        self._scan_transcripts_to_entries(entries)
        return entries

    def _scan_sec_files_to_entries(self, entries: dict[str, CacheEntry]) -> None:
        for pdf_path in self._sec_data_dir.glob("*/*.pdf"):
            parsed = self._parse_sec_filing_path(pdf_path)
            if parsed is None:
                continue
            key = _sec_cache_key(parsed.ticker, parsed.year, parsed.item)
            entries[key] = CacheEntry(
                exists=True,
                mtime=int(pdf_path.stat().st_mtime),
                entry_type="sec_filings",
            )

    def _scan_sec_markdown_to_entries(self, entries: dict[str, CacheEntry]) -> None:
        for markdown_path in self._sec_markdown_dir.glob("*/*.md"):
            parsed = self._parse_sec_markdown_path(markdown_path)
            if parsed is None:
                continue
            key = _sec_cache_key(parsed.ticker, parsed.year, parsed.item)
            entries[key] = CacheEntry(
                exists=True,
                mtime=int(markdown_path.stat().st_mtime),
                entry_type="sec_markdown_filings",
            )

    def _scan_transcripts_to_entries(self, entries: dict[str, CacheEntry]) -> None:
        for transcript_path in self._transcripts_dir.glob("*/*/Q*.md"):
            parsed = self._parse_transcript_path(transcript_path)
            if parsed is None:
                continue
            key = _transcript_cache_key(parsed.ticker, parsed.year, parsed.item)
            entries[key] = CacheEntry(
                exists=True,
                mtime=int(transcript_path.stat().st_mtime),
                entry_type="earnings_data",
            )

    def _parse_sec_filing_path(self, pdf_path: Path) -> ProcessedDataKey | None:
        case_match = _SEC_CASE_DIR_RE.match(pdf_path.parent.name)
        if case_match is None:
            return None
        ticker = case_match.group("ticker").upper().strip()
        year = case_match.group("year").strip()
        filing_type = pdf_path.stem.upper().strip()
        if not ticker or not year or not filing_type:
            return None
        return ProcessedDataKey(ticker=ticker, year=year, item=filing_type)

    def _parse_transcript_path(
        self, transcript_path: Path
    ) -> ProcessedDataKey | None:
        quarter_match = _TRANSCRIPT_FILENAME_RE.match(transcript_path.name)
        if quarter_match is None:
            return None
        ticker = transcript_path.parent.parent.name.upper().strip()
        year = transcript_path.parent.name.strip()
        quarter = f"Q{quarter_match.group('quarter')}"
        if not ticker or not year:
            return None
        return ProcessedDataKey(ticker=ticker, year=year, item=quarter)

    def _parse_sec_markdown_path(
        self,
        markdown_path: Path,
    ) -> ProcessedDataKey | None:
        case_match = _SEC_CASE_DIR_RE.match(markdown_path.parent.name)
        if case_match is None:
            return None
        ticker = case_match.group("ticker").upper().strip()
        year = case_match.group("year").strip()
        filing_type = markdown_path.stem.upper().strip()
        if not ticker or not year or not filing_type:
            return None
        return ProcessedDataKey(ticker=ticker, year=year, item=filing_type)

    def _snapshot_from_entries(
        self,
        entries: dict[str, CacheEntry],
    ) -> ProcessedDataSnapshot:
        sec_filings = {
            key
            for key, value in entries.items()
            if value.exists and value.entry_type == "sec_filings"
        }
        sec_markdown_filings = {
            key
            for key, value in entries.items()
            if value.exists and value.entry_type == "sec_markdown_filings"
        }
        transcript_quarters = {
            key
            for key, value in entries.items()
            if value.exists and value.entry_type == "earnings_data"
        }
        return ProcessedDataSnapshot(
            sec_filings=frozenset(sec_filings),
            sec_markdown_filings=frozenset(sec_markdown_filings),
            transcript_quarters=frozenset(transcript_quarters),
        )

    def _deserialize_cache(
        self,
        payload: dict[object, object],
    ) -> dict[str, CacheEntry]:
        entries: dict[str, CacheEntry] = {}
        for key, raw_value in payload.items():
            if not isinstance(key, str) or not isinstance(raw_value, dict):
                continue
            exists = bool(raw_value.get("exists", False))
            mtime = int(raw_value.get("mtime", 0))
            entry_type = str(raw_value.get("type", "")).strip()
            if not entry_type:
                continue
            entries[key] = CacheEntry(exists=exists, mtime=mtime, entry_type=entry_type)
        return entries

    def _serialize_cache(self) -> dict[str, dict[str, int | bool | str]]:
        with self._lock:
            serialized = {
                key: {
                    "exists": value.exists,
                    "mtime": value.mtime,
                    "type": value.entry_type,
                }
                for key, value in self._cache_entries.items()
            }
        return serialized

    def _save_cache_to_file(self) -> None:
        payload = self._serialize_cache()
        self._cache_file.parent.mkdir(parents=True, exist_ok=True)
        encoded = orjson.dumps(
            payload,
            option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS,
        )
        self._cache_file.write_bytes(encoded)
        logger.info(f"{self._cache_file=}")

    def _start_watcher(self) -> None:
        handler = _ProcessedDataEventHandler(index=self)
        observer = Observer()
        self._schedule_if_exists(observer, self._sec_data_dir, handler)
        self._schedule_if_exists(observer, self._sec_markdown_dir, handler)
        self._schedule_if_exists(observer, self._transcripts_dir, handler)
        observer.daemon = True
        observer.start()
        self._observer = observer
        logger.info("Processed-data watcher started.")

    def _schedule_if_exists(
        self,
        observer: Observer,
        target_dir: Path,
        handler: _ProcessedDataEventHandler,
    ) -> None:
        if not target_dir.exists():
            logger.info(f"Skipping watcher for missing path. {target_dir=}")
            return
        observer.schedule(handler, target_dir.as_posix(), recursive=True)
        logger.info(f"Watcher scheduled. {target_dir=}")

    def handle_filesystem_event(self, path: Path, is_deleted: bool = False) -> None:
        cache_update = self._cache_update_for_path(path, is_deleted=is_deleted)
        if cache_update is None:
            return
        if is_deleted:
            self._remove_cache_entry(cache_update.key)
            return
        self._upsert_cache_entry(
            key=cache_update.key,
            entry_type=cache_update.entry_type,
            mtime=cache_update.mtime,
        )

    def _cache_update_for_path(
        self,
        path: Path,
        is_deleted: bool,
    ) -> _CacheUpdate | None:
        if self._is_path_under(path, self._sec_data_dir) and path.suffix.lower() == ".pdf":
            parsed = self._parse_sec_filing_path(path)
            if parsed is None:
                return None
            mtime = 0 if is_deleted else self._safe_mtime(path)
            key = _sec_cache_key(parsed.ticker, parsed.year, parsed.item)
            return _CacheUpdate(key=key, entry_type="sec_filings", mtime=mtime)
        if self._is_path_under(path, self._sec_markdown_dir) and path.suffix.lower() == ".md":
            parsed = self._parse_sec_markdown_path(path)
            if parsed is None:
                return None
            mtime = 0 if is_deleted else self._safe_mtime(path)
            key = _sec_cache_key(parsed.ticker, parsed.year, parsed.item)
            return _CacheUpdate(key=key, entry_type="sec_markdown_filings", mtime=mtime)
        if self._is_path_under(path, self._transcripts_dir) and path.suffix.lower() == ".md":
            parsed = self._parse_transcript_path(path)
            if parsed is None:
                return None
            mtime = 0 if is_deleted else self._safe_mtime(path)
            key = _transcript_cache_key(parsed.ticker, parsed.year, parsed.item)
            return _CacheUpdate(key=key, entry_type="earnings_data", mtime=mtime)
        return None

    def _upsert_cache_entry(
        self,
        key: str,
        entry_type: str,
        mtime: int | None = None,
    ) -> None:
        file_mtime = int(mtime) if mtime is not None else 0
        with self._lock:
            self._cache_entries[key] = CacheEntry(
                exists=True,
                mtime=file_mtime,
                entry_type=entry_type,
            )
            self._snapshot = self._snapshot_from_entries(self._cache_entries)
        logger.info(f"{key=}, {entry_type=}, {file_mtime=}")
        self._save_cache_to_file()

    def _remove_cache_entry(self, key: str) -> None:
        with self._lock:
            self._cache_entries.pop(key, None)
            self._snapshot = self._snapshot_from_entries(self._cache_entries)
        logger.info(f"{key=}")
        self._save_cache_to_file()

    def _is_path_under(self, child_path: Path, parent_path: Path) -> bool:
        try:
            child_path.resolve().relative_to(parent_path.resolve())
        except ValueError:
            return False
        return True

    def _safe_mtime(self, file_path: Path) -> int:
        try:
            return int(file_path.stat().st_mtime)
        except OSError:
            logger.info(f"Could not read mtime. {file_path=}")
            return 0


class _CacheUpdate(NamedTuple):
    key: str
    entry_type: str
    mtime: int


def _sec_cache_key(ticker: str, year: str, filing_type: str) -> str:
    normalized = ProcessedDataKey(
        ticker=ticker.upper().strip(),
        year=str(year).strip(),
        item=filing_type.upper().strip(),
    )
    return _join_key(normalized)


def _transcript_cache_key(ticker: str, year: str, quarter: str) -> str:
    normalized = ProcessedDataKey(
        ticker=ticker.upper().strip(),
        year=str(year).strip(),
        item=quarter.upper().strip(),
    )
    return _join_key(normalized)


def _join_key(key: ProcessedDataKey) -> str:
    return f"{key.ticker}|{key.year}|{key.item}"


def _parse_key(raw_key: str) -> ProcessedDataKey:
    ticker, year, item = raw_key.split("|", 2)
    return ProcessedDataKey(ticker=ticker, year=year, item=item)


def _has_ticker_year_match(raw_key: str, ticker: str, year: str) -> bool:
    parsed = _parse_key(raw_key)
    return parsed.ticker == ticker and parsed.year == year


processed_data_index = ProcessedDataIndex(
    sec_data_dir=sec_settings.sec_data_dir,
    sec_markdown_dir=(
        Path(sec_settings.olmocr_workspace) / "markdown" / sec_settings.sec_data_dir
    ).as_posix(),
    transcripts_dir=sec_settings.earnings_transcripts_dir,
    max_workers=sec_settings.processed_index_max_workers,
    ignore_ocr=sec_settings.ignore_ocr,
    cache_file=sec_settings.processed_index_cache_file,
)


def rebuild_processed_data_cache() -> str:
    """Refreshes the persisted processed-data ORJSON cache from disk."""
    processed_data_index.refresh()
    cache_file = sec_settings.processed_index_cache_file
    logger.info(f"{cache_file=}")
    return cache_file
