import re
from dataclasses import dataclass

import markdownify
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

_PAGE_BLOCK_RE = re.compile(
    r"<PAGE-NUM-(\d+)>\s*([\s\S]*?)\s*</PAGE-NUM-\1>",
    re.IGNORECASE,
)
_PAGE_TAG_RE = re.compile(r"</?PAGE-NUM-\d+>", re.IGNORECASE)
_TABLE_RE = re.compile(r"<table[\s\S]*?</table>", re.IGNORECASE)
_SECTION_TITLE_RE = re.compile(
    r"^(Item\s+\d+[A-C]?\..*|Part\s+[IV]+.*)",
    re.MULTILINE | re.IGNORECASE,
)

_SEC_CHUNK_SIZE = 1024
_SEC_CHUNK_OVERLAP = 128
_EARNINGS_TRANSCRIPT_CHUNK_SIZE = 1024
_EARNINGS_TRANSCRIPT_OVERLAP = 128
_MIN_CHUNK_LENGTH = 100


@dataclass
class Chunk:
    """A single logical unit extracted from a markdown document."""

    text: str
    chunk_type: str  # "table" | "text"
    page_num: int | None
    section_title: str | None
    index: int


def _extract_section(text: str, current_section: str | None) -> str | None:
    title_match = _SECTION_TITLE_RE.search(text)
    if title_match:
        return title_match.group(0).strip()
    return current_section


def alnum_length(text: str) -> int:
    return sum(1 for c in text if c.isalnum())


def _build_splitter(chunk_size: int, overlap: int) -> RecursiveCharacterTextSplitter:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0:
        raise ValueError("overlap must be non-negative")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=RecursiveCharacterTextSplitter.get_separators_for_language(
            Language.MARKDOWN
        ),
        keep_separator=False,
        strip_whitespace=True,
        length_function=alnum_length,
    )


def _extract_pages(markdown_text: str) -> list[tuple[int | None, str]]:
    matches = list(_PAGE_BLOCK_RE.finditer(markdown_text))
    if not matches:
        cleaned = _PAGE_TAG_RE.sub("", markdown_text).strip()
        return [(None, cleaned)] if cleaned else []

    pages: list[tuple[int | None, str]] = []
    for match in matches:
        page_num = int(match.group(1))
        page_text = match.group(2).strip()
        if page_text:
            pages.append((page_num, page_text))
    return pages


def _split_tables(page_text: str) -> tuple[list[str], list[str]]:
    return _TABLE_RE.split(page_text), _TABLE_RE.findall(page_text)


def _last_line(text: str) -> str | None:
    """Return the last non-empty line immediately preceding a table."""
    stripped = text.rstrip()
    if not stripped:
        return None

    last_newline = stripped.rfind("\n")
    line = stripped[last_newline + 1:].strip()
    return line or None


def _strip_last_line(text: str) -> str:
    """Return text with the last line removed."""
    stripped = text.rstrip()
    last_newline = stripped.rfind("\n")
    if last_newline == -1:
        return ""
    return stripped[:last_newline].rstrip()


def _merge_small_chunks(chunks: list[Chunk]) -> list[Chunk]:
    """Merge chunks shorter than _MIN_CHUNK_LENGTH into adjacent chunks.

    A small chunk is prepended to the next chunk. If no next chunk exists,
    it is appended to the previous one. Chunks with no eligible neighbour
    are left as-is.
    """
    merged: list[Chunk] = list(chunks)
    i = 0
    while i < len(merged):
        if len(merged[i].text) >= _MIN_CHUNK_LENGTH:
            i += 1
            continue

        if i + 1 < len(merged):
            next_chunk = merged[i + 1]
            merged[i + 1] = Chunk(
                text=f"{merged[i].text}\n\n{next_chunk.text}",
                chunk_type=next_chunk.chunk_type,
                page_num=next_chunk.page_num,
                section_title=next_chunk.section_title,
                index=next_chunk.index,
            )
            merged.pop(i)
        elif i > 0:
            prev = merged[i - 1]
            merged[i - 1] = Chunk(
                text=f"{prev.text}\n\n{merged[i].text}",
                chunk_type=prev.chunk_type,
                page_num=prev.page_num,
                section_title=prev.section_title,
                index=prev.index,
            )
            merged.pop(i)
        else:
            i += 1

    return merged


def chunk_markdown(
    text: str,
    *,
    chunk_size: int = _SEC_CHUNK_SIZE,
    overlap: int = _SEC_CHUNK_OVERLAP,
) -> list[Chunk]:
    """Chunk SEC markdown by page using langchain recursive splitting.

    Tables are extracted and emitted as standalone markdown table chunks.
    Remaining text content is split per-page with overlap.
    """
    splitter = _build_splitter(chunk_size=chunk_size, overlap=overlap)
    pages = _extract_pages(text)

    chunks: list[Chunk] = []
    current_section: str | None = None

    for page_num, page_text in pages:
        non_table_parts, tables = _split_tables(page_text)
        page_chunks: list[Chunk] = []

        for part_idx, part in enumerate(non_table_parts):
            clean_text = _PAGE_TAG_RE.sub("", part).strip()
            preceding_line = _last_line(clean_text)
            text_body = _strip_last_line(clean_text) if preceding_line else clean_text

            if clean_text:
                current_section = _extract_section(clean_text, current_section)

            if text_body:
                for split_text in splitter.split_text(text_body):
                    page_chunks.append(
                        Chunk(
                            text=split_text,
                            chunk_type="text",
                            page_num=page_num,
                            section_title=current_section,
                            index=0,
                        )
                    )

            if part_idx < len(tables):
                table_markdown = markdownify.markdownify(
                    tables[part_idx].strip(), heading_style="ATX"
                ).strip()
                if table_markdown:
                    table_text = (
                        f"{preceding_line}\n\n{table_markdown}"
                        if preceding_line
                        else table_markdown
                    )
                    page_chunks.append(
                        Chunk(
                            text=table_text,
                            chunk_type="table",
                            page_num=page_num,
                            section_title=current_section,
                            index=0,
                        )
                    )

        for chunk in _merge_small_chunks(page_chunks):
            chunk.index = len(chunks)
            chunks.append(chunk)

    return chunks


def chunk_transcript_rows(
    rows: list[tuple[str, str]],
    chunk_size: int = _EARNINGS_TRANSCRIPT_CHUNK_SIZE,
    overlap: int = _EARNINGS_TRANSCRIPT_OVERLAP,
) -> list[Chunk]:
    """Create transcript chunks using RecursiveCharacterTextSplitter overlap."""
    splitter = _build_splitter(chunk_size=chunk_size, overlap=overlap)

    chunks: list[Chunk] = []
    index = 0

    for speaker, text in rows:
        clean_speaker = speaker.strip()
        clean_text = text.strip()
        if not clean_text:
            continue

        for part in splitter.split_text(clean_text):
            chunk_text = (
                f"Speaker: {clean_speaker}\nText: {part}" if clean_speaker else part
            )
            chunks.append(
                Chunk(
                    text=chunk_text,
                    chunk_type="text",
                    page_num=None,
                    section_title=clean_speaker or None,
                    index=index,
                )
            )
            index += 1

    return chunks
