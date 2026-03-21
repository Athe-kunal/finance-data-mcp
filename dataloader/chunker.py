import re
from dataclasses import dataclass
import markdownify

_PAGE_TAG_RE = re.compile(r"</?PAGE-NUM-(\d+)>")
_PAGE_SPLIT_RE = re.compile(r"(</?PAGE-NUM-(\d+)>)")
_TABLE_RE = re.compile(r"<table[\s\S]*?</table>", re.IGNORECASE)
_SECTION_TITLE_RE = re.compile(
    r"^(Item\s+\d+[A-C]?\..*|Part\s+[IV]+.*)",
    re.MULTILINE | re.IGNORECASE,
)
_MIN_CHUNK_CHARS = 1024
_MIN_PAGE_BREAK_CHARS = 512
_EARNINGS_TRANSCRIPT_CHUNK_SIZE = 1024
_EARNINGS_TRANSCRIPT_OVERLAP = 256


@dataclass
class Chunk:
    """A single logical unit extracted from a markdown document."""

    text: str
    chunk_type: str  # "table" | "text"
    page_num: int | None
    section_title: str | None
    index: int

    def __repr__(self) -> str:
        preview = self.text[:80].replace("\n", " ")
        return (
            f"Chunk(index={self.index}, type={self.chunk_type!r}, "
            f"page={self.page_num}, section={self.section_title!r}, "
            f"text={preview!r}...) "
            f"len(text)={len(self.text)}"
        )


def _visible_len(text: str) -> int:
    return len(text.replace(" ", ""))


def _split_tables(text: str) -> tuple[list[str], list[str]]:
    return _TABLE_RE.split(text), _TABLE_RE.findall(text)


def _split_paragraphs(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"\n{2,}", text) if part.strip()]


def _remove_page_tags(para: str) -> str:
    return _PAGE_TAG_RE.sub("", para).strip()


def _extract_section(para: str, current_section: str | None) -> str | None:
    title_match = _SECTION_TITLE_RE.search(para)
    if title_match:
        return title_match.group(0).strip()
    return current_section


def _split_on_page_tags(
    text: str, current_page: int | None
) -> list[tuple[str, int | None]]:
    """Split a text block into (content, page_num) segments at every <PAGE-NUM-X> tag.

    Each segment inherits the page number of the most recently seen tag.
    Content before the first tag inherits current_page.
    """
    # re.split with a capturing group keeps the delimiters in the result list.
    # Result pattern: [pre, full_tag, num, inter, full_tag, num, inter, ...]
    raw_parts = _PAGE_SPLIT_RE.split(text)
    segments: list[tuple[str, int | None]] = []
    page = current_page
    i = 0
    while i < len(raw_parts):
        part = raw_parts[i]
        # Every tag produces two extra entries: the full tag and the digit group.
        # Detect by checking if the part matches a page tag.
        if _PAGE_SPLIT_RE.fullmatch(part):
            # Next entry is the captured digit group
            page = int(raw_parts[i + 1])
            i += 2  # skip full_tag + digit entries
        else:
            clean = _PAGE_TAG_RE.sub("", part).strip()
            if clean:
                segments.append((clean, page))
            i += 1
    return segments


def _make_chunk(
    *,
    text: str,
    chunk_type: str,
    page_num: int | None,
    section_title: str | None,
    index: int,
) -> Chunk:
    return Chunk(
        text=text,
        chunk_type=chunk_type,
        page_num=page_num,
        section_title=section_title,
        index=index,
    )


def chunk_markdown(text: str) -> list[Chunk]:
    """Split markdown text into Chunk objects.

    Strategy:
    1. Track current page via <PAGE-NUM-X> tags.
    2. HTML <table>...</table> blocks are kept atomic as table chunks.
    3. Non-table text is split on blank lines, then each paragraph is further
       split on page tags so that page boundaries can flush the buffer.
    4. A page boundary only flushes the buffer when it has already accumulated
       at least _MIN_PAGE_BREAK_CHARS; otherwise the new page's content is
       appended to the same chunk (preventing tiny orphan chunks on short pages).
    5. Section titles (Item 1., Part II, ...) are attached as metadata.
    """
    chunks: list[Chunk] = []
    current_page: int | None = None
    current_section: str | None = None
    index = 0

    parts, tables = _split_tables(text)

    buffer_parts: list[str] = []
    buffer_page: int | None = None
    buffer_section: str | None = None

    def flush_buffer() -> None:
        nonlocal index, buffer_parts, buffer_page, buffer_section
        if not buffer_parts:
            return
        chunks.append(
            _make_chunk(
                text="\n\n".join(buffer_parts),
                chunk_type="text",
                page_num=buffer_page,
                section_title=buffer_section,
                index=index,
            )
        )
        index += 1
        buffer_parts = []
        buffer_page = None
        buffer_section = None

    def append_paragraph(
        para_text: str,
        para_page: int | None,
        para_section: str | None,
    ) -> None:
        nonlocal buffer_page, buffer_section

        if not buffer_parts:
            buffer_page = para_page
            buffer_section = para_section
        elif buffer_page != para_page:
            # Only flush at a page boundary when the buffer is already large
            # enough; otherwise just absorb the new page into the same chunk.
            if len("\n\n".join(buffer_parts)) >= _MIN_PAGE_BREAK_CHARS:
                flush_buffer()
            buffer_page = para_page
            buffer_section = para_section

        buffer_parts.append(para_text)
        buffer_section = para_section

        if len("\n\n".join(buffer_parts)) >= _MIN_CHUNK_CHARS:
            flush_buffer()

    for part_idx, non_table_text in enumerate(parts):
        paragraphs = _split_paragraphs(non_table_text)

        for raw_para in paragraphs:
            # Split the paragraph on page tags so each sub-segment carries
            # the correct page number and a page boundary always flushes.
            sub_segments = _split_on_page_tags(raw_para, current_page)

            for seg_text, seg_page in sub_segments:
                para_section = _extract_section(seg_text, current_section)
                append_paragraph(
                    para_text=seg_text,
                    para_page=seg_page,
                    para_section=para_section,
                )
                current_page = seg_page
                current_section = para_section

            # If the paragraph had no content segments, still advance the page
            # by scanning for the last tag in the raw paragraph.
            if not sub_segments:
                for m in _PAGE_TAG_RE.finditer(raw_para):
                    current_page = int(m.group(1))

        if part_idx < len(tables):
            flush_buffer()
            chunks.append(
                _make_chunk(
                    text=markdownify.markdownify(
                        tables[part_idx].strip(), heading_style="ATX"
                    ),
                    chunk_type="table",
                    page_num=current_page,
                    section_title=current_section,
                    index=index,
                )
            )
            index += 1

    flush_buffer()
    return chunks


def chunk_text_with_overlap(
    text: str,
    chunk_size: int = _EARNINGS_TRANSCRIPT_CHUNK_SIZE,
    overlap: int = _EARNINGS_TRANSCRIPT_OVERLAP,
) -> list[str]:
    """Split text into overlapping character windows."""
    cleaned = text.strip()
    if not cleaned:
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0:
        raise ValueError("overlap must be non-negative")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    windows: list[str] = []
    step = chunk_size - overlap
    for start in range(0, len(cleaned), step):
        window = cleaned[start : start + chunk_size]
        if not window:
            break
        windows.append(window)
        if start + chunk_size >= len(cleaned):
            break
    return windows


def chunk_transcript_rows(
    rows: list[tuple[str, str]],
    chunk_size: int = 2048,
    overlap: int = 256,
) -> list[Chunk]:
    """Create transcript chunks with speaker-prefixed overlapping text."""
    chunks: list[Chunk] = []
    index = 0

    for speaker, text in rows:
        clean_speaker = speaker.strip()
        clean_text = text.strip()
        if not clean_text:
            continue

        for part in chunk_text_with_overlap(
            clean_text, chunk_size=chunk_size, overlap=overlap
        ):
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
