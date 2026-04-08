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
_MARKDOWN_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+(.+?)\s*$", re.MULTILINE)
_OPERATOR_SPEAKER_RE = re.compile(r"\boperator\b", re.IGNORECASE)
_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(])")

_SEC_CHUNK_SIZE = 1024
_SEC_CHUNK_OVERLAP = 256
_EARNINGS_TRANSCRIPT_CHUNK_SIZE = 1024
_EARNINGS_TRANSCRIPT_OVERLAP = 256
_MIN_CHUNK_LENGTH = 200


@dataclass
class Chunk:
    """A single logical unit extracted from a markdown document."""

    text: str
    chunk_type: str  # "table" | "text"
    page_num: int | None
    section_title: str | None
    index: int


def _extract_section(text: str, current_section: str | None) -> str | None:
    heading_match = _MARKDOWN_HEADING_RE.search(text)
    if heading_match:
        heading_text = heading_match.group(1).strip()
        if heading_text:
            return heading_text

    title_match = _SECTION_TITLE_RE.search(text)
    if title_match:
        return title_match.group(0).strip()
    return current_section


def _is_operator_speaker(speaker: str) -> bool:
    return bool(_OPERATOR_SPEAKER_RE.search(speaker))


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
    line = stripped[last_newline + 1 :].strip()
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
        if alnum_length(merged[i].text) >= _MIN_CHUNK_LENGTH:
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


def _split_into_sentences(text: str) -> list[str]:
    """Split text into sentence-like segments while preserving punctuation."""
    compact = re.sub(r"\s+", " ", text).strip()
    if not compact:
        return []

    parts = _SENTENCE_BOUNDARY_RE.split(compact)
    return [part.strip() for part in parts if part.strip()]


def _trim_sentence_list_by_overlap(
    sentences: list[str], overlap: int
) -> list[str]:
    """Keep trailing sentences whose cumulative length does not exceed overlap."""
    if overlap <= 0:
        return []

    kept: list[str] = []
    total = 0
    for sentence in reversed(sentences):
        sentence_length = alnum_length(sentence)
        if kept and total + sentence_length > overlap:
            break
        kept.insert(0, sentence)
        total += sentence_length
        if total >= overlap:
            break
    return kept


def _sentence_aware_split(
    *,
    text: str,
    splitter: RecursiveCharacterTextSplitter,
    chunk_size: int,
    overlap: int,
) -> list[str]:
    """Build chunks with sentence boundaries when possible."""
    raw_sentences = _split_into_sentences(text)
    if not raw_sentences:
        return []

    normalized_sentences: list[str] = []
    for sentence in raw_sentences:
        if alnum_length(sentence) <= chunk_size:
            normalized_sentences.append(sentence)
            continue
        normalized_sentences.extend(splitter.split_text(sentence))

    chunks: list[str] = []
    current_sentences: list[str] = []
    current_length = 0

    for sentence in normalized_sentences:
        sentence_length = alnum_length(sentence)
        if current_sentences and current_length + sentence_length > chunk_size:
            chunks.append(" ".join(current_sentences).strip())
            overlap_sentences = _trim_sentence_list_by_overlap(
                current_sentences, overlap
            )
            current_sentences = list(overlap_sentences)
            current_length = sum(alnum_length(item) for item in current_sentences)

        current_sentences.append(sentence)
        current_length += sentence_length

    if current_sentences:
        chunks.append(" ".join(current_sentences).strip())

    return [chunk for chunk in chunks if chunk]


def _split_non_table_text(
    *,
    splitter: RecursiveCharacterTextSplitter,
    text_body: str,
    chunk_size: int,
    overlap: int,
    page_num: int | None,
    section_title: str | None,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    sentence_chunks = _sentence_aware_split(
        text=text_body,
        splitter=splitter,
        chunk_size=chunk_size,
        overlap=overlap,
    )
    for split_text in sentence_chunks:
        chunks.append(
            Chunk(
                text=split_text,
                chunk_type="text",
                page_num=page_num,
                section_title=section_title,
                index=0,
            )
        )
    return chunks


def _build_table_chunk(
    *,
    table_html: str,
    preceding_line: str | None,
    page_num: int | None,
    section_title: str | None,
) -> Chunk | None:
    table_markdown = markdownify.markdownify(
        table_html.strip(), heading_style="ATX"
    ).strip()
    if not table_markdown:
        return None

    table_text = (
        f"{preceding_line}\n\n{table_markdown}" if preceding_line else table_markdown
    )
    return Chunk(
        text=table_text,
        chunk_type="table",
        page_num=page_num,
        section_title=section_title,
        index=0,
    )


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
                page_chunks.extend(
                    _split_non_table_text(
                        splitter=splitter,
                        text_body=text_body,
                        chunk_size=chunk_size,
                        overlap=overlap,
                        page_num=page_num,
                        section_title=current_section,
                    )
                )

            if part_idx < len(tables):
                table_chunk = _build_table_chunk(
                    table_html=tables[part_idx],
                    preceding_line=preceding_line,
                    page_num=page_num,
                    section_title=current_section,
                )
                if table_chunk is not None:
                    page_chunks.append(table_chunk)

        for chunk in _merge_small_chunks(page_chunks):
            chunk.index = len(chunks)
            chunks.append(chunk)

    return chunks


def chunk_transcript_rows(
    rows: list[tuple[str, str]],
    chunk_size: int = _EARNINGS_TRANSCRIPT_CHUNK_SIZE,
    overlap: int = _EARNINGS_TRANSCRIPT_OVERLAP,
) -> list[Chunk]:
    """Create transcript chunks using RecursiveCharacterTextSplitter overlap.

    Operator turns (speaker name matching 'operator') are not emitted as their
    own chunks. Instead, their formatted text is buffered and prepended to the
    first chunk of the following non-operator speaker, providing context for
    what the operator said before the next participant speaks.
    """
    splitter = _build_splitter(chunk_size=chunk_size, overlap=overlap)

    chunks: list[Chunk] = []
    index = 0
    pending_operator_text: str = ""

    for speaker, text in rows:
        clean_speaker = speaker.strip()
        clean_text = text.strip()
        if not clean_text:
            continue

        if _is_operator_speaker(clean_speaker):
            operator_line = (
                f"Speaker: {clean_speaker}\nText: {clean_text}"
                if clean_speaker
                else clean_text
            )
            pending_operator_text = (
                f"{pending_operator_text}\n\n{operator_line}"
                if pending_operator_text
                else operator_line
            )
            continue

        for part_idx, part in enumerate(splitter.split_text(clean_text)):
            chunk_text = (
                f"Speaker: {clean_speaker}\nText: {part}" if clean_speaker else part
            )
            if pending_operator_text and part_idx == 0:
                chunk_text = f"{pending_operator_text}\n\n{chunk_text}"
                pending_operator_text = ""

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

    if pending_operator_text:
        chunks.append(
            Chunk(
                text=pending_operator_text,
                chunk_type="text",
                page_num=None,
                section_title=None,
                index=index,
            )
        )

    return chunks
