from __future__ import annotations

import argparse
import asyncio
import datetime
import dataclasses
import json
import re
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from bs4 import BeautifulSoup
from bs4.element import Tag
from loguru import logger
from playwright.async_api import (
    Browser,
    BrowserContext,
    Page,
    Playwright,
    TimeoutError as PlaywrightTimeoutError,
    async_playwright,
)

from finance_data.common import processed_data_index
from finance_data.settings import sec_settings


@dataclasses.dataclass
class SpeakerText:
    speaker: str
    text: str


@dataclasses.dataclass
class Transcript:
    ticker: str
    year: int
    quarter_num: int
    date: str
    speaker_texts: list[SpeakerText]

    def to_markdown(self) -> str:
        """Format the transcript as markdown with parseable speaker tags."""
        quarter_label = f"Q{self.quarter_num}"
        date_display = self.date.strip() or "—"
        parts: list[str] = [
            f"# {self.ticker} · {self.year} · {quarter_label}",
            "",
            f"**Date:** {date_display}",
            "",
            "---",
            "",
            "## Transcript",
            "",
        ]
        for block in self.speaker_texts:
            speaker = block.speaker.strip() or "(Unknown speaker)"
            body = block.text.strip() or "_(empty)_"
            parts.extend(
                [
                    "<speaker-start>",
                    f"### {speaker}",
                    "",
                    body,
                    "<speaker-end>",
                    "",
                ]
            )
        return "\n".join(parts).rstrip() + "\n"

    @classmethod
    def from_markdown(cls, markdown_path: str | Path) -> "Transcript":
        path = Path(markdown_path)
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            raise ValueError(f"Empty transcript file: {path}")

        filename_match = re.fullmatch(
            r"Q(?P<quarter>[1-4])(?:_(?P<date>\d{4}-\d{2}-\d{2}))?\.md",
            path.name,
            flags=re.IGNORECASE,
        )
        if not filename_match:
            raise ValueError(
                f"Invalid transcript filename {path.name!r}; expected "
                f"'Q1_YYYY-MM-DD.md' (date optional)."
            )

        try:
            year = int(path.parent.name)
        except ValueError as exc:
            raise ValueError(
                f"Invalid transcript year directory {path.parent.name!r} for {path}."
            ) from exc

        ticker = path.parent.parent.name

        block_matches = re.findall(
            r"<speaker-start>\s*###\s*(.*?)\n(.*?)<speaker-end>",
            text,
            flags=re.DOTALL | re.IGNORECASE,
        )
        if not block_matches:
            raise ValueError(f"Missing transcript speaker blocks in markdown: {path}")

        utterances = [
            SpeakerText(speaker=speaker.strip(), text=body.strip())
            for speaker, body in block_matches
        ]

        return cls(
            ticker=ticker,
            year=year,
            quarter_num=int(filename_match.group("quarter")),
            date=filename_match.group("date") or "",
            speaker_texts=utterances,
        )


class TranscriptUrlDoesNotExistError(Exception):
    pass


class TranscriptSourceForbiddenError(Exception):
    """Raised when the primary transcript host returns HTTP 403 (use fallback)."""

    pass


def quarter_label_to_num(quarter: str) -> int:
    """Parse a quarter label (e.g. ``Q1``, ``q2``, ``Q 3``) into 1–4."""
    match = re.fullmatch(r"\s*Q\s*([1-4])\s*", quarter.strip(), flags=re.IGNORECASE)
    if not match:
        raise ValueError(
            f"Invalid quarter label {quarter!r}; expected Q1, Q2, Q3, or Q4"
        )
    return int(match.group(1))


def _assert_transcript_params(ticker: str, year: int, quarter_num: int) -> None:
    curr_year = datetime.datetime.now().year
    assert year <= curr_year, f"{year=} is in the future for {ticker=} in {curr_year=}"
    assert quarter_num in [1, 2, 3, 4], f"{quarter_num=} is not a valid quarter number"


def _make_url(ticker: str, year: int, quarter_num: int) -> str:
    _assert_transcript_params(ticker, year, quarter_num)
    return f"https://discountingcashflows.com/company/{ticker}/transcripts/{year}/{quarter_num}/"


def _make_earningscall_url(
    ticker: str, year: int, quarter_num: int, exchange: str
) -> str:
    _assert_transcript_params(ticker, year, quarter_num)
    slug = ticker.strip().lower()
    return f"https://earningscall.biz/e/{exchange}/s/{slug}/y/{year}/q/q{quarter_num}"


def _probe_transcript_url(url: str, *, timeout_sec: float = 20.0) -> None:
    request = Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            ),
        },
    )
    try:
        with urlopen(request, timeout=timeout_sec) as response:
            _ = response.status
    except HTTPError as exc:
        if exc.code == 404:
            raise TranscriptUrlDoesNotExistError(
                f"Transcript page does not exist (HTTP 404): {url}"
            ) from exc
        if exc.code == 403:
            logger.warning(
                f"DCF transcript probe forbidden url={url} status={exc.code}"
            )
            raise TranscriptSourceForbiddenError(
                f"Transcript page forbidden (HTTP 403): {url}"
            ) from exc
        raise
    except URLError as exc:
        raise TranscriptUrlDoesNotExistError(
            f"Transcript URL unreachable: {url} ({exc.reason!r})"
        ) from exc


def _chromium_launch_args() -> list[str]:
    return [
        "--disable-gpu",
        "--no-sandbox",
        "--disable-dev-shm-usage",
    ]


async def _new_browser_context(
    playwright: Playwright,
) -> tuple[Browser, BrowserContext]:
    browser = await playwright.chromium.launch(
        headless=True,
        args=_chromium_launch_args(),
    )
    context = await browser.new_context(
        viewport={"width": 1920, "height": 1080},
        user_agent=(
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
    )
    context.set_default_timeout(20_000)
    context.set_default_navigation_timeout(30_000)
    return browser, context


async def _wait_for_transcript_dom(page: Page) -> None:
    locator = page.locator("div.flex.flex-col.my-5").first
    await locator.wait_for(state="visible", timeout=20_000)


async def _wait_for_earningscall_dom(page: Page) -> None:
    locator = page.locator("div.content.without-focus").first
    await locator.wait_for(state="visible", timeout=20_000)


def _parse_transcript_metadata(
    soup: BeautifulSoup, default_quarter: int
) -> tuple[int, str]:
    metadata_container = soup.select_one(
        "div.flex.flex-col.place-content-center.sm\\:ms-2"
    )
    parsed_quarter = default_quarter
    date_iso = ""

    if not metadata_container:
        return parsed_quarter, date_iso

    spans = metadata_container.find_all("span")

    if len(spans) > 0:
        q_text = spans[0].get_text(strip=True)
        match = re.search(r"(?:Quarter|Q)\s*(\d+)", q_text, re.I)
        if match:
            parsed_quarter = int(match.group(1))

    if len(spans) > 1:
        date_text = spans[1].get_text(strip=True)
        try:
            parsed_date = datetime.datetime.strptime(date_text, "%B %d, %Y")
            date_iso = parsed_date.strftime("%Y-%m-%d")
        except ValueError:
            date_iso = ""

    return parsed_quarter, date_iso


def _parse_speaker_texts(soup: BeautifulSoup) -> list[SpeakerText]:
    blocks = soup.select("div.flex.flex-col.my-5")
    speaker_texts: list[SpeakerText] = []

    for block in blocks:
        speaker_tag = block.select_one("span")
        speaker = speaker_tag.get_text(strip=True) if speaker_tag else ""

        text_tag = block.select_one("div.p-4")
        text = text_tag.get_text(" ", strip=True) if text_tag else ""

        if speaker or text:
            speaker_texts.append(SpeakerText(speaker=speaker, text=text))

    return speaker_texts


def _parse_us_mmddyyyy_to_iso(text: str) -> str:
    """Parse a US MM/DD/YYYY string to YYYY-MM-DD, or return an empty string."""
    match = re.fullmatch(
        r"\s*(\d{1,2})/(\d{1,2})/(\d{4})\s*",
        text.strip(),
    )
    if not match:
        return ""
    month_s, day_s, year_s = match.groups()
    try:
        parsed = datetime.datetime(int(year_s), int(month_s), int(day_s))
    except ValueError:
        return ""
    return parsed.strftime("%Y-%m-%d")


def _parse_earningscall_date(soup: BeautifulSoup) -> str:
    for el in soup.select(".text-date"):
        iso = _parse_us_mmddyyyy_to_iso(el.get_text())
        if iso:
            logger.info(f"{iso=} from earningscall.biz element .text-date")
            return iso

    pattern = re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b")
    for node in soup.find_all(string=pattern):
        match = pattern.search(str(node))
        if not match:
            continue
        month_s, day_s, year_s = match.groups()
        try:
            parsed = datetime.datetime(int(year_s), int(month_s), int(day_s))
        except ValueError:
            continue
        iso_fb = parsed.strftime("%Y-%m-%d")
        logger.info(f"{iso_fb=} earningscall.biz date from regex fallback")
        return iso_fb
    return ""


def _earningscall_call_text_for_speaker(
    speaker_div: Tag,
    next_speaker: Tag | None,
) -> str:
    for el in speaker_div.find_all_next():
        if next_speaker is not None and el is next_speaker:
            break
        name = getattr(el, "name", None)
        if name != "p":
            continue
        classes = el.get("class") or []
        if "call-text" not in classes:
            continue
        return el.get_text(" ", strip=True)
    return ""


def _parse_earningscall_speaker_texts(content_root: Tag) -> list[SpeakerText]:
    """Parse speaker rows inside one ``div.content.without-focus`` section."""
    speakers = content_root.select("div.speaker")
    speaker_texts: list[SpeakerText] = []

    for i, speaker_div in enumerate(speakers):
        name_el = speaker_div.select_one(".speaker-name")
        desig_el = speaker_div.select_one(".designation")
        name = name_el.get_text(strip=True) if name_el else ""
        desig = desig_el.get_text(strip=True) if desig_el else ""
        if name and desig:
            speaker_label = f"{name} · {desig}"
        else:
            speaker_label = name or desig

        next_sp = speakers[i + 1] if i + 1 < len(speakers) else None
        text = _earningscall_call_text_for_speaker(speaker_div, next_sp)

        if speaker_label or text:
            speaker_texts.append(SpeakerText(speaker=speaker_label, text=text))

    return speaker_texts


def _parse_earningscall_speaker_texts_from_sections(
    sections: list[Tag],
) -> list[SpeakerText]:
    """Collect utterances from every ``div.content.without-focus`` block."""
    logger.info(f"{len(sections)=} for earningscall.biz transcript page")
    combined: list[SpeakerText] = []
    for section in sections:
        combined.extend(_parse_earningscall_speaker_texts(section))
    return combined


def save_transcript_markdown(transcript: Transcript) -> Path:
    out_dir = (
        Path(sec_settings.earnings_transcripts_dir)
        / transcript.ticker
        / str(transcript.year)
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    date_suffix = transcript.date.strip() or "unknown-date"
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date_suffix):
        date_suffix = "unknown-date"
    path = out_dir / f"Q{transcript.quarter_num}_{date_suffix}.md"
    with path.open("w", encoding="utf-8") as f:
        f.write(transcript.to_markdown())
    processed_data_index.mark_transcript(
        ticker=transcript.ticker,
        year=str(transcript.year),
        quarter=f"Q{transcript.quarter_num}",
    )
    return path


def convert_transcript_jsonl_to_markdown(
    jsonl_path: str | Path, *, delete_jsonl: bool = False
) -> Path:
    """Convert one transcript JSONL file into tagged markdown format."""
    source_path = Path(jsonl_path)
    text = source_path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Empty transcript file: {source_path}")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        first = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
        data = json.loads(first)

    transcript = Transcript(
        ticker=data["ticker"],
        year=int(data["year"]),
        quarter_num=int(data["quarter_num"]),
        date=data.get("date", ""),
        speaker_texts=[SpeakerText(**row) for row in data["speaker_texts"]],
    )
    out_path = source_path.with_suffix(".md")
    out_path.write_text(transcript.to_markdown(), encoding="utf-8")
    if delete_jsonl:
        source_path.unlink(missing_ok=True)
    return out_path


async def _load_transcript_discounting_cashflows_page(
    page: Page,
    dcf_url: str,
    ticker: str,
    year: int,
    quarter_num: int,
) -> Transcript | None:
    """Load DCF transcript in an open page. Returns None when navigation returns HTTP 403."""
    response = await page.goto(dcf_url, wait_until="domcontentloaded", timeout=30_000)
    if response is not None and response.status == 403:
        logger.warning(
            f"DCF transcript navigation forbidden url={dcf_url} status={response.status}"
        )
        return None

    await _wait_for_transcript_dom(page)

    soup = BeautifulSoup(await page.content(), "html.parser")
    parsed_quarter, date_iso = _parse_transcript_metadata(soup, quarter_num)
    speaker_texts = _parse_speaker_texts(soup)

    if not speaker_texts:
        raise ValueError(
            f"No speaker blocks parsed for {ticker=} {year=} {quarter_num=}"
        )

    return Transcript(
        ticker=ticker,
        year=year,
        quarter_num=parsed_quarter,
        date=date_iso,
        speaker_texts=speaker_texts,
    )


async def _load_transcript_earningscall(
    context: BrowserContext,
    ticker: str,
    year: int,
    quarter_num: int,
) -> Transcript:
    last_error: str | None = None
    for exchange in ("nasdaq", "nyse"):
        url = _make_earningscall_url(ticker, year, quarter_num, exchange)
        logger.info(
            f"Trying earningscall.biz transcript "
            f"ticker={ticker} year={year} quarter={quarter_num} "
            f"exchange={exchange} url={url}"
        )
        page = await context.new_page()
        try:
            response = await page.goto(
                url, wait_until="domcontentloaded", timeout=30_000
            )
            if response is not None and response.status in (403, 404):
                logger.warning(
                    f"Earningscall page not available url={url} status={response.status}"
                )
                last_error = f"HTTP {response.status}"
                continue
            try:
                await _wait_for_earningscall_dom(page)
            except PlaywrightTimeoutError as exc:
                logger.warning(
                    f"Earningscall DOM timeout exchange={exchange} url={url} error={exc}"
                )
                last_error = str(exc).strip()
                continue

            soup = BeautifulSoup(await page.content(), "html.parser")
            content_sections = soup.select("div.content.without-focus")
            if not content_sections:
                logger.warning(
                    f"Earningscall missing content sections exchange={exchange} url={url}"
                )
                last_error = "missing content root"
                continue

            date_iso = _parse_earningscall_date(soup)
            speaker_texts = _parse_earningscall_speaker_texts_from_sections(
                content_sections
            )
            if not speaker_texts:
                logger.warning(
                    f"No speaker blocks from earningscall exchange={exchange} url={url}"
                )
                last_error = "no speaker blocks"
                continue

            transcript = Transcript(
                ticker=ticker,
                year=year,
                quarter_num=quarter_num,
                date=date_iso,
                speaker_texts=speaker_texts,
            )
            save_transcript_markdown(transcript)
            return transcript
        finally:
            await page.close()

    raise ValueError(
        f"Earningscall.biz: no transcript for {ticker=} {year=} {quarter_num=}; "
        f"last_error={last_error!r}"
    )


async def _load_transcript_with_new_page(
    context: BrowserContext,
    ticker: str,
    year: int,
    quarter_num: int,
) -> Transcript:
    dcf_url = _make_url(ticker, year, quarter_num)

    try:
        await asyncio.to_thread(_probe_transcript_url, dcf_url)
    except TranscriptSourceForbiddenError as exc:
        logger.warning(
            f"DCF forbidden; using earningscall.biz fallback "
            f"ticker={ticker} year={year} quarter={quarter_num} detail={exc}"
        )
        return await _load_transcript_earningscall(context, ticker, year, quarter_num)

    page = await context.new_page()
    try:
        transcript = await _load_transcript_discounting_cashflows_page(
            page, dcf_url, ticker, year, quarter_num
        )
        if transcript is not None:
            save_transcript_markdown(transcript)
            return transcript
    finally:
        await page.close()

    return await _load_transcript_earningscall(context, ticker, year, quarter_num)


async def _fetch_one_quarter(
    context: BrowserContext,
    semaphore: asyncio.Semaphore,
    ticker: str,
    year: int,
    quarter_num: int,
    retries: int = 2,
) -> Transcript | None:
    async with semaphore:
        for attempt in range(1, retries + 2):
            try:
                return await _load_transcript_with_new_page(
                    context=context,
                    ticker=ticker,
                    year=year,
                    quarter_num=quarter_num,
                )
            except TranscriptUrlDoesNotExistError as exc:
                logger.error(
                    f"Skipping transcript: URL missing or unreachable. "
                    f"ticker={ticker} year={year} quarter={quarter_num} error={exc}"
                )
                return None
            except (PlaywrightTimeoutError, ValueError) as exc:
                if attempt > retries:
                    logger.error(
                        f"Skipping transcript after retries. "
                        f"ticker={ticker} year={year} quarter={quarter_num} "
                        f"attempt={attempt} error={str(exc).strip()}"
                    )
                    return None
                backoff_sec = float(attempt)
                logger.warning(
                    f"Retrying transcript. "
                    f"ticker={ticker} year={year} quarter={quarter_num} "
                    f"attempt={attempt} backoff_sec={backoff_sec} "
                    f"error={str(exc).strip()}"
                )
                await asyncio.sleep(backoff_sec)
            except Exception as exc:
                logger.exception(
                    f"Unexpected error. "
                    f"ticker={ticker} year={year} quarter={quarter_num} error={exc}"
                )
                return None
    return None


async def get_transcript_from_dcf_async(
    ticker: str,
    year: int,
    quarter_num: int,
) -> Transcript | None:
    """Fetch transcript data from discountingcashflows.com for one quarter."""
    dcf_url = _make_url(ticker, year, quarter_num)
    logger.info(
        "Pulling DCF transcript " f"{ticker=} {year=} {quarter_num=} {dcf_url=}"
    )

    try:
        await asyncio.to_thread(_probe_transcript_url, dcf_url)
    except (TranscriptUrlDoesNotExistError, TranscriptSourceForbiddenError) as exc:
        logger.warning(
            "DCF transcript probe failed " f"{ticker=} {year=} {quarter_num=} {exc=}"
        )
        return None

    async with async_playwright() as playwright:
        browser, context = await _new_browser_context(playwright)
        try:
            page = await context.new_page()
            try:
                transcript = await _load_transcript_discounting_cashflows_page(
                    page=page,
                    dcf_url=dcf_url,
                    ticker=ticker,
                    year=year,
                    quarter_num=quarter_num,
                )
            finally:
                await page.close()
        except (PlaywrightTimeoutError, ValueError) as exc:
            logger.warning(
                "DCF transcript load failed " f"{ticker=} {year=} {quarter_num=} {exc=}"
            )
            return None
        finally:
            await context.close()
            await browser.close()

    if transcript is None:
        return None
    save_transcript_markdown(transcript)
    return transcript


async def get_transcript_from_earnings_biz_async(
    ticker: str,
    year: int,
    quarter_num: int,
) -> Transcript | None:
    """Fetch transcript data from earningscall.biz for one quarter."""
    logger.info(
        "Pulling earningscall.biz transcript " f"{ticker=} {year=} {quarter_num=}"
    )
    async with async_playwright() as playwright:
        browser, context = await _new_browser_context(playwright)
        try:
            return await _load_transcript_earningscall(
                context=context,
                ticker=ticker,
                year=year,
                quarter_num=quarter_num,
            )
        except (PlaywrightTimeoutError, ValueError) as exc:
            logger.warning(
                "earningscall.biz transcript load failed "
                f"{ticker=} {year=} {quarter_num=} {exc=}"
            )
            return None
        finally:
            await context.close()
            await browser.close()


def _find_transcript_path(ticker: str, year: int, quarter_num: int) -> Path | None:
    """Return the first matching transcript markdown file on disk, or None."""
    transcript_dir = (
        Path(sec_settings.earnings_transcripts_dir)
        / ticker
        / str(year)
    )
    candidates = sorted(transcript_dir.glob(f"Q{quarter_num}*.md"))
    return candidates[0] if candidates else None


async def get_transcript_for_quarter_async(
    ticker: str,
    year: int,
    quarter: str,
) -> Transcript | None:
    """Fetch a single earnings-call transcript for one fiscal quarter.

    ``quarter`` must be a label such as ``Q1``, ``Q2``, ``Q3``, or ``Q4``
    (case-insensitive; optional spaces after ``Q``).
    """
    quarter_num = quarter_label_to_num(quarter)
    quarter_label = f"Q{quarter_num}"

    if processed_data_index.has_transcript(ticker, str(year), quarter_label):
        path = _find_transcript_path(ticker, year, quarter_num)
        if path is not None:
            logger.info(
                f"Cache hit. Loading transcript from disk. "
                f"{ticker=} {year=} {quarter_label=} {path=}"
            )
            return Transcript.from_markdown(path)
        logger.warning(
            f"Cache hit but transcript file missing on disk. "
            f"{ticker=} {year=} {quarter_label=}"
        )

    async with async_playwright() as playwright:
        browser, context = await _new_browser_context(playwright)
        try:
            semaphore = asyncio.Semaphore(1)
            return await _fetch_one_quarter(
                context, semaphore, ticker, year, quarter_num
            )
        finally:
            await context.close()
            await browser.close()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch earnings call transcripts (discountingcashflows.com; "
            "earningscall.biz on HTTP 403)."
        ),
    )
    parser.add_argument(
        "ticker",
        nargs="?",
        default="CSCO",
        help="Stock ticker symbol (default: %(default)s)",
    )
    parser.add_argument(
        "year",
        nargs="?",
        type=int,
        # default=datetime.datetime.now().year,
        default=2023,
        help="Fiscal year (default: current year - 1)",
    )
    parser.add_argument(
        "--quarter",
        type=str,
        default="Q1",
        metavar="QX",
        help="Quarter label: Q1, Q2, Q3, or Q4 (case-insensitive). "
        "If omitted, fetches Q1–Q4 sequentially.",
    )
    return parser.parse_args()


async def _fetch_single_quarter(
    ticker: str, year: int, quarter: str
) -> Transcript | None:
    """Fetch transcript for a single quarter, or None if unavailable."""
    return await get_transcript_for_quarter_async(ticker, year, quarter)


def _main(args: argparse.Namespace) -> None:
    logger.info(
        "Fetching transcript for ticker={} year={} quarter={}".format(
            args.ticker, args.year, args.quarter
        )
    )
    if args.quarter is None:
        raise SystemExit(
            "Please provide a quarter using --quarter (e.g., Q1, Q2, Q3, Q4)"
        )
    try:
        quarter_label_to_num(args.quarter)
    except ValueError as exc:
        raise SystemExit(f"error: {exc}") from exc

    transcript = asyncio.run(
        _fetch_single_quarter(args.ticker, args.year, args.quarter)
    )
    if transcript is None:
        logger.warning("No transcript found for the specified period")
        return
    logger.info(
        "Got Q{} date={} speaker_blocks={}",
        transcript.quarter_num,
        transcript.date or "(none)",
        len(transcript.speaker_texts),
    )
    logger.info("Done: 1 quarter loaded")


if __name__ == "__main__":
    _main(_parse_args())
