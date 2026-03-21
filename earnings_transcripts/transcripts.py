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
from loguru import logger
from playwright.sync_api import (
    Browser,
    BrowserContext,
    Page,
    Playwright,
    TimeoutError as PlaywrightTimeoutError,
    sync_playwright,
)

from settings import sec_settings


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

    @classmethod
    def from_file(cls, jsonl_path: str | Path) -> Transcript:
        path = Path(jsonl_path)
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            raise ValueError(f"Empty transcript file: {path}")
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            first = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
            data = json.loads(first)
        utterances = [SpeakerText(**row) for row in data["speaker_texts"]]
        return cls(
            ticker=data["ticker"],
            year=int(data["year"]),
            quarter_num=int(data["quarter_num"]),
            date=data["date"],
            speaker_texts=utterances,
        )


class TranscriptUrlDoesNotExistError(Exception):
    """Raised when the transcript URL returns 404 or cannot be reached via HTTP."""


def _make_url(ticker: str, year: int, quarter_num: int) -> str:
    curr_year = datetime.datetime.now().year
    assert year <= curr_year, f"{year=} is in the future for {ticker=} in {curr_year=}"
    assert quarter_num in [1, 2, 3, 4], f"{quarter_num=} is not a valid quarter number"
    return f"https://discountingcashflows.com/company/{ticker}/transcripts/{year}/{quarter_num}/"


def _probe_transcript_url(url: str, *, timeout_sec: float = 20.0) -> None:
    """HEAD/GET the URL with urllib so missing pages fail before starting Chrome."""
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


def _new_browser_context(playwright: Playwright) -> tuple[Browser, BrowserContext]:
    browser = playwright.chromium.launch(
        headless=True,
        args=_chromium_launch_args(),
    )
    context = browser.new_context(
        viewport={"width": 1920, "height": 1080},
        user_agent=(
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
    )
    return browser, context


def _wait_for_transcript_dom(page: Page) -> None:
    page.wait_for_selector("div.flex.flex-col.my-5", timeout=20_000)


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


def _write_transcript_jsonl(transcript: Transcript) -> Path:
    out_dir = (
        Path(sec_settings.earnings_transcripts_dir)
        / f"{transcript.ticker}-{transcript.year}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"q{transcript.quarter_num}.jsonl"
    with path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(dataclasses.asdict(transcript), ensure_ascii=False) + "\n")
    return path


def _load_transcript_with_page(
    page: Page,
    ticker: str,
    year: int,
    quarter_num: int,
) -> Transcript:
    url = _make_url(ticker, year, quarter_num)
    _probe_transcript_url(url)

    page.goto(url, wait_until="domcontentloaded")
    _wait_for_transcript_dom(page)

    soup = BeautifulSoup(page.content(), "html.parser")
    parsed_quarter, date_iso = _parse_transcript_metadata(soup, quarter_num)
    speaker_texts = _parse_speaker_texts(soup)

    transcript = Transcript(
        ticker=ticker,
        year=year,
        quarter_num=parsed_quarter,
        date=date_iso,
        speaker_texts=speaker_texts,
    )
    _write_transcript_jsonl(transcript)
    return transcript


def get_transcripts_for_year_sync(ticker: str, year: int) -> list[Transcript]:
    transcripts: list[Transcript] = []
    with sync_playwright() as playwright:
        browser, context = _new_browser_context(playwright)
        try:
            page = context.new_page()
            for quarter_num in (1, 2, 3, 4):
                try:
                    transcript = _load_transcript_with_page(
                        page, ticker, year, quarter_num
                    )
                    transcripts.append(transcript)
                except TranscriptUrlDoesNotExistError as exc:
                    logger.error(
                        f"Skipping transcript: URL missing or unreachable. "
                        f"ticker={ticker} year={year} quarter={quarter_num} error={exc}"
                    )
                except PlaywrightTimeoutError as exc:
                    logger.error(
                        f"Skipping transcript: timeout waiting for transcript DOM. "
                        f"ticker={ticker} year={year} quarter={quarter_num} "
                        f"error={str(exc).strip()}"
                    )
        finally:
            context.close()
            browser.close()

    return transcripts


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch earnings call transcripts (discountingcashflows.com).",
    )
    parser.add_argument(
        "ticker",
        nargs="?",
        default="AMZN",
        help="Stock ticker symbol (default: %(default)s)",
    )
    parser.add_argument(
        "year",
        nargs="?",
        type=int,
        default=datetime.datetime.now().year - 1,
        help="Fiscal year (default: current year)",
    )
    return parser.parse_args()


def _main(args: argparse.Namespace) -> None:
    logger.info("Fetching transcripts for ticker={} year={}", args.ticker, args.year)
    transcripts = get_transcripts_for_year_sync(args.ticker, args.year)
    for item in transcripts:
        logger.info(
            "Got Q{} date={} speaker_blocks={}",
            item.quarter_num,
            item.date or "(none)",
            len(item.speaker_texts),
        )
    logger.info("Done: {} quarter(s) loaded", len(transcripts))


if __name__ == "__main__":
    _main(_parse_args())
