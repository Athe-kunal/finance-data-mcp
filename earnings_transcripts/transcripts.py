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
from playwright.async_api import (
    Browser,
    BrowserContext,
    Page,
    Playwright,
    TimeoutError as PlaywrightTimeoutError,
    async_playwright,
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
    def from_file(cls, jsonl_path: str | Path) -> "Transcript":
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
    pass


def _make_url(ticker: str, year: int, quarter_num: int) -> str:
    curr_year = datetime.datetime.now().year
    assert year <= curr_year, f"{year=} is in the future for {ticker=} in {curr_year=}"
    assert quarter_num in [1, 2, 3, 4], f"{quarter_num=} is not a valid quarter number"
    return f"https://discountingcashflows.com/company/{ticker}/transcripts/{year}/{quarter_num}/"


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
        / transcript.ticker
        / str(transcript.year)
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"Q{transcript.quarter_num}.jsonl"
    with path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(dataclasses.asdict(transcript), ensure_ascii=False) + "\n")
    return path


async def _load_transcript_with_new_page(
    context: BrowserContext,
    ticker: str,
    year: int,
    quarter_num: int,
) -> Transcript:
    url = _make_url(ticker, year, quarter_num)

    # urllib is blocking, so push it off the event loop.
    await asyncio.to_thread(_probe_transcript_url, url)

    page = await context.new_page()
    try:
        await page.goto(url, wait_until="domcontentloaded", timeout=30_000)
        await _wait_for_transcript_dom(page)

        soup = BeautifulSoup(await page.content(), "html.parser")
        parsed_quarter, date_iso = _parse_transcript_metadata(soup, quarter_num)
        speaker_texts = _parse_speaker_texts(soup)

        if not speaker_texts:
            raise ValueError(
                f"No speaker blocks parsed for {ticker=} {year=} {quarter_num=}"
            )

        transcript = Transcript(
            ticker=ticker,
            year=year,
            quarter_num=parsed_quarter,
            date=date_iso,
            speaker_texts=speaker_texts,
        )
        _write_transcript_jsonl(transcript)
        return transcript
    finally:
        await page.close()


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


async def get_transcripts_for_year_async(
    ticker: str,
    year: int,
    max_concurrency: int = 3,
) -> list[Transcript]:
    async with async_playwright() as playwright:
        browser, context = await _new_browser_context(playwright)
        try:
            semaphore = asyncio.Semaphore(max_concurrency)
            tasks = [
                _fetch_one_quarter(context, semaphore, ticker, year, quarter_num)
                for quarter_num in (1, 2, 3, 4)
            ]
            results = await asyncio.gather(*tasks)
        finally:
            await context.close()
            await browser.close()

    transcripts = [item for item in results if item is not None]
    transcripts.sort(key=lambda item: item.quarter_num)
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
        help="Fiscal year (default: current year - 1)",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=4,
        help="Max number of concurrent quarter fetches",
    )
    return parser.parse_args()


def _main(args: argparse.Namespace) -> None:
    logger.info("Fetching transcripts for ticker={} year={}", args.ticker, args.year)
    transcripts = asyncio.run(
        get_transcripts_for_year_async(
            args.ticker,
            args.year,
            max_concurrency=args.max_concurrency,
        )
    )
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
