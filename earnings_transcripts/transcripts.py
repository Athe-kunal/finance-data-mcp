from __future__ import annotations
import argparse
import asyncio
import concurrent.futures
import datetime
import dataclasses
import functools
import json
import re
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from bs4 import BeautifulSoup
from loguru import logger
from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

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


def _headless_chrome_options() -> Options:
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1920,1080")
    return options


def _new_chrome_driver() -> webdriver.Chrome:
    return webdriver.Chrome(options=_headless_chrome_options())


def _wait_for_transcript_dom(driver: webdriver.Chrome) -> None:
    WebDriverWait(driver, 20).until(
        EC.presence_of_all_elements_located((By.CSS_SELECTOR, "div.flex.flex-col.my-5"))
    )


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


def _get_transcript_sync(ticker: str, year: int, quarter_num: int) -> Transcript:
    url = _make_url(ticker, year, quarter_num)
    _probe_transcript_url(url)

    driver = _new_chrome_driver()
    try:
        driver.get(url)
        _wait_for_transcript_dom(driver)

        soup = BeautifulSoup(driver.page_source, "html.parser")
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
    finally:
        driver.quit()


async def get_transcript(
    ticker: str,
    year: int,
    quarter_num: int,
    *,
    executor: concurrent.futures.Executor | None = None,
) -> Transcript:
    """Fetches one transcript; blocking I/O runs on ``executor`` (or the loop default)."""
    loop = asyncio.get_running_loop()
    fn = functools.partial(_get_transcript_sync, ticker, year, quarter_num)
    return await loop.run_in_executor(executor, fn)


async def get_transcripts_for_year(ticker: str, year: int) -> list[Transcript]:
    """Fetches transcripts for Q1–Q4 in parallel via a :class:`ThreadPoolExecutor`.

    Quarters that are missing over HTTP (urllib) or time out in WebDriverWait are
    skipped after logging. Other errors propagate.
    """
    loop = asyncio.get_running_loop()
    quarters = (1, 2, 3, 4)
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(quarters)) as pool:
        tasks = [
            loop.run_in_executor(
                pool,
                functools.partial(_get_transcript_sync, ticker, year, quarter_num),
            )
            for quarter_num in quarters
        ]
        outcomes = await asyncio.gather(*tasks, return_exceptions=True)

    transcripts: list[Transcript] = []
    for quarter_num, outcome in zip(quarters, outcomes, strict=True):
        if isinstance(outcome, Transcript):
            transcripts.append(outcome)
            continue
        if isinstance(outcome, AssertionError):
            raise outcome
        if isinstance(outcome, TranscriptUrlDoesNotExistError):
            url = _make_url(ticker, year, quarter_num)
            logger.error(
                "Skipping transcript: URL missing or unreachable (urllib probe). "
                "{} | ticker={} year={} quarter={} url={}",
                outcome,
                ticker,
                year,
                quarter_num,
                url,
            )
            continue
        if isinstance(outcome, TimeoutException):
            url = _make_url(ticker, year, quarter_num)
            logger.error(
                "Skipping transcript: WebDriverWait timed out waiting for "
                "'div.flex.flex-col.my-5' (missing transcript or wrong page). "
                "TimeoutException: {} | ticker={} year={} quarter={} url={}",
                str(outcome).strip(),
                ticker,
                year,
                quarter_num,
                url,
            )
            continue
        raise outcome
    return transcripts


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch earnings call transcripts (discountingcashflows.com).",
    )
    parser.add_argument(
        "ticker",
        nargs="?",
        default="AAPL",
        help="Stock ticker symbol (default: %(default)s)",
    )
    parser.add_argument(
        "year",
        nargs="?",
        type=int,
        default=datetime.datetime.now().year,
        help="Fiscal year (default: current year)",
    )
    return parser.parse_args()


async def _main_async(args: argparse.Namespace) -> None:
    logger.info("Fetching transcripts for ticker={} year={}", args.ticker, args.year)
    transcripts = await get_transcripts_for_year(args.ticker, args.year)
    for item in transcripts:
        logger.info(
            "Got Q{} date={} speaker_blocks={}",
            item.quarter_num,
            item.date or "(none)",
            len(item.speaker_texts),
        )
    logger.info("Done: {} quarter(s) loaded", len(transcripts))


def main() -> None:
    asyncio.run(_main_async(_parse_args()))


if __name__ == "__main__":
    main()
