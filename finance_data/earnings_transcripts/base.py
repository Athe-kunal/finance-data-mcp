"""Base abstractions and vendor pullers for earnings transcript retrieval."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, TypeAlias

from loguru import logger

if TYPE_CHECKING:
    from finance_data.earnings_transcripts.transcripts import Transcript


TranscriptPullFn: TypeAlias = Callable[[str, int, int], Awaitable["Transcript | None"]]


class TranscriptDataPuller(ABC):
    """Abstract interface for loading one transcript period."""

    @abstractmethod
    async def pull_data_for_period(
        self,
        ticker: str,
        year: int,
        quarter_num: int,
    ) -> Transcript | None:
        """Pull transcript data for one ticker, fiscal year, and quarter."""


class DCFDataPull(TranscriptDataPuller):
    """Loads transcript data from the Discounting Cash Flows vendor."""

    def __init__(self, dcf_pull_fn: TranscriptPullFn) -> None:
        self._dcf_pull_fn = dcf_pull_fn

    async def pull_data_for_period(
        self,
        ticker: str,
        year: int,
        quarter_num: int,
    ) -> Transcript | None:
        logger.info(f"Using DCFDataPull {ticker=} {year=} {quarter_num=}")
        return await self._dcf_pull_fn(ticker, year, quarter_num)


class EarningsBizDataPull(TranscriptDataPuller):
    """Loads transcript data from the earningscall.biz vendor."""

    def __init__(self, earnings_biz_pull_fn: TranscriptPullFn) -> None:
        self._earnings_biz_pull_fn = earnings_biz_pull_fn

    async def pull_data_for_period(
        self,
        ticker: str,
        year: int,
        quarter_num: int,
    ) -> Transcript | None:
        logger.info(f"Using EarningsBizDataPull {ticker=} {year=} {quarter_num=}")
        return await self._earnings_biz_pull_fn(ticker, year, quarter_num)


class TranscriptFallbackDataPull(TranscriptDataPuller):
    """Pulls transcripts with earningscall.biz first, then DCF as fallback."""

    def __init__(
        self,
        primary_pull: EarningsBizDataPull,
        fallback_pull: DCFDataPull,
    ) -> None:
        self._primary_pull = primary_pull
        self._fallback_pull = fallback_pull

    async def pull_data_for_period(
        self,
        ticker: str,
        year: int,
        quarter_num: int,
    ) -> Transcript | None:
        logger.info(
            "Pulling transcript with fallback order "
            f"{ticker=} {year=} {quarter_num=}"
        )
        primary_result = await self._primary_pull.pull_data_for_period(
            ticker=ticker,
            year=year,
            quarter_num=quarter_num,
        )
        if primary_result is not None:
            return primary_result

        logger.info(
            "Primary vendor returned no data. Trying DCF fallback "
            f"{ticker=} {year=} {quarter_num=}"
        )
        return await self._fallback_pull.pull_data_for_period(
            ticker=ticker,
            year=year,
            quarter_num=quarter_num,
        )
