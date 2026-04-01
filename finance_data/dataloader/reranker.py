"""Reranker client for OpenAI-compatible vLLM rerank endpoints."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import httpx

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class RerankResult:
    """A single reranked candidate."""

    index: int
    score: float


class VllmRerankerClient:
    """HTTP client for the vLLM OpenAI-compatible rerank endpoint."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        timeout_seconds: float = 30.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout_seconds = timeout_seconds

    def rerank(
        self, query: str, documents: list[str], top_k: int
    ) -> list[RerankResult]:
        """Return reranked document indices and scores.

        Args:
            query: The original user query.
            documents: Candidate texts to rerank.
            top_k: Number of final items to return.
        """
        if not documents:
            return []

        response_json = self._post_rerank_request(query=query, documents=documents)
        parsed = self._parse_rerank_response(response_json)
        reranked = sorted(parsed, key=lambda item: item.score, reverse=True)
        return reranked[: max(0, top_k)]

    def _post_rerank_request(self, query: str, documents: list[str]) -> dict:
        payload = {
            "model": self._model,
            "query": query,
            "documents": documents,
            "top_n": len(documents),
        }
        rerank_url = f"{self._base_url}/rerank"
        _log.info(f"{rerank_url=}, {self._model=}, {len(documents)=}")

        try:
            with httpx.Client(timeout=self._timeout_seconds) as client:
                response = client.post(rerank_url, json=payload)
                response.raise_for_status()
        except httpx.HTTPError as exc:
            raise RuntimeError(
                f"Reranker request failed for {rerank_url=} with {exc!r}."
            ) from exc
        return response.json()

    def _parse_rerank_response(self, response_json: dict) -> list[RerankResult]:
        candidates = response_json.get("results")
        if candidates is None:
            candidates = response_json.get("data")
        if not isinstance(candidates, list):
            raise RuntimeError(
                "Reranker response is missing a list under 'results' or 'data'."
            )

        parsed: list[RerankResult] = []
        for item in candidates:
            index = item.get("index")
            score = item.get("relevance_score")
            if score is None:
                score = item.get("score")
            if not isinstance(index, int):
                continue
            if score is None:
                continue
            parsed.append(RerankResult(index=index, score=float(score)))

        _log.info(f"{len(parsed)=}")
        return parsed
