import asyncio
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

import httpx

_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


@dataclass(frozen=True)
class RetryConfig:
    max_retries: int
    base_seconds: float
    max_seconds: float


def _backoff_seconds(attempt: int, retry: RetryConfig) -> float:
    # Exponential backoff with full jitter.
    ceiling = min(retry.max_seconds, retry.base_seconds * (2**attempt))
    return random.uniform(0.0, ceiling)


async def request_with_retry(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    *,
    retry: RetryConfig,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    **kwargs,
) -> httpx.Response:
    """Issue a request, retrying 429/5xx with exponential backoff + full jitter.

    Returns the first non-retryable response (raising for 4xx other than 429 via
    ``raise_for_status``). Raises ``httpx.HTTPStatusError`` if retries are exhausted.
    """
    attempt = 0
    while True:
        response = await client.request(method, url, **kwargs)
        if response.status_code not in _RETRYABLE_STATUS:
            response.raise_for_status()
            return response
        if attempt >= retry.max_retries:
            response.raise_for_status()
            return response  # pragma: no cover - raise_for_status always raises here
        await sleep(_backoff_seconds(attempt, retry))
        attempt += 1
