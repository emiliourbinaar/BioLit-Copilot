import httpx
import pytest
import respx

from biolit.clients.http import RetryConfig, request_with_retry


@pytest.fixture
def no_sleep():
    async def _sleep(_seconds: float) -> None:
        return None

    return _sleep


@respx.mock
async def test_retries_429_then_succeeds(no_sleep):
    route = respx.get("https://api.example/thing").mock(
        side_effect=[
            httpx.Response(429, headers={"Retry-After": "0"}),
            httpx.Response(200, json={"ok": True}),
        ]
    )
    retry = RetryConfig(max_retries=4, base_seconds=0.01, max_seconds=0.02)
    async with httpx.AsyncClient() as client:
        resp = await request_with_retry(
            client, "GET", "https://api.example/thing", retry=retry, sleep=no_sleep
        )
    assert resp.status_code == 200
    assert resp.json() == {"ok": True}
    assert route.call_count == 2


@respx.mock
async def test_gives_up_after_max_retries(no_sleep):
    respx.get("https://api.example/thing").mock(return_value=httpx.Response(429))
    retry = RetryConfig(max_retries=2, base_seconds=0.01, max_seconds=0.02)
    async with httpx.AsyncClient() as client:
        with pytest.raises(httpx.HTTPStatusError):
            await request_with_retry(
                client, "GET", "https://api.example/thing", retry=retry, sleep=no_sleep
            )


@respx.mock
async def test_success_first_try_no_retry(no_sleep):
    route = respx.get("https://api.example/ok").mock(return_value=httpx.Response(200, json={}))
    retry = RetryConfig(max_retries=4, base_seconds=0.01, max_seconds=0.02)
    async with httpx.AsyncClient() as client:
        resp = await request_with_retry(
            client, "GET", "https://api.example/ok", retry=retry, sleep=no_sleep
        )
    assert resp.status_code == 200
    assert route.call_count == 1
