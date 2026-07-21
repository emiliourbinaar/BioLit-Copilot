from pathlib import Path

import httpx
import pytest
import respx

from biolit.clients.biorxiv import BiorxivClient, dedupe
from biolit.config import Settings
from biolit.domain.enums import LicenseTier, Source, TextType
from biolit.domain.paper import Paper

CASSETTES = Path(__file__).parent.parent / "cassettes"


def _cassette(name: str) -> str:
    return (CASSETTES / name).read_text(encoding="utf-8")


@pytest.fixture
def settings() -> Settings:
    return Settings(http_backoff_base_seconds=0.001, http_backoff_max_seconds=0.002)


@respx.mock
async def test_details_fulltext(settings):
    respx.get(url__regex=r"https://api\.biorxiv\.org/details/biorxiv/.*").mock(
        return_value=httpx.Response(200, text=_cassette("biorxiv_fulltext.json"))
    )
    async with httpx.AsyncClient() as http:
        papers = await BiorxivClient(http, settings).details("biorxiv", "10.1101/2020.12.30.424878")
    paper = papers[0]
    assert paper.source is Source.biorxiv
    assert paper.text_type is TextType.full_text_unverified
    assert paper.full_text_pointer.endswith(".source.xml")  # type: ignore[union-attr]
    assert paper.license_tier is LicenseTier.open
    assert paper.extraction_allowed is True
    assert paper.published_doi == "10.1038/s41586-021-00000-0"
    assert [a.name for a in paper.authors] == ["Doe, J.", "Roe, R."]


@respx.mock
async def test_details_missing_jatsxml_is_abstract_only(settings):
    respx.get(url__regex=r"https://api\.biorxiv\.org/details/biorxiv/.*").mock(
        return_value=httpx.Response(200, text=_cassette("biorxiv_no_jatsxml.json"))
    )
    async with httpx.AsyncClient() as http:
        papers = await BiorxivClient(http, settings).details("biorxiv", "10.1101/2021.02.02.111111")
    paper = papers[0]
    assert paper.text_type is TextType.abstract_only
    assert paper.full_text_pointer is None
    assert paper.published_doi is None  # "NA" normalized to None


@respx.mock
async def test_details_fulltext_but_restricted_license(settings):
    respx.get(url__regex=r"https://api\.biorxiv\.org/details/medrxiv/.*").mock(
        return_value=httpx.Response(200, text=_cassette("biorxiv_restricted.json"))
    )
    async with httpx.AsyncClient() as http:
        papers = await BiorxivClient(http, settings).details(
            "medrxiv", "10.1101/2020.09.09.20191205"
        )
    paper = papers[0]
    assert paper.source is Source.medrxiv
    assert paper.text_type is TextType.full_text_unverified
    assert paper.license_tier is LicenseTier.restricted
    assert paper.extraction_allowed is False


def test_dedupe_keeps_first_by_doi():
    a = Paper(
        id="10.1/x",
        source=Source.biorxiv,
        doi="10.1/x",
        title="A",
        text_type=TextType.abstract_only,
    )
    b = Paper(
        id="10.1/x",
        source=Source.biorxiv,
        doi="10.1/x",
        title="A dup",
        text_type=TextType.abstract_only,
    )
    c = Paper(id="p9", source=Source.pubmed, pmid="9", title="C", text_type=TextType.abstract_only)
    result = dedupe([a, b, c])
    assert [p.title for p in result] == ["A", "C"]
