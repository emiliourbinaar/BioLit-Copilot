from pathlib import Path

import httpx
import pytest
import respx

from biolit.clients.pubmed import PubMedClient
from biolit.config import Settings
from biolit.domain.enums import LicenseTier, Source, TextType

CASSETTES = Path(__file__).parent.parent / "cassettes"


def _cassette(name: str) -> str:
    return (CASSETTES / name).read_text(encoding="utf-8")


@pytest.fixture
def settings() -> Settings:
    return Settings(http_backoff_base_seconds=0.001, http_backoff_max_seconds=0.002)


@respx.mock
async def test_efetch_open_access(settings):
    respx.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi").mock(
        return_value=httpx.Response(200, text=_cassette("pubmed_efetch_pmc_oa.xml"))
    )
    respx.get("https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi").mock(
        return_value=httpx.Response(200, text=_cassette("pmc_oa_open.xml"))
    )
    async with httpx.AsyncClient() as http:
        papers = await PubMedClient(http, settings).efetch(["11111111"])
    paper = papers[0]
    assert paper.source is Source.pubmed
    assert paper.pmid == "11111111"
    assert paper.title == "Open access paper"
    assert paper.year == 2021
    assert paper.mesh_terms == ["Metformin"]
    assert paper.text_type is TextType.full_text_unverified
    assert paper.license_tier is LicenseTier.open
    assert paper.extraction_allowed is True


@respx.mock
async def test_efetch_pmc_but_restricted_license(settings):
    # In PMC, but under a restrictive license: must NOT be treated as extractable.
    respx.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi").mock(
        return_value=httpx.Response(200, text=_cassette("pubmed_efetch_pmc_restricted.xml"))
    )
    respx.get("https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi").mock(
        return_value=httpx.Response(200, text=_cassette("pmc_oa_restricted.xml"))
    )
    async with httpx.AsyncClient() as http:
        papers = await PubMedClient(http, settings).efetch(["22222222"])
    paper = papers[0]
    assert paper.text_type is TextType.full_text_unverified
    assert paper.license_tier is LicenseTier.restricted
    assert paper.extraction_allowed is False


@respx.mock
async def test_efetch_pmc_present_but_not_oa(settings):
    respx.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi").mock(
        return_value=httpx.Response(200, text=_cassette("pubmed_efetch_pmc_restricted.xml"))
    )
    respx.get("https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi").mock(
        return_value=httpx.Response(200, text=_cassette("pmc_oa_not_open.xml"))
    )
    async with httpx.AsyncClient() as http:
        papers = await PubMedClient(http, settings).efetch(["22222222"])
    paper = papers[0]
    assert paper.text_type is TextType.abstract_only
    assert paper.extraction_allowed is False


@respx.mock
async def test_efetch_no_pmc_is_abstract_only(settings):
    respx.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi").mock(
        return_value=httpx.Response(200, text=_cassette("pubmed_efetch_no_pmc.xml"))
    )
    async with httpx.AsyncClient() as http:
        papers = await PubMedClient(http, settings).efetch(["33333333"])
    paper = papers[0]
    assert paper.text_type is TextType.abstract_only
    assert paper.full_text_pointer is None
    assert paper.license_tier is LicenseTier.unknown


@respx.mock
async def test_esearch_returns_pmids(settings):
    respx.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi").mock(
        return_value=httpx.Response(
            200,
            text='<?xml version="1.0"?><eSearchResult><IdList>'
            "<Id>11111111</Id><Id>22222222</Id></IdList></eSearchResult>",
        )
    )
    async with httpx.AsyncClient() as http:
        pmids = await PubMedClient(http, settings).esearch("metformin PCOS")
    assert pmids == ["11111111", "22222222"]
