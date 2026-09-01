from pathlib import Path

import httpx
import pytest
import respx

from biolit.clients.pubmed import PubMedClient
from biolit.config import Settings
from biolit.domain.enums import LicenseTier, TextType

CASSETTES = Path(__file__).parent.parent / "cassettes"
EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


def _cassette(name: str) -> str:
    return (CASSETTES / name).read_text(encoding="utf-8")


@pytest.fixture
def settings() -> Settings:
    return Settings(http_backoff_base_seconds=0.001, http_backoff_max_seconds=0.002)


@respx.mock
async def test_efetch_classifies_from_the_permissions_block(settings):
    """The replacement path end to end: PubMed article -> its PMC id -> the permissions
    block -> a tier. Routes key on `db` because both calls now share efetch.fcgi."""
    respx.get(EFETCH, params__contains={"db": "pubmed"}).mock(
        return_value=httpx.Response(200, text=_cassette("pubmed_efetch_pmc_oa.xml"))
    )
    respx.get(EFETCH, params__contains={"db": "pmc"}).mock(
        return_value=httpx.Response(200, text=_cassette("pmc_efetch_cc_by.xml"))
    )
    async with httpx.AsyncClient() as http:
        papers = await PubMedClient(http, settings).efetch(["11111111"])
    paper = papers[0]
    assert paper.license == "cc_by"
    assert paper.license_tier is LicenseTier.open
    assert paper.extraction_allowed is True
    assert paper.text_type is TextType.full_text_unverified


@respx.mock
async def test_a_restricted_stub_is_refused_not_permitted(settings):
    """The PMC1401093 shape. The dangerous failure here is over-permitting, so this pins
    the refusal rather than merely pinning that something was returned."""
    respx.get(EFETCH, params__contains={"db": "pubmed"}).mock(
        return_value=httpx.Response(200, text=_cassette("pubmed_efetch_pmc_restricted.xml"))
    )
    respx.get(EFETCH, params__contains={"db": "pmc"}).mock(
        return_value=httpx.Response(200, text=_cassette("pmc_efetch_restricted_stub.xml"))
    )
    async with httpx.AsyncClient() as http:
        papers = await PubMedClient(http, settings).efetch(["22222222"])
    paper = papers[0]
    assert paper.license is None
    assert paper.license_tier is LicenseTier.unknown
    assert paper.extraction_allowed is False
    assert paper.text_type is TextType.abstract_only


@respx.mock
async def test_permissions_are_fetched_in_one_batched_call(settings):
    """One request for N papers, not N. The old per-article lookup was the dominant cost of
    a multi-thousand-paper fetch at NCBI's unkeyed 3 req/s, and was the 404's blast radius:
    one article outside the OA subset killed the whole call."""
    pubmed_route = respx.get(EFETCH, params__contains={"db": "pubmed"}).mock(
        return_value=httpx.Response(200, text=_cassette("pubmed_efetch_pmc_oa.xml"))
    )
    pmc_route = respx.get(EFETCH, params__contains={"db": "pmc"}).mock(
        return_value=httpx.Response(200, text=_cassette("pmc_efetch_cc_by.xml"))
    )
    async with httpx.AsyncClient() as http:
        await PubMedClient(http, settings).efetch(["11111111"])
    assert pubmed_route.call_count == 1
    assert pmc_route.call_count == 1


@respx.mock
async def test_no_pmc_call_is_made_when_no_article_carries_a_pmc_id(settings):
    """respx fails any unmocked request, so leaving the db=pmc route unmocked is what
    proves the call is skipped rather than merely ignored."""
    respx.get(EFETCH, params__contains={"db": "pubmed"}).mock(
        return_value=httpx.Response(200, text=_cassette("pubmed_efetch_no_pmc.xml"))
    )
    async with httpx.AsyncClient() as http:
        papers = await PubMedClient(http, settings).efetch(["33333333"])
    assert papers[0].text_type is TextType.abstract_only
    assert papers[0].extraction_allowed is False


@respx.mock
async def test_an_http_error_from_the_permissions_call_refuses_rather_than_crashing(settings):
    """THE CASSETTE WHOSE ABSENCE SHIPPED THE DEFECT.

    The suite covered the 200-with-<error> body and had no 404 case, so
    `request_with_retry`'s raise_for_status propagated and one article killed the whole
    fetch -- the same shape as the structured-abstract truncation that survived an
    extensive suite because every cassette happened to hold an unstructured abstract.

    A permissions lookup that fails must degrade to "no licence" (refuse), never to a
    crash and never to a permit.
    """
    respx.get(EFETCH, params__contains={"db": "pubmed"}).mock(
        return_value=httpx.Response(200, text=_cassette("pubmed_efetch_pmc_oa.xml"))
    )
    respx.get(EFETCH, params__contains={"db": "pmc"}).mock(return_value=httpx.Response(404))
    async with httpx.AsyncClient() as http:
        papers = await PubMedClient(http, settings).efetch(["11111111"])
    assert papers[0].extraction_allowed is False
    assert papers[0].license_tier is LicenseTier.unknown


@respx.mock
async def test_efetch_no_pmc_is_abstract_only(settings):
    respx.get(EFETCH, params__contains={"db": "pubmed"}).mock(
        return_value=httpx.Response(200, text=_cassette("pubmed_efetch_no_pmc.xml"))
    )
    async with httpx.AsyncClient() as http:
        papers = await PubMedClient(http, settings).efetch(["33333333"])
    paper = papers[0]
    assert paper.text_type is TextType.abstract_only
    assert paper.full_text_pointer is None
    assert paper.license_tier is LicenseTier.unknown
    # Unstructured abstract (single unlabelled AbstractText) must pass through
    # unchanged: plain text, no label prefix, no whitespace changes.
    assert paper.abstract == "Open abstract."


@respx.mock
async def test_efetch_structured_abstract_concatenates_all_sections(settings):
    # PubMed structured abstracts have multiple <AbstractText Label="..."> sections
    # (BACKGROUND/METHODS/RESULTS/CONCLUSIONS). All must be captured, in order,
    # each prefixed with its label and joined by newlines -- not just the first.
    respx.get(EFETCH, params__contains={"db": "pubmed"}).mock(
        return_value=httpx.Response(200, text=_cassette("pubmed_efetch_abstract_structured.xml"))
    )
    async with httpx.AsyncClient() as http:
        papers = await PubMedClient(http, settings).efetch(["44444444"])
    paper = papers[0]
    assert paper.abstract == (
        "BACKGROUND: Diabetes affects millions worldwide.\n"
        "METHODS: We conducted a randomized controlled trial.\n"
        # Nested inline markup (<i>...</i>) inside a section must be included
        # via itertext(), not truncated at the first child tag.
        "RESULTS: Response rate improved by 12% (p < 0.05).\n"
        "CONCLUSIONS: The intervention was effective."
    )


@respx.mock
async def test_efetch_no_abstract_element_is_none(settings):
    # Absence must still yield None, not "", preserving the pre-existing contract.
    respx.get(EFETCH, params__contains={"db": "pubmed"}).mock(
        return_value=httpx.Response(200, text=_cassette("pubmed_efetch_no_abstract.xml"))
    )
    async with httpx.AsyncClient() as http:
        papers = await PubMedClient(http, settings).efetch(["55555555"])
    paper = papers[0]
    assert paper.abstract is None


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


@respx.mock
async def test_efetch_abstracts_makes_no_pmc_call(settings):
    """Phase 5 needs only pmid -> (abstract, year); it judges abstracts and never consults
    full text, so the license/text-type classification `efetch` performs is work it does not
    use. That work is not free: `_classify_pmc` issues ONE request per paper carrying a PMC
    id, which against a 5,400-paper corpus at NCBI's unkeyed 3 req/s dominates the whole
    fetch -- and it is also currently fatal, because the OA service answers 404 for an
    article outside the OA subset while `request_with_retry` raises for status.

    The PMC route is deliberately left unmocked: respx fails any unmocked request, so this
    test fails loudly if the lean path ever starts classifying. Asserting on the returned
    values alone would not catch it -- the classification's results are simply discarded
    here, so a version that made 5,400 pointless requests would return exactly the same
    mapping.

    The cassette's article carries a PMC id, so the temptation to classify is present."""
    route = respx.get(EFETCH, params__contains={"db": "pubmed"}).mock(
        return_value=httpx.Response(200, text=_cassette("pubmed_efetch_pmc_oa.xml"))
    )
    async with httpx.AsyncClient() as http:
        got = await PubMedClient(http, settings).efetch_abstracts(["11111111"])

    assert got == {"11111111": ("Open abstract.", 2021)}
    assert route.called
