import re
from collections.abc import Mapping
from dataclasses import dataclass
from xml.etree import ElementTree as ET

import httpx

from biolit.clients.http import RetryConfig, request_with_retry
from biolit.clients.pmc import licences_by_pmcid, normalize_pmcid
from biolit.config import Settings
from biolit.domain.enums import Source, TextType
from biolit.domain.licensing import (
    extraction_allowed_for,
    license_token_from_url,
    normalize_license,
)
from biolit.domain.paper import Author, Paper

_EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
_PMC_ARTICLE_URL = "https://pmc.ncbi.nlm.nih.gov/articles/PMC{pmcid}/"

#: `"acidosis, lactic"[MeSH Terms]` -> ("acidosis, lactic", "MeSH Terms").
_QUOTED_TERM = re.compile(r'"([^"]+)"\[([^\]]+)\]')

#: EVERY quoted term is kept, whatever field tag it carries, and the dictionary decides.
#:
#: ⚠️ SUPERSEDED DESIGN, recorded because the reasoning was wrong in an instructive way. This
#: first restricted to {MeSH Terms, Supplementary Concept, Pharmacological Action} on the
#: argument that `[All Fields]` carries only inflections ("depressed", "depression's") and
#: `[Subheading]` only qualifiers ("physiopathology"), so a variant colliding with an
#: unrelated alias could widen the concept set. That risk was hypothetical. Measured on
#: 2026-09-05 across the eight frozen queries it never occurred: dropping the restriction
#: adds exactly three concepts, and all three are correct -- Hemorrhage, Acidosis, and
#: 3-hydroxy-3-methylglutaryl-coenzyme A. The restriction's cost was real and larger: it
#: drops five on-query clusters the wide policy keeps (65/83 vs 70/83), among them
#: `Aspirin | Hemorrhage` and `SSRI | Hemorrhage` for an NSAIDs/GI-bleeding question and the
#: 9-paper `Lactic Acid | Acidosis` mechanism cluster for metformin/lactic acidosis.
#: A measured cost beats a hypothetical risk; the inflections simply NIL and cost nothing.


@dataclass(frozen=True)
class SearchResult:
    """PMIDs plus NCBI's own reading of what the query is about."""

    pmids: list[str]
    concept_terms: tuple[str, ...]


def _pmc_id_of(article: ET.Element) -> str | None:
    for article_id in article.findall(".//ArticleIdList/ArticleId"):
        if article_id.get("IdType") == "pmc":
            return normalize_pmcid(article_id.text)
    return None


class PubMedClient:
    def __init__(self, client: httpx.AsyncClient, settings: Settings) -> None:
        self._client = client
        self._settings = settings
        self._retry = RetryConfig(
            max_retries=settings.http_max_retries,
            base_seconds=settings.http_backoff_base_seconds,
            max_seconds=settings.http_backoff_max_seconds,
        )

    def _params(self, **extra: str) -> dict[str, str]:
        params = {"tool": self._settings.ncbi_tool}
        if self._settings.ncbi_api_key:
            params["api_key"] = self._settings.ncbi_api_key
        if self._settings.ncbi_email:
            params["email"] = self._settings.ncbi_email
        params.update(extra)
        return params

    async def esearch(self, query: str, retmax: int = 20) -> list[str]:
        return (await self.esearch_detailed(query, retmax=retmax)).pmids

    async def esearch_detailed(self, query: str, retmax: int = 20) -> SearchResult:
        """`esearch`, plus the concept terms NCBI translated the query into.

        The translation is free -- it rides on the search request that already happens -- and
        it is the only query-side concept resolution available that does not depend on the
        local alias dictionary. That matters because the dictionary NILs on exactly the terms
        a user is most likely to type: measured over the eight frozen queries, `depression`,
        `gastrointestinal bleeding` and `thyroid dysfunction` all resolve to nothing locally,
        while NCBI maps them to "depressive disorder", "gastrointestinal hemorrhage" and
        "thyroid gland" respectively.

        NOTE the element is `TranslationSet`, NOT `TranslationStack`: the stack is absent from
        every response measured on 2026-09-05, so parsing it would have silently produced an
        empty concept set on every real query while the unit tests passed against a fixture.
        """
        resp = await request_with_retry(
            self._client,
            "GET",
            f"{_EUTILS}/esearch.fcgi",
            retry=self._retry,
            params=self._params(db="pubmed", term=query, retmax=str(retmax)),
        )
        root = ET.fromstring(resp.text)
        terms: list[str] = []
        for translation in root.findall(".//TranslationSet/Translation"):
            for term, _field in _QUOTED_TERM.findall(translation.findtext("To") or ""):
                if term not in terms:
                    terms.append(term)
        return SearchResult(
            pmids=[el.text or "" for el in root.findall(".//IdList/Id") if el.text],
            concept_terms=tuple(terms),
        )

    async def efetch(self, pmids: list[str]) -> list[Paper]:
        if not pmids:
            return []
        resp = await request_with_retry(
            self._client,
            "GET",
            f"{_EUTILS}/efetch.fcgi",
            retry=self._retry,
            params=self._params(db="pubmed", id=",".join(pmids), retmode="xml"),
        )
        root = ET.fromstring(resp.text)
        articles = root.findall(".//PubmedArticle")
        pmc_ids = sorted({p for article in articles if (p := _pmc_id_of(article))})
        licences = await self._fetch_licences(pmc_ids)
        return [self._parse_article(article, licences) for article in articles]

    async def efetch_abstracts(self, pmids: list[str]) -> dict[str, tuple[str | None, int | None]]:
        """{pmid: (abstract, publication year)} for consumers that judge abstracts only.

        Deliberately does NOT build `Paper` and does not consult the PMC OA service. A
        consumer that never reads full text has no use for text_type, license, or a full-text
        pointer, and obtaining them costs one extra HTTP request per article carrying a PMC
        id -- which at NCBI's unkeyed 3 req/s is the dominant cost of a multi-thousand-paper
        fetch. (It was also fatal when this was written: the dead OA service answered 404 for
        any article outside the OA subset and `request_with_retry` raises for status. That is
        no longer true -- `_fetch_licences` degrades to an empty map instead -- so cost is now
        the whole reason, not merely the surviving one.)

        Year comes from JournalIssue/PubDate/Year and may be None; it is here because the
        spec requires per-class publication-year distributions to be reported regardless of
        outcome, as the check on the availability confound that topping up can introduce.
        """
        if not pmids:
            return {}
        resp = await request_with_retry(
            self._client,
            "GET",
            f"{_EUTILS}/efetch.fcgi",
            retry=self._retry,
            params=self._params(db="pubmed", id=",".join(pmids), retmode="xml"),
        )
        root = ET.fromstring(resp.text)
        out: dict[str, tuple[str | None, int | None]] = {}
        for article in root.findall(".//PubmedArticle"):
            pmid = article.findtext(".//MedlineCitation/PMID") or ""
            if not pmid:
                continue
            year_text = article.findtext(".//JournalIssue/PubDate/Year")
            year = int(year_text) if year_text and year_text.isdigit() else None
            out[pmid] = (self._parse_abstract(article), year)
        return out

    async def _fetch_licences(self, pmc_ids: list[str]) -> dict[str, str | None]:
        """One batched request for every PMC id in the response, or none at all.

        Degrades to an empty map on any HTTP failure, which refuses rather than crashing.
        A licence lookup that cannot answer must never be read as permission, and must
        never take the whole fetch down with it -- which is exactly what the dead OA
        endpoint did.
        """
        if not pmc_ids:
            return {}
        try:
            resp = await request_with_retry(
                self._client,
                "GET",
                f"{_EUTILS}/efetch.fcgi",
                retry=self._retry,
                params=self._params(db="pmc", id=",".join(pmc_ids), retmode="xml"),
            )
        except httpx.HTTPError:
            return {}
        try:
            return licences_by_pmcid(ET.fromstring(resp.text))
        except ET.ParseError:
            return {}

    def _parse_article(self, article: ET.Element, licences: Mapping[str, str | None]) -> Paper:
        pmid = article.findtext(".//MedlineCitation/PMID") or ""
        title = article.findtext(".//Article/ArticleTitle") or ""
        abstract = self._parse_abstract(article)
        journal = article.findtext(".//Journal/Title")
        year_text = article.findtext(".//JournalIssue/PubDate/Year")
        year = int(year_text) if year_text and year_text.isdigit() else None

        authors: list[Author] = []
        for author in article.findall(".//AuthorList/Author"):
            last = author.findtext("LastName")
            fore = author.findtext("ForeName")
            if last or fore:
                authors.append(Author(name=" ".join(p for p in (fore, last) if p)))

        mesh = [
            el.text
            for el in article.findall(".//MeshHeadingList/MeshHeading/DescriptorName")
            if el.text
        ]

        doi = None
        pmc_id = _pmc_id_of(article)

        for aid in article.findall(".//ArticleIdList/ArticleId"):
            id_type = aid.get("IdType")
            if id_type == "doi":
                doi = aid.text

        raw_licence = licences.get(pmc_id) if pmc_id else None
        token, tier = normalize_license(license_token_from_url(raw_licence))
        if raw_licence:
            text_type = TextType.full_text_unverified
            pointer = _PMC_ARTICLE_URL.format(pmcid=pmc_id)
        else:
            text_type = TextType.abstract_only
            pointer = None

        return Paper(
            id=doi or pmid,
            source=Source.pubmed,
            pmid=pmid or None,
            doi=doi,
            title=title,
            abstract=abstract,
            authors=authors,
            journal=journal,
            year=year,
            mesh_terms=mesh,
            text_type=text_type,
            full_text_pointer=pointer,
            license=token,
            license_tier=tier,
            extraction_allowed=extraction_allowed_for(tier),
            raw={"pmid": pmid, "pmc_id": pmc_id},
        )

    @staticmethod
    def _parse_abstract(article: ET.Element) -> str | None:
        """Concatenate all AbstractText sections in document order.

        PubMed *structured* abstracts contain multiple <AbstractText> elements,
        one per section (e.g. Label="BACKGROUND"/"METHODS"/"RESULTS"/"CONCLUSIONS").
        Using findtext (or el.text) on the first match alone silently drops every
        section after the first. Labelled sections are prefixed "<LABEL>: " to
        match how PubMed itself renders structured abstracts; an unstructured,
        unlabelled single-section abstract passes through unchanged.
        """
        sections: list[str] = []
        for el in article.findall(".//Abstract/AbstractText"):
            text = "".join(el.itertext()).strip()
            if not text:
                continue
            label = el.get("Label")
            sections.append(f"{label}: {text}" if label else text)
        return "\n".join(sections) if sections else None
