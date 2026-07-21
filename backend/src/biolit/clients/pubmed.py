from xml.etree import ElementTree as ET

import httpx

from biolit.clients.http import RetryConfig, request_with_retry
from biolit.config import Settings
from biolit.domain.enums import Source, TextType
from biolit.domain.licensing import extraction_allowed_for, normalize_license
from biolit.domain.paper import Author, Paper

_EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
_PMC_OA = "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi"


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
        resp = await request_with_retry(
            self._client,
            "GET",
            f"{_EUTILS}/esearch.fcgi",
            retry=self._retry,
            params=self._params(db="pubmed", term=query, retmax=str(retmax)),
        )
        root = ET.fromstring(resp.text)
        return [el.text or "" for el in root.findall(".//IdList/Id") if el.text]

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
        papers: list[Paper] = []
        for article in root.findall(".//PubmedArticle"):
            papers.append(await self._parse_article(article))
        return papers

    async def _parse_article(self, article: ET.Element) -> Paper:
        pmid = article.findtext(".//MedlineCitation/PMID") or ""
        title = article.findtext(".//Article/ArticleTitle") or ""
        abstract = article.findtext(".//Abstract/AbstractText")
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
        pmc_id = None
        for aid in article.findall(".//ArticleIdList/ArticleId"):
            id_type = aid.get("IdType")
            if id_type == "doi":
                doi = aid.text
            elif id_type == "pmc":
                pmc_id = aid.text

        text_type, pointer, license_raw = await self._classify_pmc(pmc_id)
        token, tier = normalize_license(license_raw)

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

    async def _classify_pmc(self, pmc_id: str | None) -> tuple[TextType, str | None, str | None]:
        """Cross-reference the PMC OA Web Service; never infer rights from PMC presence."""
        if not pmc_id:
            return TextType.abstract_only, None, None
        resp = await request_with_retry(
            self._client,
            "GET",
            _PMC_OA,
            retry=self._retry,
            params={"id": pmc_id},
        )
        root = ET.fromstring(resp.text)
        if root.find(".//error") is not None:
            return TextType.abstract_only, None, None
        record = root.find(".//records/record")
        if record is None:
            return TextType.abstract_only, None, None
        license_raw = record.get("license")
        link = record.find("link")
        pointer = link.get("href") if link is not None else None
        return TextType.full_text_unverified, pointer, license_raw
