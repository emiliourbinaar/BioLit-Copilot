import httpx

from biolit.clients.http import RetryConfig, request_with_retry
from biolit.config import Settings
from biolit.domain.enums import Source, TextType
from biolit.domain.licensing import extraction_allowed_for, normalize_license
from biolit.domain.paper import Author, Paper

_API = "https://api.biorxiv.org/details"
_SERVER_SOURCE = {"biorxiv": Source.biorxiv, "medrxiv": Source.medrxiv}


def _clean(value: str | None) -> str | None:
    if value is None:
        return None
    trimmed = value.strip()
    if not trimmed or trimmed.upper() == "NA":
        return None
    return trimmed


def _authors(raw: str | None) -> list[Author]:
    if not raw:
        return []
    return [Author(name=name.strip()) for name in raw.split(";") if name.strip()]


class BiorxivClient:
    def __init__(self, client: httpx.AsyncClient, settings: Settings) -> None:
        self._client = client
        self._retry = RetryConfig(
            max_retries=settings.http_max_retries,
            base_seconds=settings.http_backoff_base_seconds,
            max_seconds=settings.http_backoff_max_seconds,
        )

    async def details(self, server: str, doi: str) -> list[Paper]:
        resp = await request_with_retry(
            self._client,
            "GET",
            f"{_API}/{server}/{doi}",
            retry=self._retry,
        )
        payload = resp.json()
        return [self._parse(record) for record in payload.get("collection", [])]

    def _parse(self, record: dict) -> Paper:
        server = (record.get("server") or "").lower()
        source = _SERVER_SOURCE.get(server, Source.biorxiv)

        jatsxml = _clean(record.get("jatsxml"))
        if jatsxml:
            text_type = TextType.full_text_unverified
            pointer = jatsxml
        else:
            text_type = TextType.abstract_only
            pointer = None

        token, tier = normalize_license(record.get("license"))
        year_text = (record.get("date") or "")[:4]
        year = int(year_text) if year_text.isdigit() else None
        doi = _clean(record.get("doi"))

        return Paper(
            id=doi or record.get("title", ""),
            source=source,
            doi=doi,
            title=record.get("title", ""),
            abstract=_clean(record.get("abstract")),
            authors=_authors(record.get("authors")),
            year=year,
            categories=[c for c in [_clean(record.get("category"))] if c],
            published_doi=_clean(record.get("published")),
            text_type=text_type,
            full_text_pointer=pointer,
            license=token,
            license_tier=tier,
            extraction_allowed=extraction_allowed_for(tier),
            raw=record,
        )


def dedupe(papers: list[Paper]) -> list[Paper]:
    """Drop duplicates, keying on DOI then PMID; first occurrence wins."""
    seen: set[str] = set()
    result: list[Paper] = []
    for paper in papers:
        key = paper.doi or paper.pmid or paper.id
        if key in seen:
            continue
        seen.add(key)
        result.append(paper)
    return result
