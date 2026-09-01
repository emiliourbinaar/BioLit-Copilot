"""Pure parsing of PMC article XML. No HTTP — the caller supplies a parsed root.

Replaces the PMC OA Web Service, whose endpoint is dead (404 with no parameters at all,
serving an NCBI error page rather than the service's own XML error document). This reads
the article's own `<permissions>` block instead, which is the PUBLISHER'S licence
statement — so the Phase 1 rule "never infer rights from PMC presence" holds exactly as
written: presence in PMC yields nothing here; only a licence identifier does.
"""

from xml.etree import ElementTree as ET

_ALI = "{http://www.niso.org/schemas/ali/1.0/}"
_XLINK = "{http://www.w3.org/1999/xlink}"


def normalize_pmcid(raw: str | None) -> str | None:
    """Bare digits, no `PMC` prefix, so both id sources key alike.

    PubMed's ArticleIdList reports `PMC8917620`; PMC's own article-id reports `8917620`.
    Joining the two without this yields an empty intersection and therefore zero permitted
    papers — silently, and indistinguishably from "nothing is licensed".
    """
    if not raw or not raw.strip():
        return None
    return raw.strip().upper().removeprefix("PMC") or None


def _licence_of(article: ET.Element) -> str | None:
    """The licence IDENTIFIER, or None. Never the element's prose."""
    permissions = article.find(".//permissions")
    if permissions is None:
        return None
    ref = permissions.find(f".//{_ALI}license_ref")
    if ref is not None and (ref.text or "").strip():
        return (ref.text or "").strip()
    licence = permissions.find(".//license")
    if licence is not None:
        href = licence.get(f"{_XLINK}href")
        if href and href.strip():
            return href.strip()
    return None


def licences_by_pmcid(root: ET.Element) -> dict[str, str | None]:
    """{normalized PMCID: licence identifier or None} for every article in the response.

    An article present with no licence maps to None rather than being omitted, so the
    caller can distinguish "PMC returned it and it carries no licence" (refuse) from
    "PMC did not return it at all" (also refuse, different drop reason).
    """
    out: dict[str, str | None] = {}
    for article in root.findall(".//article"):
        pmcid = None
        for article_id in article.findall(".//article-id"):
            # `pmcid` is what the service actually emits, carrying the PMC-prefixed value
            # (`PMC8917620`). Verified against efetch db=pmc across 10 articles spanning the
            # id range: every one emits pmcid, pmcid-ver, pmcaid and pmcaiid, and not one
            # emits a bare `pmc`. Matching `pmc` as well would be dead surface that exists
            # only to accommodate a fixture, which is backwards -- the fixture follows the
            # service. The match is exact, so the `pmcid-ver` sibling (`PMC8917620.1`) is
            # correctly skipped rather than parsed as a different article.
            if article_id.get("pub-id-type") == "pmcid":
                pmcid = normalize_pmcid(article_id.text)
                break
        if pmcid is None:
            continue
        out[pmcid] = _licence_of(article)
    return out
