import re
from urllib.parse import urlsplit

from biolit.domain.enums import LicenseTier

# Canonical token -> tier. Tokens are lowercase with separators collapsed to "_".
_TIER_BY_TOKEN: dict[str, LicenseTier] = {
    "cc0": LicenseTier.open,
    "cc_by": LicenseTier.open,
    "cc_by_sa": LicenseTier.open,
    "cc_by_nc": LicenseTier.non_commercial,
    "cc_by_nc_sa": LicenseTier.non_commercial,
    "cc_by_nd": LicenseTier.non_commercial,
    "cc_by_nc_nd": LicenseTier.non_commercial,
    "cc_no": LicenseTier.restricted,
    "no_cc_code": LicenseTier.restricted,
}

_EXTRACTION_ALLOWED: dict[LicenseTier, bool] = {
    LicenseTier.open: True,
    LicenseTier.non_commercial: True,
    LicenseTier.restricted: False,
    LicenseTier.unknown: False,
}


def _canonicalize(raw: str) -> str:
    token = raw.strip().lower()
    token = re.sub(r"[\s\-]+", "_", token)
    token = re.sub(r"_+", "_", token)
    return token.strip("_")


def normalize_license(raw: str | None) -> tuple[str | None, LicenseTier]:
    """Map a source license string to (canonical_token, tier).

    Presence of a token never implies extraction rights on its own; the tier does.
    """
    if raw is None or not raw.strip():
        return None, LicenseTier.unknown
    token = _canonicalize(raw)
    return token, _TIER_BY_TOKEN.get(token, LicenseTier.unknown)


def extraction_allowed_for(tier: LicenseTier) -> bool:
    return _EXTRACTION_ALLOWED[tier]


def license_token_from_url(raw: str | None) -> str | None:
    """Map a Creative Commons licence URL to the canonical token vocabulary above.

    The dead PMC OA service returned tokens like `CC BY`, which `_canonicalize` handled.
    Its replacement returns URLs, so this is the new front half of the same pipeline; the
    tier table and `extraction_allowed_for` are untouched.

    MATCHING IS BY EXACT PATH SEGMENT, never by substring. `by-nc` is a proper substring of
    `by-nc-nd` and `by-nc-sa`, both of which appear in the live sample, so `in` would
    mislabel them. Anything that is not a recognised Creative Commons URL returns None and
    is therefore refused -- prose asserting reuse rights is not a licence identifier.
    """
    if not raw or not raw.strip():
        return None
    parsed = urlsplit(raw.strip())
    if parsed.netloc.lower().removeprefix("www.") != "creativecommons.org":
        return None
    segments = [segment for segment in parsed.path.split("/") if segment]
    if len(segments) < 2:
        return None
    if segments[0] == "publicdomain" and segments[1] == "zero":
        return "cc0"
    if segments[0] == "licenses":
        return "cc_" + segments[1].replace("-", "_")
    return None


def license_deed_url(raw: str | None) -> str | None:
    """The Creative Commons deed for exactly the licence version the publisher granted.

    Unlike `license_token_from_url`, the VERSION IS KEPT: 3.0 and 4.0 share a tier but not
    their attribution terms, so a deed link must name the one actually granted. A URL with
    no version segment returns None rather than a guessed current version -- an invented
    version is a false statement of the terms. `legalcode` and `deed.xx` suffixes point at
    the same licence and are dropped; a jurisdiction port (`/3.0/us/`) is a different
    licence and is kept.
    """
    if license_token_from_url(raw) is None:
        return None
    segments = [segment for segment in urlsplit((raw or "").strip()).path.split("/") if segment]
    if len(segments) < 3 or not re.fullmatch(r"\d+(\.\d+)*", segments[2]):
        return None
    kept = segments[:3]
    if len(segments) > 3 and not re.match(r"(legalcode|deed)", segments[3]):
        kept.append(segments[3])
    return "https://creativecommons.org/" + "/".join(kept) + "/"
