import re

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
