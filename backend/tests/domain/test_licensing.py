import pytest

from biolit.domain.enums import LicenseTier
from biolit.domain.licensing import extraction_allowed_for, normalize_license


@pytest.mark.parametrize(
    "raw, token, tier",
    [
        ("CC0", "cc0", LicenseTier.open),
        ("cc0", "cc0", LicenseTier.open),
        ("CC BY", "cc_by", LicenseTier.open),
        ("cc_by", "cc_by", LicenseTier.open),
        ("CC BY-NC", "cc_by_nc", LicenseTier.non_commercial),
        ("cc_by_nc_nd", "cc_by_nc_nd", LicenseTier.non_commercial),
        ("cc_no", "cc_no", LicenseTier.restricted),
        ("NO-CC CODE", "no_cc_code", LicenseTier.restricted),
        (None, None, LicenseTier.unknown),
        ("weird-unknown", "weird_unknown", LicenseTier.unknown),
    ],
)
def test_normalize_license(raw, token, tier):
    assert normalize_license(raw) == (token, tier)


@pytest.mark.parametrize(
    "tier, allowed",
    [
        (LicenseTier.open, True),
        (LicenseTier.non_commercial, True),
        (LicenseTier.restricted, False),
        (LicenseTier.unknown, False),
    ],
)
def test_extraction_allowed(tier, allowed):
    assert extraction_allowed_for(tier) is allowed
