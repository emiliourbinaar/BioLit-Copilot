import pytest

from biolit.domain.enums import LicenseTier
from biolit.domain.licensing import (
    extraction_allowed_for,
    license_token_from_url,
    normalize_license,
)


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


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://creativecommons.org/licenses/by/4.0/", "cc_by"),
        ("https://creativecommons.org/licenses/by-nc/4.0/", "cc_by_nc"),
        ("https://creativecommons.org/licenses/by-nc-nd/4.0/", "cc_by_nc_nd"),
        ("https://creativecommons.org/licenses/by-nc-sa/4.0/", "cc_by_nc_sa"),
        ("https://creativecommons.org/licenses/by-nc/3.0/", "cc_by_nc"),
        ("https://creativecommons.org/licenses/by-nc-nd/3.0/", "cc_by_nc_nd"),
        ("http://creativecommons.org/publicdomain/zero/1.0/", "cc0"),
    ],
)
def test_every_licence_form_observed_in_the_live_sample_maps_to_its_token(url, expected):
    """One case per form seen in the spec's 41-article live sample, both versions included.

    The version suffix must be stripped: 3.0 and 4.0 of the same licence are the same
    licence for tier purposes, and the live sample contains both.
    """
    assert license_token_from_url(url) == expected


def test_by_nc_nd_is_not_read_as_by_nc():
    """THE SUBSTRING TRAP, and the reason matching is by exact path segment.

    `by-nc` is a proper substring of both `by-nc-nd` and `by-nc-sa`. A shortest-first
    substring match silently mislabels six of the 41 articles in the live sample. That the
    two tokens happen to share a tier today is COINCIDENCE, not safety -- the tier table can
    change, and this function must be correct independently of it.

    Mutation-verified by running the weakened case, not by predicting it: a shortest-first
    substring match fails this test and also the by-nc-nd and by-nc-sa cases of the
    parametrized test above, while every by-nc and plain-by case stays green. That green is
    the point -- the mislabel returns a real token from the vocabulary, so it reads as a
    correct answer for the shorter code rather than as an error.
    """
    assert (
        license_token_from_url("https://creativecommons.org/licenses/by-nc-nd/4.0/") != "cc_by_nc"
    )
    assert (
        license_token_from_url("https://creativecommons.org/licenses/by-nc-sa/4.0/") != "cc_by_nc"
    )


@pytest.mark.parametrize(
    "raw",
    [
        None,
        "",
        "   ",
        "This file is available for text mining. It may also be used consistent with fair use.",
        "https://www.cochranelibrary.com/cdsr/editorial-policies",
        "https://www.diabetesjournals.org/content/license",
        "Article reuse guidelines: sagepub.com/journals-permissions",
    ],
)
def test_non_creative_commons_terms_yield_no_token_and_are_therefore_refused(raw):
    """Fails CLOSED, which is the whole safety property.

    Four articles in the live sample carry the NIH author-manuscript statement, whose PROSE
    asserts text mining is permitted while carrying no licence identifier. Permitting on
    parsed prose would weaken a deliberate compliance safeguard; it is refused deliberately,
    and this test is where that decision is enforced. The publisher-specific terms
    (Cochrane, Sage, diabetesjournals) are the same shape.
    """
    token = license_token_from_url(raw)
    assert token is None
    _, tier = normalize_license(token)
    assert tier is LicenseTier.unknown
    assert extraction_allowed_for(tier) is False
