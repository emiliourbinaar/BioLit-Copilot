from datetime import date, timedelta

import pytest

from biolit.config import (
    _PRICES_MAX_AGE_DAYS,
    PRICES_VERIFIED_ON,
    StalePriceTableError,
    UnverifiedModelPriceError,
    critic_prices_per_token,
)
from biolit_evals.critic_cost import cost_of

# A fixed "today" inside the verified window (PRICES_VERIFIED_ON is 2026-08-21, max age 30
# days) so these tests are deterministic and never rot as real time passes -- see the
# staleness tests below for the injectable-date mechanism itself.
_FRESH_TODAY = date(2026, 8, 21)


def test_critic_prices_per_token_pins_all_four_fields_for_a_known_model():
    """Pins the per-MTok -> per-token conversion for every field `cost_of` requires, including
    the two cache fields priced at the OUTPUT rate (the higher of the two known rates, so an
    unverified cache assumption overestimates rather than undercounts). A mutant that forgets
    to divide by 1e6, divides by the wrong constant, or prices a cache field at the input rate
    instead of the output rate would all miss this pinned dict."""
    prices = critic_prices_per_token("claude-sonnet-5", today=_FRESH_TODAY)
    assert prices == {
        "input_tokens": 0.000003,
        "output_tokens": 0.000015,
        "cache_creation_input_tokens": 0.000015,
        "cache_read_input_tokens": 0.000015,
    }


def test_a_price_table_exactly_at_the_age_limit_is_still_accepted():
    """The BOUNDARY, pinned on the passing side. The refusal is `age > _PRICES_MAX_AGE_DAYS`,
    so a table verified exactly `_PRICES_MAX_AGE_DAYS` ago is still fresh. Without this, a
    `>` -> `>=` mutant refuses a table that is precisely at the limit and every other test
    stays green -- the two staleness tests together are what distinguish the operators.

    `today` is INJECTED rather than mocked: a staleness test that reads the real clock passes
    today and fails once real time drifts past the window, which would be its own silent
    wrongness bug (ADR-0016, rule 4)."""
    at_limit = PRICES_VERIFIED_ON + timedelta(days=_PRICES_MAX_AGE_DAYS)
    prices = critic_prices_per_token("claude-opus-5", today=at_limit)
    assert prices["input_tokens"] == 0.000005


def test_a_price_table_one_day_past_the_limit_refuses_rather_than_warns():
    """The BOUNDARY, pinned on the refusing side, one day past the limit -- the minimum step
    that distinguishes `>` from `>=` together with the at-limit test above.

    It REFUSES rather than warns on purpose: a warning scrolls past, and the entire reason this
    check exists is that an unverified rate must not be allowed to price a real run. The message
    must name the recorded date so a human knows what to re-verify and bump."""
    stale = PRICES_VERIFIED_ON + timedelta(days=_PRICES_MAX_AGE_DAYS + 1)
    with pytest.raises(StalePriceTableError, match=PRICES_VERIFIED_ON.isoformat()):
        critic_prices_per_token("claude-opus-5", today=stale)


def test_critic_prices_per_token_raises_a_named_error_for_an_unverified_model():
    """A named exception, not a bare `None`, is the refusal signal: a future caller that
    forgets to check for it gets a message stating the ACTUAL problem (no verified rate) via
    `str(exc)`, rather than a `TypeError` out of `cost_of`'s iteration over `prices=None` that
    names nothing about a missing price -- exactly the 'loud but confusing' failure the
    louder-not-quieter posture is meant to avoid."""
    with pytest.raises(UnverifiedModelPriceError, match="gpt-4o"):
        critic_prices_per_token("gpt-4o", today=_FRESH_TODAY)


def test_a_million_input_and_output_tokens_costs_exactly_the_published_mtok_rates():
    """A round, human-checkable magnitude (1 MTok of each field) that ties the per-MTok table
    straight to a dollar figure a reader can verify by eye: $3 + $15 = $18 for sonnet-5. An
    off-by-1e6 divisor error would make this either ~$0.000018 (guard fires instantly on any
    real run) or ~$18,000,000 (guard never fires) -- both far outside this pin's tolerance."""
    usage = {
        "input_tokens": 1_000_000,
        "output_tokens": 1_000_000,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
    }
    prices = critic_prices_per_token("claude-sonnet-5", today=_FRESH_TODAY)
    assert cost_of(usage, prices) == pytest.approx(18.00)
