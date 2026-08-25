import pytest

from biolit.config import critic_prices_per_token
from biolit_evals.critic_cost import cost_of


def test_critic_prices_per_token_pins_all_four_fields_for_a_known_model():
    """Pins the per-MTok -> per-token conversion for every field `cost_of` requires, including
    the two cache fields priced at the OUTPUT rate (the higher of the two known rates, so an
    unverified cache assumption overestimates rather than undercounts). A mutant that forgets
    to divide by 1e6, divides by the wrong constant, or prices a cache field at the input rate
    instead of the output rate would all miss this pinned dict."""
    prices = critic_prices_per_token("claude-sonnet-5")
    assert prices == {
        "input_tokens": 0.000003,
        "output_tokens": 0.000015,
        "cache_creation_input_tokens": 0.000015,
        "cache_read_input_tokens": 0.000015,
    }


def test_critic_prices_per_token_returns_none_for_an_unverified_model():
    """`None` is the refusal signal a caller uses to reject running a paid arm against a model
    whose rates were never hand-verified -- a mutant that falls back to a default price instead
    of `None` would silently defeat that refusal."""
    assert critic_prices_per_token("gpt-4o") is None


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
    prices = critic_prices_per_token("claude-sonnet-5")
    assert prices is not None
    assert cost_of(usage, prices) == pytest.approx(18.00)
