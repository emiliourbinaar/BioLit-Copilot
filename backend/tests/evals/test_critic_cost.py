import pytest

from biolit_evals.critic_cost import (
    ArmPilot,
    BudgetExceeded,
    KillSwitch,
    authorization_bound,
    cost_of,
)


def test_cost_of_weights_each_usage_field_by_its_own_price():
    """Three non-zero fields, three DIFFERENT prices: a mutant that ignores a field, sums raw
    token counts unweighted, or applies one field's price to every field would all miss this
    pinned total. A fixture with equal prices or only one non-zero field would not catch any
    of those."""
    usage = {"input_tokens": 1000, "output_tokens": 200, "cache_read_input_tokens": 500}
    prices = {
        "input_tokens": 0.000003,
        "output_tokens": 0.000015,
        "cache_read_input_tokens": 0.0000003,
    }
    expected = 1000 * 0.000003 + 200 * 0.000015 + 500 * 0.0000003
    assert cost_of(usage, prices) == pytest.approx(expected)


def test_cost_of_raises_loudly_on_a_usage_field_with_no_price_rather_than_skipping_it():
    """A silent skip would UNDERCOUNT the run -- exactly the bias this module exists to avoid.
    A price map missing one of two present usage fields must not just quietly total the other."""
    usage = {"input_tokens": 1000, "output_tokens": 200}
    prices = {"input_tokens": 0.000003}
    with pytest.raises(ValueError, match="output_tokens"):
        cost_of(usage, prices)


def test_kill_switch_aborts_above_1_25x_the_bound():
    switch = KillSwitch(limit=10.0)
    switch.record(9.0)
    with pytest.raises(BudgetExceeded, match="12.5"):
        switch.record(4.0)


def test_kill_switch_does_not_raise_exactly_at_1_25x_the_limit():
    """'exceeds' means strictly greater than -- landing exactly on the threshold is not yet an
    overrun. Distinguishes a `>` implementation from a `>=` one, which the above test (which
    lands strictly past the threshold, at 13.0 > 12.5) cannot do on its own."""
    switch = KillSwitch(limit=10.0)
    switch.record(12.5)


def test_zero_variance_pilot_gives_a_bound_equal_to_the_point_estimate():
    """No formula needed for this answer: if every call costs the same, the total is exact."""
    arms = [ArmPilot("a", mean_cost=0.002, sd_cost=0.0, n_pilot=30, n_full=900)]
    bound = authorization_bound(arms)
    assert bound.bound == pytest.approx(bound.point_estimate)
    assert bound.point_estimate == pytest.approx(1.8)


def test_bound_exceeds_the_point_estimate_but_not_by_a_per_call_percentile():
    """The bill is a SUM of thousands of draws, so it concentrates. Assuming every call is a
    95th-percentile call would bound a scenario that cannot occur."""
    arms = [
        ArmPilot("abstract", mean_cost=0.001, sd_cost=0.001, n_pilot=30, n_full=900),
        ArmPilot("findings", mean_cost=0.0006, sd_cost=0.0006, n_pilot=30, n_full=900),
        ArmPilot("direction", mean_cost=0.0005, sd_cost=0.0005, n_pilot=60, n_full=1800),
    ]
    bound = authorization_bound(arms)
    assert bound.bound > bound.point_estimate
    assert 0.05 < bound.inflation < 0.30


def test_bound_is_pinned_to_the_worked_example_not_merely_bracketed():
    """A tight numeric pin, not just an inequality: catches a mutant that POOLS the three arms
    into one distribution before computing V (pooling would misstate the spread, since the
    arms' `mean_cost`/`sd_cost`/`n_pilot`/`n_full` all differ), and catches a mutant that
    collapses the `N_a^2/n_a` (mean-uncertainty) term with the `+N_a` (realized-sum) term --
    for this fixture the two terms are far apart in magnitude (27000 vs 900 for the first arm
    alone), so confusing them moves the answer far outside this pin's tolerance."""
    arms = [
        ArmPilot("abstract", mean_cost=0.001, sd_cost=0.001, n_pilot=30, n_full=900),
        ArmPilot("findings", mean_cost=0.0006, sd_cost=0.0006, n_pilot=30, n_full=900),
        ArmPilot("direction", mean_cost=0.0005, sd_cost=0.0005, n_pilot=60, n_full=1800),
    ]
    bound = authorization_bound(arms)
    assert bound.point_estimate == pytest.approx(2.34)
    assert bound.bound == pytest.approx(2.727065, abs=1e-4)
    assert bound.inflation == pytest.approx(0.165413, abs=1e-4)
