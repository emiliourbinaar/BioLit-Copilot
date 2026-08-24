"""Cost accounting, the authorization bound, and the kill-switch for a paid Critic run.

TWO ERROR SOURCES, HANDLED SEPARATELY, BECAUSE THEY NEED DIFFERENT REMEDIES.

(a) STATISTICAL UNCERTAINTY -- covered by `authorization_bound`. A per-call percentile
multiplied by the call count is the WRONG estimator: the bill is a SUM of ~3,600 draws, and a
sum of many draws concentrates around its mean (its relative spread shrinks as the count
grows), so assuming every one of those draws is a 95th-percentile draw bounds a scenario that
cannot occur. `authorization_bound` instead puts a one-sided prediction bound on the REALIZED
TOTAL. Arms combine through the variance rather than being pooled into one distribution,
because their cost distributions differ materially (two abstracts read per call against one,
full text against extracted sentences) and pooling would misstate the spread.

(b) SYSTEMATIC ERROR -- which `authorization_bound` deliberately does NOT cover, and this is
the point. At the pilot sizes this eval plans, the bound lands only ~8-16% above the point
estimate (the worked example below lands at ~16.5%), so it would NOT have caught a prior
phase's cost estimate that came in 24% low -- because that miss was BIAS, not variance, and no
percentile fixes a measurement bug. Widening the interval until it happens to cover a bias is
the wrong repair: it masks the defect while inflating every future estimate, on every future
run, forever. Bias is eliminated at the source instead -- tokens read from the API's own usage
fields (`cost_of` below), every call counted including retries and refusals, prices pinned at
run time and passed in rather than hardcoded, and the estimate reconciled against the actual
bill after the run. A future reader who "fixes" a bound that looks too tight by widening it has
mistaken (a) for (b) and re-introduced exactly the failure mode this module exists to prevent.

THE FORMULA. Per arm `a`, with pilot mean `m_a`, pilot SD `s_a`, pilot size `n_a`, and the full
run's call count `N_a` for that arm:

    point estimate  T = sum_a  N_a * m_a
    variance        V = sum_a  s_a^2 * ( N_a^2 / n_a  +  N_a )
    bound             = T + t(confidence, min_a n_a - 1) * sqrt(V)

`N_a^2 / n_a` is the uncertainty in the ESTIMATED MEAN (shrinks as the pilot grows); `+ N_a` is
the REALIZED SUM's own variation (present even with a perfectly known mean, because the total
is a sum of `N_a` individually-random calls, not `N_a` copies of the mean). Both terms matter
and are of very different magnitude for a realistic pilot ratio (N_a >> n_a makes the first
term dominate), which is why collapsing them, or pooling arms into a single distribution before
computing V, would silently mis-state the bound.

`t` IS COMPUTED WITHOUT A NEW DEPENDENCY, via `statistics.NormalDist` corrected toward the
Student-t quantile by a third-order Cornish-Fisher expansion (see `_t_critical`) -- not a
lookup table, because the eval's pilot sizes are not fixed in advance and a table would need an
entry, or an interpolation policy, for every one of them.

WHAT THIS MODULE DOES NOT DO: it never makes an API call, reads a credential, or invents a
price. `cost_of` converts already-measured usage into dollars using a price map the CALLER
supplies -- see `biolit.extract.llm`'s "NO PRICES HERE, DELIBERATELY" note for why per-token
prices live nowhere in `biolit/` itself. Every function here consumes numbers that were already
measured; none of them can make an estimate come out wrong by talking to the network.
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from statistics import NormalDist


@dataclass(frozen=True)
class ArmPilot:
    """One arm's pilot-run cost statistics -- already MEASURED, never estimated or hardcoded.

    `mean_cost` and `sd_cost` are per-call dollar costs, typically computed by `cost_of` over a
    pilot's actually-measured usage. `n_pilot` is the pilot sample size (drives the t
    degrees-of-freedom and the `N^2/n` mean-uncertainty term in `authorization_bound`); `n_full`
    is the call count the FULL run will make in this arm (drives the point estimate and the
    `+N` realized-sum term).
    """

    name: str
    mean_cost: float
    sd_cost: float
    n_pilot: int
    n_full: int


@dataclass(frozen=True)
class Bound:
    """The result of `authorization_bound`. A paid run is authorized against `.bound`, never
    against `.point_estimate` alone -- the point estimate carries no margin for the sampling
    noise in the pilot's own `mean_cost`/`sd_cost`. `.inflation` is `(bound - point_estimate) /
    point_estimate`, reported so a reviewer can see at a glance how much margin the bound adds.
    """

    point_estimate: float
    bound: float
    inflation: float


def _t_critical(confidence: float, df: int) -> float:
    """One-sided Student-t critical value, via a Cornish-Fisher-corrected normal quantile.

    `statistics.NormalDist().inv_cdf(confidence)` gives the EXACT quantile of the *normal*
    distribution, but the normal distribution understates the Student-t tail at small degrees
    of freedom -- using it unmodified would silently narrow the authorization bound exactly
    where the pilot is thinnest (small `n_pilot`) and the bound matters most. The third-order
    Cornish-Fisher expansion below corrects the normal quantile `z` toward the true Student-t
    quantile using only `df`, so no new dependency (e.g. scipy) is needed for a value that
    otherwise requires the inverse of the regularized incomplete beta function.

    Verified against the standard one-sided-95% Student-t table: df=24 -> 1.7109 (table 1.711),
    df=29 -> 1.6991 (table 1.699), df=39 -> 1.6849 (table 1.685), df=59 -> 1.6711 (table 1.671)
    -- all four round to the tabulated three-decimal value.
    """
    z = NormalDist().inv_cdf(confidence)
    z3, z5, z7 = z**3, z**5, z**7
    g1 = (z3 + z) / 4
    g2 = (5 * z5 + 16 * z3 + 3 * z) / 96
    g3 = (3 * z7 + 19 * z5 + 17 * z3 - 15 * z) / 384
    return z + g1 / df + g2 / df**2 + g3 / df**3


def authorization_bound(arms: Sequence[ArmPilot], *, confidence: float = 0.95) -> Bound:
    """One-sided prediction bound on the REALIZED TOTAL, not a per-call percentile times N.

    See the module docstring for the formula, and for why this bound covers variance but
    deliberately not bias.
    """
    point_estimate = sum(arm.n_full * arm.mean_cost for arm in arms)
    variance = sum(arm.sd_cost**2 * (arm.n_full**2 / arm.n_pilot + arm.n_full) for arm in arms)
    df = min(arm.n_pilot for arm in arms) - 1
    critical = _t_critical(confidence, df)
    bound = point_estimate + critical * math.sqrt(variance)
    inflation = (bound - point_estimate) / point_estimate
    return Bound(point_estimate=point_estimate, bound=bound, inflation=inflation)


def cost_of(usage: Mapping[str, int], prices: Mapping[str, float]) -> float:
    """Converts measured token counts to a dollar figure, using a price map the CALLER supplies.

    `biolit/extract/llm.py` carries a "NO PRICES HERE, DELIBERATELY" note: per-token pricing
    changes, and a rate committed to the repo rots silently into a wrong estimate. This helper
    is the named conversion seam the spec requires -- tokens read from the API's own usage
    fields, prices pinned at run time -- so it lives in `biolit_evals`, and `prices` is passed
    in rather than hardcoded here.

    Every key in `usage` must have a matching entry in `prices`. A usage field present in the
    API's own payload but missing from the price map is exactly the kind of silent undercount
    this module exists to prevent, so it is a loud `ValueError`, not a skipped term.
    """
    total = 0.0
    for token_field, count in usage.items():
        if token_field not in prices:
            raise ValueError(
                f"cost_of: no price for usage field {token_field!r}; every measured usage "
                "field must be priced, or the total silently undercounts the run."
            )
        total += count * prices[token_field]
    return total


class BudgetExceeded(Exception):
    """Raised by `KillSwitch.record` when cumulative recorded spend exceeds 1.25x its limit."""


@dataclass
class KillSwitch:
    """Aborts a run once cumulative recorded spend exceeds 1.25x `limit`.

    The 25% margin exists so ordinary sampling noise in `authorization_bound` (which is not,
    and does not claim to be, a hard ceiling) does not trip the switch on ordinary variance --
    while still keeping a SYSTEMATIC error (a price mis-pin, a retry loop not being counted)
    from running away across the ~3,600 calls a full run makes. This class only accumulates
    numbers it is handed; it makes no API call and reads no credential.
    """

    limit: float
    _spent: float = field(default=0.0, init=False)

    def record(self, amount: float) -> None:
        """Adds `amount` to the cumulative spend, raising `BudgetExceeded` if it now exceeds
        1.25x `limit`. Strictly EXCEEDS: spend landing exactly on the 1.25x threshold is not
        yet an overrun."""
        self._spent += amount
        threshold = 1.25 * self.limit
        if self._spent > threshold:
            raise BudgetExceeded(
                f"cumulative spend ${self._spent:.2f} exceeds 1.25x the ${self.limit:.2f} "
                f"budget limit (threshold ${threshold:.2f})"
            )
