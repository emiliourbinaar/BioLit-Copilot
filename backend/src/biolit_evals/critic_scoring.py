"""Scoring for the Critic arms: macro-F1, per-class confusion, Wilson intervals, and the
natural-prevalence precision projection.

MACRO, not micro: this eval's drop-rate rule (elsewhere in this project) permits unequal class
N, and a micro-style average would let the largest class dominate the headline figure -- exactly
the failure this fixture is built to catch (see `score`'s docstring).

WILSON, not normal-approximation, for every interval computed here: the report carries
proportions near 0 and 1 (refusal rates, per-class recalls), and the normal interval runs
outside [0, 1] exactly there.
"""

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import comb, sqrt

from biolit.domain.records import ContradictionLabel

_Z95 = 1.959963984540054  # two-sided 95% normal quantile


@dataclass(frozen=True)
class ClassMetrics:
    """One-vs-rest metrics for a single `ContradictionLabel`, over one `score()` call."""

    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int
    # Controller ruling: TN / (TN + FP) over the one-vs-rest collapse for this class. Needed so
    # Task 10's run log can write `natural_prevalence_precision`, which requires BOTH sensitivity
    # (== recall here) and specificity for the same class; computing specificity ad hoc there
    # would duplicate this formula outside the module that owns it.
    specificity: float


@dataclass(frozen=True)
class CriticScores:
    accuracy: float
    macro_f1: float
    per_class: Mapping[ContradictionLabel, ClassMetrics]
    # confusion[gold][pred] = count. Every (gold, pred) cell is present, even when zero.
    confusion: Mapping[ContradictionLabel, Mapping[ContradictionLabel, int]]


def score(gold: Sequence[ContradictionLabel], pred: Sequence[ContradictionLabel]) -> CriticScores:
    """Score critic predictions against gold labels.

    MACRO-averaged: the drop-rate rule permits unequal class N elsewhere in this eval, and a
    micro-style average -- weighting by instance count rather than by class -- would let the
    largest class dominate the headline number. `macro_f1` is the unweighted mean of per-class
    F1 over ALL THREE `ContradictionLabel` members, INCLUDING a class absent from both gold and
    pred for this call (which then contributes f1 = 0.0 by the zero-denominator rule below).
    That is a real choice of denominator -- dropping an absent class instead would silently
    change what the number means, so it is spelled out here rather than left implicit.

    precision / recall / f1 / specificity each return 0.0 rather than raising when their
    denominator is zero (e.g. a class with no predicted or no gold instances at all).
    """
    if len(gold) != len(pred):
        raise ValueError(
            f"score: gold and pred must be the same length, got {len(gold)} and "
            f"{len(pred)}. Scoring against misaligned gold is meaningless."
        )
    confusion: dict[ContradictionLabel, dict[ContradictionLabel, int]] = {
        label: dict.fromkeys(ContradictionLabel, 0) for label in ContradictionLabel
    }
    counts: Counter[tuple[ContradictionLabel, ContradictionLabel]] = Counter(
        zip(gold, pred, strict=True)
    )
    for (g, p), n in counts.items():
        confusion[g][p] = n

    n_total = len(gold)
    correct = sum(confusion[label][label] for label in ContradictionLabel)
    accuracy = correct / n_total if n_total else 0.0

    per_class: dict[ContradictionLabel, ClassMetrics] = {}
    for label in ContradictionLabel:
        tp = confusion[label][label]
        fp = sum(confusion[g][label] for g in ContradictionLabel if g != label)
        fn = sum(confusion[label][p] for p in ContradictionLabel if p != label)
        tn = n_total - tp - fp - fn
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        specificity = tn / (tn + fp) if (tn + fp) else 0.0
        per_class[label] = ClassMetrics(
            precision=precision,
            recall=recall,
            f1=f1,
            tp=tp,
            fp=fp,
            fn=fn,
            specificity=specificity,
        )

    macro_f1 = sum(m.f1 for m in per_class.values()) / len(ContradictionLabel)
    return CriticScores(
        accuracy=accuracy, macro_f1=macro_f1, per_class=per_class, confusion=confusion
    )


def project_to_prevalence(sensitivity: float, specificity: float, prevalence: float) -> float:
    """Precision the measured arm would have at NATURAL prevalence.

    Balanced sampling is forced by this eval's low base rate (a natural sample scores ~97.5%
    by always answering `agreement`), but a figure measured on a balanced sample then
    OVERSTATES deployment performance -- the favourably-selected-population trap this project
    has recorded prior instances of. This projection is reported beside every balanced figure
    for that reason; it is not optional supplementary reporting.
    """
    tp = sensitivity * prevalence
    fp = (1.0 - specificity) * (1.0 - prevalence)
    return tp / (tp + fp) if tp + fp else 0.0


@dataclass(frozen=True)
class McNemarResult:
    """Outcome of a paired `mcnemar()` comparison. `b` and `c` are the two discordant-pair
    counts (arm-a-only-correct, arm-b-only-correct); `statistic` is |b - c|."""

    b: int
    c: int
    statistic: float
    p_value: float


def mcnemar(correct_a: Sequence[bool], correct_b: Sequence[bool]) -> McNemarResult:
    """Paired comparison of two arms over the SAME units.

    VALID FOR EVERY ARM PAIR IN THIS EVAL -- both LLM input modes, the direction arm, and the
    three baselines -- because each emits a label for every one of the same pairs. The
    direction arm's per-paper calls are composed into pair labels BEFORE scoring, so once
    composed it is the same units as everything else.

    NOT valid for the direction arm's per-paper diagnostics, whose unit is a paper rather than
    a pair. Those are reported on their own terms.

    EXACT BINOMIAL, not the chi-square approximation: with b+c often under 25 on the
    contradiction class and on the human-labelled subsample, chi-square is unreliable exactly
    where the comparison matters most.

    Only discordant pairs carry information -- concordant pairs (both arms right, or both
    wrong, on the same unit) cancel and are not counted. `b == c == 0` returns `p_value = 1.0`:
    no discordant pairs means no information to test. The branch is NOT required to avoid a
    division by zero -- `2**n` is never zero for n >= 0, and the general formula returns this
    exact value at n = 0 anyway. It is kept as an explicit, documented statement of the
    no-discordance case, and is provably equivalent to falling through (ADR-0014 category (c):
    flagged, deliberately untested, deliberately not deleted).
    """
    if len(correct_a) != len(correct_b):
        raise ValueError(
            f"mcnemar: correct_a and correct_b must be the same length, got {len(correct_a)} "
            f"and {len(correct_b)}. A paired test over misaligned arms is meaningless."
        )
    b = sum(1 for x, y in zip(correct_a, correct_b, strict=True) if x and not y)
    c = sum(1 for x, y in zip(correct_a, correct_b, strict=True) if y and not x)
    n = b + c
    if n == 0:
        return McNemarResult(b=0, c=0, statistic=0.0, p_value=1.0)
    tail = sum(comb(n, i) for i in range(min(b, c) + 1)) / (2**n)
    return McNemarResult(b=b, c=c, statistic=float(abs(b - c)), p_value=min(1.0, 2 * tail))


def wilson_interval(k: int, n: int) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion k / n.

    WILSON, not normal-approximation: this report's proportions -- refusal rates, per-class
    recalls -- sit near 0 and 1, exactly where a normal interval overshoots outside [0, 1]. The
    endpoints are exact 0.0 (k == 0) and exact 1.0 (k == n): those are the interval's true
    closed-form limits at phat == 0 and phat == 1, not a clamp papering over floating-point
    drift, so they are computed directly rather than via the general formula, which can miss
    the exact value by a rounding ulp when the two paths to it (division vs. sqrt-then-multiply)
    don't cancel bit-for-bit.
    """
    if n == 0:
        return (0.0, 1.0)
    phat = k / n
    denom = 1 + _Z95 * _Z95 / n
    center = phat + _Z95 * _Z95 / (2 * n)
    margin = _Z95 * sqrt(phat * (1 - phat) / n + _Z95 * _Z95 / (4 * n * n))
    lo = 0.0 if k == 0 else (center - margin) / denom
    hi = 1.0 if k == n else (center + margin) / denom
    return (lo, hi)
