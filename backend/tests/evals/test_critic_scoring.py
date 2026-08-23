import pytest

from biolit.domain.records import ContradictionLabel
from biolit_evals.critic_scoring import mcnemar, project_to_prevalence, score, wilson_interval


def test_macro_f1_averages_classes_not_instances():
    """Macro because the drop-rate rule permits unequal N; micro would let the largest class
    dominate the headline."""
    gold = [ContradictionLabel.contradiction] * 2 + [ContradictionLabel.agreement] * 6
    pred = [ContradictionLabel.agreement] * 8
    scores = score(gold, pred)
    assert scores.accuracy == pytest.approx(0.75)
    assert scores.macro_f1 < 0.45  # micro-style weighting would read far higher


def test_projection_to_natural_prevalence_reproduces_the_spec_figure():
    """The spec's worked illustration: 80% sensitivity, 95% specificity, 2.52% prevalence
    yields precision 0.29 -- seven in ten flagged contradictions would be false."""
    assert project_to_prevalence(0.80, 0.95, 0.0252) == pytest.approx(0.2926, abs=1e-4)


def test_wilson_interval_is_bounded_and_tightens_with_n():
    lo_small, hi_small = wilson_interval(15, 30)
    lo_big, hi_big = wilson_interval(150, 300)
    assert 0.0 <= lo_small < 0.5 < hi_small <= 1.0
    assert (hi_big - lo_big) < (hi_small - lo_small)
    assert wilson_interval(0, 10)[0] == 0.0
    assert wilson_interval(10, 10)[1] == 1.0


def test_specificity_is_the_true_negative_rate_not_recall_restated():
    """Controller ruling: ClassMetrics carries specificity = TN / (TN + FP) over the one-vs-rest
    collapse. Non-degenerate fixture (2/3) chosen so it cannot coincidentally equal precision
    or recall on the same class, which are both 1/2 here."""
    gold = [
        ContradictionLabel.contradiction,  # idx0: TP
        ContradictionLabel.contradiction,  # idx1: FN (predicted agreement)
        ContradictionLabel.agreement,  # idx2: FP (predicted contradiction)
        ContradictionLabel.agreement,  # idx3: TN
        ContradictionLabel.insufficient_overlap,  # idx4: TN
    ]
    pred = [
        ContradictionLabel.contradiction,
        ContradictionLabel.agreement,
        ContradictionLabel.contradiction,
        ContradictionLabel.agreement,
        ContradictionLabel.insufficient_overlap,
    ]
    scores = score(gold, pred)
    contradiction_metrics = scores.per_class[ContradictionLabel.contradiction]
    assert contradiction_metrics.precision == pytest.approx(0.5)
    assert contradiction_metrics.recall == pytest.approx(0.5)
    assert contradiction_metrics.specificity == pytest.approx(2 / 3)


def test_a_class_absent_from_both_gold_and_pred_still_divides_macro_f1_by_three():
    """Design point: macro_f1 averages over all three ContradictionLabel members, including one
    with zero instances anywhere in this call -- it contributes f1 = 0.0 rather than being
    dropped from the denominator. Reusing the brief's own fixture, where insufficient_overlap
    never appears in gold or pred."""
    gold = [ContradictionLabel.contradiction] * 2 + [ContradictionLabel.agreement] * 6
    pred = [ContradictionLabel.agreement] * 8
    scores = score(gold, pred)
    absent = scores.per_class[ContradictionLabel.insufficient_overlap]
    assert (absent.tp, absent.fp, absent.fn) == (0, 0, 0)
    assert absent.f1 == 0.0
    # If the absent class were dropped instead of counted as 0.0, macro_f1 would be the mean of
    # only the two present classes' f1 -- a larger number than dividing by three.
    absent_label = ContradictionLabel.insufficient_overlap
    present_only_mean = (
        sum(m.f1 for label, m in scores.per_class.items() if label != absent_label) / 2
    )
    assert scores.macro_f1 < present_only_mean


def test_specificity_is_zero_not_error_when_the_class_is_the_entire_gold_set():
    """Zero-denominator branch of specificity: when every gold instance IS this class, there
    are no negatives at all (TN + FP == 0), so specificity is undefined by the TN/(TN+FP)
    formula -- the guard returns 0.0 rather than raising ZeroDivisionError."""
    gold = [ContradictionLabel.contradiction] * 3
    pred = [
        ContradictionLabel.contradiction,
        ContradictionLabel.agreement,
        ContradictionLabel.contradiction,
    ]
    scores = score(gold, pred)
    assert scores.per_class[ContradictionLabel.contradiction].specificity == 0.0


def test_score_of_empty_sequences_is_all_zero_not_an_error():
    """Zero-denominator branch of accuracy (and, transitively, every per-class metric): an
    empty gold/pred pair has n_total == 0, which must not raise ZeroDivisionError."""
    scores = score([], [])
    assert scores.accuracy == 0.0
    assert scores.macro_f1 == 0.0


def test_wilson_interval_of_zero_trials_is_the_full_unit_interval():
    """n == 0 branch: no trials means no information, so the interval is maximally wide
    rather than dividing by zero."""
    assert wilson_interval(0, 0) == (0.0, 1.0)


def test_projection_is_zero_not_error_when_sensitivity_and_specificity_predict_nothing_positive():
    """Zero-denominator branch: sensitivity 0.0 and specificity 1.0 means TP == FP == 0 --
    the arm never flags anything positive at natural prevalence either."""
    assert project_to_prevalence(0.0, 1.0, 0.0252) == 0.0


@pytest.mark.parametrize("n", [3, 7])
def test_wilson_lower_bound_guard_is_load_bearing_at_n_where_the_general_formula_does_not_cancel(
    n,
):
    """At phat = 0, `center` and `margin` are mathematically identical (both reduce to
    z^2 / (2n)), so `center - margin` cancels to exactly 0.0 for EVERY n in real arithmetic --
    the k==0 guard is not needed by the mathematics. What this test actually proves is
    narrower: that under THIS module's specific operation order (division for `center`'s
    term vs. sqrt-then-multiply for `margin`'s), rounding does NOT cancel bit-for-bit at these
    particular n, so the guard is load-bearing against a mutant that deletes it -- for this
    formula's rounding behaviour, not for every algebraically equivalent rewrite of it. A swept
    check (n = 3, 7, 11, 23, 97) found only 3 and 7 leave a nonzero residue here; 11, 23, and 97
    cancel exactly under this ordering too and would not distinguish the mutant, so they are not
    used as witnesses. A future reordering of this formula could silently make even n=3 and n=7
    stop proving anything, with no failing test to signal the loss -- this is a known limit of
    pinning floating-point rounding behaviour rather than the underlying mathematics."""
    assert wilson_interval(0, n)[0] == 0.0


def test_score_rejects_gold_and_pred_of_different_lengths():
    with pytest.raises(ValueError, match="same length"):
        score(
            [ContradictionLabel.agreement],
            [ContradictionLabel.agreement, ContradictionLabel.agreement],
        )


def test_mcnemar_ignores_pairs_the_arms_agree_on():
    """Only discordant pairs carry information; agreements cancel. Seven elements."""
    a = [True, True, True, True, True, True, True]
    b = [False, False, False, True, True, True, True]
    result = mcnemar(a, b)
    assert (result.b, result.c) == (3, 0)
    # n = 3, min(b, c) = 0: one-sided tail = C(3,0)/2**3 = 1/8; doubled = 0.25 exactly.
    assert result.p_value == pytest.approx(0.25)


def test_identical_arms_are_not_distinguishable():
    """No discordant pairs at all -- b == c == 0 -- returns p_value 1.0 rather than dividing by
    zero. Seven elements, mixing True/False so this also exercises the x-and-not-y /
    y-and-not-x conditions with x == y == False present, not just x == y == True."""
    same = [True, False, True, True, False, True, False]
    result = mcnemar(same, list(same))
    assert (result.b, result.c) == (0, 0)
    assert result.p_value == 1.0


def test_mcnemar_rejects_arms_over_different_unit_counts():
    """Controller ruling (carried from Task 8's review): mcnemar's length guard must match
    score's shape -- same structure, same style of message, naming both lengths and why they
    must match. Matched here on the same real substring `score`'s own test matches on."""
    with pytest.raises(ValueError, match="same length"):
        mcnemar([True, False], [True])


def test_mcnemar_counts_discordant_pairs_in_both_directions():
    """ADR-0014: the two compound conditions (`x and not y` feeding b, `y and not x` feeding
    c) each need both operands independently exercised. Earlier fixtures never produced a
    discordant pair favouring arm b (x False, y True), so a bug that always returned c == 0
    would have passed every prior test undetected. Seven elements, b != c so the two counts
    cannot be swapped without the assertion catching it."""
    a = [True, True, False, False, True, True, False]
    b = [False, False, True, True, True, False, False]
    result = mcnemar(a, b)
    assert (result.b, result.c) == (3, 2)


def test_mcnemar_p_value_is_capped_at_one_when_doubling_the_tail_overshoots():
    """Design point: doubling a one-sided tail can exceed 1.0 when b == c > 0 (unlike the
    b == c == 0 case, which is handled by its own early return). b == c cannot distinguish the
    b/c formulas from each other -- that is covered by the other fixtures above -- but this
    fixture is testing the min(1.0, ...) cap, not which array is which."""
    a = [True, True, False, True, True, True, True]
    b = [False, True, True, True, True, True, True]
    result = mcnemar(a, b)
    assert (result.b, result.c) == (1, 1)
    assert result.p_value == 1.0
