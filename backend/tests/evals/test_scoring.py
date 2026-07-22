import pytest

from biolit.domain.records import Entity
from biolit_evals.scoring import PRF, score_corpus


def _e(start, end, label):
    return Entity(text="x", label=label, start=start, end=end)


def test_perfect_match():
    gold = [_e(0, 9, "CHEMICAL")]
    prf = score_corpus([(gold, list(gold))])
    assert prf == PRF(1.0, 1.0, 1.0, tp=1, fp=0, fn=0)


def test_boundary_off_is_wrong():
    gold = [_e(0, 9, "CHEMICAL")]
    pred = [_e(0, 8, "CHEMICAL")]  # off-by-one end
    prf = score_corpus([(gold, pred)])
    assert (prf.tp, prf.fp, prf.fn) == (0, 1, 1)


def test_wrong_type_is_wrong():
    gold = [_e(0, 4, "DISEASE")]
    pred = [_e(0, 4, "CHEMICAL")]
    prf = score_corpus([(gold, pred)])
    assert (prf.tp, prf.fp, prf.fn) == (0, 1, 1)


def test_missing_and_spurious():
    prf = score_corpus([([_e(0, 4, "DISEASE")], []), ([], [_e(0, 4, "CHEMICAL")])])
    assert (prf.tp, prf.fp, prf.fn) == (0, 1, 1)


def test_micro_average_across_examples():
    # Two examples each with the same (0,4,DISEASE) span must NOT collide/dedupe.
    gold = [_e(0, 4, "DISEASE")]
    prf = score_corpus([(gold, list(gold)), (gold, list(gold))])
    assert (prf.tp, prf.fp, prf.fn) == (2, 0, 0)
    assert prf.f1 == 1.0


def test_empty_corpus_is_zero():
    prf = score_corpus([])
    assert prf == PRF(0.0, 0.0, 0.0, 0, 0, 0)


def test_duplicate_predicted_span_is_not_deduped():
    # Set semantics would collapse two identical (start, end, label) predictions into
    # one element, silently dropping the duplicate false positive. Multiset (Counter)
    # semantics must count it: 1 TP (matches the single gold span) + 1 FP (the extra
    # duplicate prediction has nothing left in gold to match).
    gold = [_e(0, 4, "DISEASE")]
    pred = [_e(0, 4, "DISEASE"), _e(0, 4, "DISEASE")]
    prf = score_corpus([(gold, pred)])
    assert (prf.tp, prf.fp, prf.fn) == (1, 1, 0)


def test_f1_is_harmonic_mean_not_arithmetic_mean():
    # precision != recall here, so a transposed formula or an arithmetic-mean mix-up
    # (which every other test's 1.0/1.0 or 0.0/0.0 case cannot distinguish from the
    # correct harmonic mean) would produce a different, wrong f1 value.
    gold = [_e(0, 1, "A"), _e(1, 2, "A")]
    pred = [_e(0, 1, "A"), _e(1, 2, "A"), _e(2, 3, "A")]
    prf = score_corpus([(gold, pred)])
    assert (prf.tp, prf.fp, prf.fn) == (2, 1, 0)
    assert prf.precision == pytest.approx(2 / 3)
    assert prf.recall == pytest.approx(1.0)
    assert prf.f1 == pytest.approx(0.8)


def test_pure_false_positives_no_gold_at_all():
    # No gold anywhere in the corpus, so recall's guard (tp+fn == 0) fires, while
    # precision still takes the real-division path (denominator > 0, numerator 0).
    # Isolates the recall guard from the empty-corpus test, which trips every guard.
    pred = [_e(0, 4, "DISEASE"), _e(5, 9, "CHEMICAL")]
    prf = score_corpus([([], pred)])
    assert (prf.tp, prf.fp, prf.fn) == (0, 2, 0)
    assert prf.precision == pytest.approx(0.0)
    assert prf.recall == pytest.approx(0.0)
    assert prf.f1 == pytest.approx(0.0)


def test_pure_false_negatives_no_predictions_at_all():
    # No predictions anywhere in the corpus, so precision's guard (tp+fp == 0) fires,
    # while recall still takes the real-division path (denominator > 0, numerator 0).
    # Isolates the precision guard from the empty-corpus test, which trips every guard.
    gold = [_e(0, 4, "DISEASE"), _e(5, 9, "CHEMICAL")]
    prf = score_corpus([(gold, [])])
    assert (prf.tp, prf.fp, prf.fn) == (0, 0, 2)
    assert prf.precision == pytest.approx(0.0)
    assert prf.recall == pytest.approx(0.0)
    assert prf.f1 == pytest.approx(0.0)
