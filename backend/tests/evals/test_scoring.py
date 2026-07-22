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
