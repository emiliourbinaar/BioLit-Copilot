from biolit.domain.records import ContradictionLabel
from biolit_evals.contradiction_gold import label_for_directions
from biolit_evals.ctd_directions import Direction

_MM = frozenset({Direction.marker_mechanism})
_TH = frozenset({Direction.therapeutic})


def test_opposite_directions_are_a_contradiction():
    assert label_for_directions(_MM, _TH) is ContradictionLabel.contradiction
    assert label_for_directions(_TH, _MM) is ContradictionLabel.contradiction


def test_same_direction_is_agreement():
    assert label_for_directions(_MM, _MM) is ContradictionLabel.agreement
    assert label_for_directions(_TH, _TH) is ContradictionLabel.agreement


def test_paper_with_both_directions_on_one_key_is_not_gold():
    assert label_for_directions(_MM | _TH, _TH) is None


def test_both_directions_on_the_SECOND_paper_is_also_not_gold():
    """The second disjunct. Without it a b-side double-direction pair is labelled
    `contradiction` because the frozensets merely differ."""
    assert label_for_directions(_MM, _MM | _TH) is None
