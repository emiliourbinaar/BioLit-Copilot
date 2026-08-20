from biolit.domain.records import ContradictionLabel
from biolit_evals.contradiction_gold import build_candidates, label_for_directions
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


def test_excluded_pmid_drops_its_pairs():
    """p2 is a BC5CDR training pmid. Without the exclusion it would pair with p1 and p3
    (all three share key C1/D1); with it, no candidate may name p2."""
    directions = {
        "p1": {("C1", "D1"): _MM},
        "p2": {("C1", "D1"): _MM},
        "p3": {("C1", "D1"): _MM},
    }
    candidates = build_candidates(directions, excluded={"p2"})
    named_papers = {pid for c in candidates for pid in (c.paper_id_a, c.paper_id_b)}
    assert "p2" not in named_papers
    assert named_papers == {"p1", "p3"}


def test_curated_key_pairs_get_both_endpoint_ids_set():
    """A co-keyed pair with agreeing directions and one with opposing directions must both
    carry the full (chemical, disease) key -- that is what distinguishes them from a hard
    negative, which has exactly one endpoint set."""
    directions = {
        "p1": {("C1", "D1"): _MM},
        "p2": {("C1", "D1"): _MM},
        "p3": {("C2", "D2"): _MM},
        "p4": {("C2", "D2"): _TH},
    }
    candidates = build_candidates(directions, excluded=set())
    agreement = next(c for c in candidates if c.label is ContradictionLabel.agreement)
    contradiction = next(c for c in candidates if c.label is ContradictionLabel.contradiction)
    assert (agreement.chemical_id, agreement.disease_id) == ("C1", "D1")
    assert (contradiction.chemical_id, contradiction.disease_id) == ("C2", "D2")


def test_hard_negative_has_exactly_one_endpoint_id_and_no_directions():
    """p1 and p2 share chemical C1 (via different diseases) but no curated key joins them --
    that is the definition of insufficient_overlap. Exactly one endpoint id is set."""
    directions = {
        "p1": {("C1", "D1"): _MM},
        "p2": {("C1", "D2"): _TH},
    }
    candidates = build_candidates(directions, excluded=set())
    assert len(candidates) == 1
    negative = candidates[0]
    assert negative.label is ContradictionLabel.insufficient_overlap
    assert (negative.chemical_id is None) != (negative.disease_id is None)
    assert negative.chemical_id == "C1"
    assert negative.direction_a is None
    assert negative.direction_b is None


def test_max_per_key_caps_emission():
    """8 papers on one key, all agreeing, form 28 combinations -- real CTD has ~2.78M such
    pairs on its most popular keys. Only max_per_key may be materialised."""
    directions = {f"p{i}": {("C1", "D1"): _MM} for i in range(8, 0, -1)}
    candidates = build_candidates(directions, excluded=set(), max_per_key=3)
    assert len(candidates) == 3
    assert all(c.label is ContradictionLabel.agreement for c in candidates)


def test_build_candidates_is_deterministic():
    directions = {
        f"p{i}": {("C1", "D1"): _MM} if i % 2 else {("C1", "D1"): _TH} for i in range(7, 0, -1)
    }
    first = build_candidates(directions, excluded=set())
    second = build_candidates(directions, excluded=set())
    assert first == second
