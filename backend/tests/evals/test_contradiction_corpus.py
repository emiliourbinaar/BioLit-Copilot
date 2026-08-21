from biolit.domain.records import ContradictionLabel
from biolit_evals.contradiction_corpus import drop_report, usable_pairs
from biolit_evals.contradiction_gold import GoldPair


def test_drop_report_counts_per_class_not_just_in_aggregate():
    """Limitation 5: differential availability by class is a confound, so the rate must be
    visible per class. An aggregate number cannot show it."""
    pairs = [
        GoldPair(
            "1",
            "2",
            "C000001",
            "D000001",
            ContradictionLabel.contradiction,
            "marker/mechanism",
            "therapeutic",
        ),
        GoldPair(
            "3",
            "4",
            "C000002",
            "D000002",
            ContradictionLabel.agreement,
            "therapeutic",
            "therapeutic",
        ),
    ]
    abstracts = {"1": "a", "2": "b", "3": "c"}  # paper 4 missing
    report = drop_report(pairs, abstracts)
    assert report.per_class["contradiction"] == (1, 1)
    assert report.per_class["agreement"] == (0, 1)


def test_pair_missing_only_its_SECOND_abstract_is_dropped():
    """`a in abstracts and b in abstracts` -- the b-side conjunct."""
    pairs = [
        GoldPair(
            "1",
            "2",
            "C000001",
            "D000001",
            ContradictionLabel.contradiction,
            "marker/mechanism",
            "therapeutic",
        )
    ]
    assert usable_pairs(pairs, {"1": "only a"}) == []


def test_pair_missing_only_its_FIRST_abstract_is_also_dropped():
    """`a in abstracts and b in abstracts` -- the a-side conjunct. Neither of the two tests
    above exercises this on its own: the first (per-class) test's dropped pair has its
    a-side present and its b-side missing, and the SECOND-abstract test above is the same
    shape. Mutation-verified: weakening the guard to `p.paper_id_b in abstracts` alone left
    both of those tests green, so this test is required to catch that regression."""
    pairs = [
        GoldPair(
            "1",
            "2",
            "C000001",
            "D000001",
            ContradictionLabel.contradiction,
            "marker/mechanism",
            "therapeutic",
        )
    ]
    assert usable_pairs(pairs, {"2": "only b"}) == []


def test_drop_report_drops_a_pair_missing_only_its_first_abstract():
    """The same `a in abstracts and b in abstracts` guard is duplicated inline inside
    drop_report -- it is not implemented in terms of usable_pairs -- so its own a-side
    conjunct needs its own witness. Mutation-verified: weakening drop_report's guard to
    `pair.paper_id_b in abstracts` alone left test_drop_report_counts_per_class... green,
    because that test's only dropped pair already had its a-side present."""
    pairs = [
        GoldPair(
            "1",
            "2",
            "C000001",
            "D000001",
            ContradictionLabel.contradiction,
            "marker/mechanism",
            "therapeutic",
        )
    ]
    report = drop_report(pairs, {"2": "only b"})
    assert report.per_class["contradiction"] == (0, 1)
