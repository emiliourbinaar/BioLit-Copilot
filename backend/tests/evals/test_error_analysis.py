from biolit.domain.records import Entity
from biolit_evals.error_analysis import analyze


def test_analyze_empty_pairs_reports_all_zero():
    report = analyze([])
    assert report.n_fn == 0
    assert report.n_fp == 0
    assert report.fn_overlapping == 0
    assert report.fp_overlapping == 0
    assert report.fn_common == []
    assert report.fp_common == []


def test_analyze_exact_matches_are_not_errors():
    gold = [Entity(text="metformin", label="CHEMICAL", start=0, end=9)]
    pred = [Entity(text="metformin", label="CHEMICAL", start=0, end=9)]
    report = analyze([(gold, pred)])
    assert report.n_fn == 0
    assert report.n_fp == 0


def test_analyze_counts_pure_miss_and_pure_hallucination_as_non_overlapping():
    # Gold "PCOS" at 0:4 is missed entirely; predicted "cancer" at 20:26 has no gold
    # counterpart anywhere nearby -- neither error touches the other's span.
    gold = [Entity(text="PCOS", label="DISEASE", start=0, end=4)]
    pred = [Entity(text="cancer", label="DISEASE", start=20, end=26)]
    report = analyze([(gold, pred)])
    assert report.n_fn == 1
    assert report.n_fp == 1
    assert report.fn_overlapping == 0
    assert report.fp_overlapping == 0


def test_analyze_counts_boundary_disagreement_as_overlapping_on_both_sides():
    # Gold "CFD" at 0:3 vs predicted "CF" at 0:2 -- classic boundary-disagreement
    # case (see docs/EVAL_REPORT.md "Interpreting the gap"): the model found
    # *something* in the right place, but a strict exact-span matcher scores this
    # as one FN (missed "CFD") and one FP (spurious "CF"), and the two overlap in
    # character range.
    gold = [Entity(text="CFD", label="CHEMICAL", start=0, end=3)]
    pred = [Entity(text="CF", label="CHEMICAL", start=0, end=2)]
    report = analyze([(gold, pred)])
    assert report.n_fn == 1
    assert report.n_fp == 1
    assert report.fn_overlapping == 1
    assert report.fp_overlapping == 1


def test_analyze_most_common_ranks_repeated_surface_label_pairs_first():
    pairs = [
        (
            [Entity(text="CFD", label="CHEMICAL", start=0, end=3)],
            [Entity(text="CF", label="CHEMICAL", start=0, end=2)],
        ),
        (
            [Entity(text="CFD", label="CHEMICAL", start=10, end=13)],
            [Entity(text="CF", label="CHEMICAL", start=10, end=12)],
        ),
        (
            [Entity(text="PCOS", label="DISEASE", start=0, end=4)],
            [],
        ),
    ]
    report = analyze(pairs)
    assert report.fn_common[0] == (("CFD", "CHEMICAL"), 2)
    assert report.fp_common[0] == (("CF", "CHEMICAL"), 2)
