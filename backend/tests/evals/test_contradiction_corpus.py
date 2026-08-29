import zipfile

import pytest

from biolit.domain.records import ContradictionLabel
from biolit_evals.contradiction_corpus import (
    ClassOutcome,
    available_abstracts,
    bc5cdr_pmids_from_zip,
    compose_corpus,
    drop_report,
    usable_pairs,
)
from biolit_evals.contradiction_gold import GoldPair
from biolit_evals.mesh_gold_download import (
    DEVELOPMENT_MEMBER,
    TEST_MEMBER,
    TRAINING_MEMBER,
)


def _contradiction(a: str, b: str) -> GoldPair:
    """A contradiction pair with a key unique to its papers, so nothing but the abstract
    filter and the per-class cap can remove it during composition."""
    return GoldPair(
        a,
        b,
        f"C{a}",
        f"D{a}",
        ContradictionLabel.contradiction,
        "marker/mechanism",
        "therapeutic",
    )


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


def test_year_by_class_records_only_pmids_present_in_years_without_crashing():
    """`if years and pmid in years` -- both operands. `years` truthy (a real mapping was
    passed) is exercised by paper "1" being recorded at all; `pmid in years` is exercised by
    paper "2" being absent from `years` yet not crashing and not appearing in the output.
    Limitation 5's confound check depends on this field: an unpopulated or crashing
    year_by_class would silently hide a class-correlated availability skew."""
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
    abstracts = {"1": "a", "2": "bb"}
    years = {"1": 1990}  # paper "2" deliberately absent from years
    report = drop_report(pairs, abstracts, years)
    assert report.year_by_class["contradiction"] == [1990]


def test_length_by_class_records_kept_pairs_abstract_lengths():
    """length_by_class is populated on every drop_report run (no injected `years` needed to
    exercise it) but no prior test asserted on it -- only per_class was checked. Limitation
    5's confound check depends on this field just as much as year_by_class: a length skew by
    class is the same kind of availability confound a year skew is."""
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
    abstracts = {"1": "abc", "2": "de"}
    report = drop_report(pairs, abstracts)
    assert report.length_by_class["contradiction"] == [3, 2]


def _agreement(a: str, b: str) -> GoldPair:
    return GoldPair(
        a,
        b,
        f"C{a}",
        f"D{a}",
        ContradictionLabel.agreement,
        "therapeutic",
        "therapeutic",
    )


def test_the_per_class_cap_is_counted_PER_CLASS_not_globally():
    """`per_class` names a per-class quota, and the spec's rule table reasons about each
    class's resulting N independently ("leave other classes at 300"). A single global
    counter satisfies the recorded-order test above -- that pool holds one class only -- but
    here it would stop after two pairs total and return the two contradictions, silently
    starving every class after the first in the pool order.

    Macro-averaged metrics make this failure quiet rather than loud: a class present at
    n=0 does not obviously misreport, it just vanishes from the average."""
    pairs = [
        _contradiction("1", "2"),
        _contradiction("3", "4"),
        _agreement("5", "6"),
        _agreement("7", "8"),
    ]
    abstracts = {str(i): "text" for i in range(1, 9)}
    composed = compose_corpus(pairs, abstracts, per_class=2)
    assert composed.pairs == pairs


@pytest.mark.parametrize(
    ("n_usable", "expected"),
    [
        (300, ClassOutcome.at_target),
        (200, ClassOutcome.accepted_smaller),
        (199, ClassOutcome.below_floor),
    ],
)
def test_the_pre_committed_drop_rule_classifies_each_class_by_its_RESULTING_N(
    n_usable: int, expected: ClassOutcome
):
    """The spec's rule table, stated deliberately in terms of resulting N rather than drop
    rate -- "defining materiality on the drop rate would leave the judgement call exactly
    where it must not be". These three cases are the table's three rows.

    200 and 199 are the discriminating pair: the floor is INCLUSIVE, so a class landing on
    exactly 200 is accepted at its smaller N, and only 199 triggers topping up. A `> floor`
    implementation instead of `>= floor` agrees with this test everywhere except at exactly
    200, so the boundary needs both sides or the operator is unpinned.

    The floor is an absolute N, not a fraction of the target: the spec argues it from
    interval width at p = 0.5 (+/-0.069 at N=200 vs +/-0.057 at N=300, against +/-0.098 at
    N=100, where the eval can no longer separate arms differing by ~0.10)."""
    pairs = [_contradiction(str(2 * i), str(2 * i + 1)) for i in range(n_usable)]
    abstracts = {p: "text" for pair in pairs for p in (pair.paper_id_a, pair.paper_id_b)}
    composed = compose_corpus(pairs, abstracts, per_class=300, floor=200)
    assert composed.n_by_class["contradiction"] == n_usable
    assert composed.outcome_by_class["contradiction"] is expected


def test_a_class_landing_short_does_NOT_downsample_the_classes_that_reached_target():
    """The spec's rule table is explicit: when a class lands under target, "accept the
    smaller N, document it, LEAVE OTHER CLASSES AT 300. Do not downsample the others."

    Equalising to the smallest surviving class is the intuitive thing to write and is wrong
    here, because metrics are macro-averaged precisely so unequal class sizes cannot
    reweight the headline -- the balance the equalising implementation would be buying has
    already been bought a different way. Worse, it fails quietly: every class would still be
    non-empty and the run would simply be smaller than it needed to be, throwing away
    fetched abstracts and widening every interval with nothing to show for it."""
    pairs = [
        _contradiction("1", "2"),
        _contradiction("3", "4"),
        _contradiction("5", "6"),
        _agreement("7", "8"),
        _agreement("9", "10"),
        _agreement("11", "12"),
    ]
    abstracts = {str(i): "text" for i in range(1, 13) if i != 6}  # one contradiction short
    composed = compose_corpus(pairs, abstracts, per_class=3)
    kept = [p for p in composed.pairs]
    assert sum(1 for p in kept if p.label is ContradictionLabel.contradiction) == 2
    assert sum(1 for p in kept if p.label is ContradictionLabel.agreement) == 3


def test_composition_takes_the_first_n_usable_pairs_in_RECORDED_order():
    """The spec's topping-up rule: draw a pre-ordered pool, drop pairs whose abstracts are
    missing, and take the first `per_class` IN THE RECORDED ORDER -- topping up means
    consuming more of a pre-ordered list, never re-drawing. A composition that took the LAST
    n usable, or re-sorted by pmid, or re-shuffled, would return the right COUNT with the
    right label, so a count-only assertion cannot distinguish any of them. This pins which
    pairs are taken, and in what order.

    The pool order here (50, 10, 30, 20) is deliberately not sorted, so a sort-by-pmid
    implementation returns [pd, pc] and a take-the-last implementation returns [pc, pd] --
    both different from the correct [pa, pc]."""
    pa, pb, pc, pd = (
        _contradiction("50", "51"),
        _contradiction("10", "11"),
        _contradiction("30", "31"),
        _contradiction("20", "21"),
    )
    pool = [pa, pb, pc, pd]
    abstracts = {p: "text" for p in ("50", "51", "30", "31", "20", "21")}  # "11" missing
    assert compose_corpus(pool, abstracts, per_class=2).pairs == [pa, pc]


def test_a_class_wiped_out_entirely_reports_zero_rather_than_vanishing():
    """The counters are seeded from the POOL's classes, not the survivors'. Seeding from
    survivors is the natural way to write it and loses this case: a class every one of whose
    pairs failed the abstract filter would be absent from `n_by_class` and
    `outcome_by_class` altogether, so the drop rule would be applied to a report that simply
    does not mention it -- and `below_floor`, the row that halts the design, is exactly the
    row a wiped-out class belongs in.

    A KeyError at the point of use would at least be loud. The quiet reading is worse: code
    that iterates the report and finds every class it contains at or above the floor
    concludes the corpus is fine.

    Mutation-verified: seeding the counter from the survivors (`Counter()`) instead of the
    pool raises KeyError here and leaves the other 12 tests green."""
    pairs = [_contradiction("1", "2"), _agreement("3", "4")]
    abstracts = {"1": "text", "2": "text"}  # the agreement pair loses both abstracts
    composed = compose_corpus(pairs, abstracts, per_class=1, floor=1)
    assert composed.n_by_class["agreement"] == 0
    assert composed.outcome_by_class["agreement"] is ClassOutcome.below_floor


def test_excluded_pmids_come_from_ALL_THREE_bc5cdr_splits_not_only_test(tmp_path):
    """The anchor says no sampled PMID may appear in BC5CDR's 1,500 -- because the NER
    checkpoint was fine-tuned on that corpus, so a paper from ANY split is contaminated, not
    just the test split.

    Reading only CDR_TestSet is the natural mistake: every other consumer of CDR_Data.zip in
    this project defaults to the test member, and mesh_gold_download's three loaders all take
    `member=TEST_MEMBER` by default. That would exclude 500 pmids and leave 1,000 contaminated
    ones eligible for sampling -- and the anchor would still pass, because it checks the pool
    against whatever set it was given. A too-small exclusion set makes the anchor agree with
    itself, which is ADR-0016 rule 4's "a check that cannot run" wearing a different hat.

    Each split here contributes exactly one document, so a loader reading one member returns
    one pmid and a loader reading all three returns three."""
    zip_path = tmp_path / "CDR_Data.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for member, pmid in (
            (TRAINING_MEMBER, "111"),
            (DEVELOPMENT_MEMBER, "222"),
            (TEST_MEMBER, "333"),
        ):
            zf.writestr(member, f"{pmid}|t|Title here\n{pmid}|a|Abstract here.\n\n")
    assert bc5cdr_pmids_from_zip(zip_path) == frozenset({"111", "222", "333"})


def test_a_paper_returned_without_abstract_text_counts_as_MISSING():
    """PubMed answers for a pmid it knows even when that record carries no abstract -- older
    papers, editorials, letters. `efetch_abstracts` reports those as (None, year), so the
    pmid IS a key in the fetch result.

    Handing that result straight to `usable_pairs` would be a silent corruption rather than
    an error: `p.paper_id_a in abstracts` is True, the pair survives the filter, and the
    corpus gains a pair the Critic is asked to judge with no text on one side. It would score
    as a wrong answer attributable to the model instead of a missing input, and the drop rate
    -- the very number this step exists to measure -- would be understated by exactly the
    count of these.

    A year with no abstract is the discriminating case: keying on presence alone, or on the
    tuple being truthy, both keep it."""
    fetched = {
        "1": ("Real abstract text.", 1999),
        "2": (None, 2004),  # known to PubMed, no abstract
        "3": ("", 2010),  # present but empty
    }
    assert available_abstracts(fetched) == {"1": "Real abstract text."}
