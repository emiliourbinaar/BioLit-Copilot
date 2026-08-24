import random

import pytest

from biolit.domain.records import ContradictionLabel
from biolit_evals.annotation_export import evaluate_gate1, evaluate_gate2, export_blind_sheet
from biolit_evals.contradiction_gold import GoldPair
from biolit_evals.critic_scoring import wilson_interval

_PAIRS = [
    GoldPair(
        "p1",
        "p2",
        "C001",
        "D001",
        ContradictionLabel.contradiction,
        "marker/mechanism",
        "therapeutic",
    ),
    GoldPair(
        "p3", "p4", "C002", "D002", ContradictionLabel.agreement, "therapeutic", "therapeutic"
    ),
]
_ABSTRACTS = {
    "p1": "Abstract text for p1.",
    "p2": "Abstract text for p2.",
    "p3": "Abstract text for p3.",
    "p4": "Abstract text for p4.",
}


def test_the_blind_sheet_carries_no_gold_label():
    """Blind, per the ADR-0006 domain_sample precedent. A leaked label makes pi
    unmeasurable, and nothing downstream could detect that it had happened."""
    rows = export_blind_sheet(
        _PAIRS, _ABSTRACTS, n_contradiction=1, n_other=1, rng=random.Random(0)
    )
    for row in rows:
        assert "label" not in row
        assert "direction_a" not in row and "direction_b" not in row
        assert set(row) == {"pair_id", "chemical_id", "disease_id", "abstract_a", "abstract_b"}


def _pool(label: ContradictionLabel, count: int) -> list[GoldPair]:
    # 20 candidates per label, reverse-inserted, so a request for 15 must actually SELECT a
    # subset rather than just returning whatever happens to be there.
    return [
        GoldPair(
            f"{label}-a{i}",
            f"{label}-b{i}",
            f"C{i:03d}",
            f"D{i:03d}",
            label,
            "marker/mechanism",
            "therapeutic",
        )
        for i in range(count, 0, -1)
    ]


def test_first_batch_is_fifteen_contradiction_and_fifteen_split_across_the_other_two_classes():
    """15 contradiction + 15 spread across agreement/insufficient_overlap -- deliberately
    weighted, since an even three-way split would yield only ~10 contradiction pairs, too few
    for Gate 2 to fire on."""
    pool = (
        _pool(ContradictionLabel.contradiction, 20)
        + _pool(ContradictionLabel.agreement, 20)
        + _pool(ContradictionLabel.insufficient_overlap, 20)
    )
    label_of_pair_id = {f"{p.paper_id_a}_{p.paper_id_b}": p.label for p in pool}
    abstracts = {pid: f"Abstract for {pid}." for p in pool for pid in (p.paper_id_a, p.paper_id_b)}

    rows = export_blind_sheet(pool, abstracts, n_contradiction=15, n_other=15, rng=random.Random(0))

    assert len(rows) == 30
    counts: dict[ContradictionLabel, int] = {}
    for row in rows:
        label = label_of_pair_id[row["pair_id"]]
        counts[label] = counts.get(label, 0) + 1
    assert counts[ContradictionLabel.contradiction] == 15
    assert counts[ContradictionLabel.agreement] == 8
    assert counts[ContradictionLabel.insufficient_overlap] == 7


def test_rows_are_actually_shuffled_not_left_in_selection_order():
    """Class must not be inferable from position. A no-op shuffle would leave the rows in
    selection order (all 7 pairs, in the reverse-inserted order they were built in) --
    asserting mere length or set-equality would pass a broken no-shuffle implementation, so
    this pins that the ORDER differs from that unshuffled baseline."""
    pool = _pool(ContradictionLabel.contradiction, 7)  # 7, reverse-inserted
    unshuffled_order = [f"{p.paper_id_a}_{p.paper_id_b}" for p in pool]
    abstracts = {pid: f"Abstract for {pid}." for p in pool for pid in (p.paper_id_a, p.paper_id_b)}

    rows = export_blind_sheet(pool, abstracts, n_contradiction=7, n_other=0, rng=random.Random(0))

    shuffled_order = [row["pair_id"] for row in rows]
    assert sorted(shuffled_order) == sorted(unshuffled_order)  # same 7 pairs
    assert shuffled_order != unshuffled_order  # but a different order


def test_gate2_stops_at_seven_of_fifteen_and_continues_at_eight():
    """The cut is at 7 because under a true pi of 0.8 that outcome has probability 0.0042,
    and under 0.7 it is 0.0500 -- a strong signal against pi >= 0.7 that fires rarely when
    the proxy is sound. Boundary pinned on both sides."""
    assert evaluate_gate2(genuine=7, n=15).verdict == "STOP"
    assert evaluate_gate2(genuine=8, n=15).verdict == "CONTINUE_FLAGGED"
    assert evaluate_gate2(genuine=11, n=15).verdict == "CONTINUE"


def test_gate2_continue_flagged_upper_boundary_is_ten():
    """The band is the two-sided `8 <= g <= 10` (ADR-0014: both operands of a compound
    condition independently exercised). The prior test pins g=8, the lower operand; this
    pins g=10, the upper one -- without it, an `8 <= g < 10` off-by-one mutant would survive
    since g=8 alone cannot distinguish the two."""
    assert evaluate_gate2(genuine=10, n=15).verdict == "CONTINUE_FLAGGED"


def test_gate2_reports_a_wide_interval_it_does_not_hide():
    """At 15 pairs the 95% interval is about +/-0.25, so a batch reading 0.6 cannot be
    distinguished from one reading 0.8. Gate 2 is a STOP RULE, not an estimate of pi."""
    gate = evaluate_gate2(genuine=9, n=15)
    lo, hi = gate.interval
    assert (hi - lo) > 0.4


def test_evaluate_gate2_refuses_an_uncalibrated_n():
    """The verdict bands are literal counts derived from binomial tail probabilities computed
    at n=15, not a proportional rule -- a Gate2 built for n=30 (the whole first batch, which
    is Gate 1's domain) or any other n would present with exactly as much apparent authority
    as a correct one, so evaluate_gate2 must refuse rather than silently mis-band it."""
    with pytest.raises(ValueError, match="n=15"):
        evaluate_gate2(genuine=10, n=30)
    with pytest.raises(ValueError, match="n=15"):
        evaluate_gate2(genuine=50, n=100)


def test_evaluate_gate2_rejects_a_genuine_count_out_of_range():
    """A mis-transcribed count is a smaller version of the same silent-wrong-answer risk as
    an uncalibrated n, and cheap to catch. Both directions checked: negative, and above n."""
    with pytest.raises(ValueError, match="genuine"):
        evaluate_gate2(genuine=-1, n=15)
    with pytest.raises(ValueError, match="genuine"):
        evaluate_gate2(genuine=16, n=15)


def test_gate1_boundary_ten_of_thirty_is_tractable_and_eleven_revises():
    """Gate 1 is a tractability check over the WHOLE first batch (all 30 pairs across all
    three classes) -- unlike Gate 2, which runs over the 15 contradiction pairs only; the two
    gates have different n by design. A cant_tell rate above ~1/3 means the protocol needs
    revision before more time is spent. At n=30, 10 cant_tell is exactly 1/3 and passes; 11
    exceeds it and fails -- mutation-verifies `>` vs `>=`."""
    assert evaluate_gate1(cant_tell=10, n=30).verdict == "TRACTABLE"
    assert evaluate_gate1(cant_tell=11, n=30).verdict == "REVISE_PROTOCOL"


def test_gate1_interval_is_the_wilson_interval_of_the_cant_tell_rate():
    """Guards against a hardcoded/disconnected interval -- Gate 1 mirrors Gate 2's shape,
    including reporting a real Wilson interval on its own rate, not a stand-in constant."""
    gate = evaluate_gate1(cant_tell=6, n=30)
    assert gate.interval == wilson_interval(6, 30)
