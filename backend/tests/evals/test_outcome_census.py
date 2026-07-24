from biolit.domain.enums import EntityLabel
from biolit.domain.records import Entity
from biolit_evals.mesh_gold import GoldMention
from biolit_evals.outcome_census import Outcome, TruncationKind, classify_outcome

CHEMICAL = EntityLabel.CHEMICAL
DISEASE = EntityLabel.DISEASE


def _gold(start, end, text, label=CHEMICAL):
    return GoldMention(
        pmid="1", start=start, end=end, text=text, label=label, mesh_ids=("MESH:D1",)
    )


def _pred(start, end, text, label=CHEMICAL):
    return Entity(text=text, label=label, start=start, end=end)


def test_exact_match():
    rec = classify_outcome(_gold(0, 9, "metformin"), [_pred(0, 9, "metformin")])
    assert rec.outcome is Outcome.EXACT
    assert rec.truncation is TruncationKind.NOT_TRUNCATED
    assert rec.char_delta == 0


def test_mergeable_when_two_adjacent_predictions_cover_gold():
    # Real case: gold "GLP-1RAs" predicted as "GLP" + "1RA".
    rec = classify_outcome(_gold(0, 8, "GLP-1RAs"), [_pred(0, 3, "GLP"), _pred(4, 7, "1RA")])
    assert rec.outcome is Outcome.MERGEABLE


def test_truncated_prefix_of_gold_is_a_dropped_suffix():
    # Real case: gold "CFD" predicted as "CF".
    rec = classify_outcome(_gold(0, 3, "CFD"), [_pred(0, 2, "CF")])
    assert rec.outcome is Outcome.TRUNCATED
    assert rec.truncation is TruncationKind.PREFIX_OF_GOLD
    assert rec.char_delta == 1


def test_truncated_suffix_of_gold_is_a_dropped_prefix():
    # Real case: gold "insulin resistance" predicted as "resistance".
    rec = classify_outcome(_gold(0, 18, "insulin resistance"), [_pred(8, 18, "resistance")])
    assert rec.outcome is Outcome.TRUNCATED
    assert rec.truncation is TruncationKind.SUFFIX_OF_GOLD
    assert rec.char_delta == 8


def test_truncated_interior_when_neither_boundary_matches():
    rec = classify_outcome(_gold(0, 20, "a" * 20), [_pred(5, 15, "b" * 10)])
    assert rec.outcome is Outcome.TRUNCATED
    assert rec.truncation is TruncationKind.INTERIOR_OR_OTHER


def test_missed_when_no_overlapping_prediction():
    rec = classify_outcome(_gold(0, 9, "metformin"), [_pred(20, 26, "cancer")])
    assert rec.outcome is Outcome.MISSED


def test_other_label_predictions_are_ignored():
    # A DISEASE prediction sitting exactly on a CHEMICAL gold span is not a match.
    rec = classify_outcome(_gold(0, 9, "metformin", CHEMICAL), [_pred(0, 9, "metformin", DISEASE)])
    assert rec.outcome is Outcome.MISSED
