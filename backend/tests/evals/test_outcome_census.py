from biolit.domain.enums import EntityLabel
from biolit.domain.records import Entity
from biolit_evals.mesh_gold import GoldMention
from biolit_evals.outcome_census import (
    ExactLinkRecord,
    ExactLinkStatus,
    Outcome,
    OutcomeRecord,
    TruncationKind,
    census,
    classify_exact_link,
    classify_outcome,
    exact_link_audit,
)

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


def test_census_counts_pooled_and_per_label():
    records = [
        OutcomeRecord(CHEMICAL, Outcome.EXACT, TruncationKind.NOT_TRUNCATED, 0),
        OutcomeRecord(CHEMICAL, Outcome.TRUNCATED, TruncationKind.PREFIX_OF_GOLD, 1),
        OutcomeRecord(DISEASE, Outcome.TRUNCATED, TruncationKind.SUFFIX_OF_GOLD, 8),
        OutcomeRecord(DISEASE, Outcome.MISSED, TruncationKind.NOT_TRUNCATED, 0),
    ]
    c = census(records)
    assert c.total == 4
    assert c.outcomes == {"EXACT": 1, "TRUNCATED": 2, "MISSED": 1}
    # only TRUNCATED records contribute to the truncation breakdown
    assert c.truncation == {"PREFIX_OF_GOLD": 1, "SUFFIX_OF_GOLD": 1}
    assert c.by_label["CHEMICAL"] == {"EXACT": 1, "TRUNCATED": 1}
    assert c.by_label["DISEASE"] == {"TRUNCATED": 1, "MISSED": 1}
    assert c.truncation_by_label["CHEMICAL"] == {"PREFIX_OF_GOLD": 1}
    assert c.truncation_by_label["DISEASE"] == {"SUFFIX_OF_GOLD": 1}


def test_exact_span_that_did_not_link_is_nil():
    # The fallback-addressable population: NER got the span exactly right and the
    # dictionary still had nothing for it.
    gold = _gold(0, 9, "metformin")
    canon = [Entity(text="metformin", label=CHEMICAL, start=0, end=9, canonical_id=None)]
    rec = classify_exact_link(gold, canon)
    assert rec is not None
    assert rec.status is ExactLinkStatus.NIL
    assert rec.label is CHEMICAL


def test_exact_span_linked_to_a_gold_id_is_correct():
    gold = _gold(0, 9, "metformin")  # mesh_ids == ("MESH:D1",)
    canon = [Entity(text="metformin", label=CHEMICAL, start=0, end=9, canonical_id="MESH:D1")]
    rec = classify_exact_link(gold, canon)
    assert rec is not None and rec.status is ExactLinkStatus.LINKED_CORRECT


def test_exact_span_linked_to_the_wrong_concept_is_not_nil():
    # Separating this from NIL is what keeps "the dictionary lacks the alias" (fixable by a
    # fallback) apart from "the dictionary had a confident wrong answer" (a fallback would
    # never even be consulted).
    gold = _gold(0, 9, "metformin")
    canon = [Entity(text="metformin", label=CHEMICAL, start=0, end=9, canonical_id="MESH:D999")]
    rec = classify_exact_link(gold, canon)
    assert rec is not None and rec.status is ExactLinkStatus.LINKED_WRONG


def test_exact_span_on_unlinkable_gold_is_not_counted_as_nil():
    # BC5CDR annotates 91 of 9809 mentions with `-1`. Abstaining on those is correct, so
    # they must not inflate the fallback-addressable count.
    gold = GoldMention(pmid="1", start=0, end=9, text="metformin", label=CHEMICAL, mesh_ids=())
    canon = [Entity(text="metformin", label=CHEMICAL, start=0, end=9, canonical_id=None)]
    rec = classify_exact_link(gold, canon)
    assert rec is not None and rec.status is ExactLinkStatus.GOLD_UNLINKABLE


def test_non_exact_span_has_no_exact_link_verdict():
    # A truncated prediction ("CF" for gold "CFD") linking or not says nothing about the
    # dictionary's coverage of "CFD", so it is excluded rather than scored.
    gold = _gold(0, 3, "CFD")
    canon = [Entity(text="CF", label=CHEMICAL, start=0, end=2, canonical_id=None)]
    assert classify_exact_link(gold, canon) is None


def test_exact_link_audit_counts_pooled_and_per_label():
    records = [
        ExactLinkRecord(CHEMICAL, ExactLinkStatus.LINKED_CORRECT),
        ExactLinkRecord(CHEMICAL, ExactLinkStatus.NIL),
        ExactLinkRecord(DISEASE, ExactLinkStatus.NIL),
        ExactLinkRecord(DISEASE, ExactLinkStatus.GOLD_UNLINKABLE),
    ]
    a = exact_link_audit(records)
    assert a.total == 4
    assert a.statuses == {"LINKED_CORRECT": 1, "NIL": 2, "GOLD_UNLINKABLE": 1}
    assert a.by_label["CHEMICAL"] == {"LINKED_CORRECT": 1, "NIL": 1}
    assert a.by_label["DISEASE"] == {"NIL": 1, "GOLD_UNLINKABLE": 1}


def test_census_of_no_records_is_empty():
    c = census([])
    assert c.total == 0
    assert c.outcomes == {} and c.truncation == {}
    assert c.by_label == {} and c.truncation_by_label == {}
