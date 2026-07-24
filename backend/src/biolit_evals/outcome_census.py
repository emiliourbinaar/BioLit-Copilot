from dataclasses import dataclass
from enum import StrEnum

from biolit.domain.enums import EntityLabel
from biolit.domain.records import Entity
from biolit_evals.mesh_gold import GoldMention


class Outcome(StrEnum):
    EXACT = "EXACT"
    MERGEABLE = "MERGEABLE"
    TRUNCATED = "TRUNCATED"
    MISSED = "MISSED"


class TruncationKind(StrEnum):
    PREFIX_OF_GOLD = "PREFIX_OF_GOLD"  # prediction starts at gold, ends early: suffix dropped
    SUFFIX_OF_GOLD = "SUFFIX_OF_GOLD"  # prediction ends at gold, starts late: prefix dropped
    INTERIOR_OR_OTHER = "INTERIOR_OR_OTHER"
    NOT_TRUNCATED = "NOT_TRUNCATED"


@dataclass(frozen=True)
class OutcomeRecord:
    label: EntityLabel
    outcome: Outcome
    truncation: TruncationKind
    char_delta: int  # gold length - prediction length; 0 unless TRUNCATED


def _overlaps(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return max(a_start, b_start) < min(a_end, b_end)


def classify_outcome(gold: GoldMention, predictions: list[Entity]) -> OutcomeRecord:
    """Classify how the NER predictions covered one gold mention.

    This is the categorical breakdown behind the end-to-end number: it separates spans the
    model got right from the three distinct ways it can get them wrong, so a change in the
    headline score can be attributed to a mechanism rather than guessed at.
    """
    same_label = [
        p
        for p in predictions
        if p.label is gold.label and p.start is not None and p.end is not None
    ]
    if any(p.start == gold.start and p.end == gold.end for p in same_label):
        return OutcomeRecord(gold.label, Outcome.EXACT, TruncationKind.NOT_TRUNCATED, 0)

    overlapping = [
        p
        for p in same_label
        if p.start is not None
        and p.end is not None
        and _overlaps(p.start, p.end, gold.start, gold.end)
    ]
    if not overlapping:
        return OutcomeRecord(gold.label, Outcome.MISSED, TruncationKind.NOT_TRUNCATED, 0)
    if len(overlapping) >= 2:
        return OutcomeRecord(gold.label, Outcome.MERGEABLE, TruncationKind.NOT_TRUNCATED, 0)

    pred = overlapping[0]
    assert pred.start is not None and pred.end is not None
    if pred.start == gold.start and pred.end < gold.end:
        kind = TruncationKind.PREFIX_OF_GOLD
    elif pred.end == gold.end and pred.start > gold.start:
        kind = TruncationKind.SUFFIX_OF_GOLD
    else:
        kind = TruncationKind.INTERIOR_OR_OTHER
    delta = (gold.end - gold.start) - (pred.end - pred.start)
    return OutcomeRecord(gold.label, Outcome.TRUNCATED, kind, delta)
