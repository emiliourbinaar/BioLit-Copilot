from dataclasses import dataclass


@dataclass(frozen=True)
class ConceptMetrics:
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float


def concept_counts(gold_ids: set[str], pred_ids: set[str]) -> tuple[int, int, int]:
    """(tp, fp, fn) over the concept SETS of a single document.

    Set semantics are deliberate: at concept level the question is whether a paper mentions
    a concept at all, so an entity repeated five times in one abstract counts once. Multiset
    counting would let one frequently-repeated entity dominate the corpus score.
    """
    return (
        len(gold_ids & pred_ids),
        len(pred_ids - gold_ids),
        len(gold_ids - pred_ids),
    )


def metrics_from_counts(tp: int, fp: int, fn: int) -> ConceptMetrics:
    """Build metrics from MICRO-averaged totals: tp/fp/fn summed over all documents, with
    precision/recall/F1 computed once from those sums -- not a macro average of
    per-document scores."""
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return ConceptMetrics(tp=tp, fp=fp, fn=fn, precision=precision, recall=recall, f1=f1)
