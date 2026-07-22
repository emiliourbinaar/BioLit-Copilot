from dataclasses import dataclass

from biolit.domain.records import Entity


@dataclass(frozen=True)
class PRF:
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int


def _keys(entities: list[Entity]) -> set[tuple[int | None, int | None, str]]:
    return {(e.start, e.end, e.label) for e in entities}


def score_corpus(pairs: list[tuple[list[Entity], list[Entity]]]) -> PRF:
    """Micro-averaged strict entity-level P/R/F1.

    Correct = exact (start, end, label). TP/FP/FN are counted per example and summed,
    so identical spans in different examples never collide.
    """
    tp = fp = fn = 0
    for gold, pred in pairs:
        gold_keys = _keys(gold)
        pred_keys = _keys(pred)
        tp += len(gold_keys & pred_keys)
        fp += len(pred_keys - gold_keys)
        fn += len(gold_keys - pred_keys)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return PRF(precision, recall, f1, tp, fp, fn)
