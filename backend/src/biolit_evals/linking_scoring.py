from dataclasses import dataclass

from biolit.canon.linker import Linker
from biolit_evals.mesh_gold import GoldMention


@dataclass(frozen=True)
class LinkingMetrics:
    n: int
    correct: int
    linked: int
    tiebroken: int
    precision: float
    recall: float
    f1: float
    nil_rate: float
    tiebreak_rate: float


def score_linking(gold: list[GoldMention], linker: Linker) -> LinkingMetrics:
    """Score linking on gold *mentions* (surface fed straight to the linker), isolating
    linking quality from NER. Evaluated on linkable gold only (mesh_ids non-empty)."""
    items = [g for g in gold if g.mesh_ids]
    n = len(items)
    correct = linked = tiebroken = 0
    for g in items:
        r = linker.link(g.text)
        if r.tiebroken:
            tiebroken += 1
        if r.concept is not None:
            linked += 1
            if r.concept.id in g.mesh_ids:
                correct += 1
    precision = correct / linked if linked else 0.0
    recall = correct / n if n else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    nil_rate = (n - linked) / n if n else 0.0
    tiebreak_rate = tiebroken / n if n else 0.0
    return LinkingMetrics(
        n=n,
        correct=correct,
        linked=linked,
        tiebroken=tiebroken,
        precision=precision,
        recall=recall,
        f1=f1,
        nil_rate=nil_rate,
        tiebreak_rate=tiebreak_rate,
    )
