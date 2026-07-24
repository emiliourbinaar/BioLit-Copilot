from collections.abc import Callable
from dataclasses import dataclass

from biolit.canon.canonicalize import canonicalize
from biolit.canon.fragments import merge_fragments
from biolit.canon.linker import Linker
from biolit.domain.enums import EntityLabel
from biolit.domain.records import Entity
from biolit_evals.mesh_gold import GoldDocument
from biolit_evals.outcome_census import Census, OutcomeRecord, census, classify_outcome


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


@dataclass(frozen=True)
class E2EMetrics:
    n_documents: int
    concepts: ConceptMetrics
    concepts_by_label: dict[str, ConceptMetrics]
    n_predicted: int
    n_predicted_linked: int
    e2e_nil_rate: float
    census: Census
    merge_candidates: int
    merge_candidates_linked: int
    merge_candidates_matching_gold: int
    merged_constituents: int


def score_end_to_end(
    documents: list[GoldDocument],
    *,
    predict: Callable[[str], list[Entity]],
    linker: Linker,
) -> E2EMetrics:
    """Score the real path: predictions -> canonicalize (merge + link) -> canonical ids.

    `predict` is injected so this runs offline in tests. Concept scoring is micro-averaged
    over documents with set semantics inside each document; the census explains the result
    categorically, and the merge audit is raw counts (n is expected to be small).
    """
    totals = [0, 0, 0]
    label_totals: dict[str, list[int]] = {}
    records: list[OutcomeRecord] = []
    n_predicted = n_linked = 0
    cand_total = cand_linked = cand_gold = merged_constituents = 0

    for doc in documents:
        preds = predict(doc.text)
        individual = [linker.link(p.text) for p in preds]
        candidates = merge_fragments(preds, doc.text)
        gold_spans = {(m.start, m.end, m.label) for m in doc.mentions}
        for cand in candidates:
            cand_total += 1
            if linker.link(cand.text).concept is not None:
                cand_linked += 1
                merged_constituents += sum(
                    1 for i in cand.source_indices if individual[i].concept is None
                )
            if (cand.start, cand.end, cand.label) in gold_spans:
                cand_gold += 1

        canon = canonicalize(preds, doc.text, linker=linker)
        n_predicted += len(canon)
        n_linked += sum(1 for e in canon if e.canonical_id is not None)

        records.extend(classify_outcome(m, preds) for m in doc.mentions)

        gold_ids = {i for m in doc.mentions for i in m.mesh_ids}
        pred_ids = {e.canonical_id for e in canon if e.canonical_id is not None}
        counts = concept_counts(gold_ids, pred_ids)
        totals = [totals[j] + counts[j] for j in range(3)]

        for label in (EntityLabel.CHEMICAL, EntityLabel.DISEASE):
            g = {i for m in doc.mentions if m.label is label for i in m.mesh_ids}
            p = {e.canonical_id for e in canon if e.label is label and e.canonical_id is not None}
            slot = label_totals.setdefault(label.value, [0, 0, 0])
            per = concept_counts(g, p)
            for j in range(3):
                slot[j] += per[j]

    return E2EMetrics(
        n_documents=len(documents),
        concepts=metrics_from_counts(totals[0], totals[1], totals[2]),
        concepts_by_label={
            k: metrics_from_counts(v[0], v[1], v[2]) for k, v in sorted(label_totals.items())
        },
        n_predicted=n_predicted,
        n_predicted_linked=n_linked,
        e2e_nil_rate=(n_predicted - n_linked) / n_predicted if n_predicted else 0.0,
        census=census(records),
        merge_candidates=cand_total,
        merge_candidates_linked=cand_linked,
        merge_candidates_matching_gold=cand_gold,
        merged_constituents=merged_constituents,
    )
