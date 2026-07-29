from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from biolit.domain.records import Cluster
from biolit_evals.end_to_end import ConceptMetrics, concept_counts, metrics_from_counts


def _key(pair: tuple[str, str]) -> str:
    return f"{pair[0]}|{pair[1]}"


def key_metrics(
    pred: Mapping[str, set[tuple[str, str]]], gold: Mapping[str, set[tuple[str, str]]]
) -> ConceptMetrics:
    """Micro-averaged P/R/F1 over (pmid, key), set-per-document -- DIAGNOSTIC, not primary.

    Reuses concept_counts so the set semantics match every other concept-level number in
    this project.
    """
    tp = fp = fn = 0
    for pmid in set(pred) | set(gold):
        d_tp, d_fp, d_fn = concept_counts(
            {_key(p) for p in gold.get(pmid, set())},
            {_key(p) for p in pred.get(pmid, set())},
        )
        tp, fp, fn = tp + d_tp, fp + d_fp, fn + d_fn
    return metrics_from_counts(tp, fp, fn)


def cluster_key_metrics(
    pred_clusters: Sequence[Cluster], gold_clusters: Sequence[Cluster]
) -> ConceptMetrics:
    """Is a predicted multi-paper cluster's KEY a gold multi-paper cluster key? DIAGNOSTIC.

    Blind to which papers landed in the cluster -- that is exactly what paper_pair_metrics
    measures and why this one cannot stand in for it.
    """
    pred_keys = {c.key for c in pred_clusters}
    gold_keys = {c.key for c in gold_clusters}
    return metrics_from_counts(*concept_counts(gold_keys, pred_keys))


def paper_pairs(clusters: Sequence[Cluster]) -> set[tuple[str, str]]:
    """Every unordered paper pair a cluster set implies, as sorted 2-tuples.

    This is the Critic's actual unit of work: ContradictionFinding is
    (paper_id_a, paper_id_b, label, rationale).
    """
    out: set[tuple[str, str]] = set()
    for cluster in clusters:
        ids = sorted(cluster.paper_ids)
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                out.add((ids[i], ids[j]))
    return out


def paper_pair_metrics(
    pred_clusters: Sequence[Cluster], gold_clusters: Sequence[Cluster]
) -> ConceptMetrics:
    """PRIMARY metric: did the right papers end up comparable?

    Primary because ContradictionFinding's unit of work is a pair of papers, so every
    false pair is a wasted or wrong Critic comparison -- and the Critic will be LLM-backed,
    making that direct cost. Deliberately NOT cluster-key agreement: a prediction can match
    a gold key exactly while recovering only part of its paper set, and this level sees
    that where cluster_key_metrics cannot.
    """
    return metrics_from_counts(
        *concept_counts(
            {f"{a}|{b}" for a, b in paper_pairs(gold_clusters)},
            {f"{a}|{b}" for a, b in paper_pairs(pred_clusters)},
        )
    )


def gold_clusters_from_relations(
    relations: Mapping[str, set[tuple[str, str]]], min_size: int = 2
) -> list[Cluster]:
    """Gold clusters: papers sharing a gold CID pair. Same min_size rule as cluster_papers."""
    by_key: dict[str, set[str]] = {}
    for pmid, pairs in relations.items():
        for pair in pairs:
            by_key.setdefault(_key(pair), set()).add(pmid)
    return [
        Cluster(key=key, paper_ids=sorted(pmids))
        for key, pmids in sorted(by_key.items())
        if len(pmids) >= min_size
    ]


@dataclass(frozen=True)
class Workload:
    """What clustering hands the Critic. Raw counts, not only aggregates.

    top5_pair_share prices the quadratic risk directly: a cluster of n papers is
    n*(n-1)/2 comparisons, so a single 25-paper cluster is 300 on its own. An aggregate
    pair count cannot show how much of the Critic's budget one oversized cluster burns.
    """

    n_clusters: int
    n_paper_pairs: int
    largest_cluster: int
    top5_pair_share: float


def workload(clusters: Sequence[Cluster]) -> Workload:
    """Measure clustering's cost to the Critic and concentration risk.

    A cluster of n papers creates n*(n-1)/2 Critic comparisons, a quadratic cost. This
    function logs both the aggregate pair count and the top-5 concentration ratio, which
    exposes when a single oversized cluster dominates the budget in a way a total count
    cannot show. Essential for identifying degenerate pairing arms where one bad key
    inflates the Critic's workload.
    """
    sizes = sorted((len(c.paper_ids) for c in clusters), reverse=True)
    pairs = [n * (n - 1) // 2 for n in sizes]
    total = sum(pairs)
    return Workload(
        n_clusters=len(clusters),
        n_paper_pairs=total,
        largest_cluster=sizes[0] if sizes else 0,
        top5_pair_share=sum(pairs[:5]) / total if total else 0.0,
    )


_GOLD_CLUSTER_ANCHORS = {500: 80, 1500: 325}


def assert_key_recall_anchor(metrics: ConceptMetrics, *, arm: str) -> None:
    """HARNESS correctness. Cross-product pairing on GOLD entities must recall every gold
    CID pair, because a gold pair always connects two annotated entities and the
    cross-product of those entities' ids necessarily contains it.

    A deviation means the harness is wrong -- gold parsing, MeSH id prefixing, or label
    assignment -- not that the result is interesting. Do NOT relax this to match an
    observation.
    """
    if round(metrics.recall, 4) != 1.0:
        raise SystemExit(
            f"{arm}: cross-product key recall is {metrics.recall:.4f}, expected exactly "
            f"1.0000 (fn={metrics.fn}). A gold CID pair connects two annotated entities, "
            "so the cross-product cannot miss one. The harness is wrong."
        )


def assert_gold_cluster_anchor(n_documents: int, n_gold_clusters: int) -> None:
    """LOADER correctness -- NOT a clustering-quality check.

    This validates `load_bc5cdr_cid_relations` against independently established gold
    statistics. "The loader reproduces known gold counts" says NOTHING about whether the
    pairing strategy is accurate; do not read a passing anchor as evidence about clustering.
    """
    expected = _GOLD_CLUSTER_ANCHORS.get(n_documents)
    if expected is None:
        return
    if n_gold_clusters != expected:
        raise SystemExit(
            f"gold cluster anchor: {n_documents} documents yielded {n_gold_clusters} "
            f"multi-paper gold clusters, expected {expected}. The CID loader is wrong; "
            "this says nothing about clustering quality either way."
        )
