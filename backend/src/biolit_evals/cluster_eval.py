import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from biolit.cluster.group import cluster_papers, pairing_diagnostics
from biolit.cluster.pairing import CrossProductPairing, SameSentencePairing
from biolit.domain.records import Cluster, Entity, ExtractedRecord
from biolit_evals.end_to_end import ConceptMetrics, concept_counts, metrics_from_counts
from biolit_evals.mesh_gold import GoldDocument


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


def synthesize_records(
    documents: Sequence[GoldDocument],
) -> tuple[list[ExtractedRecord], dict[str, str]]:
    """Build ExtractedRecords from gold mentions so BOTH arms run the same production code
    path (`cluster_papers`) instead of a parallel gold-only implementation.

    INTERFACE DECISION -- multi-id gold mentions: `Entity.canonical_id` stays `str | None`
    and each (mention, id) becomes its own Entity at the same span. Widening canonical_id
    to a set would be a Phase-1 domain change for one eval's convenience; packing ids into
    one delimited string would collide with "|", the cluster-key delimiter, and push id-set
    logic into the PairingStrategy both arms share. Consequence, stated not hidden:
    duplicate spans inflate Arm A's raw entity counts in pairing_diagnostics.

    Zero-id gold mentions (`mesh_ids == ()`, BC5CDR's unlinkable/-1 case) still become
    exactly one `Entity(canonical_id=None)` -- there is no id to iterate, but the mention
    itself must not vanish. Dropping it instead would be invisible to
    `pairing_diagnostics`, which can only count NIL entities that exist, making Arm A's
    NIL-mention counters structurally 0 regardless of how many gold mentions are actually
    unlinkable.
    """
    records: list[ExtractedRecord] = []
    texts: dict[str, str] = {}
    for document in documents:
        entities = [
            Entity(
                text=mention.text,
                label=mention.label,
                start=mention.start,
                end=mention.end,
                canonical_id=mesh_id,
            )
            for mention in document.mentions
            for mesh_id in (mention.mesh_ids or (None,))
        ]
        records.append(ExtractedRecord(paper_id=document.pmid, entities=entities))
        texts[document.pmid] = document.text
    return records, texts


def run_cluster_eval(
    *,
    documents: Sequence[GoldDocument] | None = None,
    records: Sequence[ExtractedRecord] | None = None,
    texts: Mapping[str, str] | None = None,
    relations: Mapping[str, set[tuple[str, str]]],
    arm: str,
    dataset: str,
    log_path: str,
    git_sha: str,
    now: str,
) -> dict:
    """Score both pairing strategies for one arm and append one JSON line to `log_path`.

    Pass `documents` for Arm A (records are synthesized from gold mentions) or
    `records`+`texts` for Arm B (produced by the real pipeline). Every impure input is
    injected -- log_path, git_sha, now -- so this stays offline-testable, matching
    run_e2e_eval and run_canon_eval.
    """
    if documents is not None:
        records, texts = synthesize_records(documents)
    if records is None or texts is None:
        # Not an assert: asserts vanish under `python -O`, and this is exactly the guard
        # that must hold when a caller supplies neither arm's inputs.
        raise ValueError("pass documents= (Arm A) or records= and texts= (Arm B)")

    gold = gold_clusters_from_relations(relations)
    assert_gold_cluster_anchor(len(records), len(gold))

    strategies: dict[str, object] = {}
    for name, strategy in (
        ("cross_product", CrossProductPairing()),
        ("same_sentence", SameSentencePairing()),
    ):
        predicted_pairs = {
            r.paper_id: strategy.pairs(r.entities, texts.get(r.paper_id, "")) for r in records
        }
        clusters = cluster_papers(records, texts=texts, pairing=strategy)
        km = key_metrics(predicted_pairs, relations)
        if name == "cross_product" and documents is not None:
            assert_key_recall_anchor(km, arm=arm)
        strategies[name] = {
            "key": asdict(km),
            "cluster_key": asdict(cluster_key_metrics(clusters, gold)),
            "paper_pair": asdict(paper_pair_metrics(clusters, gold)),
            "workload": asdict(workload(clusters)),
        }

    line = {
        "timestamp": now,
        "git_sha": git_sha,
        "dataset": dataset,
        "arm": arm,
        "n_documents": len(records),
        "n_gold_clusters": len(gold),
        "n_gold_paper_pairs": len(paper_pairs(gold)),
        "diagnostics": asdict(pairing_diagnostics(records, texts=texts)),
        "strategies": strategies,
    }
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(line) + "\n")
    return line


DEFAULT_LOG = "evals/cluster_runs.jsonl"


def main(argv: list[str] | None = None) -> None:
    # Heavy imports are local so importing this module for scoring stays cheap and offline,
    # same pattern as end_to_end.main.
    import argparse
    from datetime import UTC, datetime

    from biolit.canon.canonicalize import canonicalize
    from biolit.canon.linker import DictionaryLinker
    from biolit.canon.mesh import MeshDictionary
    from biolit.config import get_settings
    from biolit.ner.extract import extract_entities
    from biolit.ner.model import NerModel
    from biolit_evals._meta import git_sha
    from biolit_evals.mesh_gold_download import (
        DEVELOPMENT_MEMBER,
        TEST_MEMBER,
        TRAINING_MEMBER,
        load_bc5cdr_cid_relations,
        load_bc5cdr_documents,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=["A", "B"], required=True)
    args = parser.parse_args(argv)
    settings = get_settings()
    url = settings.bc5cdr_cdr_zip_url

    if args.arm == "A":
        # All three splits: no model runs, so the NER checkpoint's training split is not a
        # leakage risk here, and multi-document clusters are sparse enough to want 1500.
        members = (TRAINING_MEMBER, DEVELOPMENT_MEMBER, TEST_MEMBER)
        documents = [d for m in members for d in load_bc5cdr_documents(url, m)]
        relations: dict[str, set[tuple[str, str]]] = {}
        for member in members:
            relations.update(load_bc5cdr_cid_relations(url, member))
        line = run_cluster_eval(
            documents=documents,
            relations=relations,
            arm="A",
            dataset="bc5cdr_all1500",
            log_path=DEFAULT_LOG,
            git_sha=git_sha(),
            now=datetime.now(UTC).isoformat(),
        )
    else:
        # Test split ONLY: the NER checkpoint was fine-tuned on BC5CDR's training split, so
        # any arm running real NER must stay held out or the number is contaminated.
        documents = load_bc5cdr_documents(url, TEST_MEMBER)
        relations = load_bc5cdr_cid_relations(url, TEST_MEMBER)
        dictionary = MeshDictionary.from_artifact(settings.mesh_artifact_path)
        linker = DictionaryLinker(dictionary)
        model = NerModel.load(settings)
        records = []
        texts = {}
        for document in documents:
            preds = extract_entities(
                document.text, model, score_threshold=settings.ner_score_threshold
            )
            canon = canonicalize(preds, document.text, linker=linker)
            records.append(ExtractedRecord(paper_id=document.pmid, entities=list(canon)))
            texts[document.pmid] = document.text
        line = run_cluster_eval(
            records=records,
            texts=texts,
            relations=relations,
            arm="B",
            dataset="bc5cdr_test500",
            log_path=DEFAULT_LOG,
            git_sha=git_sha(),
            now=datetime.now(UTC).isoformat(),
        )

    print(
        f"arm={line['arm']} docs={line['n_documents']} "
        f"gold_clusters={line['n_gold_clusters']} gold_pairs={line['n_gold_paper_pairs']}"
    )
    print(f"  diagnostics: {line['diagnostics']}")
    for name, s in line["strategies"].items():
        pp, ck, k, w = s["paper_pair"], s["cluster_key"], s["key"], s["workload"]
        print(f"\n=== {name} ===")
        print(
            f"  PAPER-PAIR (primary): P={pp['precision']:.4f} R={pp['recall']:.4f} "
            f"F1={pp['f1']:.4f} (tp={pp['tp']} fp={pp['fp']} fn={pp['fn']})"
        )
        print(
            f"  cluster-key (diag):   P={ck['precision']:.4f} R={ck['recall']:.4f} "
            f"F1={ck['f1']:.4f}"
        )
        print(
            f"  key (diag):           P={k['precision']:.4f} R={k['recall']:.4f} F1={k['f1']:.4f}"
        )
        print(
            f"  Critic workload: clusters={w['n_clusters']} pairs={w['n_paper_pairs']} "
            f"largest={w['largest_cluster']} top5_share={w['top5_pair_share']:.3f}"
        )


if __name__ == "__main__":
    main()
