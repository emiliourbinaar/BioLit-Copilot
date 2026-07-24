import argparse
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from biolit.canon.canonicalize import canonicalize
from biolit.canon.fragments import merge_fragments
from biolit.canon.linker import Linker
from biolit.domain.enums import EntityLabel
from biolit.domain.records import Entity
from biolit_evals._meta import git_sha
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


DEFAULT_LOG = "evals/e2e_runs.jsonl"
DOMAIN_NORM_GOLD = "evals/gold/domain_normalization_sample.jsonl"


def run_e2e_eval(
    *,
    documents: list[GoldDocument],
    predict: Callable[[str], list[Entity]],
    linker: Linker,
    dataset: str,
    artifact_source: str,
    n_aliases: int,
    log_path: str,
    git_sha: str,
    now: str,
) -> E2EMetrics:
    """Score end to end and append one JSON line to `log_path`.

    Every impure input is injected (documents, predict, linker, log_path, git_sha, now) so
    this stays offline-testable, matching `run_eval` and `run_canon_eval`.
    """
    m = score_end_to_end(documents, predict=predict, linker=linker)
    line = {
        "timestamp": now,
        "git_sha": git_sha,
        "dataset": dataset,
        "artifact_source": artifact_source,
        "n_aliases": n_aliases,
        "n_documents": m.n_documents,
        "tp": m.concepts.tp,
        "fp": m.concepts.fp,
        "fn": m.concepts.fn,
        "precision": m.concepts.precision,
        "recall": m.concepts.recall,
        "f1": m.concepts.f1,
        "e2e_nil_rate": m.e2e_nil_rate,
        "n_predicted": m.n_predicted,
        "n_predicted_linked": m.n_predicted_linked,
        "census": asdict(m.census),
        "concepts_by_label": {k: asdict(v) for k, v in m.concepts_by_label.items()},
        "merge_audit": {
            "candidates": m.merge_candidates,
            "candidates_linked": m.merge_candidates_linked,
            "candidates_matching_gold": m.merge_candidates_matching_gold,
            "merged_constituents": m.merged_constituents,
        },
    }
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(line) + "\n")
    return m


def main(argv: list[str] | None = None) -> None:
    # Heavy imports are local so importing this module for scoring stays cheap and offline.
    from biolit.canon.linker import DictionaryLinker
    from biolit.canon.mesh import MeshDictionary
    from biolit.config import get_settings
    from biolit.ner.extract import extract_entities
    from biolit.ner.model import NerModel
    from biolit_evals.mesh_gold import load_domain_norm_documents
    from biolit_evals.mesh_gold_download import load_bc5cdr_documents

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["bc5cdr", "domain"], required=True)
    args = parser.parse_args(argv)

    settings = get_settings()
    dictionary = MeshDictionary.from_artifact(settings.mesh_artifact_path)
    linker = DictionaryLinker(dictionary)
    model = NerModel.load(settings)

    if args.dataset == "bc5cdr":
        documents = load_bc5cdr_documents(settings.bc5cdr_cdr_zip_url)
    else:
        documents = load_domain_norm_documents(DOMAIN_NORM_GOLD)

    m = run_e2e_eval(
        documents=documents,
        predict=lambda text: extract_entities(
            text, model, score_threshold=settings.ner_score_threshold
        ),
        linker=linker,
        dataset=args.dataset,
        artifact_source=settings.mesh_artifact_path,
        n_aliases=len(dictionary),
        log_path=DEFAULT_LOG,
        git_sha=git_sha(),
        now=datetime.now(UTC).isoformat(),
    )
    c = m.concepts
    print(
        f"{args.dataset}: concept P={c.precision:.4f} R={c.recall:.4f} F1={c.f1:.4f} "
        f"(tp={c.tp} fp={c.fp} fn={c.fn}, docs={m.n_documents}) "
        f"e2e_NIL={m.e2e_nil_rate:.3f}"
    )
    print(f"  census: {m.census.outcomes}")
    print(f"  truncation: {m.census.truncation}")
    print(f"  by label: {m.census.by_label}")
    print(
        f"  merge audit: candidates={m.merge_candidates} linked={m.merge_candidates_linked} "
        f"matching_gold={m.merge_candidates_matching_gold} "
        f"constituents={m.merged_constituents}"
    )


if __name__ == "__main__":
    main()
