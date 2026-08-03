from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from biolit.cluster.pairing import linked_ids, sentence_index
from biolit.domain.enums import EntityLabel
from biolit.domain.records import Entity
from biolit.ner.windowing import sentence_spans
from biolit_evals.end_to_end import ConceptMetrics, metrics_from_counts
from biolit_evals.mesh_gold import GoldDocument


def gold_finding_sentences(
    documents: Sequence[GoldDocument],
    relations: Mapping[str, set[tuple[str, str]]],
) -> dict[str, set[int]]:
    """Sentences where BOTH endpoints of at least one gold CID relation are gold-annotated.

    PROXY, NOT GROUND TRUTH. BC5CDR annotates CID relations at DOCUMENT level; sentence-level
    co-occurrence is this project's inference about where the relation is asserted. Some
    qualifying sentences state background rather than a finding, and a paper's actual key
    finding may concern efficacy, which CID does not annotate at all. A high score against
    this gold means "selects sentences containing the annotated relation" -- NOT "selects the
    paper's key finding". Cite it that way.

    The resulting sentence COUNT is a regression pin, not an independent validation: no
    published corpus statistic exists for it. The independent invariant is that the number
    of relations with >=1 gold sentence is <= 1066 (BC5CDR Test-500's published gold CID
    relation count) -- every gold sentence traces back to a real relation, never invented.
    """
    gold: dict[str, set[int]] = {}
    for document in documents:
        pairs = relations.get(document.pmid)
        if not pairs:
            continue
        spans = sentence_spans(document.text)
        by_label: dict[EntityLabel, dict[int, set[str]]] = {
            EntityLabel.CHEMICAL: {},
            EntityLabel.DISEASE: {},
        }
        for mention in document.mentions:
            index = sentence_index(spans, mention.start)
            if index is None or mention.label not in by_label:
                continue
            by_label[mention.label].setdefault(index, set()).update(mention.mesh_ids)
        chemicals = by_label[EntityLabel.CHEMICAL]
        diseases = by_label[EntityLabel.DISEASE]
        for index in set(chemicals) & set(diseases):
            if any(c in chemicals[index] and d in diseases[index] for c, d in pairs):
                gold.setdefault(document.pmid, set()).add(index)
    return gold


def sentence_metrics(pred: Mapping[str, set[int]], gold: Mapping[str, set[int]]) -> ConceptMetrics:
    """Micro-averaged P/R/F1 over (paper_id, sentence_index) pairs.

    PRECISION IS LOAD-BEARING AND RECALL IS NOT QUOTABLE ALONE: selecting every sentence
    scores recall 1.0. Report `mean_sentences_per_paper` beside every arm.

    Iterates the UNION of pmids so a paper present on only one side still counts -- an
    intersection would drop whole-document misses and whole-document hallucinations from
    both denominators.
    """
    tp = fp = fn = 0
    for pmid in set(pred) | set(gold):
        predicted, expected = pred.get(pmid, set()), gold.get(pmid, set())
        tp += len(predicted & expected)
        fp += len(predicted - expected)
        fn += len(expected - predicted)
    return metrics_from_counts(tp, fp, fn)


_GOLD_SENTENCE_PINS: dict[int, int] = {}


def assert_gold_sentence_recall_anchor(metrics: ConceptMetrics, *, arm: str) -> None:
    """HARNESS correctness. A co-occurrence selector running on GOLD mentions must recall
    every gold sentence, because a gold sentence is DEFINED as one holding both gold
    endpoints. A miss means the harness is wrong -- sentence splitting, offset handling, or
    MeSH id prefixing -- not that the selector underperformed. Do NOT relax to match an
    observation.
    """
    if round(metrics.recall, 4) != 1.0:
        raise SystemExit(
            f"{arm}: gold-sentence recall is {metrics.recall:.4f}, expected exactly 1.0000 "
            f"(fn={metrics.fn}). A gold sentence holds both gold endpoints by definition, so "
            "a co-occurrence selector on gold mentions cannot miss one. The harness is wrong."
        )


def assert_gold_sentence_regression_pin(n_documents: int, n_gold_sentences: int) -> None:
    """REGRESSION PIN -- deliberately weaker than the cluster-count anchor it resembles.

    Gold cluster counts (500->80, 1500->325) come from independently published corpus
    statistics. No published statistic exists for gold SENTENCES: the value is established
    by this project, so this catches a CHANGE in gold construction, not an ERROR in it.
    Labelled honestly rather than dressed up as loader validation.
    """
    expected = _GOLD_SENTENCE_PINS.get(n_documents)
    if expected is not None and n_gold_sentences != expected:
        raise SystemExit(
            f"gold-sentence pin: {n_documents} documents yielded {n_gold_sentences} gold "
            f"sentences, pinned at {expected}. Gold construction changed. If the change was "
            "deliberate, update the pin IN THE SAME COMMIT as the change and say why."
        )


@dataclass(frozen=True)
class MissBuckets:
    """Why control-real missed each gold sentence. Three buckets, three different fixes.

    endpoint_lost           -- >=1 endpoint has no linked mention anywhere in the paper.
                               UNRECOVERABLE by any window or pairing mechanism. This is the
                               40.3% population from ADR-0013, and the subset on which the
                               LLM arm's recall is the direct proof of bottleneck escape.
    never_co_sentential     -- both endpoints linked somewhere, no sentence holds both.
                               Recoverable by a wider window; prices same-paragraph variants.
    co_sentential_elsewhere -- both linked AND co-sentential, but in a sentence other than
                               the gold one. Gold-vs-real span disagreement; costs a false
                               positive as well as this false negative.

    Evaluated in that order and mutually exclusive, so the three sum to total.
    """

    endpoint_lost: int
    never_co_sentential: int
    co_sentential_elsewhere: int
    total: int


def classify_misses(
    documents: Sequence[GoldDocument],
    relations: Mapping[str, set[tuple[str, str]]],
    gold: Mapping[str, set[int]],
    pred: Mapping[str, set[int]],
    *,
    entities_by_paper: Mapping[str, Sequence[Entity]],
) -> MissBuckets:
    lost = never = elsewhere = 0
    for document in documents:
        missed = gold.get(document.pmid, set()) - pred.get(document.pmid, set())
        if not missed:
            continue
        entities = entities_by_paper.get(document.pmid, ())
        chemicals = linked_ids(entities, EntityLabel.CHEMICAL)
        diseases = linked_ids(entities, EntityLabel.DISEASE)
        spans = sentence_spans(document.text)
        per_sentence: dict[int, tuple[set[str], set[str]]] = {}
        for entity in entities:
            if entity.canonical_id is None or entity.start is None:
                continue
            index = sentence_index(spans, entity.start)
            if index is None:
                continue
            chem, dis = per_sentence.setdefault(index, (set(), set()))
            if entity.label is EntityLabel.CHEMICAL:
                chem.add(entity.canonical_id)
            elif entity.label is EntityLabel.DISEASE:
                dis.add(entity.canonical_id)
        pairs = relations.get(document.pmid, set())
        for _index in sorted(missed):
            reachable = [(c, d) for c, d in pairs if c in chemicals and d in diseases]
            if not reachable:
                lost += 1
                continue
            co_sentential = any(
                c in chem and d in dis for chem, dis in per_sentence.values() for c, d in reachable
            )
            if not co_sentential:
                never += 1
            else:
                elsewhere += 1
    return MissBuckets(
        endpoint_lost=lost,
        never_co_sentential=never,
        co_sentential_elsewhere=elsewhere,
        total=lost + never + elsewhere,
    )


def assert_bucket_closure(buckets: MissBuckets, *, n_false_negatives: int) -> None:
    """HARNESS correctness: the three buckets must account for every false negative.

    A cheap identity that catches a misclassified bucket, in the spirit of the three-way
    reachable-share / oracle-recall / cross-product-recall agreement at 0.5966. Nothing in
    the code forces this to hold, so its holding is evidence.
    """
    if buckets.total != n_false_negatives:
        raise SystemExit(
            f"bucket closure: {buckets.total} classified misses "
            f"({buckets.endpoint_lost} lost + {buckets.never_co_sentential} never "
            f"co-sentential + {buckets.co_sentential_elsewhere} elsewhere) != "
            f"{n_false_negatives} false negatives. A miss was misclassified or double-counted."
        )
    bucket_sum = (
        buckets.endpoint_lost + buckets.never_co_sentential + buckets.co_sentential_elsewhere
    )
    if bucket_sum != buckets.total:
        raise SystemExit(
            f"bucket sum: {buckets.endpoint_lost} lost + {buckets.never_co_sentential} never "
            f"co-sentential + {buckets.co_sentential_elsewhere} elsewhere = {bucket_sum}, but "
            f"total is {buckets.total}. MissBuckets is internally inconsistent -- `total` is a "
            "free field, not a computed sum, and this construction violated the identity."
        )
