from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from biolit.cluster.pairing import linked_ids, sentence_index
from biolit.domain.enums import EntityLabel
from biolit.domain.records import Entity
from biolit.ner.windowing import sentence_spans
from biolit_evals.end_to_end import ConceptMetrics, metrics_from_counts
from biolit_evals.mesh_gold import GoldDocument


def _gold_pairs_by_sentence(
    document: GoldDocument, pairs: set[tuple[str, str]]
) -> dict[int, set[tuple[str, str]]]:
    """Per sentence index, the gold relations whose BOTH endpoints are annotated in it.

    Shared by `gold_finding_sentences` (a sentence is gold iff it has a non-empty entry) and
    `classify_misses` (a miss is classified against its own entry, never the document's whole
    relation set). Factored out so the two cannot drift: if classification asked a different
    question than gold construction answered, buckets would describe misses that are not the
    misses the gold produced, and no test or anchor would see it.

    Sentences with no qualifying pair are omitted, so the returned keys ARE the gold
    sentences. Mentions whose start falls in no sentence span are dropped, as are labels
    outside CHEMICAL/DISEASE and mentions with empty `mesh_ids`.
    """
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
    qualifying: dict[int, set[tuple[str, str]]] = {}
    for index in set(chemicals) & set(diseases):
        matched = {(c, d) for c, d in pairs if c in chemicals[index] and d in diseases[index]}
        if matched:
            qualifying[index] = matched
    return qualifying


def _reject_repeated_pmids(documents: Sequence[GoldDocument], *, caller: str) -> None:
    """CHECKED CONTRACT: at most one `GoldDocument` per pmid, for the whole module.

    Every mapping this module produces or consumes is keyed `pmid -> set[sentence index]`,
    and a sentence index is only meaningful relative to ONE `document.text`. Two documents
    sharing a pmid therefore denote DIFFERENT sentences under the same key, and every way of
    reconciling them is silently wrong: unioning the index sets conflates two unrelated
    sentences, while assigning drops all but the last, order-dependently. `classify_misses`
    is affected the same way -- it looks up one `gold[pmid]` per document, so a repeated pmid
    re-classifies the same misses once per duplicate.

    So this raises instead of choosing. `ValueError`, not the `SystemExit` the anchors and
    pins use: those report a RESULT that fails an eval invariant and must halt a run, whereas
    this is an argument-shape violation detected before anything is measured -- the same
    category as `mesh_gold`'s own `ValueError`s for out-of-bounds spans and mismatched span
    text, and catchable by an ordinary `except Exception`.
    """
    seen: set[str] = set()
    for document in documents:
        if document.pmid in seen:
            raise ValueError(
                f"{caller}: repeated pmid {document.pmid!r} in `documents`. A sentence index "
                "is only meaningful relative to a single document text, so two documents "
                "sharing a pmid denote different sentences under one key -- unioning them "
                "conflates unrelated sentences and assigning silently drops all but one. "
                "`parse_pubtator_documents` yields one document per pmid; "
                "`load_domain_norm_documents` yields one per annotated record and so CAN "
                "repeat a pmid. Merge those records into one document per pmid first."
            )
        seen.add(document.pmid)


def gold_finding_sentences(
    documents: Sequence[GoldDocument],
    relations: Mapping[str, set[tuple[str, str]]],
) -> dict[str, set[int]]:
    """Sentences where BOTH endpoints of at least one gold CID relation are gold-annotated.

    Raises `ValueError` if two documents share a pmid -- see `_reject_repeated_pmids`.

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
    _reject_repeated_pmids(documents, caller="gold_finding_sentences")
    gold: dict[str, set[int]] = {}
    for document in documents:
        pairs = relations.get(document.pmid)
        if not pairs:
            continue
        qualifying = _gold_pairs_by_sentence(document, pairs)
        if qualifying:
            gold[document.pmid] = set(qualifying)
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

    Every bucket is decided PER MISSED SENTENCE against only the gold relations that made
    THAT sentence gold -- never against the document's whole relation set. At ~2.13 gold CID
    relations per Test-500 document, pooling would let one reachable relation anywhere in a
    paper suppress `endpoint_lost` for a miss it had nothing to do with.

    endpoint_lost           -- no relation qualifying that sentence has both endpoints linked
                               anywhere in the paper. UNRECOVERABLE by any window or pairing
                               mechanism, and the sentence-level counterpart of ADR-0013's
                               PER-RELATION 40.3% (430 of 1066 gold CID relations in Test-500
                               lose an endpoint). It is the subset on which the LLM arm's
                               recall is the direct proof of bottleneck escape. Not the same
                               number as 40.3%: that counts relations, this counts gold
                               sentences, and one sentence can be made gold by several
                               relations. Do not quote them as the same statistic.
    never_co_sentential     -- >=1 qualifying relation is reachable, but no sentence in the
                               paper holds both endpoints of any reachable qualifying
                               relation. Recoverable by a wider window; prices same-paragraph
                               variants.
    co_sentential_elsewhere -- a reachable qualifying relation IS co-sentential somewhere,
                               just not at the gold sentence. Gold-vs-real span disagreement;
                               costs a false positive as well as this false negative.

    Evaluated in that order and mutually exclusive, so the three sum to total.

    The direction of any error here is not neutral. Misrouting an unrecoverable miss into
    either recoverable bucket inflates the headroom a downstream component appears to have,
    which is exactly what ADR-0013's standing finding says to guard against: price
    downstream reasoning against an entity-conditioned ceiling, not an oracle-entity one.
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
    """Bucket every false negative in `pred` against `gold`. See `MissBuckets` for the rules.

    ONE CHECKED PRECONDITION: `documents` must hold at most one entry per pmid, or this
    raises `ValueError` -- see `_reject_repeated_pmids`. A duplicate would re-classify the
    same `gold[pmid]` misses once per copy.

    TWO UNENFORCED ONES, below. Nothing in the signature checks either, and a
    caller that violates one gets silently corrupted buckets: no exception is raised, and
    `assert_bucket_closure` still passes, because closure only checks that the misses were
    counted, not that each landed in the right bucket.

    1. SAME SELECTOR, SAME ENTITIES. `pred` must have been produced by the same
       same-sentence co-occurrence rule over the SAME `entities_by_paper` passed here. The
       name `co_sentential_elsewhere` asserts that the selector saw a co-sentential pair and
       chose a different sentence. Score a different selector, or the same one over a
       different entity set, and "elsewhere" describes nothing real.
    2. SAME TEXT. This function splits `sentence_spans(document.text)`, while
       `SameSentenceAsEntitiesExtractor` splits `paper.abstract or ""`. For real BC5CDR,
       `GoldDocument.text` is `title + " " + abstract` (see `parse_pubtator_documents`), so a
       `main()` that naively builds `Paper(title=..., abstract=...)` desynchronises every
       sentence index between `pred` and `gold` -- silently, and the more so the longer the
       title. The caller must feed the extractor the SAME string as `document.text`.
    """
    _reject_repeated_pmids(documents, caller="classify_misses")
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
        gold_pairs = _gold_pairs_by_sentence(document, pairs)
        for index in sorted(missed):
            pairs_at_index = gold_pairs.get(index, set())
            reachable = [(c, d) for c, d in pairs_at_index if c in chemicals and d in diseases]
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

    WHAT IT ACTUALLY DETECTS, stated no stronger than the code supports. `classify_misses`
    computes `total` as `lost + never + elsewhere` in the same pass, and the `continue` after
    `endpoint_lost` makes the branches mutually exclusive, so on that path the identity holds
    BY CONSTRUCTION. It is not independent evidence in the way the three-way reachable-share
    / oracle-recall / cross-product-recall agreement at 0.5966 was. Its real value is as a
    mutation detector for exactly that exclusivity -- a dropped `continue`, a branch that
    increments two counters, or a hand-built `MissBuckets` whose `total` is a free field --
    plus the population check below. Labelled honestly rather than dressed up as an
    independent invariant, the same way `assert_gold_sentence_regression_pin` is.

    IT ALSO COMPARES TWO POPULATIONS THAT CAN DIFFER. `buckets.total` sums over the
    `documents` sequence; `n_false_negatives` normally comes from `sentence_metrics`, whose
    `fn` sums over `set(pred) | set(gold)`. One way they diverge remains live: a gold pmid
    absent from `documents` contributes its misses to `fn` but has none classified here.
    Suspect that caller shape first when the two sides differ by a whole document's worth of
    misses, rather than hunting for the misclassification the message below blames.

    The OTHER divergence an earlier version of this docstring described -- a pmid repeated in
    `documents`, double-counting its misses here -- can no longer reach this check. Both
    `gold_finding_sentences` and `classify_misses` now raise `ValueError` on a repeated pmid
    (see `_reject_repeated_pmids`), because a sentence index is only meaningful relative to
    one document text. Do not re-add "duplicate pmids are benign caller shape" guidance: that
    shape is now rejected at the source, not tolerated and diagnosed here.
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
