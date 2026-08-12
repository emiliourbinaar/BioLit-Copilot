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


def _freeze(sentences: Mapping[str, set[int]]) -> dict[str, frozenset[int]]:
    """Make a bucket's accumulated membership immutable, matching the frozen dataclass."""
    return {pmid: frozenset(indices) for pmid, indices in sentences.items()}


def _n_sentences(sentences: Mapping[str, frozenset[int]]) -> int:
    """Total (pmid, sentence index) pairs in a bucket's membership mapping."""
    return sum(len(indices) for indices in sentences.values())


@dataclass(frozen=True)
class MissBuckets:
    """Why control-real missed each gold sentence. Four buckets, four different fixes.

    CARRIES IDENTITY, NOT ONLY COUNTS. Each bucket is stored as its membership -- a
    `pmid -> frozenset[sentence index]` mapping -- and every count, `total` included, is a
    PROPERTY derived from it. There is no separate counter to disagree with the membership,
    so count/membership drift is impossible by construction rather than merely tested
    against. Task 8's `recall_on_endpoint_lost` restricts gold to bucket (a) and scores the
    LLM arm against that restriction alone; a count cannot express which sentences those are,
    which is why membership is stored rather than reconstructed. Membership is exposed for
    ALL FOUR buckets, not just (a): a reader given (a) alone cannot check that the buckets
    partition the misses rather than overlap. A pmid with no miss in a bucket contributes no
    key at all, so a restricted gold names exactly the papers that bucket touches.

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
    endpoint_unlocatable    -- >=1 qualifying relation has both endpoints LINKED, but every
                               such relation has >=1 endpoint whose only linked entities
                               carry no offset (`Entity.start is None`). ALSO UNRECOVERABLE,
                               and for a stronger reason than (a): the endpoint's position is
                               undefined, not merely distant, so no window variant and no
                               pairing rule can place it. Kept SEPARATE from (a) rather than
                               folded into it, because (a) is defined as "no qualifying
                               relation has both endpoints linked anywhere" and these
                               endpoints ARE linked -- folding would make (a)'s own
                               definition false. See `classify_misses` for why this is a
                               bucket rather than a rejected precondition.
    never_co_sentential     -- >=1 qualifying relation is reachable AND both its endpoints
                               are locatable, but no sentence in the paper holds both.
                               Recoverable by a wider window; prices same-paragraph variants.
                               An endpoint whose integer offset falls in the inter-sentence
                               gap `sentence_spans` leaves uncovered belongs HERE, not in
                               (a'): its position is known, so a different splitter or a
                               paragraph window reaches it.
    co_sentential_elsewhere -- a reachable, locatable qualifying relation IS co-sentential
                               somewhere, just not at the gold sentence. Gold-vs-real span
                               disagreement; costs a false positive as well as this false
                               negative.

    Evaluated in that order and mutually exclusive, so the four sum to total. (a') sits
    immediately after (a) for the same reason (a) precedes (b): unrecoverable-by-any-
    mechanism outranks fixable-by-windowing, and a miss must be priced by its most
    fundamental obstacle.

    The direction of any error here is not neutral. Misrouting an unrecoverable miss into
    either recoverable bucket inflates the headroom a downstream component appears to have,
    which is exactly what ADR-0013's standing finding says to guard against: price
    downstream reasoning against an entity-conditioned ceiling, not an oracle-entity one.
    """

    endpoint_lost_sentences: Mapping[str, frozenset[int]]
    endpoint_unlocatable_sentences: Mapping[str, frozenset[int]]
    never_co_sentential_sentences: Mapping[str, frozenset[int]]
    co_sentential_elsewhere_sentences: Mapping[str, frozenset[int]]

    @property
    def endpoint_lost(self) -> int:
        return _n_sentences(self.endpoint_lost_sentences)

    @property
    def endpoint_unlocatable(self) -> int:
        return _n_sentences(self.endpoint_unlocatable_sentences)

    @property
    def never_co_sentential(self) -> int:
        return _n_sentences(self.never_co_sentential_sentences)

    @property
    def co_sentential_elsewhere(self) -> int:
        return _n_sentences(self.co_sentential_elsewhere_sentences)

    @property
    def total(self) -> int:
        return (
            self.endpoint_lost
            + self.endpoint_unlocatable
            + self.never_co_sentential
            + self.co_sentential_elsewhere
        )


def classify_misses(
    documents: Sequence[GoldDocument],
    relations: Mapping[str, set[tuple[str, str]]],
    gold: Mapping[str, set[int]],
    pred: Mapping[str, set[int]],
    *,
    entities_by_paper: Mapping[str, Sequence[Entity]],
) -> MissBuckets:
    """Bucket every false negative in `pred` against `gold`. See `MissBuckets` for the rules.

    TWO CHECKED PRECONDITIONS, both raising `ValueError` before any bucket is reported:

    A. AT MOST ONE DOCUMENT PER PMID in `documents` -- see `_reject_repeated_pmids`. A
       duplicate would re-classify the same `gold[pmid]` misses once per copy.
    B. `gold` CAME FROM `gold_finding_sentences(documents, relations)` OVER THESE SAME
       ARGUMENTS. Checked, for EVERY document in `documents` and before any of it is used, as
       the full equality `gold.get(pmid, set()) == set(_gold_pairs_by_sentence(document,
       relations.get(pmid, set())))`. That is exactly the predicate `gold_finding_sentences`
       computes, so over the pmids in `documents` the check is NECESSARY AND SUFFICIENT for
       the precondition, not merely necessary. Both directions of divergence corrupt a
       reported number, which is why neither is tolerated:
         - an index in `gold` that NO gold relation qualifies has no qualifying pairs, hence
           nothing reachable. Missed by `pred`, it is bucketed `endpoint_lost` -- the
           UNRECOVERABLE bucket, inflating apparent downstream headroom. Selected by `pred`,
           it is never bucketed at all, yet still scores as a true positive in
           `sentence_metrics`, whose precision then describes a gold that does not exist.
         - a genuine gold sentence MISSING from `gold` never becomes a miss, so it shrinks
           WHICHEVER bucket that miss would have landed in. `endpoint_lost` is the
           plurality, not the rule: measured over the omission population it took 54% of it
           in one sample and 41% in another, differently shaped one, with
           `never_co_sentential` and `co_sentential_elsewhere` splitting the remainder.
           Quote neither figure as fixed -- the split tracks the arm's linking rate -- and
           note that the fixture pinning this direction happens to be a pure
           `endpoint_lost` case, a 2-of-2 population reported as 1. A further quarter to a
           third of omissions shrink NO bucket at all, because `pred` had selected the
           omitted sentence, which was therefore never a miss; those still corrupt a
           number, just not one of these three -- in `sentence_metrics` the sentence scores
           as a false positive instead of the true positive it is.
           `assert_bucket_closure` sees none of it, in either direction: it compares
           against a false-negative count computed from the same wrong `gold`, which
           deflates in lockstep.
       WHAT IT STILL DOES NOT COVER, stated no stronger than the code supports: a pmid in
       `gold` but ABSENT from `documents` is never visited, so its entry is unchecked. That
       is the same population `assert_bucket_closure` warns about, where `sentence_metrics`
       counts misses this function never classifies.
       COST. The gold-pair mapping is now built for every document, including those with no
       misses, instead of only inside the has-misses branch. Measured on a synthetic
       500-document corpus shaped like Test-500 (9 sentences, 20 mentions, 2 relations per
       document): 26.7 ms -> 31.8 ms with half the gold sentences missed, and 0.2 ms ->
       19.5 ms in the degenerate case where NOTHING is missed and the old placement skipped
       every document. This is an offline harness that runs once per arm behind NER over the
       same 500 abstracts; tens of milliseconds buy an exact check, and the trade is not
       close.

    TWO UNENFORCED ONES, below. Both are properties of how `pred` was PRODUCED, and neither
    `pred` nor this signature carries any record of that, so nothing here can check them: a
    caller that violates one gets silently corrupted buckets: no exception is raised, and
    `assert_bucket_closure` still passes, because closure only checks that the misses were
    counted, not that each landed in the right bucket.

    1. SAME SELECTOR, SAME ENTITIES. `pred` must have been produced by the same
       same-sentence co-occurrence rule over the SAME `entities_by_paper` passed here. The
       name `co_sentential_elsewhere` asserts that the selector saw a co-sentential pair and
       chose a different sentence. Score a different selector, or the same one over a
       different entity set, and "elsewhere" describes nothing real.
    THE LINKED-BUT-UNLOCATABLE POPULATION, and why it is a BUCKET and not a REJECTION.
    `linked_ids` filters on `canonical_id is not None` alone and ignores `start`, so an
    entity with `canonical_id is not None and start is None` makes `reachable` non-empty --
    escaping `endpoint_lost` -- while being unable to appear in `per_sentence` at all. Before
    this was separated out, such a miss landed in `never_co_sentential`: a bucket meaning
    "the endpoints exist and are placed, they merely never share a sentence", which is quoted
    as headroom a windowing change could recover. No window variant recovers an endpoint with
    no offset, so that inflated the recoverable population -- ADR-0013's standing finding in
    miniature.

    IS THE POPULATION REACHABLE? Established from source, not assumed, because the answer
    decides between a fourth bucket and a rejected precondition.
      - NOT on the gold-mention-derived path. `GoldMention.start` is a required `int` (parsed
        `int(start)` in `parse_pubtator`), and the gold-entity construction passes
        `start=mention.start` straight through (`cluster_eval.synthesize_records`, the shape
        the runner reuses for `control-gold`). Every gold-derived entity therefore has an
        integer offset, and a linked one is always locatable.
      - YES on the real NER + linking path, which is the one that feeds this function's
        `entities_by_paper` for `control-real`. `extract_entities` assigns
        `start = span.get("start")` and branches on `start is not None`, falling back to the
        pipeline's `word` field for the surface -- it KEEPS such a span rather than dropping
        it, and sorts with an explicit `e.start if e.start is not None else 0`.
        `predict_windowed` likewise propagates a `None` offset rather than discarding the
        span, and `_is_contained` returns False for it so it survives de-duplication.
        `canonicalize` then links on `entity.text` alone and writes back through
        `model_copy(update={"canonical_id": ..., "canonical_name": ...})`, which never
        touches `start`. So a span the model returns without offsets, whose surface links,
        arrives here linked and unlocatable.
      - The repository already treats this population as ROUTINE rather than malformed:
        `pairing_diagnostics` counts it in a dedicated `unplaceable_entities` field,
        `SameSentencePairing` documents that it "FAILS CLOSED" on it, and
        `SameSentenceAsEntitiesExtractor` -- the very selector that produces `pred` -- skips
        it with the identical guard. All three have fixtures constructing it directly.

    REJECTED ALTERNATIVE: raise `ValueError` on it as an argument-shape violation, the way
    `_reject_repeated_pmids` does. Rejected because the population is reachable on the
    production path, so this would halt a real 500-document run over an occurrence the
    selector standing beside it handles as a matter of course, and would make this function
    stricter about its entities than the arm being measured. A repeated pmid has no correct
    interpretation; an unlocatable endpoint has one, and it is bucket (a').

    ALSO REJECTED: folding it into `endpoint_lost`. That bucket is defined as "no relation
    qualifying that sentence has both endpoints linked anywhere in the paper", and a
    linked-but-unlocatable endpoint IS linked, so folding would make (a)'s own definition
    false while hiding a distinct diagnosis (linking succeeded, offsets did not) inside one
    that blames linking.

    2. SAME TEXT. This function splits `sentence_spans(document.text)`, while
       `SameSentenceAsEntitiesExtractor` splits `paper.abstract or ""`. For real BC5CDR,
       `GoldDocument.text` is `title + " " + abstract` (see `parse_pubtator_documents`), so a
       `main()` that naively builds `Paper(title=..., abstract=...)` desynchronises every
       sentence index between `pred` and `gold` -- silently, and the more so the longer the
       title. The caller must feed the extractor the SAME string as `document.text`.
    """
    _reject_repeated_pmids(documents, caller="classify_misses")
    lost: dict[str, set[int]] = {}
    unlocatable: dict[str, set[int]] = {}
    never: dict[str, set[int]] = {}
    elsewhere: dict[str, set[int]] = {}
    for document in documents:
        pairs = relations.get(document.pmid, set())
        gold_pairs = _gold_pairs_by_sentence(document, pairs)
        declared = gold.get(document.pmid, set())
        invented = sorted(declared - set(gold_pairs))
        if invented:
            raise ValueError(
                f"classify_misses: sentence {invented[0]} of pmid {document.pmid!r} is in "
                "`gold` but no gold relation qualifies it, so it is not a gold finding "
                "sentence for these `documents` and `relations`. `gold` must be the output of "
                "`gold_finding_sentences(documents, relations)` over the SAME arguments. A "
                "miss at a sentence this function does not consider gold has no qualifying "
                "pairs, hence nothing reachable, and is silently bucketed as `endpoint_lost` "
                "-- the UNRECOVERABLE bucket, so the error inflates apparent downstream "
                "headroom -- while `assert_bucket_closure` still passes, because closure "
                "checks only that misses were counted, never WHICH bucket each landed in. An "
                "invented index `pred` also selected is never bucketed at all, but still "
                "scores as a true positive in `sentence_metrics`."
            )
        omitted = sorted(set(gold_pairs) - declared)
        if omitted:
            raise ValueError(
                f"classify_misses: sentence {omitted[0]} of pmid {document.pmid!r} is a gold "
                "finding sentence for these `documents` and `relations` but is absent from "
                "`gold`. `gold` must be the output of "
                "`gold_finding_sentences(documents, relations)` over the SAME arguments. An "
                "omitted gold sentence can never become a miss, so it silently shrinks every "
                "bucket -- including `endpoint_lost`, the UNRECOVERABLE population this eval "
                "reports -- while `assert_bucket_closure` still passes, because the false "
                "negative count it compares against is computed from the same `gold` and "
                "deflates in lockstep."
            )
        missed = declared - pred.get(document.pmid, set())
        if not missed:
            continue
        entities = entities_by_paper.get(document.pmid, ())
        chemicals = linked_ids(entities, EntityLabel.CHEMICAL)
        diseases = linked_ids(entities, EntityLabel.DISEASE)
        # `linked_ids` filters on `canonical_id is not None` and IGNORES `start`, so the two
        # sets above admit an entity that is linked but carries no offset. Re-running the SAME
        # definition over the located entities alone gives the ids an offset-based mechanism
        # can actually reach. Restricting on `start is not None` and NOT on
        # `sentence_index(...) is not None` is deliberate: an entity whose integer offset
        # lands in the inter-sentence gap `sentence_spans` leaves uncovered has a KNOWN
        # position that a different splitter or a wider window recovers, so it belongs in
        # `never_co_sentential`; an entity with no offset at all does not.
        located = [entity for entity in entities if entity.start is not None]
        locatable_chemicals = linked_ids(located, EntityLabel.CHEMICAL)
        locatable_diseases = linked_ids(located, EntityLabel.DISEASE)
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
        for index in sorted(missed):
            pairs_at_index = gold_pairs.get(index, set())
            reachable = [(c, d) for c, d in pairs_at_index if c in chemicals and d in diseases]
            if not reachable:
                lost.setdefault(document.pmid, set()).add(index)
                continue
            locatable = [
                (c, d) for c, d in reachable if c in locatable_chemicals and d in locatable_diseases
            ]
            if not locatable:
                unlocatable.setdefault(document.pmid, set()).add(index)
                continue
            co_sentential = any(
                c in chem and d in dis for chem, dis in per_sentence.values() for c, d in locatable
            )
            if not co_sentential:
                never.setdefault(document.pmid, set()).add(index)
            else:
                elsewhere.setdefault(document.pmid, set()).add(index)
    return MissBuckets(
        endpoint_lost_sentences=_freeze(lost),
        endpoint_unlocatable_sentences=_freeze(unlocatable),
        never_co_sentential_sentences=_freeze(never),
        co_sentential_elsewhere_sentences=_freeze(elsewhere),
    )


def assert_bucket_closure(buckets: MissBuckets, *, n_false_negatives: int) -> None:
    """HARNESS correctness: the four buckets must account for every false negative, once each.

    WHAT IT ACTUALLY DETECTS, stated no stronger than the code supports, and REVISED because
    `MissBuckets` changed underneath it. It used to claim, among its detections, "a hand-built
    `MissBuckets` whose `total` is a free field". THAT FAILURE MODE NO LONGER EXISTS: `total`
    and every bucket count are now properties derived from the membership mappings, there is
    no count to set independently of the sentences it counts, and the sum check that guarded
    it became provably unfireable and has been deleted rather than kept as a check that cannot
    fail. Do not re-add it, and do not let this docstring keep claiming a detection the code
    no longer performs -- being labelled honestly, the same way
    `assert_gold_sentence_regression_pin` is, is this function's entire stated virtue.

    WHAT MEMBERSHIP MADE CHECKABLE FOR THE FIRST TIME is mutual exclusivity. Previously the
    `continue` after each bucket made the branches exclusive BY CONSTRUCTION, with nothing
    observable to compare, so the claim rested on reading `classify_misses`. The buckets now
    carry which (pmid, sentence index) pairs they hold, so exclusivity is CHECKED DIRECTLY
    below, over every unordered pair of the four buckets: a dropped `continue` or a branch
    that records into two mappings is caught by evidence, not by inspection.

    WHAT REMAINS BY CONSTRUCTION, and is therefore still not evidence: that each miss landed
    in the RIGHT bucket. Closure and disjointness together prove the misses were partitioned,
    never that the partition is the one the bucket definitions describe -- a swapped pair of
    branches passes both checks. That is why the misclassification failures this module cares
    about (a `gold` that did not come from `gold_finding_sentences`, an unlocatable endpoint
    routed to `never_co_sentential`) are guarded by their own raises and fixtures rather than
    here. It is not independent evidence in the way the three-way reachable-share /
    oracle-recall / cross-product-recall agreement at 0.5966 was.

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
    named = (
        ("endpoint_lost", buckets.endpoint_lost_sentences),
        ("endpoint_unlocatable", buckets.endpoint_unlocatable_sentences),
        ("never_co_sentential", buckets.never_co_sentential_sentences),
        ("co_sentential_elsewhere", buckets.co_sentential_elsewhere_sentences),
    )
    # Disjointness FIRST: an overlap also inflates `total`, so checking closure first would
    # report the derived symptom ("a miss was misclassified or double-counted") and bury the
    # exact pair of buckets and the exact sentence that caused it.
    for position, (name, sentences) in enumerate(named):
        for other_name, other in named[position + 1 :]:
            shared = {
                (pmid, index)
                for pmid in set(sentences) & set(other)
                for index in sentences[pmid] & other[pmid]
            }
            if shared:
                pmid, index = sorted(shared)[0]
                raise SystemExit(
                    f"bucket overlap: sentence {index} of pmid {pmid!r} is in BOTH {name} and "
                    f"{other_name} ({len(shared)} such sentence(s)). The buckets must partition "
                    "the misses -- each is a different fix, and a miss counted twice inflates "
                    "`total` past the false-negative count as well as pricing one obstacle as "
                    "two. Suspect a missing `continue` in `classify_misses`."
                )
    if buckets.total != n_false_negatives:
        raise SystemExit(
            f"bucket closure: {buckets.total} classified misses "
            f"({buckets.endpoint_lost} lost + {buckets.endpoint_unlocatable} unlocatable + "
            f"{buckets.never_co_sentential} never co-sentential + "
            f"{buckets.co_sentential_elsewhere} elsewhere) != "
            f"{n_false_negatives} false negatives. A miss was misclassified or double-counted."
        )
