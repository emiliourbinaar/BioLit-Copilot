import pytest

from biolit.cluster.pairing import sentence_index
from biolit.domain.enums import EntityLabel, Source, TextType
from biolit.domain.paper import Paper
from biolit.domain.records import Entity
from biolit.extract.deterministic import SameSentenceAsEntitiesExtractor
from biolit.ner.windowing import sentence_spans
from biolit_evals import extract_eval
from biolit_evals.end_to_end import metrics_from_counts
from biolit_evals.extract_eval import (
    MissBuckets,
    assert_bucket_closure,
    assert_gold_sentence_recall_anchor,
    assert_gold_sentence_regression_pin,
    classify_misses,
    gold_finding_sentences,
    sentence_metrics,
)
from biolit_evals.mesh_gold import GoldDocument, GoldMention

TEXT = "Metformin was given. Acidosis followed metformin use."

# sentence_spans(MULTI_TEXT) == [(0, 20), (21, 53), (54, 75)] -- verified with the real
# splitter, not assumed. "Aspirin" is at [54, 61), "fever" is at [69, 74).
MULTI_TEXT = "Metformin was given. Acidosis followed metformin use. Aspirin caused fever."


def _m(start: int, end: int, label: EntityLabel, mesh_ids: tuple[str, ...]) -> GoldMention:
    return GoldMention(
        pmid="1", start=start, end=end, text=TEXT[start:end], label=label, mesh_ids=mesh_ids
    )


def test_a_gold_sentence_needs_both_endpoints_of_one_relation_in_it():
    # Sentence 0 holds the chemical alone. Sentence 1 holds BOTH endpoints -> gold = {1}.
    # A construction that only required one endpoint would return {0, 1}.
    doc = GoldDocument(
        pmid="1",
        text=TEXT,
        mentions=[
            _m(0, 9, EntityLabel.CHEMICAL, ("MESH:D008687",)),
            _m(21, 29, EntityLabel.DISEASE, ("MESH:D000138",)),
            _m(39, 48, EntityLabel.CHEMICAL, ("MESH:D008687",)),
        ],
    )
    relations = {"1": {("MESH:D008687", "MESH:D000138")}}
    assert gold_finding_sentences([doc], relations) == {"1": {1}}


def test_two_endpoints_present_but_not_of_the_same_relation_is_not_a_gold_sentence():
    # Sentence 1 holds chemical A and disease B, but the only gold relation is (A, C).
    # A construction that checked "any chemical and any disease co-occur" would wrongly
    # return {1} -- that is the cross_product error, one level down.
    doc = GoldDocument(
        pmid="1",
        text=TEXT,
        mentions=[
            _m(0, 9, EntityLabel.CHEMICAL, ("MESH:D008687",)),
            _m(21, 29, EntityLabel.DISEASE, ("MESH:D000138",)),
            _m(39, 48, EntityLabel.CHEMICAL, ("MESH:D008687",)),
        ],
    )
    relations = {"1": {("MESH:D008687", "MESH:D011085")}}
    assert gold_finding_sentences([doc], relations) == {}


def test_chemical_endpoint_alone_without_the_disease_is_not_gold():
    # Sentence 0 holds only the chemical endpoint of the gold relation -- no disease at all
    # anywhere in sentence 0 -- so it must not be marked gold.
    doc = GoldDocument(
        pmid="1",
        text=TEXT,
        mentions=[
            _m(0, 9, EntityLabel.CHEMICAL, ("MESH:D008687",)),
        ],
    )
    relations = {"1": {("MESH:D008687", "MESH:D000138")}}
    assert gold_finding_sentences([doc], relations) == {}


def test_disease_endpoint_alone_without_the_chemical_is_not_gold():
    # Sentence 1 holds only the disease endpoint of the gold relation -- no chemical at all
    # anywhere in sentence 1 -- so it must not be marked gold.
    doc = GoldDocument(
        pmid="1",
        text=TEXT,
        mentions=[
            _m(21, 29, EntityLabel.DISEASE, ("MESH:D000138",)),
        ],
    )
    relations = {"1": {("MESH:D008687", "MESH:D000138")}}
    assert gold_finding_sentences([doc], relations) == {}


def test_a_mention_with_empty_mesh_ids_contributes_no_endpoint():
    # The chemical mention is present in sentence 1 but unlinkable (mesh_ids == ()), so it
    # cannot match either endpoint of the gold relation -- sentence 1 must not be gold even
    # though both labels are physically present in it.
    doc = GoldDocument(
        pmid="1",
        text=TEXT,
        mentions=[
            _m(21, 29, EntityLabel.DISEASE, ("MESH:D000138",)),
            _m(39, 48, EntityLabel.CHEMICAL, ()),
        ],
    )
    relations = {"1": {("MESH:D008687", "MESH:D000138")}}
    assert gold_finding_sentences([doc], relations) == {}


def test_a_mention_starting_in_the_inter_sentence_gap_is_dropped():
    # sentence_spans(TEXT) == [(0, 20), (21, 53)] -- verified with the real splitter. The
    # separator at position 20 (the single space after "given.") belongs to neither span.
    # A mention starting there cannot be placed in a sentence and is silently dropped: it
    # contributes no endpoint, same as any other absent mention, rather than raising or
    # being attributed to a neighboring sentence.
    doc = GoldDocument(
        pmid="1",
        text=TEXT,
        mentions=[
            GoldMention(
                pmid="1",
                start=20,
                end=21,
                text=TEXT[20:21],
                label=EntityLabel.CHEMICAL,
                mesh_ids=("MESH:D008687",),
            ),
            _m(21, 29, EntityLabel.DISEASE, ("MESH:D000138",)),
        ],
    )
    relations = {"1": {("MESH:D008687", "MESH:D000138")}}
    assert gold_finding_sentences([doc], relations) == {}


def test_realized_pairs_trace_to_real_gold_relations_and_never_outnumber_them():
    # Independent invariant, computed from OUTPUT (which sentences gold_finding_sentences
    # actually returned) plus raw mention presence -- NOT sourced from `relations` directly,
    # which is what let the prior version of this test pass unconditionally regardless of
    # what the function returned. 3 sentences, verified via sentence_spans(MULTI_TEXT):
    # [(0, 20), (21, 53), (54, 75)]. Sentence 0: chemical alone. Sentence 1: chemical +
    # disease of a REAL relation -> the only true gold sentence. Sentence 2: a DIFFERENT
    # chemical + disease pair that co-occurs but is not itself a gold relation -- the
    # cross_product trap one level up, sitting right next to 3 declared relations so the
    # <= bound is not vacuously satisfied at cardinality 1.
    doc = GoldDocument(
        pmid="1",
        text=MULTI_TEXT,
        mentions=[
            GoldMention(
                pmid="1",
                start=0,
                end=9,
                text=MULTI_TEXT[0:9],
                label=EntityLabel.CHEMICAL,
                mesh_ids=("MESH:D008687",),
            ),
            GoldMention(
                pmid="1",
                start=21,
                end=29,
                text=MULTI_TEXT[21:29],
                label=EntityLabel.DISEASE,
                mesh_ids=("MESH:D000138",),
            ),
            GoldMention(
                pmid="1",
                start=39,
                end=48,
                text=MULTI_TEXT[39:48],
                label=EntityLabel.CHEMICAL,
                mesh_ids=("MESH:D008687",),
            ),
            GoldMention(
                pmid="1",
                start=54,
                end=61,
                text=MULTI_TEXT[54:61],
                label=EntityLabel.CHEMICAL,
                mesh_ids=("MESH:D000568",),
            ),
            GoldMention(
                pmid="1",
                start=69,
                end=74,
                text=MULTI_TEXT[69:74],
                label=EntityLabel.DISEASE,
                mesh_ids=("MESH:D005334",),
            ),
        ],
    )
    relations = {
        "1": {
            ("MESH:D008687", "MESH:D000138"),  # real: both endpoints co-occur in sentence 1
            ("MESH:D008687", "MESH:D005334"),  # decoy: never co-occur in one sentence
            ("MESH:D000568", "MESH:D000138"),  # decoy: never co-occur in one sentence
        }
    }
    gold = gold_finding_sentences([doc], relations)

    spans = sentence_spans(MULTI_TEXT)
    realized: set[tuple[str, str]] = set()
    for index in gold.get("1", set()):
        chemicals = {
            mesh_id
            for m in doc.mentions
            if m.label is EntityLabel.CHEMICAL and sentence_index(spans, m.start) == index
            for mesh_id in m.mesh_ids
        }
        diseases = {
            mesh_id
            for m in doc.mentions
            if m.label is EntityLabel.DISEASE and sentence_index(spans, m.start) == index
            for mesh_id in m.mesh_ids
        }
        realized |= {(c, d) for c in chemicals for d in diseases}

    assert realized <= relations["1"]
    assert len(realized) <= len(relations["1"])


def test_sentence_metrics_cover_papers_present_on_only_one_side():
    # pmid "1" shared -> tp. pmid "2" gold-only (the arm produced nothing for it, e.g. NER
    # found no entities) -> fn. pmid "3" pred-only (selected a sentence in a paper with no
    # gold relation) -> fp. An `&` mutant on `set(pred) | set(gold)` iterates only "1" and
    # silently drops BOTH the whole-document miss and the whole-document hallucination.
    gold = {"1": {0, 1}, "2": {4}}
    pred = {"1": {1, 2}, "3": {0}}
    m = sentence_metrics(pred, gold)
    assert (m.tp, m.fp, m.fn) == (1, 2, 2)


def test_the_recall_anchor_catches_even_a_one_in_a_thousand_miss():
    # control-gold recall is 1.0000 BY CONSTRUCTION: every gold sentence holds both gold
    # endpoints, so a co-occurrence selector on gold mentions cannot miss one.
    # The near-miss case (999 tp, 1 fn -> 0.9990) is what discriminates against a loosened
    # tolerance: round(.,4) raises, round(.,2) would not.
    assert_gold_sentence_recall_anchor(metrics_from_counts(10, 5, 0), arm="control-gold")
    with pytest.raises(SystemExit, match="gold-sentence recall"):
        assert_gold_sentence_recall_anchor(metrics_from_counts(999, 0, 1), arm="control-gold")


def test_the_regression_pin_passes_untabulated_sizes_and_catches_a_changed_count():
    # A pin, not a validation: it catches a CHANGE in gold construction, not an ERROR in it.
    # Untabulated sizes must pass so unit fixtures need no entry.
    assert_gold_sentence_regression_pin(3, 99)
    assert_gold_sentence_regression_pin(0, 0)


def test_the_regression_pin_raises_on_a_changed_count_for_a_tabulated_size(monkeypatch):
    # The happy-path test above never populates a pin entry, so the `!=` comparison and the
    # entire SystemExit branch of assert_gold_sentence_regression_pin have zero coverage
    # there: `==` instead of `!=`, or the `raise` deleted outright, would still pass it.
    # Populate ONE entry locally (never touch the shipped `_GOLD_SENTENCE_PINS`, which stays
    # `{}` until Task 8) and exercise both the raising and non-raising paths.
    monkeypatch.setitem(extract_eval._GOLD_SENTENCE_PINS, 500, 1234)
    assert_gold_sentence_regression_pin(500, 1234)
    with pytest.raises(SystemExit, match="gold-sentence pin"):
        assert_gold_sentence_regression_pin(500, 1235)


def _paper(pmid: str, text: str) -> Paper:
    return Paper(
        id=pmid,
        source=Source.pubmed,
        title="t",
        abstract=text,
        text_type=TextType.abstract_only,
        extraction_allowed=True,
    )


def test_each_miss_bucket_is_populated_distinctly():
    # THREE papers, one per bucket, with DISTINCT non-zero counts so a swapped or combined
    # implementation fails. Conflating them would make "the ceiling" unactionable, the same
    # defect as Phase 3C's diluted mentions_wrong/fp ratio.
    #   pA -> endpoint_lost: the disease is never linked anywhere in entities_by_paper.
    #   pB -> never_co_sentential: both endpoints linked, but in different sentences.
    #   pC -> co_sentential_elsewhere: both co-occur in sentence 0, gold sentences are 1/2/3.
    chem, dis = "MESH:D008687", "MESH:D000138"

    # --- pA: endpoint_lost (count 1) ----------------------------------------------------
    # sentence_spans(text_a) == [(0, 20), (21, 53)] -- verified with the real splitter
    # (text_a is TEXT, reused from test_a_gold_sentence_needs_both_endpoints_of_one_relation
    # _in_it, same offsets). Gold sentence 1 holds both endpoints. entities_by_paper for
    # "pA" never links a disease AT ALL, so the disease endpoint is unreachable anywhere.
    text_a = TEXT
    doc_a = GoldDocument(
        pmid="pA",
        text=text_a,
        mentions=[
            GoldMention(
                pmid="pA",
                start=0,
                end=9,
                text=text_a[0:9],
                label=EntityLabel.CHEMICAL,
                mesh_ids=(chem,),
            ),
            GoldMention(
                pmid="pA",
                start=21,
                end=29,
                text=text_a[21:29],
                label=EntityLabel.DISEASE,
                mesh_ids=(dis,),
            ),
            GoldMention(
                pmid="pA",
                start=39,
                end=48,
                text=text_a[39:48],
                label=EntityLabel.CHEMICAL,
                mesh_ids=(chem,),
            ),
        ],
    )
    entities_a = [
        Entity(text="Metformin", label=EntityLabel.CHEMICAL, start=0, end=9, canonical_id=chem),
    ]

    # --- pB: never_co_sentential (count 2) -----------------------------------------------
    # sentence_spans(text_b) == [(0, 20), (21, 53), (54, 74), (75, 107)] -- verified.
    # Gold sentences 1 and 3 each hold both endpoints. entities_by_paper links the chemical
    # in sentence 0 and the disease in sentence 2 -- both reachable, but never co-sentential
    # anywhere in the paper, including at either gold sentence.
    text_b = (
        "Metformin was given. Acidosis followed metformin use. "
        "Nausea was reported. Metformin caused acidosis again."
    )
    doc_b = GoldDocument(
        pmid="pB",
        text=text_b,
        mentions=[
            GoldMention(
                pmid="pB",
                start=21,
                end=29,
                text=text_b[21:29],
                label=EntityLabel.DISEASE,
                mesh_ids=(dis,),
            ),
            GoldMention(
                pmid="pB",
                start=39,
                end=48,
                text=text_b[39:48],
                label=EntityLabel.CHEMICAL,
                mesh_ids=(chem,),
            ),
            GoldMention(
                pmid="pB",
                start=75,
                end=84,
                text=text_b[75:84],
                label=EntityLabel.CHEMICAL,
                mesh_ids=(chem,),
            ),
            GoldMention(
                pmid="pB",
                start=92,
                end=100,
                text=text_b[92:100],
                label=EntityLabel.DISEASE,
                mesh_ids=(dis,),
            ),
        ],
    )
    entities_b = [
        Entity(text="Metformin", label=EntityLabel.CHEMICAL, start=0, end=9, canonical_id=chem),
        Entity(text="Nausea", label=EntityLabel.DISEASE, start=54, end=60, canonical_id=dis),
    ]

    # --- pC: co_sentential_elsewhere (count 3) ---------------------------------------------
    # sentence_spans(text_c) == [(0, 26), (27, 59), (60, 94), (95, 130)] -- verified.
    # Gold sentences 1, 2, 3 each hold both endpoints (sentence 0 carries no gold mentions).
    # entities_by_paper links both endpoints together ONLY in sentence 0 -- reachable and
    # co-sentential, but never at any of the three gold sentences, so all three misses land
    # in this bucket.
    text_c = (
        "Metformin caused acidosis. Acidosis followed metformin use. "
        "Metformin caused further acidosis. Acidosis persisted after metformin."
    )
    doc_c = GoldDocument(
        pmid="pC",
        text=text_c,
        mentions=[
            GoldMention(
                pmid="pC",
                start=27,
                end=35,
                text=text_c[27:35],
                label=EntityLabel.DISEASE,
                mesh_ids=(dis,),
            ),
            GoldMention(
                pmid="pC",
                start=45,
                end=54,
                text=text_c[45:54],
                label=EntityLabel.CHEMICAL,
                mesh_ids=(chem,),
            ),
            GoldMention(
                pmid="pC",
                start=60,
                end=69,
                text=text_c[60:69],
                label=EntityLabel.CHEMICAL,
                mesh_ids=(chem,),
            ),
            GoldMention(
                pmid="pC",
                start=85,
                end=93,
                text=text_c[85:93],
                label=EntityLabel.DISEASE,
                mesh_ids=(dis,),
            ),
            GoldMention(
                pmid="pC",
                start=95,
                end=103,
                text=text_c[95:103],
                label=EntityLabel.DISEASE,
                mesh_ids=(dis,),
            ),
            GoldMention(
                pmid="pC",
                start=120,
                end=129,
                text=text_c[120:129],
                label=EntityLabel.CHEMICAL,
                mesh_ids=(chem,),
            ),
        ],
    )
    entities_c = [
        Entity(text="Metformin", label=EntityLabel.CHEMICAL, start=0, end=9, canonical_id=chem),
        Entity(text="acidosis", label=EntityLabel.DISEASE, start=17, end=25, canonical_id=dis),
    ]

    docs = [doc_a, doc_b, doc_c]
    relations = {"pA": {(chem, dis)}, "pB": {(chem, dis)}, "pC": {(chem, dis)}}
    entities = {"pA": entities_a, "pB": entities_b, "pC": entities_c}

    gold = gold_finding_sentences(docs, relations)
    assert gold == {"pA": {1}, "pB": {1, 3}, "pC": {1, 2, 3}}

    # pred is what the actual control arm selects on these same entities -- not a
    # hand-rolled stand-in for it.
    extractor = SameSentenceAsEntitiesExtractor(entities)
    pred = {
        doc.pmid: {f.sentence_index for f in extractor.findings(_paper(doc.pmid, doc.text))}
        for doc in docs
    }
    assert pred == {"pA": set(), "pB": set(), "pC": {0}}

    buckets = classify_misses(docs, relations, gold, pred, entities_by_paper=entities)
    assert (
        buckets.endpoint_lost,
        buckets.never_co_sentential,
        buckets.co_sentential_elsewhere,
    ) == (1, 2, 3)
    assert buckets.total == 6


def test_bucket_closure_raises_when_a_miss_is_unaccounted():
    assert_bucket_closure(MissBuckets(1, 2, 3, 6), n_false_negatives=6)
    with pytest.raises(SystemExit, match="bucket closure"):
        assert_bucket_closure(MissBuckets(1, 2, 3, 6), n_false_negatives=7)


def test_bucket_closure_raises_when_the_buckets_do_not_sum_to_total():
    # `total` is a plain field, freely settable independent of the three buckets --
    # MissBuckets(1, 2, 3, 99) is constructible. Here total(99) == n_false_negatives(99), so
    # the closure check above alone would NOT catch this: the buckets summing to 6 while
    # total claims 99 needs its own check, with a message distinct from "bucket closure" so
    # the two failures are distinguishable.
    assert_bucket_closure(MissBuckets(1, 2, 3, 6), n_false_negatives=6)
    with pytest.raises(SystemExit, match="bucket sum"):
        assert_bucket_closure(MissBuckets(1, 2, 3, 99), n_false_negatives=99)
