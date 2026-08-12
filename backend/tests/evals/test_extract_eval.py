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
    # THIS FIXTURE DOES NOT PIN THE `index is None` GUARD, and never did: without it the
    # chemical lands under key None and the disease under key 1, the intersection is still
    # empty, and {} comes back either way. It documents the intended behaviour; the guard
    # itself is pinned by test_both_endpoints_in_inter_sentence_gaps_do_not_pool_into_one
    # _pseudo_sentence below.
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


def test_both_endpoints_in_inter_sentence_gaps_do_not_pool_into_one_pseudo_sentence():
    # THIS is the fixture that actually pins `if index is None` in _gold_pairs_by_sentence.
    # The test above cannot: its chemical lands under key None and its disease under key 1,
    # so `set(chemicals) & set(diseases)` is empty with OR without the guard and it returns
    # {} either way -- verified by replaying it against a guard-deleted copy of the helper.
    # Here BOTH endpoints of the gold pair start in gaps (positions 20 and 53 of MULTI_TEXT,
    # the separators between the three spans [(0, 20), (21, 53), (54, 75)], verified with the
    # real splitter). Without the guard they pool under the single pseudo-key None, the
    # intersection is {None}, the pair matches, and gold_finding_sentences returns
    # {"1": {None}} -- None entering a set typed set[int], and a "gold sentence" that is not
    # a sentence. With the guard, neither mention is placeable and nothing is gold.
    doc = GoldDocument(
        pmid="1",
        text=MULTI_TEXT,
        mentions=[
            GoldMention(
                pmid="1",
                start=20,
                end=21,
                text=MULTI_TEXT[20:21],
                label=EntityLabel.CHEMICAL,
                mesh_ids=("MESH:D008687",),
            ),
            GoldMention(
                pmid="1",
                start=53,
                end=54,
                text=MULTI_TEXT[53:54],
                label=EntityLabel.DISEASE,
                mesh_ids=("MESH:D000138",),
            ),
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


def test_two_documents_sharing_a_pmid_are_rejected_not_merged_and_not_dropped():
    # A sentence index is only meaningful relative to ONE document text. These two documents
    # share pmid "p" but carry DIFFERENT texts, so index 0 in `first` ("Metformin caused
    # acidosis.") and index 1 in `second` ("Acidosis followed metformin use.") are different
    # sentences that would collide under one key.
    # Two wrong answers are both silently wrong, in opposite directions, and this repository
    # really produces the shape: domain_normalization_sample.jsonl holds 49 records over 3
    # pmids and load_domain_norm_documents emits one GoldDocument per record.
    #   union    -> {"p": {0, 1}}: lossless but conflates two unrelated sentences.
    #   assign   -> {"p": {1}} or {"p": {0}} depending on ARGUMENT ORDER: silent data loss.
    # Both orders are asserted so an assign-style regression cannot hide behind whichever
    # order happens to survive.
    # sentence_spans(first_text) == [(0, 26), (27, 48)] and sentence_spans(TEXT) ==
    # [(0, 20), (21, 53)] -- both verified with the real splitter.
    chem, dis = "MESH:D008687", "MESH:D000138"
    first_text = "Metformin caused acidosis. Aspirin caused fever."
    first = GoldDocument(
        pmid="p",
        text=first_text,
        mentions=[
            GoldMention(
                pmid="p",
                start=0,
                end=9,
                text=first_text[0:9],
                label=EntityLabel.CHEMICAL,
                mesh_ids=(chem,),
            ),
            GoldMention(
                pmid="p",
                start=17,
                end=25,
                text=first_text[17:25],
                label=EntityLabel.DISEASE,
                mesh_ids=(dis,),
            ),
        ],
    )
    second = GoldDocument(
        pmid="p",
        text=TEXT,
        mentions=[
            _m(21, 29, EntityLabel.DISEASE, (dis,)),
            _m(39, 48, EntityLabel.CHEMICAL, (chem,)),
        ],
    )
    relations = {"p": {(chem, dis)}}

    # Each document ALONE is gold at a different index -- so the collision is real, not an
    # artefact of one of them being empty.
    assert gold_finding_sentences([first], relations) == {"p": {0}}
    assert gold_finding_sentences([second], relations) == {"p": {1}}

    # The `caller` prefix is pinned, not just "repeated pmid": `caller` exists solely to make
    # the message name the function that actually rejected the input, and a pattern matching
    # only the shared tail lets either call site report itself under the other's name.
    with pytest.raises(ValueError, match="gold_finding_sentences: repeated pmid"):
        gold_finding_sentences([first, second], relations)
    with pytest.raises(ValueError, match="gold_finding_sentences: repeated pmid"):
        gold_finding_sentences([second, first], relations)


def test_a_repeated_pmid_is_rejected_even_when_it_carries_no_gold_relations():
    # The contract is on the SHAPE of `documents`, not on which documents happen to survive
    # the `if not pairs: continue` filter. Both functions therefore screen the whole sequence
    # BEFORE any per-document work. Fold the screen into the loop body instead -- so it only
    # sees documents that have relations -- and a malformed input is silently half-accepted:
    # this call returns {} with no complaint even though it was handed two documents under
    # one pmid, and the very next run, with a relation added for "p", starts silently losing
    # or conflating sentences. `relations` is empty here, so nothing is gold either way and
    # the raise is the ONLY observable difference.
    doc = GoldDocument(
        pmid="p",
        text=TEXT,
        mentions=[_m(21, 29, EntityLabel.DISEASE, ("MESH:D000138",))],
    )
    relations: dict[str, set[tuple[str, str]]] = {}
    assert gold_finding_sentences([doc], relations) == {}
    with pytest.raises(ValueError, match="gold_finding_sentences: repeated pmid"):
        gold_finding_sentences([doc, doc], relations)
    with pytest.raises(ValueError, match="classify_misses: repeated pmid"):
        classify_misses([doc, doc], relations, {}, {}, entities_by_paper={})


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


def test_a_miss_is_classified_against_only_the_relations_that_made_its_sentence_gold():
    # ADR-0013's 40.3% is a PER-RELATION statistic (430 of 1066 gold CID relations in
    # Test-500 lose an endpoint), and at ~2.13 relations/document a per-document pooling
    # cannot reproduce it. Two relations here: A (metformin/acidosis) is fully linked,
    # B (aspirin/fever) has no linked disease anywhere. Sentence 1 is gold ONLY via B, so
    # its miss is UNRECOVERABLE -- endpoint_lost. Pooling reachability over the document's
    # whole relation set lets reachable relation A suppress endpoint_lost for a miss it had
    # nothing to do with, reporting the miss as the recoverable co_sentential_elsewhere and
    # inflating the LLM arm's apparent headroom -- the exact failure ADR-0013's standing
    # finding warns against.
    # sentence_spans(text) == [(0, 26), (27, 48)] -- verified with the real splitter.
    text = "Metformin caused acidosis. Aspirin caused fever."
    metformin, acidosis = "MESH:D008687", "MESH:D000138"
    aspirin, fever = "MESH:D000568", "MESH:D005334"
    doc = GoldDocument(
        pmid="pD",
        text=text,
        mentions=[
            GoldMention(
                pmid="pD",
                start=0,
                end=9,
                text=text[0:9],
                label=EntityLabel.CHEMICAL,
                mesh_ids=(metformin,),
            ),
            GoldMention(
                pmid="pD",
                start=17,
                end=25,
                text=text[17:25],
                label=EntityLabel.DISEASE,
                mesh_ids=(acidosis,),
            ),
            GoldMention(
                pmid="pD",
                start=27,
                end=34,
                text=text[27:34],
                label=EntityLabel.CHEMICAL,
                mesh_ids=(aspirin,),
            ),
            GoldMention(
                pmid="pD",
                start=42,
                end=47,
                text=text[42:47],
                label=EntityLabel.DISEASE,
                mesh_ids=(fever,),
            ),
        ],
    )
    relations = {"pD": {(metformin, acidosis), (aspirin, fever)}}
    # "fever" is never linked, so relation B's disease endpoint is unreachable anywhere.
    entities = {
        "pD": [
            Entity(
                text="Metformin", label=EntityLabel.CHEMICAL, start=0, end=9, canonical_id=metformin
            ),
            Entity(
                text="acidosis", label=EntityLabel.DISEASE, start=17, end=25, canonical_id=acidosis
            ),
            Entity(
                text="Aspirin", label=EntityLabel.CHEMICAL, start=27, end=34, canonical_id=aspirin
            ),
        ]
    }

    gold = gold_finding_sentences([doc], relations)
    assert gold == {"pD": {0, 1}}

    extractor = SameSentenceAsEntitiesExtractor(entities)
    pred = {"pD": {f.sentence_index for f in extractor.findings(_paper("pD", text))}}
    assert pred == {"pD": {0}}

    buckets = classify_misses([doc], relations, gold, pred, entities_by_paper=entities)
    assert (
        buckets.endpoint_lost,
        buckets.never_co_sentential,
        buckets.co_sentential_elsewhere,
    ) == (1, 0, 0)
    assert buckets.total == 1


def test_a_lost_chemical_endpoint_is_endpoint_lost_just_as_a_lost_disease_one_is():
    # SIBLING of pA in test_each_miss_bucket_is_populated_distinctly, which kills the DISEASE
    # endpoint. `reachable` tests `c in chemicals AND d in diseases`, and with only the pA
    # fixture the `c in chemicals` conjunct could be deleted outright and every test still
    # passed -- the one-sided-compound-condition defect this project has shipped repeatedly.
    # Here the DISEASE is linked and the CHEMICAL is not, so it is the other conjunct that
    # has to fire. Both directions are equally unrecoverable: no window and no pairing
    # mechanism reaches an endpoint that was never extracted.
    # sentence_spans(TEXT) == [(0, 20), (21, 53)] -- verified with the real splitter.
    chem, dis = "MESH:D008687", "MESH:D000138"
    doc = GoldDocument(
        pmid="pE",
        text=TEXT,
        mentions=[
            _m(0, 9, EntityLabel.CHEMICAL, (chem,)),
            _m(21, 29, EntityLabel.DISEASE, (dis,)),
            _m(39, 48, EntityLabel.CHEMICAL, (chem,)),
        ],
    )
    relations = {"pE": {(chem, dis)}}
    # Only the disease is linked; the chemical endpoint is unreachable anywhere in the paper.
    entities = {
        "pE": [
            Entity(text="Acidosis", label=EntityLabel.DISEASE, start=21, end=29, canonical_id=dis)
        ]
    }

    gold = gold_finding_sentences([doc], relations)
    assert gold == {"pE": {1}}

    extractor = SameSentenceAsEntitiesExtractor(entities)
    pred = {"pE": {f.sentence_index for f in extractor.findings(_paper("pE", TEXT))}}
    assert pred == {"pE": set()}

    buckets = classify_misses([doc], relations, gold, pred, entities_by_paper=entities)
    assert (
        buckets.endpoint_lost,
        buckets.never_co_sentential,
        buckets.co_sentential_elsewhere,
    ) == (1, 0, 0)
    assert buckets.total == 1


def test_entities_starting_in_an_inter_sentence_gap_are_dropped_not_pooled_together():
    # The classify_misses counterpart of test_both_endpoints_in_inter_sentence_gaps_do_not
    # _pool_into_one_pseudo_sentence, which pins the same guard for gold_finding_sentences.
    # (NOT of test_a_mention_starting_in_the_inter_sentence_gap_is_dropped, which an earlier
    # version of this comment named: that fixture returns {} with the guard deleted too, so
    # it never pinned anything.) sentence_index
    # returns None for a position in no span; without the `if index is None: continue` guard
    # every such entity is pooled under the single pseudo-key None, and two entities that
    # share no sentence at all look co-sentential. That silently flips never_co_sentential
    # (recoverable only by a wider window) to co_sentential_elsewhere (a mere span
    # disagreement) -- a misclassification assert_bucket_closure cannot see, because closure
    # is blind to WHICH bucket a miss lands in.
    # sentence_spans(MULTI_TEXT) == [(0, 20), (21, 53), (54, 75)] -- verified with the real
    # splitter. Positions 20 and 53 are the separators, belonging to neither span.
    chem, dis = "MESH:D008687", "MESH:D000138"
    doc = GoldDocument(
        pmid="pF",
        text=MULTI_TEXT,
        mentions=[
            GoldMention(
                pmid="pF",
                start=21,
                end=29,
                text=MULTI_TEXT[21:29],
                label=EntityLabel.DISEASE,
                mesh_ids=(dis,),
            ),
            GoldMention(
                pmid="pF",
                start=39,
                end=48,
                text=MULTI_TEXT[39:48],
                label=EntityLabel.CHEMICAL,
                mesh_ids=(chem,),
            ),
        ],
    )
    relations = {"pF": {(chem, dis)}}
    entities = {
        "pF": [
            Entity(
                text=MULTI_TEXT[20:21],
                label=EntityLabel.CHEMICAL,
                start=20,
                end=21,
                canonical_id=chem,
            ),
            Entity(
                text=MULTI_TEXT[53:54],
                label=EntityLabel.DISEASE,
                start=53,
                end=54,
                canonical_id=dis,
            ),
        ]
    }

    gold = gold_finding_sentences([doc], relations)
    assert gold == {"pF": {1}}

    extractor = SameSentenceAsEntitiesExtractor(entities)
    pred = {"pF": {f.sentence_index for f in extractor.findings(_paper("pF", MULTI_TEXT))}}
    assert pred == {"pF": set()}

    # Both endpoints ARE linked, so the miss is reachable -- but no SENTENCE holds both,
    # because neither unplaceable entity belongs to one.
    buckets = classify_misses([doc], relations, gold, pred, entities_by_paper=entities)
    assert (
        buckets.endpoint_lost,
        buckets.never_co_sentential,
        buckets.co_sentential_elsewhere,
    ) == (0, 1, 0)
    assert buckets.total == 1


def test_classify_misses_rejects_a_repeated_pmid_rather_than_double_counting_its_misses():
    # classify_misses looks up ONE gold[pmid] per document, so a pmid appearing twice in
    # `documents` re-classifies the very same misses once per duplicate: the single miss
    # below would be counted as 2, inflating every bucket and the `total` that
    # assert_bucket_closure compares against sentence_metrics' fn (which counts each
    # (pmid, index) pair once). The exact-duplicate case is used deliberately -- it is the
    # most benign repetition imaginable, and it is still wrong, so the contract is on the
    # pmid, not on whether the texts happen to differ.
    # sentence_spans(TEXT) == [(0, 20), (21, 53)] -- verified with the real splitter.
    chem, dis = "MESH:D008687", "MESH:D000138"
    doc = GoldDocument(
        pmid="pG",
        text=TEXT,
        mentions=[
            _m(0, 9, EntityLabel.CHEMICAL, (chem,)),
            _m(21, 29, EntityLabel.DISEASE, (dis,)),
            _m(39, 48, EntityLabel.CHEMICAL, (chem,)),
        ],
    )
    relations = {"pG": {(chem, dis)}}
    # Only the disease is linked, so the one miss at sentence 1 is endpoint_lost.
    entities = {
        "pG": [
            Entity(text="Acidosis", label=EntityLabel.DISEASE, start=21, end=29, canonical_id=dis)
        ]
    }
    gold = {"pG": {1}}
    pred: dict[str, set[int]] = {"pG": set()}

    single = classify_misses([doc], relations, gold, pred, entities_by_paper=entities)
    assert (
        single.endpoint_lost,
        single.never_co_sentential,
        single.co_sentential_elsewhere,
    ) == (1, 0, 0)
    assert single.total == 1

    with pytest.raises(ValueError, match="classify_misses: repeated pmid"):
        classify_misses([doc, doc], relations, gold, pred, entities_by_paper=entities)


def test_a_gold_sentence_no_relation_qualifies_is_rejected_not_bucketed_as_endpoint_lost():
    # Precondition B: `gold` must come from gold_finding_sentences over the SAME `documents`
    # and `relations`. Unenforced, the violation is silent and lands in the worst possible
    # bucket. Here sentence 1 holds both endpoints and BOTH are linked, so the real gold is
    # {"pH": {1}}. Hand classify_misses a gold of {"pH": {0}} instead and the missed index 0 is
    # absent from _gold_pairs_by_sentence, so `pairs_at_index` is empty, `reachable` is empty,
    # and the miss is reported as endpoint_lost -- the UNRECOVERABLE bucket -- with
    # assert_bucket_closure passing, because closure only checks that misses were counted,
    # never WHERE they landed. Reporting a recoverable miss as unrecoverable inflates the LLM
    # arm's apparent headroom, the exact failure ADR-0013's standing finding warns against.
    # `pred` BELOW IS A HAND-BUILT STAND-IN THAT VIOLATES UNENFORCED PRECONDITION 1, and the
    # `consistent` assertion must be read accordingly. The real SameSentenceAsEntitiesExtractor
    # over these entities and this text returns {1} -- verified by running it -- i.e. no miss
    # at all, so a selector-consistent `pred` would leave this fixture with nothing to bucket.
    # `set()` is used so that a miss exists at the gold sentence. Its bucket is therefore the
    # MECHANICAL consequence of the stand-in and NOT an illustration of a real
    # co_sentential_elsewhere: nothing is genuinely "elsewhere" here, because the sentence the
    # selector would have chosen IS the gold one. The genuine co_sentential_elsewhere fixture
    # is pC in test_each_miss_bucket_is_populated_distinctly, whose `pred` comes from the real
    # selector. None of this weakens the pin below: the raise depends on `gold`, `documents`
    # and `relations` alone, never on how `pred` was produced.
    # sentence_spans(TEXT) == [(0, 20), (21, 53)] -- verified with the real splitter.
    chem, dis = "MESH:D008687", "MESH:D000138"
    doc = GoldDocument(
        pmid="pH",
        text=TEXT,
        mentions=[
            _m(0, 9, EntityLabel.CHEMICAL, (chem,)),
            _m(21, 29, EntityLabel.DISEASE, (dis,)),
            _m(39, 48, EntityLabel.CHEMICAL, (chem,)),
        ],
    )
    relations = {"pH": {(chem, dis)}}
    # Both endpoints linked, and together in sentence 1 -- nothing is lost or unreachable.
    entities = {
        "pH": [
            Entity(
                text="metformin", label=EntityLabel.CHEMICAL, start=39, end=48, canonical_id=chem
            ),
            Entity(text="Acidosis", label=EntityLabel.DISEASE, start=21, end=29, canonical_id=dis),
        ]
    }
    pred: dict[str, set[int]] = {"pH": set()}

    assert gold_finding_sentences([doc], relations) == {"pH": {1}}

    consistent = classify_misses([doc], relations, {"pH": {1}}, pred, entities_by_paper=entities)
    assert (
        consistent.endpoint_lost,
        consistent.never_co_sentential,
        consistent.co_sentential_elsewhere,
    ) == (0, 0, 1)

    with pytest.raises(ValueError, match="classify_misses: sentence 0 of pmid 'pH'") as excinfo:
        classify_misses([doc], relations, {"pH": {0}}, pred, entities_by_paper=entities)
    message = str(excinfo.value)
    assert "gold_finding_sentences" in message
    assert "endpoint_lost" in message


def test_a_gold_entry_that_omits_a_genuine_gold_sentence_is_rejected_not_silently_shrunk():
    # The OTHER direction of the same precondition: `gold` must not merely avoid inventing
    # sentences, it must not DROP them either. An omitted gold sentence never becomes a miss,
    # so it silently shrinks `endpoint_lost` -- the UNRECOVERABLE population that is this
    # eval's deliverable -- while assert_bucket_closure keeps passing, because the false
    # negative count it compares against is computed from the SAME wrong `gold` and deflates
    # in lockstep. Measured on this exact fixture against the pre-fix code: gold {"pI": {1}}
    # gave endpoint_lost 1 instead of 2, a 50% under-report, with closure silent; gold {}
    # gave (0, 0, 0, 0) and never even reached the consistency check, because `missed` was
    # empty and the loop skipped the document before gold pairs were computed.
    # sentence_spans(text) == [(0, 20), (21, 53), (54, 74), (75, 107)] -- verified with the
    # real splitter. Sentences 1 and 3 each hold both endpoints, so gold is {1, 3}.
    chem, dis = "MESH:D008687", "MESH:D000138"
    text = (
        "Metformin was given. Acidosis followed metformin use. "
        "Nausea was reported. Metformin caused acidosis again."
    )
    doc = GoldDocument(
        pmid="pI",
        text=text,
        mentions=[
            GoldMention(
                pmid="pI",
                start=21,
                end=29,
                text=text[21:29],
                label=EntityLabel.DISEASE,
                mesh_ids=(dis,),
            ),
            GoldMention(
                pmid="pI",
                start=39,
                end=48,
                text=text[39:48],
                label=EntityLabel.CHEMICAL,
                mesh_ids=(chem,),
            ),
            GoldMention(
                pmid="pI",
                start=75,
                end=84,
                text=text[75:84],
                label=EntityLabel.CHEMICAL,
                mesh_ids=(chem,),
            ),
            GoldMention(
                pmid="pI",
                start=92,
                end=100,
                text=text[92:100],
                label=EntityLabel.DISEASE,
                mesh_ids=(dis,),
            ),
        ],
    )
    relations = {"pI": {(chem, dis)}}
    # No disease is linked anywhere, so both gold misses are endpoint_lost -- the bucket the
    # omission deflates.
    entities = {
        "pI": [
            Entity(text="Metformin", label=EntityLabel.CHEMICAL, start=0, end=9, canonical_id=chem)
        ]
    }

    gold = gold_finding_sentences([doc], relations)
    assert gold == {"pI": {1, 3}}

    extractor = SameSentenceAsEntitiesExtractor(entities)
    pred = {"pI": {f.sentence_index for f in extractor.findings(_paper("pI", text))}}
    assert pred == {"pI": set()}

    # Anti-vacuity: with the correct gold the call succeeds and reports BOTH misses.
    correct = classify_misses([doc], relations, gold, pred, entities_by_paper=entities)
    assert (
        correct.endpoint_lost,
        correct.never_co_sentential,
        correct.co_sentential_elsewhere,
    ) == (2, 0, 0)
    assert correct.total == 2

    # Partial omission: sentence 3 dropped.
    with pytest.raises(ValueError, match="classify_misses: sentence 3 of pmid 'pI'") as partial:
        classify_misses([doc], relations, {"pI": {1}}, pred, entities_by_paper=entities)
    assert "gold_finding_sentences" in str(partial.value)
    assert "endpoint_lost" in str(partial.value)

    # Complete omission: the pmid's entry is absent entirely. This one leaves `missed` empty,
    # so the check MUST NOT be reachable only from a document that already has misses.
    with pytest.raises(ValueError, match="classify_misses: sentence 1 of pmid 'pI'") as complete:
        classify_misses([doc], relations, {}, pred, entities_by_paper=entities)
    assert "gold_finding_sentences" in str(complete.value)
    assert "endpoint_lost" in str(complete.value)


def test_the_gold_provenance_check_covers_documents_that_have_no_gold_sentence_at_all():
    # DOMAIN of the provenance check, not its operands. The single-document fixtures above
    # all have a non-empty gold, so every one of them survives a conjunction that quietly
    # restricts the check to documents that already have gold -- `if invented and gold_pairs:`
    # or `if invented and pairs:` -- while such a conjunction disables it on the DOMINANT real
    # shape: a `gold` built over a different `relations`, or over a superset of documents,
    # names sentences in papers that have no gold sentence at all.
    # "pK" is second in `documents` and carries the bogus entry, so restricting the loop to
    # the first document is caught here too.
    # Two variants, because the two conjunctions key on different values: with `relations`
    # holding a pair for "pK" that qualifies nothing, `pairs` is truthy while `gold_pairs` is
    # empty; with "pK" absent from `relations`, both are empty.
    # sentence_spans(TEXT) == [(0, 20), (21, 53)] -- verified with the real splitter.
    chem, dis, unannotated = "MESH:D008687", "MESH:D000138", "MESH:D005334"
    doc_good = GoldDocument(
        pmid="pJ",
        text=TEXT,
        mentions=[
            GoldMention(
                pmid="pJ",
                start=0,
                end=9,
                text=TEXT[0:9],
                label=EntityLabel.CHEMICAL,
                mesh_ids=(chem,),
            ),
            GoldMention(
                pmid="pJ",
                start=21,
                end=29,
                text=TEXT[21:29],
                label=EntityLabel.DISEASE,
                mesh_ids=(dis,),
            ),
            GoldMention(
                pmid="pJ",
                start=39,
                end=48,
                text=TEXT[39:48],
                label=EntityLabel.CHEMICAL,
                mesh_ids=(chem,),
            ),
        ],
    )
    # "pK" annotates a chemical only, so no sentence of it can ever hold both endpoints.
    doc_bare = GoldDocument(
        pmid="pK",
        text=TEXT,
        mentions=[
            GoldMention(
                pmid="pK",
                start=0,
                end=9,
                text=TEXT[0:9],
                label=EntityLabel.CHEMICAL,
                mesh_ids=(chem,),
            ),
        ],
    )
    docs = [doc_good, doc_bare]
    entities = {
        "pJ": [
            Entity(text="Acidosis", label=EntityLabel.DISEASE, start=21, end=29, canonical_id=dis)
        ]
    }
    pred: dict[str, set[int]] = {"pJ": set(), "pK": set()}

    # Variant 1: "pK" HAS a relation, and it qualifies nothing (its disease is never
    # annotated), so `relations["pK"]` is truthy while its gold-pair mapping is empty.
    qualifying_nothing = {"pJ": {(chem, dis)}, "pK": {(chem, unannotated)}}
    assert gold_finding_sentences(docs, qualifying_nothing) == {"pJ": {1}}

    # Anti-vacuity: the consistent gold classifies a real miss rather than raising.
    buckets = classify_misses(
        docs, qualifying_nothing, {"pJ": {1}}, pred, entities_by_paper=entities
    )
    assert (
        buckets.endpoint_lost,
        buckets.never_co_sentential,
        buckets.co_sentential_elsewhere,
    ) == (1, 0, 0)
    assert buckets.total == 1

    with pytest.raises(ValueError, match="classify_misses: sentence 0 of pmid 'pK'"):
        classify_misses(
            docs, qualifying_nothing, {"pJ": {1}, "pK": {0}}, pred, entities_by_paper=entities
        )

    # Variant 2: "pK" is absent from `relations` entirely, so `relations.get(...)` is empty
    # too and a conjunction on either value would go silent.
    absent = {"pJ": {(chem, dis)}}
    assert gold_finding_sentences(docs, absent) == {"pJ": {1}}
    with pytest.raises(ValueError, match="classify_misses: sentence 0 of pmid 'pK'"):
        classify_misses(docs, absent, {"pJ": {1}, "pK": {0}}, pred, entities_by_paper=entities)


def test_the_gold_provenance_check_covers_every_declared_index_not_just_the_smallest():
    # Second DOMAIN axis: which INDICES of a document the check examines. The single-index
    # fixtures above cannot tell "every declared index" from "the smallest one", so a check
    # narrowed to `min(...)` passes them all. Here the smallest declared index IS genuinely
    # gold and the larger one is not, so only a check that looks past the first index fires.
    # Index 2 is deliberately IN RANGE (MULTI_TEXT has three sentences) -- an out-of-range
    # index is a different, easier shape, pinned by the test below.
    # sentence_spans(MULTI_TEXT) == [(0, 20), (21, 53), (54, 75)] -- verified with the real
    # splitter. Only sentence 1 holds both endpoints; sentence 2 holds no gold mention.
    chem, dis = "MESH:D008687", "MESH:D000138"
    doc = GoldDocument(
        pmid="pL",
        text=MULTI_TEXT,
        mentions=[
            GoldMention(
                pmid="pL",
                start=0,
                end=9,
                text=MULTI_TEXT[0:9],
                label=EntityLabel.CHEMICAL,
                mesh_ids=(chem,),
            ),
            GoldMention(
                pmid="pL",
                start=21,
                end=29,
                text=MULTI_TEXT[21:29],
                label=EntityLabel.DISEASE,
                mesh_ids=(dis,),
            ),
            GoldMention(
                pmid="pL",
                start=39,
                end=48,
                text=MULTI_TEXT[39:48],
                label=EntityLabel.CHEMICAL,
                mesh_ids=(chem,),
            ),
        ],
    )
    relations = {"pL": {(chem, dis)}}
    # Only the disease is linked, so the genuine miss at sentence 1 is endpoint_lost and the
    # real selector produces nothing.
    entities = {
        "pL": [
            Entity(text="Acidosis", label=EntityLabel.DISEASE, start=21, end=29, canonical_id=dis)
        ]
    }

    gold = gold_finding_sentences([doc], relations)
    assert gold == {"pL": {1}}

    extractor = SameSentenceAsEntitiesExtractor(entities)
    pred = {"pL": {f.sentence_index for f in extractor.findings(_paper("pL", MULTI_TEXT))}}
    assert pred == {"pL": set()}

    # Anti-vacuity: the consistent gold classifies the real miss rather than raising.
    buckets = classify_misses([doc], relations, gold, pred, entities_by_paper=entities)
    assert (
        buckets.endpoint_lost,
        buckets.never_co_sentential,
        buckets.co_sentential_elsewhere,
    ) == (1, 0, 0)
    assert buckets.total == 1

    # min({1, 2}) == 1, which IS gold -- so a first-index-only check stays silent here.
    with pytest.raises(ValueError, match="classify_misses: sentence 2 of pmid 'pL'"):
        classify_misses([doc], relations, {"pL": {1, 2}}, pred, entities_by_paper=entities)


def test_a_gold_index_past_the_end_of_the_document_is_rejected_too():
    # Third DOMAIN axis: the check must not be restricted to indices that name a real
    # sentence. Narrow it to `i < len(sentence_spans(document.text))` -- a plausible-looking
    # "only consider real sentences" filter -- and every in-range fixture above still passes
    # while an out-of-range gold index sails through and is bucketed as endpoint_lost. This is
    # the shape a `gold` built over a DIFFERENT text produces: `GoldDocument.text` is
    # `title + " " + abstract` while `SameSentenceAsEntitiesExtractor` splits the abstract
    # alone, so indices computed against the longer string overrun the shorter one.
    # sentence_spans(TEXT) == [(0, 20), (21, 53)] -- verified with the real splitter, so 99
    # names no sentence at all.
    chem, dis = "MESH:D008687", "MESH:D000138"
    doc = GoldDocument(
        pmid="pM",
        text=TEXT,
        mentions=[
            _m(0, 9, EntityLabel.CHEMICAL, (chem,)),
            _m(21, 29, EntityLabel.DISEASE, (dis,)),
            _m(39, 48, EntityLabel.CHEMICAL, (chem,)),
        ],
    )
    relations = {"pM": {(chem, dis)}}
    entities = {
        "pM": [
            Entity(text="Acidosis", label=EntityLabel.DISEASE, start=21, end=29, canonical_id=dis)
        ]
    }

    gold = gold_finding_sentences([doc], relations)
    assert gold == {"pM": {1}}

    extractor = SameSentenceAsEntitiesExtractor(entities)
    pred = {"pM": {f.sentence_index for f in extractor.findings(_paper("pM", TEXT))}}
    assert pred == {"pM": set()}

    with pytest.raises(ValueError, match="classify_misses: sentence 99 of pmid 'pM'"):
        classify_misses([doc], relations, {"pM": {1, 99}}, pred, entities_by_paper=entities)


def test_an_invented_gold_index_that_pred_also_selected_is_rejected_too():
    # FOURTH DOMAIN axis, and the first about the check's OPERANDS rather than its reach.
    # Every fixture above that exercises this branch (pH, pJ/pK, pL, pM) sets `pred` to
    # `set()` for the pmid carrying the bad entry, so a check computed over `gold - pred`
    # instead of over `gold` passes every one of them. That is not hypothetical: it is
    # literally the predicate that shipped at 0cf680d --
    #     invented = sorted((declared - pred.get(document.pmid, set())) - set(gold_pairs))
    # -- and it passes the whole suite without this test. (`declared - set(gold_pairs) -
    # pred.get(...)` is the SAME set expression, since difference by two sets is difference
    # by their union, so that is one mutant and not two.) Here the invented index 0 IS in
    # `pred`, so `gold - pred` no longer contains it and only a check over `gold` fires.
    # WHAT GOES WRONG IS NOT THE BUCKETS, and this test does not pretend otherwise: an
    # invented index that `pred` selected is not a miss, so it is never bucketed either way
    # and `assert_bucket_closure` stays silent. `sentence_metrics` is what gets corrupted --
    # exactly the harm the raise message names -- and that is asserted below, not described.
    # sentence_spans(text) == [(0, 26), (27, 59)] -- verified with the real splitter.
    chem, dis = "MESH:D008687", "MESH:D000138"
    text = "Metformin caused acidosis. Acidosis followed metformin use."
    # Gold annotates ONLY sentence 1, so sentence 0 is not a gold finding sentence.
    doc = GoldDocument(
        pmid="pN",
        text=text,
        mentions=[
            GoldMention(
                pmid="pN",
                start=27,
                end=35,
                text=text[27:35],
                label=EntityLabel.DISEASE,
                mesh_ids=(dis,),
            ),
            GoldMention(
                pmid="pN",
                start=45,
                end=54,
                text=text[45:54],
                label=EntityLabel.CHEMICAL,
                mesh_ids=(chem,),
            ),
        ],
    )
    relations = {"pN": {(chem, dis)}}
    # Both endpoints linked, together in sentence 0 -- so the real selector picks 0, the one
    # sentence that is NOT gold. This is a genuine co_sentential_elsewhere shape.
    entities = {
        "pN": [
            Entity(text="Metformin", label=EntityLabel.CHEMICAL, start=0, end=9, canonical_id=chem),
            Entity(text="acidosis", label=EntityLabel.DISEASE, start=17, end=25, canonical_id=dis),
        ]
    }

    gold = gold_finding_sentences([doc], relations)
    assert gold == {"pN": {1}}

    extractor = SameSentenceAsEntitiesExtractor(entities)
    pred = {"pN": {f.sentence_index for f in extractor.findings(_paper("pN", text))}}
    # The invented index below is one the REAL selector chose, not one picked to suit.
    assert pred == {"pN": {0}}

    # Anti-vacuity: the consistent gold classifies the real miss rather than raising.
    buckets = classify_misses([doc], relations, gold, pred, entities_by_paper=entities)
    assert (
        buckets.endpoint_lost,
        buckets.never_co_sentential,
        buckets.co_sentential_elsewhere,
    ) == (0, 0, 1)
    assert buckets.total == 1

    invented = {"pN": {0, 1}}
    with pytest.raises(ValueError, match="classify_misses: sentence 0 of pmid 'pN'"):
        classify_misses([doc], relations, invented, pred, entities_by_paper=entities)

    # THE NUMBER THE RAISE PROTECTS. Sentence 0 is not gold, so the arm's one selection is a
    # false positive and precision is 0.0. Believe the invented gold and the same selection
    # scores as a true positive at precision 1.0 -- a perfect score for a sentence that no
    # gold relation qualifies. Closure is blind to it: `total` is 1 against fn 1 either way.
    honest = sentence_metrics(pred, gold)
    assert (honest.tp, honest.fp, honest.fn) == (0, 1, 1)
    corrupted = sentence_metrics(pred, invented)
    assert (corrupted.tp, corrupted.fp, corrupted.fn) == (1, 0, 1)
    assert_bucket_closure(buckets, n_false_negatives=honest.fn)
    assert_bucket_closure(buckets, n_false_negatives=corrupted.fn)


def test_an_omitted_gold_sentence_that_pred_selected_is_rejected_too():
    # The omission-branch sibling of the test above, on the same OPERAND axis: pI, the only
    # fixture that exercises this branch, sets `pred` to `set()` for its pmid, so
    #     omitted = sorted(set(gold_pairs) - declared - pred.get(document.pmid, set()))
    # passes it and the rest of the suite. Here the omitted gold sentence 3 IS in `pred`.
    # STATED NO STRONGER THAN IT IS: this omission is provably invisible in the buckets, not
    # merely unobserved. The `- pred` mutant suppresses the raise only when every omitted
    # index is in `pred`, and then `missed = declared - pred` equals `gold_pairs - pred`,
    # which is exactly what the correct gold yields -- so all three buckets are IDENTICAL
    # with and without the omission, and `assert_bucket_closure` compares 1 against 1 either
    # way. The check is what makes the docstring's NECESSARY AND SUFFICIENT claim true; the
    # number that goes wrong is in `sentence_metrics`, where a true positive at the omitted
    # sentence is rewritten into a false positive. Both facts are asserted below.
    # sentence_spans(text) == [(0, 20), (21, 53), (54, 74), (75, 107)] -- verified with the
    # real splitter. Sentences 1 and 3 each hold both endpoints, so gold is {1, 3}.
    chem, dis = "MESH:D008687", "MESH:D000138"
    text = (
        "Metformin was given. Acidosis followed metformin use. "
        "Nausea was reported. Metformin caused acidosis again."
    )
    doc = GoldDocument(
        pmid="pO",
        text=text,
        mentions=[
            GoldMention(
                pmid="pO",
                start=21,
                end=29,
                text=text[21:29],
                label=EntityLabel.DISEASE,
                mesh_ids=(dis,),
            ),
            GoldMention(
                pmid="pO",
                start=39,
                end=48,
                text=text[39:48],
                label=EntityLabel.CHEMICAL,
                mesh_ids=(chem,),
            ),
            GoldMention(
                pmid="pO",
                start=75,
                end=84,
                text=text[75:84],
                label=EntityLabel.CHEMICAL,
                mesh_ids=(chem,),
            ),
            GoldMention(
                pmid="pO",
                start=92,
                end=100,
                text=text[92:100],
                label=EntityLabel.DISEASE,
                mesh_ids=(dis,),
            ),
        ],
    )
    relations = {"pO": {(chem, dis)}}
    # Both endpoints linked, together in sentence 3 -- so the real selector picks 3, which is
    # a genuine gold sentence, and the miss at sentence 1 is co_sentential_elsewhere.
    entities = {
        "pO": [
            Entity(
                text="Metformin", label=EntityLabel.CHEMICAL, start=75, end=84, canonical_id=chem
            ),
            Entity(text="acidosis", label=EntityLabel.DISEASE, start=92, end=100, canonical_id=dis),
        ]
    }

    gold = gold_finding_sentences([doc], relations)
    assert gold == {"pO": {1, 3}}

    extractor = SameSentenceAsEntitiesExtractor(entities)
    pred = {"pO": {f.sentence_index for f in extractor.findings(_paper("pO", text))}}
    # The omitted index below is one the REAL selector chose, not one picked to suit.
    assert pred == {"pO": {3}}

    # Anti-vacuity: the consistent gold classifies the real miss rather than raising.
    buckets = classify_misses([doc], relations, gold, pred, entities_by_paper=entities)
    assert (
        buckets.endpoint_lost,
        buckets.never_co_sentential,
        buckets.co_sentential_elsewhere,
    ) == (0, 0, 1)
    assert buckets.total == 1

    truncated = {"pO": {1}}
    with pytest.raises(ValueError, match="classify_misses: sentence 3 of pmid 'pO'"):
        classify_misses([doc], relations, truncated, pred, entities_by_paper=entities)

    # THE NUMBER THE RAISE PROTECTS, and the two that it does not. Sentence 3 IS gold and the
    # arm selected it, so precision is 1.0; drop it from `gold` and the same selection scores
    # as a false positive at precision 0.0. The buckets and closure, by contrast, are the
    # same on both golds -- which is why the raise, and not a bucket assertion, is the pin.
    honest = sentence_metrics(pred, gold)
    assert (honest.tp, honest.fp, honest.fn) == (1, 0, 1)
    corrupted = sentence_metrics(pred, truncated)
    assert (corrupted.tp, corrupted.fp, corrupted.fn) == (0, 1, 1)
    assert_bucket_closure(buckets, n_false_negatives=honest.fn)
    assert_bucket_closure(buckets, n_false_negatives=corrupted.fn)


def test_a_linked_endpoint_with_no_offset_is_its_own_bucket_not_never_co_sentential():
    # THE `linked_ids` / `per_sentence` SEAM. `linked_ids` filters on `canonical_id is not
    # None` and IGNORES `start`, while the `per_sentence` loop skips `canonical_id is None or
    # start is None`. An entity that is LINKED BUT UNLOCATABLE therefore makes `reachable`
    # non-empty -- escaping endpoint_lost -- while being unable to appear in any sentence, so
    # before this commit the miss landed in `never_co_sentential`.
    # That is wrong in the direction this project is most alert to. `never_co_sentential`
    # means "the endpoints exist and are placed, they merely never share a sentence", a
    # population a wider window could plausibly recover, and it is quoted as recoverable
    # headroom. An endpoint with no offset is recoverable by NO window variant: its position
    # is undefined, not merely distant. Counting it as (b) prices downstream work against
    # headroom that does not exist -- ADR-0013's standing finding in miniature.
    # NOT FOLDED INTO endpoint_lost, which is defined as "no qualifying relation has both
    # endpoints linked anywhere in the paper": this disease IS linked, so folding would make
    # bucket (a)'s own definition false. Hence a fourth bucket, placed immediately after (a)
    # because unrecoverable-by-any-mechanism outranks fixable-by-windowing.
    # sentence_spans(TEXT) == [(0, 20), (21, 53)] -- verified with the real splitter.
    chem, dis = "MESH:D008687", "MESH:D000138"
    doc = GoldDocument(
        pmid="pQ",
        text=TEXT,
        mentions=[
            _m(0, 9, EntityLabel.CHEMICAL, (chem,)),
            _m(21, 29, EntityLabel.DISEASE, (dis,)),
            _m(39, 48, EntityLabel.CHEMICAL, (chem,)),
        ],
    )
    relations = {"pQ": {(chem, dis)}}
    # The chemical is linked AND located in the gold sentence; the disease is linked but
    # carries no offset -- the population `Entity.start: int | None` admits and the real NER
    # path produces (extract_entities keeps a span whose `start` the pipeline omitted, and
    # canonicalize links on surface text alone and never restores an offset).
    entities = {
        "pQ": [
            Entity(
                text="metformin", label=EntityLabel.CHEMICAL, start=39, end=48, canonical_id=chem
            ),
            Entity(
                text="Acidosis", label=EntityLabel.DISEASE, start=None, end=None, canonical_id=dis
            ),
        ]
    }

    gold = gold_finding_sentences([doc], relations)
    assert gold == {"pQ": {1}}

    extractor = SameSentenceAsEntitiesExtractor(entities)
    pred = {"pQ": {f.sentence_index for f in extractor.findings(_paper("pQ", TEXT))}}
    # The real selector applies the SAME `start is None` guard, so it selects nothing and the
    # gold sentence is a genuine miss -- not one manufactured by a hand-built `pred`.
    assert pred == {"pQ": set()}

    buckets = classify_misses([doc], relations, gold, pred, entities_by_paper=entities)
    assert (
        buckets.endpoint_lost,
        buckets.endpoint_unlocatable,
        buckets.never_co_sentential,
        buckets.co_sentential_elsewhere,
    ) == (0, 1, 0, 0)
    assert buckets.total == 1
    assert buckets.endpoint_unlocatable_sentences == {"pQ": frozenset({1})}


def test_an_unlocatable_chemical_endpoint_is_unlocatable_just_as_a_disease_one_is():
    # SIBLING of the test above, on the same principle as
    # test_a_lost_chemical_endpoint_is_endpoint_lost_just_as_a_lost_disease_one_is. The new
    # bucket's condition is `c in locatable_chemicals AND d in locatable_diseases`, and with
    # only the disease-side fixture the `c in locatable_chemicals` conjunct could be deleted
    # outright and every test still passed -- the one-sided-compound-condition defect this
    # project has shipped repeatedly. Here the DISEASE is located and the CHEMICAL has no
    # offset, so it is the other conjunct that has to fire.
    # sentence_spans(TEXT) == [(0, 20), (21, 53)] -- verified with the real splitter.
    chem, dis = "MESH:D008687", "MESH:D000138"
    doc = GoldDocument(
        pmid="pR",
        text=TEXT,
        mentions=[
            _m(0, 9, EntityLabel.CHEMICAL, (chem,)),
            _m(21, 29, EntityLabel.DISEASE, (dis,)),
            _m(39, 48, EntityLabel.CHEMICAL, (chem,)),
        ],
    )
    relations = {"pR": {(chem, dis)}}
    entities = {
        "pR": [
            Entity(
                text="metformin",
                label=EntityLabel.CHEMICAL,
                start=None,
                end=None,
                canonical_id=chem,
            ),
            Entity(text="Acidosis", label=EntityLabel.DISEASE, start=21, end=29, canonical_id=dis),
        ]
    }

    gold = gold_finding_sentences([doc], relations)
    assert gold == {"pR": {1}}

    extractor = SameSentenceAsEntitiesExtractor(entities)
    pred = {"pR": {f.sentence_index for f in extractor.findings(_paper("pR", TEXT))}}
    assert pred == {"pR": set()}

    buckets = classify_misses([doc], relations, gold, pred, entities_by_paper=entities)
    assert (
        buckets.endpoint_lost,
        buckets.endpoint_unlocatable,
        buckets.never_co_sentential,
        buckets.co_sentential_elsewhere,
    ) == (0, 1, 0, 0)
    assert buckets.total == 1


def test_an_unlocatable_endpoint_does_not_hide_a_relation_whose_endpoints_are_both_placed():
    # THE OTHER DIRECTION of the new bucket, and the one that keeps it from swallowing
    # recoverable misses. The bucket's condition is on the WHOLE reachable set for that
    # sentence -- every reachable qualifying relation must have an unlocatable endpoint --
    # not on whether SOME endpoint anywhere lacks an offset. Implement it as "any endpoint of
    # any qualifying relation is unlocatable" and this fixture flips from
    # co_sentential_elsewhere to endpoint_unlocatable, moving a miss that a span fix would
    # recover into the bucket the eval reports as unrecoverable. That is the same
    # headroom-inflating direction the bucket exists to prevent, so it must be pinned from
    # both sides.
    # Sentence 0 is gold via TWO relations: (metformin, acidosis), whose endpoints are both
    # linked AND located, and (aspirin, fever), whose endpoints are linked but have no
    # offsets. The located pair co-occurs in sentence 1, so the miss at sentence 0 is a
    # genuine co_sentential_elsewhere.
    # sentence_spans(text) == [(0, 48), (49, 81)] -- verified with the real splitter.
    metformin, acidosis = "MESH:D008687", "MESH:D000138"
    aspirin, fever = "MESH:D000568", "MESH:D005334"
    text = "Metformin and aspirin caused acidosis and fever. Metformin caused acidosis again."
    doc = GoldDocument(
        pmid="pS",
        text=text,
        mentions=[
            GoldMention(
                pmid="pS",
                start=0,
                end=9,
                text=text[0:9],
                label=EntityLabel.CHEMICAL,
                mesh_ids=(metformin,),
            ),
            GoldMention(
                pmid="pS",
                start=14,
                end=21,
                text=text[14:21],
                label=EntityLabel.CHEMICAL,
                mesh_ids=(aspirin,),
            ),
            GoldMention(
                pmid="pS",
                start=29,
                end=37,
                text=text[29:37],
                label=EntityLabel.DISEASE,
                mesh_ids=(acidosis,),
            ),
            GoldMention(
                pmid="pS",
                start=42,
                end=47,
                text=text[42:47],
                label=EntityLabel.DISEASE,
                mesh_ids=(fever,),
            ),
        ],
    )
    relations = {"pS": {(metformin, acidosis), (aspirin, fever)}}
    entities = {
        "pS": [
            Entity(
                text="Metformin",
                label=EntityLabel.CHEMICAL,
                start=49,
                end=58,
                canonical_id=metformin,
            ),
            Entity(
                text="acidosis", label=EntityLabel.DISEASE, start=66, end=74, canonical_id=acidosis
            ),
            Entity(
                text="aspirin",
                label=EntityLabel.CHEMICAL,
                start=None,
                end=None,
                canonical_id=aspirin,
            ),
            Entity(
                text="fever", label=EntityLabel.DISEASE, start=None, end=None, canonical_id=fever
            ),
        ]
    }

    # Only sentence 0 carries gold mentions, so only it is gold.
    gold = gold_finding_sentences([doc], relations)
    assert gold == {"pS": {0}}

    extractor = SameSentenceAsEntitiesExtractor(entities)
    pred = {"pS": {f.sentence_index for f in extractor.findings(_paper("pS", text))}}
    # The real selector places both located endpoints in sentence 1 and picks it -- the one
    # sentence that is NOT gold.
    assert pred == {"pS": {1}}

    buckets = classify_misses([doc], relations, gold, pred, entities_by_paper=entities)
    assert (
        buckets.endpoint_lost,
        buckets.endpoint_unlocatable,
        buckets.never_co_sentential,
        buckets.co_sentential_elsewhere,
    ) == (0, 0, 0, 1)
    assert buckets.total == 1


def test_the_buckets_name_which_sentences_they_hold_not_only_how_many():
    # THE TASK 8 BLOCKER. `recall_on_endpoint_lost` restricts gold to the sentences classified
    # into bucket (a) and scores the LLM arm against that restriction alone -- the eval's
    # bottleneck-escape proof. A COUNT CANNOT EXPRESS IT: the restriction needs the identity of
    # every (pmid, sentence index) pair in the bucket, not how many there are.
    # The fixture is pI's shape -- ONE pmid whose TWO gold sentences are both endpoint_lost --
    # chosen so a membership mapping keyed by pmid (1 key) and a sentence count (2) differ. A
    # count derived as `len(mapping)` would report 1 and still look plausible.
    # sentence_spans(text) == [(0, 20), (21, 53), (54, 74), (75, 107)] -- verified with the real
    # splitter. Sentences 1 and 3 each hold both endpoints, so gold is {1, 3}.
    chem, dis = "MESH:D008687", "MESH:D000138"
    text = (
        "Metformin was given. Acidosis followed metformin use. "
        "Nausea was reported. Metformin caused acidosis again."
    )
    doc = GoldDocument(
        pmid="pP",
        text=text,
        mentions=[
            GoldMention(
                pmid="pP",
                start=21,
                end=29,
                text=text[21:29],
                label=EntityLabel.DISEASE,
                mesh_ids=(dis,),
            ),
            GoldMention(
                pmid="pP",
                start=39,
                end=48,
                text=text[39:48],
                label=EntityLabel.CHEMICAL,
                mesh_ids=(chem,),
            ),
            GoldMention(
                pmid="pP",
                start=75,
                end=84,
                text=text[75:84],
                label=EntityLabel.CHEMICAL,
                mesh_ids=(chem,),
            ),
            GoldMention(
                pmid="pP",
                start=92,
                end=100,
                text=text[92:100],
                label=EntityLabel.DISEASE,
                mesh_ids=(dis,),
            ),
        ],
    )
    relations = {"pP": {(chem, dis)}}
    # No disease is linked anywhere, so both gold misses are endpoint_lost.
    entities = {
        "pP": [
            Entity(text="Metformin", label=EntityLabel.CHEMICAL, start=0, end=9, canonical_id=chem)
        ]
    }

    gold = gold_finding_sentences([doc], relations)
    assert gold == {"pP": {1, 3}}

    extractor = SameSentenceAsEntitiesExtractor(entities)
    pred = {"pP": {f.sentence_index for f in extractor.findings(_paper("pP", text))}}
    assert pred == {"pP": set()}

    buckets = classify_misses([doc], relations, gold, pred, entities_by_paper=entities)
    assert buckets.endpoint_lost_sentences == {"pP": frozenset({1, 3})}


def _mb(
    *,
    lost: dict[str, set[int]] | None = None,
    unlocatable: dict[str, set[int]] | None = None,
    never: dict[str, set[int]] | None = None,
    elsewhere: dict[str, set[int]] | None = None,
) -> MissBuckets:
    """Build a MissBuckets from plain `pmid -> indices` dicts, for the closure tests.

    `MissBuckets` no longer takes counts: every count is derived from membership, so a
    hand-built instance must supply the sentences it claims.
    """

    def freeze(sentences: dict[str, set[int]] | None) -> dict[str, frozenset[int]]:
        return {pmid: frozenset(indices) for pmid, indices in (sentences or {}).items()}

    return MissBuckets(
        endpoint_lost_sentences=freeze(lost),
        endpoint_unlocatable_sentences=freeze(unlocatable),
        never_co_sentential_sentences=freeze(never),
        co_sentential_elsewhere_sentences=freeze(elsewhere),
    )


def test_bucket_closure_raises_when_a_miss_is_unaccounted():
    # Same 1 + 2 + 3 = 6 shape the count-valued constructor expressed as MissBuckets(1,2,3,6),
    # now spelled as the sentences those counts stand for. `total` is asserted as well, which
    # the old form could not: it WAS the constructor argument, so asserting it restated the
    # input; derived from membership, it is now a claim about the class.
    buckets = _mb(lost={"a": {0}}, never={"b": {0, 1}}, elsewhere={"c": {0, 1, 2}})
    assert (
        buckets.endpoint_lost,
        buckets.never_co_sentential,
        buckets.co_sentential_elsewhere,
    ) == (1, 2, 3)
    assert buckets.total == 6
    assert_bucket_closure(buckets, n_false_negatives=6)
    with pytest.raises(SystemExit, match="bucket closure"):
        assert_bucket_closure(buckets, n_false_negatives=7)


def test_bucket_closure_rejects_a_sentence_that_two_buckets_both_claim():
    # REPLACES test_bucket_closure_raises_when_the_buckets_do_not_sum_to_total. That test
    # pinned a failure mode this commit DELETED: `total` was a free constructor field, so
    # MissBuckets(1, 2, 3, 99) was constructible and needed its own check. `total` is now
    # derived from membership and cannot be set at all, so the check it pinned became
    # provably unfireable -- a test that cannot fail.
    # What membership makes checkable for the FIRST time is mutual exclusivity, which was
    # previously true only BY CONSTRUCTION via the `continue` after each bucket and so had no
    # observable form to assert. Every unordered pair of buckets is exercised, because a check
    # written over adjacent pairs only -- or over the first two -- passes any single-pair
    # fixture while leaving four pairs unguarded.
    # `n_false_negatives` is set to 2 throughout, which is what `total` double-counts to, so
    # the closure comparison PASSES and the overlap check is the only thing that can raise.
    one: dict[str, set[int]] = {"a": {0}}
    every_pair = [
        _mb(lost=one, unlocatable=one),
        _mb(lost=one, never=one),
        _mb(lost=one, elsewhere=one),
        _mb(unlocatable=one, never=one),
        _mb(unlocatable=one, elsewhere=one),
        _mb(never=one, elsewhere=one),
    ]
    for overlapping in every_pair:
        assert overlapping.total == 2
        with pytest.raises(SystemExit, match="bucket overlap"):
            assert_bucket_closure(overlapping, n_false_negatives=2)

    # Anti-vacuity, and the discriminating shape: the SAME pmid in two buckets is fine as long
    # as the sentences differ, so the check must compare index sets and not merely pmid keys.
    disjoint = _mb(lost={"a": {0}}, never={"a": {1}})
    assert disjoint.total == 2
    assert_bucket_closure(disjoint, n_false_negatives=2)

    # ORDER OF THE TWO CHECKS, which the loop above deliberately cannot see because it feeds a
    # matching `n_false_negatives`. In the REAL failure an overlap also inflates `total`, so
    # BOTH checks are eligible and only the earlier one reports. Swap them and every case above
    # still passes while the live failure reports "a miss was misclassified or double-counted"
    # -- the derived symptom -- instead of naming the two buckets and the sentence.
    # `n_false_negatives=1` is the truth here: one real miss, double-counted to a `total` of 2.
    with pytest.raises(SystemExit, match="bucket overlap"):
        assert_bucket_closure(_mb(lost={"a": {0}}, elsewhere={"a": {0}}), n_false_negatives=1)
