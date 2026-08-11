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
    # THIRD precondition: `gold` must come from gold_finding_sentences over the SAME
    # `documents` and `relations`. Unenforced, the violation is silent and lands in the worst
    # possible bucket. Here sentence 1 holds both endpoints and BOTH are linked, so the real
    # gold is {"pH": {1}} and the miss is co_sentential_elsewhere -- reachable, recoverable.
    # Hand classify_misses a gold of {"pH": {0}} instead and the missed index 0 is absent from
    # _gold_pairs_by_sentence, so `pairs_at_index` is empty, `reachable` is empty, and the miss
    # is reported as endpoint_lost -- the UNRECOVERABLE bucket -- with assert_bucket_closure
    # passing, because closure only checks that misses were counted, never WHERE they landed.
    # Reporting a recoverable miss as unrecoverable inflates the LLM arm's apparent headroom,
    # the exact failure ADR-0013's standing finding warns against.
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
