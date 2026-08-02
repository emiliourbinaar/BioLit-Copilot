from biolit.cluster.pairing import sentence_index
from biolit.domain.enums import EntityLabel
from biolit.ner.windowing import sentence_spans
from biolit_evals.extract_eval import gold_finding_sentences
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
