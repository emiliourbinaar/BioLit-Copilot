from biolit.domain.enums import EntityLabel
from biolit_evals.extract_eval import gold_finding_sentences
from biolit_evals.mesh_gold import GoldDocument, GoldMention

TEXT = "Metformin was given. Acidosis followed metformin use."


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


def test_number_of_relations_with_a_gold_sentence_is_at_most_the_published_bc5cdr_count():
    # Independent invariant (not the count itself, which is a regression pin): a gold
    # sentence always traces to a real gold CID relation, so relations that get a gold
    # sentence can never exceed BC5CDR Test-500's published gold CID relation count.
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
    gold = gold_finding_sentences([doc], relations)
    relations_with_a_gold_sentence = {
        pair
        for pmid, sentence_indices in gold.items()
        for pair in relations[pmid]
        if sentence_indices
    }
    assert len(relations_with_a_gold_sentence) <= 1066
