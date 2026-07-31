from biolit.domain.enums import EntityLabel, Source, TextType
from biolit.domain.paper import Paper
from biolit.domain.records import Entity
from biolit.extract.deterministic import SameSentenceAsEntitiesExtractor

# "Metformin was given. Acidosis followed. Insulin fell."
#  0                 20 22               40 42
TEXT = "Metformin was given. Acidosis followed. Insulin fell."


def _paper() -> Paper:
    return Paper(
        id="p1",
        source=Source.pubmed,
        title="t",
        abstract=TEXT,
        text_type=TextType.abstract_only,
        extraction_allowed=True,
    )


def _ent(label: EntityLabel, start: int, canonical_id: str | None) -> Entity:
    return Entity(text="x", label=label, start=start, end=start + 1, canonical_id=canonical_id)


def test_only_sentences_holding_both_a_linked_chemical_and_disease_are_selected():
    # Sentence 0 has a chemical only, sentence 1 a disease only, sentence 2 both -> only 2.
    # A selector that returned any sentence with any entity would return all three.
    entities = [
        _ent(EntityLabel.CHEMICAL, 0, "MESH:D008687"),
        _ent(EntityLabel.DISEASE, 21, "MESH:D000138"),
        _ent(EntityLabel.CHEMICAL, 40, "MESH:D007328"),
        _ent(EntityLabel.DISEASE, 48, "MESH:D000138"),
    ]
    extractor = SameSentenceAsEntitiesExtractor({"p1": entities})
    assert [f.sentence_index for f in extractor.findings(_paper())] == [2]


def test_unlinked_and_unplaceable_entities_are_both_excluded():
    # TWO INDEPENDENT GUARDS -- test both, or reducing the condition to one disjunct passes.
    # This is the disjunction gap that recurred five times on the clustering branch.
    # (a) canonical_id None: sentence 2 has both labels but the chemical never linked.
    # (a-mirror) canonical_id None on the OTHER label: an unlinked entity must not count
    # toward EITHER side, not just chemicals -- the guard is one unconditional line ahead
    # of the label dispatch, so this is the same property applied to the other label.
    # (b) start None: sentence 2's disease has no offset.
    unlinked = [
        _ent(EntityLabel.CHEMICAL, 40, None),
        _ent(EntityLabel.DISEASE, 48, "MESH:D000138"),
    ]
    assert SameSentenceAsEntitiesExtractor({"p1": unlinked}).findings(_paper()) == []

    unlinked_disease = [
        _ent(EntityLabel.CHEMICAL, 40, "MESH:D007328"),
        _ent(EntityLabel.DISEASE, 48, None),
    ]
    assert SameSentenceAsEntitiesExtractor({"p1": unlinked_disease}).findings(_paper()) == []

    no_offset = Entity(
        text="x", label=EntityLabel.DISEASE, start=None, end=None, canonical_id="MESH:D000138"
    )
    unplaceable = [_ent(EntityLabel.CHEMICAL, 40, "MESH:D007328"), no_offset]
    assert SameSentenceAsEntitiesExtractor({"p1": unplaceable}).findings(_paper()) == []


def test_entity_landing_in_inter_sentence_whitespace_contributes_nothing():
    # sentence_spans(TEXT) == [(0, 20), (21, 39), (40, 53)] (verified below in the fix
    # report) -- index 20 is the space between "given." and "Acidosis", covered by no span.
    # sentence_index returns None there. Sentence 0 already holds a linked disease; if the
    # gap chemical were (wrongly) folded into sentence 0, this would wrongly select it.
    entities = [
        _ent(EntityLabel.DISEASE, 0, "MESH:D000138"),
        _ent(EntityLabel.CHEMICAL, 20, "MESH:D008687"),
    ]
    assert SameSentenceAsEntitiesExtractor({"p1": entities}).findings(_paper()) == []
