from biolit.cluster.pairing import CrossProductPairing, SameSentencePairing
from biolit.domain.enums import EntityLabel
from biolit.domain.records import Entity

CHEM = EntityLabel.CHEMICAL
DIS = EntityLabel.DISEASE


def _e(label, cid, start=0):
    return Entity(text="x", label=label, start=start, end=start + 1, canonical_id=cid)


def test_cross_product_pairs_every_chemical_with_every_disease():
    entities = [
        _e(CHEM, "MESH:D008687"),
        _e(CHEM, "MESH:D007328"),
        _e(DIS, "MESH:D011085"),
    ]
    assert CrossProductPairing().pairs(entities, "irrelevant text") == {
        ("MESH:D008687", "MESH:D011085"),
        ("MESH:D007328", "MESH:D011085"),
    }


def test_a_nil_entity_forms_no_pairs():
    # NIL endpoints are unscoreable against gold CID; their cost is reported separately.
    entities = [_e(CHEM, None), _e(DIS, "MESH:D011085")]
    assert CrossProductPairing().pairs(entities, "t") == set()


def test_a_paper_with_no_disease_forms_no_pairs():
    entities = [_e(CHEM, "MESH:D008687"), _e(CHEM, "MESH:D007328")]
    assert CrossProductPairing().pairs(entities, "t") == set()


def test_same_sentence_pairing_differs_from_cross_product_across_a_sentence_boundary():
    # Sentence 1 holds the chemical and one disease; sentence 2 holds another disease.
    # A strategy that ignores sentence boundaries returns BOTH pairs and fails here.
    text = "Metformin caused nausea. Separately, hepatic injury was observed."
    chem_at = text.index("Metformin")
    nausea_at = text.index("nausea")
    injury_at = text.index("hepatic injury")
    entities = [
        Entity(
            text="Metformin",
            label=CHEM,
            start=chem_at,
            end=chem_at + 9,
            canonical_id="MESH:D008687",
        ),
        Entity(
            text="nausea",
            label=DIS,
            start=nausea_at,
            end=nausea_at + 6,
            canonical_id="MESH:D009325",
        ),
        Entity(
            text="hepatic injury",
            label=DIS,
            start=injury_at,
            end=injury_at + 14,
            canonical_id="MESH:D056486",
        ),
    ]
    assert SameSentencePairing().pairs(entities, text) == {("MESH:D008687", "MESH:D009325")}
    # The discriminator: cross-product genuinely returns MORE here, so a same-sentence
    # implementation that just delegates to it cannot pass both assertions.
    assert CrossProductPairing().pairs(entities, text) == {
        ("MESH:D008687", "MESH:D009325"),
        ("MESH:D008687", "MESH:D056486"),
    }


def test_same_sentence_pairing_uses_sentence_spans_boundaries():
    # Two sentences, one chemical and one disease in each. Only the within-sentence pairs
    # may appear -- this pins the dependency on sentence_spans, not just on "same text".
    text = "Aspirin caused ulcers. Metformin caused acidosis."
    entities = [
        Entity(text="Aspirin", label=CHEM, start=0, end=7, canonical_id="MESH:D001241"),
        Entity(
            text="ulcers",
            label=DIS,
            start=text.index("ulcers"),
            end=text.index("ulcers") + 6,
            canonical_id="MESH:D014456",
        ),
        Entity(
            text="Metformin",
            label=CHEM,
            start=text.index("Metformin"),
            end=text.index("Metformin") + 9,
            canonical_id="MESH:D008687",
        ),
        Entity(
            text="acidosis",
            label=DIS,
            start=text.index("acidosis"),
            end=text.index("acidosis") + 8,
            canonical_id="MESH:D000138",
        ),
    ]
    assert SameSentencePairing().pairs(entities, text) == {
        ("MESH:D001241", "MESH:D014456"),
        ("MESH:D008687", "MESH:D000138"),
    }


def test_an_entity_without_offsets_fails_closed_and_forms_no_pairs():
    # start is None -> unplaceable in any sentence. It must not be silently bucketed into
    # sentence 0, which would invent pairs with the document's opening entities.
    text = "Metformin caused nausea."
    entities = [
        Entity(text="Metformin", label=CHEM, start=0, end=9, canonical_id="MESH:D008687"),
        Entity(text="nausea", label=DIS, start=None, end=None, canonical_id="MESH:D009325"),
    ]
    assert SameSentencePairing().pairs(entities, text) == set()
