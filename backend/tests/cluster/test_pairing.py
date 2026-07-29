from biolit.cluster.pairing import CrossProductPairing
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
