from biolit.cluster.group import cluster_papers, pairing_diagnostics
from biolit.cluster.pairing import CrossProductPairing
from biolit.domain.enums import EntityLabel
from biolit.domain.records import Entity, ExtractedRecord

CHEM = EntityLabel.CHEMICAL
DIS = EntityLabel.DISEASE


def _rec(pid, pairs):
    entities = []
    for i, (c, d) in enumerate(pairs):
        entities.append(Entity(text="c", label=CHEM, start=i, end=i + 1, canonical_id=c))
        entities.append(Entity(text="d", label=DIS, start=i, end=i + 1, canonical_id=d))
    return ExtractedRecord(paper_id=pid, entities=entities)


def test_a_key_held_by_only_one_paper_is_not_a_cluster():
    # min_size=2: the Critic compares papers WITHIN a cluster, so a singleton key is inert.
    # This is what makes cross-product over-generation survivable downstream.
    records = [
        _rec("A", [("MESH:D008687", "MESH:D011085")]),
        _rec("B", [("MESH:D008687", "MESH:D011085")]),
        _rec("C", [("MESH:D001241", "MESH:D014456")]),
    ]
    clusters = cluster_papers(
        records, texts={"A": "t", "B": "t", "C": "t"}, pairing=CrossProductPairing()
    )
    assert [c.key for c in clusters] == ["MESH:D008687|MESH:D011085"]
    assert clusters[0].paper_ids == ["A", "B"]


def test_nil_diagnostics_report_which_side_was_unlinked():
    # DISEASE has NIL'd at roughly double CHEMICAL's rate throughout canonicalization, so
    # the population this exclusion makes unmeasurable is probably not uniform. A single
    # combined counter would hide that.
    records = [
        ExtractedRecord(
            paper_id="A",
            entities=[
                Entity(text="c", label=CHEM, start=0, end=1, canonical_id=None),
                Entity(text="d", label=DIS, start=2, end=3, canonical_id=None),
                Entity(text="d2", label=DIS, start=4, end=5, canonical_id=None),
                Entity(text="ok", label=CHEM, start=6, end=7, canonical_id="MESH:D008687"),
                Entity(text="np", label=DIS, start=None, end=None, canonical_id="MESH:D011085"),
            ],
        )
    ]
    diagnostics = pairing_diagnostics(records, texts={"A": "Some sentence here."})
    assert diagnostics.nil_chemical_mentions == 1
    assert diagnostics.nil_disease_mentions == 2
    assert diagnostics.unplaceable_entities == 1
    assert diagnostics.n_papers == 1
