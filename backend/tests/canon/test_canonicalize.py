from biolit.canon.canonicalize import canonicalize
from biolit.canon.linker import DictionaryLinker
from biolit.canon.mesh import AliasEntry, MeshConcept, MeshDictionary
from biolit.domain.enums import EntityLabel
from biolit.domain.records import Entity

CHEMICAL = EntityLabel.CHEMICAL


def _linker() -> DictionaryLinker:
    glp = MeshConcept(id="MESH:D000067299", name="Glucagon-Like Peptide-1 Receptor Agonists")
    met = MeshConcept(id="MESH:D008687", name="Metformin")
    return DictionaryLinker(
        MeshDictionary(
            {
                "glp-1ra": [AliasEntry(glp, True)],
                "metformin": [AliasEntry(met, True)],
            }
        )
    )


def test_merged_fragment_links_all_constituents_to_one_concept():
    text = "GLP-1RA and metformin"
    ents = [
        Entity(text="GLP", label=CHEMICAL, start=0, end=3),
        Entity(text="1RA", label=CHEMICAL, start=4, end=7),
        Entity(text="metformin", label=CHEMICAL, start=12, end=21),
    ]
    out = canonicalize(ents, text, linker=_linker())
    assert out[0].canonical_id == "MESH:D000067299"
    assert out[1].canonical_id == "MESH:D000067299"  # both fragments share the merged concept
    assert out[2].canonical_id == "MESH:D008687"  # individual lookup


def test_unlinked_entity_stays_nil():
    text = "unobtainium"
    ents = [Entity(text="unobtainium", label=CHEMICAL, start=0, end=11)]
    out = canonicalize(ents, text, linker=_linker())
    assert out[0].canonical_id is None and out[0].canonical_name is None


def test_does_not_mutate_input_entities():
    text = "metformin"
    ents = [Entity(text="metformin", label=CHEMICAL, start=0, end=9)]
    canonicalize(ents, text, linker=_linker())
    assert ents[0].canonical_id is None  # original untouched


def test_spurious_merge_does_not_overwrite_correct_individual_links():
    # "aspirin/metformin"-style combination notation: merge_fragments is permissive
    # enough to propose a merge here, and the merged surface happens to resolve -- but
    # each constituent links correctly on its own, so each must KEEP its own concept.
    asa = MeshConcept(id="MESH:D001241", name="Aspirin")
    met = MeshConcept(id="MESH:D008687", name="Metformin")
    combo = MeshConcept(id="MESH:D999999", name="Wrong Combo Concept")
    linker = DictionaryLinker(
        MeshDictionary(
            {
                "asa": [AliasEntry(asa, True)],
                "metformin": [AliasEntry(met, True)],
                "asa/metformin": [AliasEntry(combo, True)],
            }
        )
    )
    text = "ASA/metformin therapy"
    ents = [
        Entity(text="ASA", label=CHEMICAL, start=0, end=3),
        Entity(text="metformin", label=CHEMICAL, start=4, end=13),
    ]
    out = canonicalize(ents, text, linker=linker)
    assert out[0].canonical_id == "MESH:D001241"
    assert out[1].canonical_id == "MESH:D008687"
