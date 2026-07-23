from biolit.canon.mesh import (
    AliasEntry,
    MeshConcept,
    MeshDictionary,
    normalize_surface,
)


def test_normalize_surface_casefolds_and_collapses_whitespace():
    assert normalize_surface("  GLP-1  Receptor   Agonists ") == "glp-1 receptor agonists"


def _dict() -> MeshDictionary:
    metformin = MeshConcept(id="MESH:D008687", name="Metformin")
    other = MeshConcept(id="MESH:D000001", name="Aspirin Variant")
    return MeshDictionary(
        {
            "metformin": [AliasEntry(metformin, True)],
            "glucophage": [AliasEntry(metformin, False)],
            # ambiguous: two distinct concepts, one via preferred name
            "ambig": [AliasEntry(metformin, False), AliasEntry(other, True)],
            # ambiguous, no preferred name -> smallest id wins
            "ambignopref": [AliasEntry(metformin, False), AliasEntry(other, False)],
        }
    )


def test_lookup_exact_and_synonym_hit_are_not_tiebroken():
    d = _dict()
    r_name = d.lookup("Metformin")
    r_syn = d.lookup("glucophage")
    assert r_name.concept is not None and r_name.concept.id == "MESH:D008687"
    assert r_syn.concept is not None and r_syn.concept.id == "MESH:D008687"
    assert r_name.tiebroken is False and r_syn.tiebroken is False


def test_lookup_miss_returns_nil():
    assert _dict().lookup("nonexistent term").concept is None


def test_lookup_ambiguous_prefers_preferred_name_and_flags_tiebreak():
    r = _dict().lookup("ambig")
    assert r.concept is not None and r.concept.id == "MESH:D000001"  # preferred-name entry
    assert r.tiebroken is True


def test_lookup_ambiguous_no_preferred_picks_smallest_id():
    r = _dict().lookup("ambignopref")
    assert r.concept is not None and r.concept.id == "MESH:D000001"  # lexicographically smallest
    assert r.tiebroken is True
