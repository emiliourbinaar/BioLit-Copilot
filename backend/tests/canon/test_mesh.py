from pathlib import Path

from biolit.canon.mesh import (
    AliasEntry,
    MeshConcept,
    MeshDictionary,
    build_alias_table,
    normalize_surface,
)

_FIX = Path(__file__).parent / "fixtures"


def _text(name: str) -> str:
    return (_FIX / name).read_text(encoding="utf-8")


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


def test_build_alias_table_prefixes_chemicals_and_keeps_disease_prefix():
    table = build_alias_table(_text("ctd_chemicals_sample.tsv"), _text("ctd_diseases_sample.tsv"))
    # chemical preferred name -> MESH-prefixed id (already prefixed in real CTD data)
    met = table["metformin"]
    assert len(met) == 1 and met[0].concept.id == "MESH:D008687" and met[0].is_preferred_name
    # chemical synonym (from MESHSynonyms column) -> non-preferred, same concept
    assert table["glucophage"][0].concept.id == "MESH:D008687"
    assert table["glucophage"][0].is_preferred_name is False
    # disease id prefix preserved as-is
    assert table["pcos"][0].concept.id == "MESH:D011085"


def test_artifact_round_trip(tmp_path):
    table = build_alias_table(_text("ctd_chemicals_sample.tsv"), _text("ctd_diseases_sample.tsv"))
    d = MeshDictionary(table)
    path = str(tmp_path / "mesh.json.gz")
    d.save_artifact(path)
    reloaded = MeshDictionary.from_artifact(path)
    r = reloaded.lookup("Glucophage")
    assert r.concept is not None and r.concept.id == "MESH:D008687"
    assert r.concept.name == "Metformin"


def test_build_alias_table_skips_header_comment_rows():
    table = build_alias_table(_text("ctd_chemicals_sample.tsv"), _text("ctd_diseases_sample.tsv"))
    assert "chemicalname" not in table and "diseasename" not in table
    assert not any(key.startswith("#") or "﻿" in key for key in table)
