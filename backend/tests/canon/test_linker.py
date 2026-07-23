from biolit.canon.linker import DictionaryLinker, Linker
from biolit.canon.mesh import AliasEntry, MeshConcept, MeshDictionary


def test_dictionary_linker_delegates_to_dictionary():
    concept = MeshConcept(id="MESH:D008687", name="Metformin")
    d = MeshDictionary({"metformin": [AliasEntry(concept, True)]})
    linker: Linker = DictionaryLinker(d)
    result = linker.link("Metformin")
    assert result.concept is not None and result.concept.id == "MESH:D008687"
    assert linker.link("nope").concept is None
