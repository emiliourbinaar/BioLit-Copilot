from typing import Protocol

from biolit.canon.mesh import LinkResult, MeshDictionary


class Linker(Protocol):
    def link(self, surface: str) -> LinkResult: ...


class DictionaryLinker:
    """Offline linker over a CTD->MeSH alias table. Normalization/tiebreak live in
    MeshDictionary.lookup; this is a thin adapter so a future embedding-based linker can
    implement the same Linker protocol without touching canonicalize()."""

    def __init__(self, dictionary: MeshDictionary) -> None:
        self._dictionary = dictionary

    def link(self, surface: str) -> LinkResult:
        return self._dictionary.lookup(surface)
