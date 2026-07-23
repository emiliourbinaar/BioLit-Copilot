import re
from dataclasses import dataclass

_WHITESPACE = re.compile(r"\s+")


def normalize_surface(s: str) -> str:
    """Casefold, strip, and collapse internal whitespace so the alias table and a
    lookup query are normalized identically (they MUST use this same function)."""
    return _WHITESPACE.sub(" ", s.strip()).casefold()


@dataclass(frozen=True)
class MeshConcept:
    id: str  # prefixed: "MESH:D008687", "OMIM:125853"
    name: str


@dataclass(frozen=True)
class AliasEntry:
    concept: MeshConcept
    is_preferred_name: bool


@dataclass(frozen=True)
class LinkResult:
    concept: MeshConcept | None
    tiebroken: bool


class MeshDictionary:
    def __init__(self, aliases: dict[str, list[AliasEntry]]) -> None:
        self._aliases = aliases

    def lookup(self, surface: str) -> LinkResult:
        entries = self._aliases.get(normalize_surface(surface))
        if not entries:
            return LinkResult(None, False)
        distinct_ids = {e.concept.id for e in entries}
        if len(distinct_ids) == 1:
            return LinkResult(entries[0].concept, False)
        preferred = [e for e in entries if e.is_preferred_name]
        pool = preferred if preferred else entries
        concept = min((e.concept for e in pool), key=lambda c: c.id)
        return LinkResult(concept, True)
