import gzip
import json
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

    def save_artifact(self, path: str) -> None:
        payload = {
            alias: [[e.concept.id, e.concept.name, e.is_preferred_name] for e in entries]
            for alias, entries in self._aliases.items()
        }
        with gzip.open(path, "wt", encoding="utf-8") as fh:
            json.dump(payload, fh)

    @classmethod
    def from_artifact(cls, path: str) -> "MeshDictionary":
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            payload = json.load(fh)
        aliases: dict[str, list[AliasEntry]] = {
            alias: [AliasEntry(MeshConcept(id=i, name=n), bool(p)) for i, n, p in rows]
            for alias, rows in payload.items()
        }
        return cls(aliases)


def _add_alias(
    table: dict[str, list[AliasEntry]], alias: str, concept: MeshConcept, is_pref: bool
) -> None:
    key = normalize_surface(alias)
    if not key:
        return
    bucket = table.setdefault(key, [])
    for existing in bucket:
        if existing.concept.id == concept.id and existing.is_preferred_name == is_pref:
            return
    bucket.append(AliasEntry(concept, is_pref))


def _ingest_rows(table: dict[str, list[AliasEntry]], rows: list[list[str]], id_prefix: str) -> None:
    for row in rows:
        if len(row) < 2:
            continue
        name = row[0].strip()
        raw_id = row[1].strip()
        if not name or not raw_id:
            continue
        concept = MeshConcept(id=f"{id_prefix}{raw_id}", name=name)
        _add_alias(table, name, concept, True)
        synonyms = row[7] if len(row) > 7 else ""
        for syn in synonyms.split("|"):
            if syn.strip():
                _add_alias(table, syn, concept, False)


def build_alias_table(
    chem_rows: list[list[str]], disease_rows: list[list[str]]
) -> dict[str, list[AliasEntry]]:
    """Build a normalized alias -> [AliasEntry] table from CTD chemical + disease rows.

    CTD chemical IDs are bare MeSH accessions (`D008687`) and get a `MESH:` prefix; CTD
    disease IDs already carry their `MESH:`/`OMIM:` prefix and are used verbatim.
    """
    table: dict[str, list[AliasEntry]] = {}
    _ingest_rows(table, chem_rows, "MESH:")
    _ingest_rows(table, disease_rows, "")
    return table
