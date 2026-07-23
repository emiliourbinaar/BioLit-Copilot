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


def _read_ctd_dump(text: str) -> tuple[list[str], list[list[str]]]:
    """Split a raw CTD TSV dump into (column_names, data_rows).

    CTD ships many '#'-prefixed comment lines; the column header is the single '#'
    line whose first field names a known column. Columns are matched BY NAME, not
    position, because CTD periodically adds columns -- a fixed-index assumption
    silently misreads ids/synonyms (which is exactly the defect this replaced).
    """
    header: list[str] = []
    rows: list[list[str]] = []
    for line in text.splitlines():
        if not line:
            continue
        if line.startswith("#"):
            fields = line.lstrip("#").strip().split("\t")
            if fields and fields[0] in ("ChemicalName", "DiseaseName"):
                header = fields
            continue
        rows.append(line.split("\t"))
    return header, rows


def _ingest_ctd(
    table: dict[str, list[AliasEntry]],
    text: str,
    *,
    name_col: str,
    id_col: str,
    synonym_cols: list[str],
) -> None:
    header, rows = _read_ctd_dump(text)
    if not header:
        raise ValueError(f"CTD header row not found (expected a '# {name_col}...' line)")
    idx = {col: i for i, col in enumerate(header)}
    name_i, id_i = idx[name_col], idx[id_col]
    syn_i = [idx[c] for c in synonym_cols if c in idx]
    for row in rows:
        if len(row) <= max(name_i, id_i):
            continue
        name = row[name_i].strip()
        raw_id = row[id_i].strip()
        if not name or not raw_id:
            continue
        # CTD ids already carry their MESH:/OMIM: prefix; keep verbatim, and only
        # prefix a bare accession defensively.
        concept_id = raw_id if ":" in raw_id else f"MESH:{raw_id}"
        concept = MeshConcept(id=concept_id, name=name)
        _add_alias(table, name, concept, True)
        for col in syn_i:
            if col < len(row):
                for syn in row[col].split("|"):
                    if syn.strip():
                        _add_alias(table, syn, concept, False)


def build_alias_table(chem_text: str, disease_text: str) -> dict[str, list[AliasEntry]]:
    """Build a normalized alias -> [AliasEntry] table from raw CTD chemical + disease
    TSV dumps (full text, including the '# ...' column header).

    Columns are resolved by name from each file's header. CTD chemical and disease IDs
    both already carry a MESH:/OMIM: prefix and are kept verbatim. Chemicals contribute
    their MESHSynonyms; diseases their Synonyms.
    """
    table: dict[str, list[AliasEntry]] = {}
    _ingest_ctd(
        table,
        chem_text,
        name_col="ChemicalName",
        id_col="ChemicalID",
        synonym_cols=["MESHSynonyms"],
    )
    _ingest_ctd(
        table,
        disease_text,
        name_col="DiseaseName",
        id_col="DiseaseID",
        synonym_cols=["Synonyms"],
    )
    return table
