from collections import defaultdict
from collections.abc import Iterable
from enum import StrEnum

# The header is the comment line containing BOTH of these. Selection is by content because
# CTD emits further '#' lines after it -- a "last comment line wins" rule picks up an empty
# one, yields no columns, and parses zero rows while looking like a clean empty result.
CTD_HEADER_MARKERS = ("ChemicalName", "DirectEvidence")


class Direction(StrEnum):
    marker_mechanism = "marker/mechanism"
    therapeutic = "therapeutic"


def normalize_mesh_id(value: str) -> str:
    """Strip a namespace prefix so ids from different sources join.

    CTD writes ChemicalID bare (`C046983`) and DiseaseID prefixed (`MESH:D054198`); BC5CDR
    prefixes both. AN UNNORMALIZED JOIN MATCHES NOTHING AND REPORTS ZERO CONTRADICTIONS,
    which reads exactly like a true negative -- it did, twice, during design probing.
    """
    return value.split(":", 1)[1] if ":" in value else value


def parse_ctd_directions(
    lines: Iterable[str],
) -> dict[str, dict[tuple[str, str], frozenset[Direction]]]:
    """{pmid: {(chemical_id, disease_id): directions}} over DIRECT-evidence rows only.

    Rows without DirectEvidence are CTD's gene-inferred associations -- no paper asserts the
    relation, so they carry no direction and are dropped.
    """
    header: list[str] | None = None
    accumulated: dict[str, dict[tuple[str, str], set[Direction]]] = defaultdict(dict)
    for line in lines:
        if line.startswith("#"):
            if all(marker in line for marker in CTD_HEADER_MARKERS):
                header = line.lstrip("#").strip().split("\t")
            continue
        if header is None:
            raise ValueError(
                "parse_ctd_directions: reached a data row before the column header. "
                f"Expected a '#' line containing all of {CTD_HEADER_MARKERS}."
            )
        row = dict(zip(header, line.rstrip("\n").split("\t"), strict=False))
        evidence = (row.get("DirectEvidence") or "").strip()
        pmids = (row.get("PubMedIDs") or "").strip()
        if not evidence or not pmids:
            continue
        key = (
            normalize_mesh_id(row["ChemicalID"].strip()),
            normalize_mesh_id(row["DiseaseID"].strip()),
        )
        directions = {Direction(d) for d in evidence.split("|") if d}
        for pmid in pmids.split("|"):
            accumulated[pmid].setdefault(key, set()).update(directions)
    return {
        pmid: {key: frozenset(values) for key, values in keys.items()}
        for pmid, keys in accumulated.items()
    }
