from collections import defaultdict
from collections.abc import Iterable
from enum import StrEnum

# The header is the comment line containing BOTH of these. Selection is by content because
# CTD emits further '#' lines after it -- a "last comment line wins" rule picks up an empty
# one, yields no columns, and parses zero rows while looking like a clean empty result.
CTD_HEADER_MARKERS = ("ChemicalName", "DirectEvidence")

# The provenance line CTD writes into every report, verified against the real
# CTD_chemicals_diseases.tsv.gz: "# Report created: Thu Jul 30 13:59:07 EDT 2026".
CTD_RELEASE_MARKER = "Report created:"


class MissingCtdReleaseStampError(RuntimeError):
    """Raised when a CTD file carries no 'Report created:' line."""


def ctd_release_stamp(lines: Iterable[str]) -> str:
    """The CTD release stamp, verbatim, from the report's own '#' header.

    CTD is republished continuously, so a corpus built from it is meaningless without a
    record of WHICH CTD produced it. The spec requires this stamp in the run log for exactly
    that reason, and the committed manifest -- not CTD -- is the frozen artifact.

    Returned as the raw string rather than a parsed datetime: `EDT` is a US-only abbreviation
    that no stdlib parser reads portably, and converting it would put a failure mode inside
    the one field whose whole job is provenance.

    Absence RAISES rather than returning a placeholder. A run logged with an unknown release
    is not reproducible, and an "unknown" string in that field would be discovered long after
    the corpus it describes had been used -- the quiet failure this project keeps paying for.
    """
    for line in lines:
        if not line.startswith("#"):
            break
        if CTD_RELEASE_MARKER in line:
            return line.split(CTD_RELEASE_MARKER, 1)[1].strip()
    raise MissingCtdReleaseStampError(
        f"ctd_release_stamp: no '{CTD_RELEASE_MARKER}' line in the CTD header. "
        "Refusing to build a corpus whose CTD release cannot be recorded."
    )


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
