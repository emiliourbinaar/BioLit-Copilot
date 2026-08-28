import pytest

from biolit_evals.ctd_directions import (
    Direction,
    MissingCtdReleaseStampError,
    ctd_release_stamp,
    parse_ctd_directions,
)

_HEADER = (
    "# ChemicalName\tChemicalID\tCasRN\tDiseaseName\tDiseaseID\tDirectEvidence"
    "\tInferenceGeneSymbol\tInferenceScore\tOmimIDs\tPubMedIDs"
)


def test_header_is_selected_by_content_not_by_last_comment_line():
    """CTD puts further '#' lines AFTER the column header, so 'last comment wins' yields an
    empty header and silently parses zero rows."""
    lines = [
        "# Report created: Thu Jul 30 13:59:07 EDT 2026",
        "# Fields:",
        _HEADER,
        "#",
        "aspirin\tC000001\t\tfever\tMESH:D000001\ttherapeutic\t\t\t\t111",
    ]
    parsed = parse_ctd_directions(lines)
    assert parsed == {"111": {("C000001", "D000001"): frozenset({Direction.therapeutic})}}


def test_data_row_before_header_raises_rather_than_returning_empty():
    """LOUDER, not quieter (ADR-0014): a silent {} here is a wrong answer that names nothing."""
    with pytest.raises(ValueError, match="before the column header"):
        parse_ctd_directions(["aspirin\tC000001\t\tfever\tMESH:D000001\ttherapeutic\t\t\t\t111"])


def test_row_with_evidence_but_no_pmids_is_dropped():
    lines = [_HEADER, "aspirin\tC000001\t\tfever\tMESH:D000001\ttherapeutic\t\t\t\t"]
    assert parse_ctd_directions(lines) == {}


def test_row_with_pmids_but_no_evidence_is_dropped():
    lines = [_HEADER, "aspirin\tC000001\t\tfever\tMESH:D000001\t\tMYC\t4.08\t\t111"]
    assert parse_ctd_directions(lines) == {}


def test_a_later_comment_naming_only_chemicalname_does_not_replace_the_header():
    """One operand of the header condition, in isolation: a comment naming ChemicalName without
    DirectEvidence. Placed AFTER the real header, not before -- a weaker line before the real
    header would just get correctly overwritten by the real header either way, hiding the
    defect. Placed after, wrongly accepting it as the header replaces the good column names
    with a single bogus column, so DirectEvidence reads as missing on every later row and the
    row is silently dropped, returning {} instead of the correct mapping."""
    lines = [
        _HEADER,
        "# ChemicalName synonyms are curated in a separate CTD report.",
        "aspirin\tC000001\t\tfever\tMESH:D000001\ttherapeutic\t\t\t\t111",
    ]
    parsed = parse_ctd_directions(lines)
    assert parsed == {"111": {("C000001", "D000001"): frozenset({Direction.therapeutic})}}


def test_a_later_comment_naming_only_directevidence_does_not_replace_the_header():
    """The other operand of the header condition, in isolation: a comment naming DirectEvidence
    without ChemicalName. Same after-the-real-header placement and same reasoning as the
    ChemicalName-only case: wrongly accepting it as the header would silently drop the row and
    return {} instead of the correct mapping."""
    lines = [
        _HEADER,
        "# DirectEvidence codes are documented at ctdbase.org/help.",
        "aspirin\tC000001\t\tfever\tMESH:D000001\ttherapeutic\t\t\t\t111",
    ]
    parsed = parse_ctd_directions(lines)
    assert parsed == {"111": {("C000001", "D000001"): frozenset({Direction.therapeutic})}}


def test_settings_has_ctd_chemicals_diseases_url():
    from biolit.config import Settings

    assert (
        Settings().ctd_chemicals_diseases_url
        == "https://ctdbase.org/reports/CTD_chemicals_diseases.tsv.gz"
    )


def test_release_stamp_is_read_from_the_REAL_ctd_header():
    """Written against the actual header of the downloaded CTD_chemicals_diseases.tsv.gz --
    these lines are copied verbatim from it, not from a remembered format. Parsing CTD by
    assumed structure has cost this project once already: fixed-index parsing produced
    `MESH:MESH:D...` ids and read Definition as synonyms, tanking a benchmark to F1 0.33
    while every unit test passed.

    The stamp is returned VERBATIM rather than parsed into a date. It is a provenance string
    for the run log, and `EDT` is a US-only abbreviation that no stdlib parser reads
    portably -- converting it would introduce a failure mode in the one field whose entire
    job is to say which CTD a corpus came from.

    The surrounding lines matter: the stamp sits among a dozen other '#' lines, several of
    which also contain a colon (`http://ctdbase.org/downloads/`, the Copyright lines), so a
    naive 'first comment line with a colon' rule returns the wrong string."""
    lines = [
        "# The Comparative Toxicogenomics Database (CTD) - http://ctdbase.org/",
        "#   Copyright 2002-2012 MDI Biological Laboratory. All rights reserved.",
        "#   Copyright 2012-2026 NC State University. All rights reserved.",
        "# Use is subject to the terms set forth at http://ctdbase.org/about/legal.jsp",
        "# More information: http://ctdbase.org/downloads/",
        "# ",
        "# Report created: Thu Jul 30 13:59:07 EDT 2026",
        "#",
        "# Fields:",
        _HEADER,
        "aspirin\tC000001\t\tfever\tMESH:D000001\ttherapeutic\t\t\t\t111",
    ]
    assert ctd_release_stamp(lines) == "Thu Jul 30 13:59:07 EDT 2026"


def test_a_ctd_file_with_no_release_line_refuses_rather_than_returning_a_placeholder():
    """The stamp's only job is to say which CTD a corpus came from. A placeholder --
    "unknown", None, or the empty string -- would be written into the run log and discovered
    long after the corpus it describes had been used for real numbers, which is precisely the
    quiet failure this project keeps paying for. So the absence is loud.

    The scan also stops at the first non-comment line rather than searching the whole file:
    CTD_chemicals_diseases is 161 MB compressed and ~110k direct-evidence rows, and a
    'Report created:' occurring in some chemical's name deep in the data is not a release
    stamp."""
    lines = [
        "# The Comparative Toxicogenomics Database (CTD) - http://ctdbase.org/",
        "# Fields:",
        _HEADER,
        "aspirin\tC000001\t\tfever\tMESH:D000001\ttherapeutic\t\t\t\t111",
    ]
    with pytest.raises(MissingCtdReleaseStampError, match="Report created:"):
        ctd_release_stamp(lines)


def test_the_release_scan_stops_at_the_first_data_row():
    """The scan breaks out of the header rather than searching the file. Without the break,
    a DATA row containing the marker is accepted as a release stamp -- and CTD chemical and
    disease names are free text, so this is not hypothetical the way a made-up string would
    be. The wrong value would be silently correct-looking: a plausible string, in the right
    field, in the run log.

    It also matters mechanically: CTD_chemicals_diseases.tsv.gz is 161 MB compressed, so a
    full-file scan for a missing stamp reads the entire report before failing."""
    lines = [
        "# The Comparative Toxicogenomics Database (CTD) - http://ctdbase.org/",
        _HEADER,
        "Report created: NOT A STAMP\tC000001\t\tfever\tMESH:D000001\ttherapeutic\t\t\t\t111",
    ]
    with pytest.raises(MissingCtdReleaseStampError):
        ctd_release_stamp(lines)
