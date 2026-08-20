import pytest

from biolit_evals.ctd_directions import Direction, parse_ctd_directions

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
