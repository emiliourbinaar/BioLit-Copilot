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


def test_settings_has_ctd_chemicals_diseases_url():
    from biolit.config import Settings

    assert (
        Settings().ctd_chemicals_diseases_url
        == "https://ctdbase.org/reports/CTD_chemicals_diseases.tsv.gz"
    )
