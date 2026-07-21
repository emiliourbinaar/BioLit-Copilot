from typing import Any

from biolit.domain.enums import LicenseTier, Source, TextType
from biolit.domain.paper import Author, Paper
from biolit.domain.records import ExtractedRecord


def _paper(**overrides: Any) -> Paper:
    base = dict(
        id="10.1101/2020.12.30.424878",
        source=Source.biorxiv,
        pmid=None,
        doi="10.1101/2020.12.30.424878",
        title="A preprint",
        abstract="Some abstract.",
        authors=[Author(name="Doe, J.")],
        journal=None,
        year=2020,
        mesh_terms=[],
        categories=["neuroscience"],
        published_doi=None,
        text_type=TextType.full_text_unverified,
        full_text_pointer="https://example/source.xml",
        license="cc0",
        license_tier=LicenseTier.open,
        extraction_allowed=True,
        raw={"doi": "10.1101/2020.12.30.424878"},
    )
    base.update(overrides)
    return Paper(**base)  # type: ignore[arg-type]


def test_paper_roundtrips_and_defaults():
    paper = _paper()
    assert paper.text_type is TextType.full_text_unverified
    assert paper.extraction_allowed is True
    dumped = paper.model_dump()
    restored = Paper(**dumped)
    assert restored == paper


def test_abstract_only_paper_can_be_restricted():
    paper = _paper(
        text_type=TextType.abstract_only,
        full_text_pointer=None,
        license="cc_no",
        license_tier=LicenseTier.restricted,
        extraction_allowed=False,
    )
    assert paper.text_type is TextType.abstract_only
    assert paper.extraction_allowed is False


def test_extracted_record_stub_defined():
    record = ExtractedRecord(paper_id="p1", entities=[], key_findings=[])
    assert record.paper_id == "p1"
