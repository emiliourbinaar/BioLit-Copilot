import pytest
from pydantic import ValidationError

from biolit.domain.records import ExtractedRecord, Finding


def test_key_findings_hold_findings_with_offsets_not_bare_strings():
    # THE DISCRIMINATING TEST for the type change. A bare string used to be valid; it must
    # not be now, or a caller could silently keep the old shape and lose the offsets a
    # Citation needs to point at a location.
    finding = Finding(text="Two patients developed acidosis.", start=57, end=89, sentence_index=2)
    record = ExtractedRecord(paper_id="1", key_findings=[finding])
    assert record.key_findings[0].sentence_index == 2
    assert record.key_findings[0].start == 57
    with pytest.raises(ValidationError):
        ExtractedRecord(paper_id="1", key_findings=["Two patients developed acidosis."])  # type: ignore[list-item]
