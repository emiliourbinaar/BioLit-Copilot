import pytest
from pydantic import ValidationError

from biolit.domain.enums import EntityLabel
from biolit.domain.records import Entity


def test_entity_label_coerces_string_to_enum():
    # The real seam: gold JSONL and model output become Entities via validation, not
    # by passing a Python str to the constructor. A canonical string must land as the
    # enum member so downstream type-based dispatch stays sound.
    ent = Entity.model_validate({"text": "metformin", "label": "CHEMICAL", "start": 0, "end": 9})
    assert ent.label is EntityLabel.CHEMICAL


def test_entity_rejects_non_canonical_label():
    # The type-checked seam: Phase 4's Extractor cannot silently admit an
    # out-of-vocabulary label (e.g. GENE) the way a bare `str` field allowed.
    with pytest.raises(ValidationError):
        Entity.model_validate({"text": "BRCA1", "label": "GENE", "start": 0, "end": 5})
