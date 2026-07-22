import pytest

from biolit.config import get_settings
from biolit.ner.extract import extract_entities
from biolit.ner.model import NerModel


@pytest.mark.heavy
def test_real_model_extracts_known_entities():
    model = NerModel.load(get_settings())  # downloads weights
    ents = extract_entities("Metformin is used to treat type 2 diabetes.", model)
    labels = {e.label for e in ents}
    assert "CHEMICAL" in labels and "DISEASE" in labels
