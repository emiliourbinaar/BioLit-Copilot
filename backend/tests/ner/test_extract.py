from biolit.domain.records import Entity
from biolit.ner.extract import extract_entities
from biolit.ner.model import NerModel


def _fake(spans):
    return NerModel(predictor=lambda text: spans)


def test_extract_maps_labels_and_spans():
    spans = [
        {"entity_group": "Chemical", "score": 0.99, "word": "metformin", "start": 0, "end": 9},
        {"entity_group": "Disease", "score": 0.97, "word": "PCOS", "start": 14, "end": 18},
    ]
    result = extract_entities("metformin in PCOS", _fake(spans))
    assert result == [
        Entity(text="metformin", label="CHEMICAL", start=0, end=9),
        Entity(text="PCOS", label="DISEASE", start=14, end=18),
    ]


def test_extract_drops_unmapped_and_subthreshold():
    spans = [
        {"entity_group": "Gene", "score": 0.99, "word": "TP53", "start": 0, "end": 4},
        {"entity_group": "Disease", "score": 0.10, "word": "cancer", "start": 8, "end": 14},
    ]
    result = extract_entities("TP53 in cancer", _fake(spans), score_threshold=0.5)
    assert result == []


def test_extract_sorts_by_start():
    spans = [
        {"entity_group": "Disease", "score": 0.9, "word": "PCOS", "start": 14, "end": 18},
        {"entity_group": "Chemical", "score": 0.9, "word": "metformin", "start": 0, "end": 9},
    ]
    result = extract_entities("metformin in PCOS", _fake(spans))
    assert [e.start for e in result] == [0, 14]


def test_extract_empty_text():
    assert (
        extract_entities(
            "   ",
            _fake([{"entity_group": "Disease", "score": 1.0, "word": "x", "start": 0, "end": 1}]),
        )
        == []
    )
