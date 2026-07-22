from biolit.domain.enums import EntityLabel
from biolit.domain.records import Entity
from biolit.ner.extract import extract_entities
from biolit.ner.model import NerModel

CHEMICAL = EntityLabel.CHEMICAL
DISEASE = EntityLabel.DISEASE


def _fake(spans):
    return NerModel(predictor=lambda text: spans)


def test_extract_maps_labels_and_spans():
    spans = [
        {"entity_group": "Chemical", "score": 0.99, "word": "metformin", "start": 0, "end": 9},
        # "PCOS" is at [13:17] in "metformin in PCOS" — an earlier version of this
        # fixture said [14:18] ("COS"), which passed only because extract_entities
        # used to trust the model's `word` field over the source slice.
        {"entity_group": "Disease", "score": 0.97, "word": "PCOS", "start": 13, "end": 17},
    ]
    result = extract_entities("metformin in PCOS", _fake(spans))
    assert result == [
        Entity(text="metformin", label=CHEMICAL, start=0, end=9),
        Entity(text="PCOS", label=DISEASE, start=13, end=17),
    ]


def test_extract_drops_unmapped_and_subthreshold():
    spans = [
        {"entity_group": "Gene", "score": 0.99, "word": "TP53", "start": 0, "end": 4},
        {"entity_group": "Disease", "score": 0.10, "word": "cancer", "start": 8, "end": 14},
    ]
    result = extract_entities("TP53 in cancer", _fake(spans), score_threshold=0.5)
    assert result == []


def test_extract_keeps_span_scoring_exactly_at_threshold():
    # The threshold is inclusive: a span scoring exactly at it is kept, not dropped.
    # Pinning the boundary because score_threshold is a tunable config knob, so a
    # `<` vs `<=` slip would silently shift every downstream entity count.
    spans = [
        {"entity_group": "Disease", "score": 0.5, "word": "cancer", "start": 8, "end": 14},
    ]
    result = extract_entities("TP53 in cancer", _fake(spans), score_threshold=0.5)
    assert result == [Entity(text="cancer", label=DISEASE, start=8, end=14)]


def test_extract_sorts_by_start():
    spans = [
        {"entity_group": "Disease", "score": 0.9, "word": "PCOS", "start": 13, "end": 17},
        {"entity_group": "Chemical", "score": 0.9, "word": "metformin", "start": 0, "end": 9},
    ]
    result = extract_entities("metformin in PCOS", _fake(spans))
    assert [e.start for e in result] == [0, 13]


def test_extract_prefers_source_slice_over_tokenizer_word_when_offsets_present():
    # PubMedBERT-base-uncased lowercases and strips accents, so under
    # aggregation_strategy="simple" the pipeline's `word` field is rebuilt from the
    # tokenizer's decoded tokens, not the source text (e.g. "Polycystic ovary
    # syndrome" -> "polycystic ovary syndrome"). Entity.text must preserve the
    # original surface form whenever character offsets are available, because later
    # phases cluster entities and ground citations on this exact string — silently
    # lowercasing/normalizing it here would corrupt both.
    text = "Polycystic ovary syndrome is a common endocrine disorder."
    spans = [
        {
            "entity_group": "Disease",
            "score": 0.99,
            "word": "polycystic ovary syndrome",  # tokenizer's lowercased reconstruction
            "start": 0,
            "end": 25,
        },
    ]
    result = extract_entities(text, _fake(spans))
    assert result == [Entity(text="Polycystic ovary syndrome", label=DISEASE, start=0, end=25)]


def test_extract_empty_text():
    assert (
        extract_entities(
            "   ",
            _fake([{"entity_group": "Disease", "score": 1.0, "word": "x", "start": 0, "end": 1}]),
        )
        == []
    )
