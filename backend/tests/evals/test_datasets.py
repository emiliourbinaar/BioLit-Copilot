import json
from pathlib import Path

import pytest

from biolit.domain.enums import EntityLabel
from biolit.domain.records import Entity
from biolit_evals.datasets import bio_tags_to_spans, load_domain_sample

CHEMICAL = EntityLabel.CHEMICAL
DISEASE = EntityLabel.DISEASE

FIXTURE = Path(__file__).parent / "fixtures" / "domain_sample_fixture.jsonl"


def test_bio_tags_to_spans_char_offsets():
    tokens = ["metformin", "treats", "PCOS"]
    tags = ["B-Chemical", "O", "B-Disease"]
    text, entities = bio_tags_to_spans(tokens, tags)
    assert text == "metformin treats PCOS"
    assert entities == [
        Entity(text="metformin", label=CHEMICAL, start=0, end=9),
        Entity(text="PCOS", label=DISEASE, start=17, end=21),
    ]


def test_bio_tags_multitoken_entity():
    tokens = ["chronic", "kidney", "disease", "improves"]
    tags = ["B-Disease", "I-Disease", "I-Disease", "O"]
    text, entities = bio_tags_to_spans(tokens, tags)
    assert entities == [Entity(text="chronic kidney disease", label=DISEASE, start=0, end=22)]


def test_bio_tags_adjacent_same_label_entities_stay_separate():
    # Two back-to-back B- tags of the SAME type are two entities, not one merged span.
    # This is the classic BIO-grouping bug: merging them would inflate recall on gold
    # and silently corrupt every downstream count.
    tokens = ["aspirin", "ibuprofen"]
    tags = ["B-Chemical", "B-Chemical"]
    text, entities = bio_tags_to_spans(tokens, tags)
    assert text == "aspirin ibuprofen"
    assert entities == [
        Entity(text="aspirin", label=CHEMICAL, start=0, end=7),
        Entity(text="ibuprofen", label=CHEMICAL, start=8, end=17),
    ]


def test_bio_tags_stray_i_tag_opens_new_span():
    # Deliberate recovery choice for noisy input: a stray I- tag with no preceding
    # B- still opens a new span (recovering the tokens) instead of silently dropping
    # them because there was never a "proper" opening tag.
    tokens = ["foo", "bar"]
    tags = ["I-Disease", "I-Disease"]
    text, entities = bio_tags_to_spans(tokens, tags)
    assert text == "foo bar"
    assert entities == [Entity(text="foo bar", label=DISEASE, start=0, end=7)]


def test_bio_tags_label_change_mid_span_without_b_tag_splits_entities():
    # Deliberate recovery choice for noisy input: a label change mid-span (I-Chemical
    # directly following an open Disease span, with no intervening B- or O) closes the
    # previous span and opens a new one rather than merging two different types into
    # one entity or silently extending the wrong label.
    tokens = ["foo", "bar"]
    tags = ["B-Disease", "I-Chemical"]
    text, entities = bio_tags_to_spans(tokens, tags)
    assert text == "foo bar"
    assert entities == [
        Entity(text="foo", label=DISEASE, start=0, end=3),
        Entity(text="bar", label=CHEMICAL, start=4, end=7),
    ]


def test_load_domain_sample_parses_provenance_and_spans():
    pairs = load_domain_sample(str(FIXTURE))
    assert len(pairs) == 2
    text0, ents0 = pairs[0]
    assert text0 == "metformin treats PCOS"
    assert {e.label for e in ents0} == {"CHEMICAL", "DISEASE"}
    assert ents0[0].text == text0[ents0[0].start : ents0[0].end]  # spans align to text
    assert pairs[1][1] == []  # no-entity line


def test_load_domain_sample_rejects_span_text_mismatch(tmp_path):
    # An annotator off-by-one: the offsets are in-bounds but point at the wrong
    # substring ("treats" instead of the recorded "metformin"). Silently trusting
    # text[start:end] here would corrupt gold with no signal anywhere downstream.
    rec = {
        "paper_id": "10.1000/z",
        "pmid": "333",
        "text": "metformin treats PCOS",
        "entities": [{"start": 10, "end": 16, "label": "CHEMICAL", "text": "metformin"}],
    }
    bad_file = tmp_path / "bad.jsonl"
    bad_file.write_text(json.dumps(rec) + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_domain_sample(str(bad_file))


def test_load_domain_sample_rejects_non_canonical_label(tmp_path):
    rec = {
        "paper_id": "10.1000/z",
        "pmid": "555",
        "text": "metformin treats PCOS",
        "entities": [{"start": 0, "end": 9, "label": "GENE", "text": "metformin"}],
    }
    bad_file = tmp_path / "bad.jsonl"
    bad_file.write_text(json.dumps(rec) + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_domain_sample(str(bad_file))


def test_load_domain_sample_rejects_out_of_bounds_span(tmp_path):
    rec = {
        "paper_id": "10.1000/z",
        "pmid": "444",
        "text": "metformin treats PCOS",
        "entities": [{"start": 15, "end": 999, "label": "DISEASE", "text": "PCOS"}],
    }
    bad_file = tmp_path / "bad.jsonl"
    bad_file.write_text(json.dumps(rec) + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_domain_sample(str(bad_file))
