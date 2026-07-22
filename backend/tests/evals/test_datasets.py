from pathlib import Path

from biolit.domain.records import Entity
from biolit_evals.datasets import bio_tags_to_spans, load_domain_sample

FIXTURE = Path(__file__).parent / "fixtures" / "domain_sample_fixture.jsonl"


def test_bio_tags_to_spans_char_offsets():
    tokens = ["metformin", "treats", "PCOS"]
    tags = ["B-Chemical", "O", "B-Disease"]
    text, entities = bio_tags_to_spans(tokens, tags)
    assert text == "metformin treats PCOS"
    assert entities == [
        Entity(text="metformin", label="CHEMICAL", start=0, end=9),
        Entity(text="PCOS", label="DISEASE", start=17, end=21),
    ]


def test_bio_tags_multitoken_entity():
    tokens = ["chronic", "kidney", "disease", "improves"]
    tags = ["B-Disease", "I-Disease", "I-Disease", "O"]
    text, entities = bio_tags_to_spans(tokens, tags)
    assert entities == [Entity(text="chronic kidney disease", label="DISEASE", start=0, end=22)]


def test_bio_tags_adjacent_same_label_entities_stay_separate():
    # Two back-to-back B- tags of the SAME type are two entities, not one merged span.
    # This is the classic BIO-grouping bug: merging them would inflate recall on gold
    # and silently corrupt every downstream count.
    tokens = ["aspirin", "ibuprofen"]
    tags = ["B-Chemical", "B-Chemical"]
    text, entities = bio_tags_to_spans(tokens, tags)
    assert text == "aspirin ibuprofen"
    assert entities == [
        Entity(text="aspirin", label="CHEMICAL", start=0, end=7),
        Entity(text="ibuprofen", label="CHEMICAL", start=8, end=17),
    ]


def test_load_domain_sample_parses_provenance_and_spans():
    pairs = load_domain_sample(str(FIXTURE))
    assert len(pairs) == 2
    text0, ents0 = pairs[0]
    assert text0 == "metformin treats PCOS"
    assert {e.label for e in ents0} == {"CHEMICAL", "DISEASE"}
    assert ents0[0].text == text0[ents0[0].start : ents0[0].end]  # spans align to text
    assert pairs[1][1] == []  # no-entity line
