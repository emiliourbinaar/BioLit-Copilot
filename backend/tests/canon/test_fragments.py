from biolit.canon.fragments import merge_fragments
from biolit.domain.enums import EntityLabel
from biolit.domain.records import Entity

CHEMICAL = EntityLabel.CHEMICAL
DISEASE = EntityLabel.DISEASE


def test_merges_hyphen_split_abbreviation_fragments():
    # "GLP-1RA" fragmented by the tokenizer into "GLP" + "1RA"
    text = "GLP-1RA therapy"
    ents = [
        Entity(text="GLP", label=CHEMICAL, start=0, end=3),
        Entity(text="1RA", label=CHEMICAL, start=4, end=7),
    ]
    cands = merge_fragments(ents, text)
    assert len(cands) == 1
    assert cands[0].text == "GLP-1RA"
    assert cands[0].start == 0 and cands[0].end == 7
    assert cands[0].source_indices == (0, 1)


def test_does_not_merge_two_distinct_hyphen_adjacent_plain_words():
    # Guard for the merged-candidate-first lookup order: two genuinely distinct,
    # correctly-split same-label entities that happen to be hyphen-adjacent must NOT be
    # merged, or a bad merge could coincidentally resolve to a real (wrong) shared MeSH id.
    text = "aspirin-warfarin interaction"
    ents = [
        Entity(text="aspirin", label=CHEMICAL, start=0, end=7),
        Entity(text="warfarin", label=CHEMICAL, start=8, end=16),
    ]
    assert merge_fragments(ents, text) == []


def test_does_not_merge_across_whitespace():
    text = "GLP 1RA"
    ents = [
        Entity(text="GLP", label=CHEMICAL, start=0, end=3),
        Entity(text="1RA", label=CHEMICAL, start=4, end=7),
    ]
    assert merge_fragments(ents, text) == []


def test_does_not_merge_different_labels():
    text = "insulin-resistance"  # even if connector-joined, a CHEMICAL + DISEASE never merge
    ents = [
        Entity(text="insulin", label=CHEMICAL, start=0, end=7),
        Entity(text="resistance", label=DISEASE, start=8, end=18),
    ]
    assert merge_fragments(ents, text) == []
