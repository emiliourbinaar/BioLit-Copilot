from biolit.domain.enums import EntityLabel, Source, TextType
from biolit.domain.paper import Paper
from biolit.domain.records import Entity, Finding
from biolit.extract.base import build_record, findings_from_sentence_indices


def test_out_of_range_indices_are_dropped_not_clamped_or_guessed():
    # Structured outputs cannot express numerical bounds, so the schema guarantees integers
    # but not that they are in range. A hallucinated index must yield NO finding -- clamping
    # to the last sentence would invent a citation the model never chose.
    text = "Metformin was given. Acidosis followed."
    out = findings_from_sentence_indices(text, [1, 7, -1])
    assert [f.sentence_index for f in out] == [1]
    assert out[0].text == "Acidosis followed."
    assert text[out[0].start : out[0].end] == out[0].text


def test_findings_come_back_ascending_by_sentence_index_with_duplicates_collapsed():
    # sorted(set(indices)) does two jobs in one call, and neither is pinned by the
    # out-of-range test above: that test's expected output is unchanged whether the indices
    # are deduplicated/sorted or passed straight through, since it supplies only one
    # in-range index. Every downstream arm comparison and run-log line depends on findings
    # coming back in a stable, ascending order -- and duplicate indices (e.g. a repeated
    # citation) must collapse to one Finding, not two.
    #
    # 32 is out of range for this 7-sentence text (valid indices are 0-6) and is dropped
    # either way by the range check -- it never appears in the output. It is included solely
    # to perturb CPython's hash-table slot assignment: a plain `set` of small contiguous
    # non-negative ints (0..6) iterates in ascending order *by construction*, since
    # hash(n) == n places each one in its own slot regardless of insertion order. Without a
    # decoy to force a slot collision, dropping `sorted()` from the implementation would go
    # undetected by any fixture built only from 0..6, no matter the insertion order or how
    # many duplicates -- this decoy is what actually makes that mutation observable.
    text = (
        "Alpha happened. Beta happened. Gamma happened. Delta happened. "
        "Epsilon happened. Zeta happened. Eta happened."
    )
    out = findings_from_sentence_indices(text, [32, 6, 5, 4, 3, 3, 2, 1, 0])
    assert [f.sentence_index for f in out] == [0, 1, 2, 3, 4, 5, 6]


class _AlwaysFinds:
    def findings(self, paper):
        return [Finding(text="x", start=0, end=1, sentence_index=0)]


def _paper(*, allowed: bool) -> Paper:
    return Paper(
        id="p1",
        source=Source.pubmed,
        title="t",
        abstract="Metformin was given. Acidosis followed.",
        text_type=TextType.abstract_only,
        extraction_allowed=allowed,
    )


def test_a_paper_whose_licence_forbids_extraction_yields_no_record_at_all():
    # NOT "no findings" -- no RECORD. Entity.text carries verbatim abstract substrings, so a
    # record with entities and zero findings still leaks the text downstream. The entity here
    # is deliberately non-empty: a gate that only emptied key_findings would pass a weaker
    # test and still leak.
    entity = Entity(
        text="Metformin", label=EntityLabel.CHEMICAL, start=0, end=9, canonical_id="MESH:D008687"
    )
    assert build_record(_paper(allowed=False), entities=[entity], extractor=_AlwaysFinds()) is None

    allowed = build_record(_paper(allowed=True), entities=[entity], extractor=_AlwaysFinds())
    assert allowed is not None
    assert allowed.entities[0].text == "Metformin"
    assert len(allowed.key_findings) == 1
