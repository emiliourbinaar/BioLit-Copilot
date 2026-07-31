from biolit.domain.enums import LicenseTier, Source, TextType
from biolit.domain.paper import Paper
from biolit.domain.records import ExtractedRecord, Finding
from biolit.state.adapters import merge_extractor, project_extractor
from biolit.state.contracts import ExtractorOutput
from biolit.state.pipeline import PipelineState


def _paper(i: int) -> Paper:
    return Paper(
        id=f"p{i}",
        source=Source.pubmed,
        title=f"Paper {i}",
        text_type=TextType.abstract_only,
        license_tier=LicenseTier.unknown,
    )


def _state_with_papers(n: int) -> PipelineState:
    return PipelineState(question="q", candidate_papers=[_paper(i) for i in range(n)])


def test_project_extractor_passes_all_candidate_papers():
    state = _state_with_papers(8)
    proj = project_extractor(state)
    assert [p.id for p in proj.papers] == [f"p{i}" for i in range(8)]


def test_merge_is_targeted_and_nondestructive():
    state = _state_with_papers(8)

    # First extractor pass touches only p0, p1, p2.
    first = ExtractorOutput(
        records=[
            ExtractedRecord(
                paper_id=f"p{i}",
                key_findings=[
                    Finding(text=f"finding {i}", start=0, end=len(f"finding {i}"), sentence_index=0)
                ],
            )
            for i in range(3)
        ]
    )
    state = merge_extractor(state, first)
    assert set(state.extracted_records) == {"p0", "p1", "p2"}
    original_p0 = state.extracted_records["p0"]
    original_p1 = state.extracted_records["p1"]

    # Second pass touches p2 (overlap, must update) and p3 (new).
    second = ExtractorOutput(
        records=[
            ExtractedRecord(
                paper_id="p2",
                key_findings=[
                    Finding(
                        text="updated finding 2",
                        start=0,
                        end=len("updated finding 2"),
                        sentence_index=0,
                    )
                ],
            ),
            ExtractedRecord(
                paper_id="p3",
                key_findings=[
                    Finding(text="finding 3", start=0, end=len("finding 3"), sentence_index=0)
                ],
            ),
        ]
    )
    state = merge_extractor(state, second)

    # Untouched records survive byte-for-byte (same object, unchanged value).
    assert state.extracted_records["p0"] is original_p0
    assert state.extracted_records["p1"] is original_p1
    # Overlapping record is updated; new record is added.
    assert state.extracted_records["p2"].key_findings[0].text == "updated finding 2"
    assert state.extracted_records["p3"].key_findings[0].text == "finding 3"
    assert set(state.extracted_records) == {"p0", "p1", "p2", "p3"}
    # The 8 candidate papers are all still present and unchanged.
    assert [p.id for p in state.candidate_papers] == [f"p{i}" for i in range(8)]
