from biolit.state.contracts import ExtractorInput, ExtractorOutput
from biolit.state.pipeline import PipelineState


def project_extractor(state: PipelineState) -> ExtractorInput:
    return ExtractorInput(papers=list(state.candidate_papers))


def merge_extractor(state: PipelineState, out: ExtractorOutput) -> PipelineState:
    """Upsert extracted records by paper_id without disturbing untouched entries.

    Targeted, non-destructive merge: records not present in `out` are preserved as-is
    (same object identity), and no other state field is replaced.
    """
    merged = dict(state.extracted_records)
    for record in out.records:
        merged[record.paper_id] = record
    return state.model_copy(update={"extracted_records": merged})
