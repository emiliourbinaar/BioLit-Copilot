from pydantic import BaseModel, Field

from biolit.domain.paper import Paper
from biolit.domain.records import Citation, Cluster, ContradictionFinding, ExtractedRecord


class PipelineState(BaseModel):
    question: str
    sub_queries: list[str] = Field(default_factory=list)
    candidate_papers: list[Paper] = Field(default_factory=list)
    extracted_records: dict[str, ExtractedRecord] = Field(default_factory=dict)
    clusters: list[Cluster] = Field(default_factory=list)
    contradictions: list[ContradictionFinding] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    answer: str | None = None
