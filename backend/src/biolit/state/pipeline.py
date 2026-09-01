from enum import StrEnum

from pydantic import BaseModel, Field

from biolit.domain.paper import Paper
from biolit.domain.records import Citation, Cluster, ContradictionFinding, ExtractedRecord


class StageStatus(StrEnum):
    """Only the two states something actually produces today.

    No `blocked`/`skipped`/`failed` member exists until a stage emits one — ADR-0013's
    "no infrastructure without a demonstrated consumer".
    """

    completed = "completed"
    not_implemented = "not_implemented"


class StageReport(BaseModel):
    """One stage's accounting, including what it dropped and why.

    Lives in PipelineState rather than only in the printed report so the JSON dump carries
    it too: `contradictions: []` alone is indistinguishable from "ran and found nothing",
    and a machine reader must be able to tell those apart.
    """

    name: str
    status: StageStatus
    n_in: int
    n_out: int
    dropped: dict[str, int] = Field(default_factory=dict)
    note: str | None = None


class PipelineState(BaseModel):
    question: str
    sub_queries: list[str] = Field(default_factory=list)
    candidate_papers: list[Paper] = Field(default_factory=list)
    extracted_records: dict[str, ExtractedRecord] = Field(default_factory=dict)
    clusters: list[Cluster] = Field(default_factory=list)
    contradictions: list[ContradictionFinding] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    answer: str | None = None
    stages: list[StageReport] = Field(default_factory=list)
