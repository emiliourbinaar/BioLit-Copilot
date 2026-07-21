from pydantic import BaseModel, Field

from biolit.domain.paper import Paper
from biolit.domain.records import (
    Citation,
    Cluster,
    ContradictionFinding,
    ExtractedRecord,
)


class PlannerInput(BaseModel):
    question: str


class PlannerOutput(BaseModel):
    sub_queries: list[str] = Field(default_factory=list)


class RetrieverInput(BaseModel):
    sub_queries: list[str]


class RetrieverOutput(BaseModel):
    papers: list[Paper] = Field(default_factory=list)


class ExtractorInput(BaseModel):
    papers: list[Paper]


class ExtractorOutput(BaseModel):
    records: list[ExtractedRecord] = Field(default_factory=list)


class ClusteringInput(BaseModel):
    records: list[ExtractedRecord]


class ClusteringOutput(BaseModel):
    clusters: list[Cluster] = Field(default_factory=list)


class CriticInput(BaseModel):
    clusters: list[Cluster]
    records: dict[str, ExtractedRecord]


class CriticOutput(BaseModel):
    contradictions: list[ContradictionFinding] = Field(default_factory=list)


class SynthesisInput(BaseModel):
    question: str
    records: dict[str, ExtractedRecord]
    contradictions: list[ContradictionFinding]


class SynthesisOutput(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)
