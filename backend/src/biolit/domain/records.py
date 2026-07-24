from enum import StrEnum

from pydantic import BaseModel, Field

from biolit.domain.enums import EntityLabel


class Entity(BaseModel):
    text: str
    label: EntityLabel  # type-checked at the Phase 4 extraction seam
    start: int | None = None
    end: int | None = None
    canonical_id: str | None = None  # e.g. "MESH:D008687" / "OMIM:125853"; None = NIL
    canonical_name: str | None = None  # CTD preferred name; None = NIL


class ExtractedRecord(BaseModel):
    paper_id: str
    entities: list[Entity] = Field(default_factory=list)
    study_type: str | None = None
    sample_size: int | None = None
    key_findings: list[str] = Field(default_factory=list)


class Cluster(BaseModel):
    key: str  # e.g. "metformin|PCOS"
    paper_ids: list[str] = Field(default_factory=list)


class ContradictionLabel(StrEnum):
    agreement = "agreement"
    contradiction = "contradiction"
    insufficient_overlap = "insufficient_overlap"


class ContradictionFinding(BaseModel):
    paper_id_a: str
    paper_id_b: str
    label: ContradictionLabel
    rationale: str


class Citation(BaseModel):
    paper_id: str
    claim: str
    excerpt: str | None = None
