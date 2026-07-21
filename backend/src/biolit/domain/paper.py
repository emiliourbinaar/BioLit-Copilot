from pydantic import BaseModel, Field

from biolit.domain.enums import LicenseTier, Source, TextType


class Author(BaseModel):
    name: str
    affiliation: str | None = None


class Paper(BaseModel):
    id: str
    source: Source
    pmid: str | None = None
    doi: str | None = None
    title: str
    abstract: str | None = None
    authors: list[Author] = Field(default_factory=list)
    journal: str | None = None
    year: int | None = None
    mesh_terms: list[str] = Field(default_factory=list)
    categories: list[str] = Field(default_factory=list)
    published_doi: str | None = None
    text_type: TextType
    full_text_pointer: str | None = None
    license: str | None = None
    license_tier: LicenseTier = LicenseTier.unknown
    extraction_allowed: bool = False
    raw: dict = Field(default_factory=dict)
