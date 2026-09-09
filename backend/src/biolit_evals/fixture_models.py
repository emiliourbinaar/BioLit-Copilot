"""The evidence viewer's fixture schema. The schema IS the licence guard.

⚠️ THERE IS NO `abstract` FIELD ANYWHERE IN THIS MODULE, and that is the point. DEF-0006
records what happened when a serialisation path outgrew an argument about which contracts see
a `Paper`: `--json-out` emitted full verbatim abstracts for 7 of 8 papers the licence gate had
refused. A field that must always be empty is a field someone eventually fills, so the unsafe
shape is made unrepresentable instead.

⭐ EVERY MODEL SETS `extra="forbid"`. Pydantic's default is to IGNORE an unknown key, which
looks identical to safety from the outside -- a caller passing `abstract=...` would be told
nothing and would reasonably assume it landed. Forbidding turns that into a loud failure at
the boundary, which is where a rights error has to surface.

Paper stubs are emitted for EVERY licence tier and text for none: "20 retrieved, 8 refused,
tiers unknown/non_commercial/open" is the interesting part of the gate and needs no abstract
to tell.
"""

from pydantic import BaseModel, ConfigDict

from biolit.state.pipeline import StageReport

#: Bumped when a field is added, removed or re-meant. The frontend reads this and refuses a
#: fixture it does not understand rather than rendering a partial one.
SCHEMA_VERSION = 1


class PaperStub(BaseModel):
    """Citation metadata and licence facts. NEVER text.

    `license` and `doi` are not optional decoration: every tier the gate allows is a Creative
    Commons licence and every one of them REQUIRES ATTRIBUTION, while the synthesis stage's own
    ledger note says citation assembly is not yet built. The viewer has to supply what the
    pipeline does not, and it can only do that if the fixture carries it.
    """

    model_config = ConfigDict(extra="forbid")

    title: str
    journal: str | None = None
    year: int | None = None
    doi: str | None = None
    pmid: str | None = None
    license: str | None = None
    license_tier: str
    extraction_allowed: bool


class FixtureCluster(BaseModel):
    """One cluster as displayed: identity, membership, and why it ranks where it does."""

    model_config = ConfigDict(extra="forbid")

    key: str
    concept_names: list[str]
    paper_ids: list[str]
    rank: int
    matched: int
    #: ⚠️ `float | None`, NOT `float`, and the None is meaningful rather than defensive.
    #: `_relevance_key` maps "no shared tree placement" to `inf` purely so `sorted` puts it
    #: last, but JSON has no Infinity: `model_dump_json` writes `null` and a `float`-typed
    #: field then REFUSES to reload it. `MeshTree.distance` already returns None for exactly
    #: this case and its docstring insists it is "a category rather than a magnitude", so the
    #: fixture restores the category instead of inventing a large number the frontend would
    #: sort numerically. Measured: 8 of 17 statins clusters carry it.
    proximity: float | None
    #: Present only where a frozen relevance label exists. The viewer MUST mark these as
    #: annotation labels from a pass whose control instrument was later found compromised
    #: (ADR-0021) -- never as ground truth the pipeline achieved.
    label: str | None = None


class FixtureFinding(BaseModel):
    """A defect pinned to this run, quoting the adjudication rather than paraphrasing it."""

    model_config = ConfigDict(extra="forbid")

    defect_id: str
    anchor: str
    headline: str
    #: Verbatim from the annotator. A paraphrase of an adjudication is a new claim.
    reason: str


class FixtureRun(BaseModel):
    """One frozen run, sanitised for publication."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int
    slug: str
    query: str
    generated_at: str
    source_pin: str
    stages: list[StageReport]
    clusters: list[FixtureCluster]
    answer: str
    papers: dict[str, PaperStub]
    findings: list[FixtureFinding]
