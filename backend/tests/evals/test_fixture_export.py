import json

from biolit.canon.mesh_actions import PharmacologicalActions
from biolit.canon.mesh_tree import MeshTree
from biolit.domain.enums import LicenseTier, Source, TextType
from biolit.domain.paper import Paper
from biolit.domain.records import Cluster
from biolit.query.concepts import QueryConcepts
from biolit.state.pipeline import PipelineState, StageReport, StageStatus
from biolit_evals.fixture_export import project_run
from biolit_evals.fixture_models import FixtureRun

SENTINEL = "ZZQX-refused-abstract-sentinel-ZZQX"


def _no_concepts() -> QueryConcepts:
    return QueryConcepts(frozenset(), {}, ())


def _refused_paper() -> Paper:
    return Paper(
        id="10.1/refused",
        source=Source.pubmed,
        title="A refused paper",
        abstract=f"BACKGROUND: {SENTINEL} and more text.",
        text_type=TextType.abstract_only,
        license=None,
        license_tier=LicenseTier.unknown,
        extraction_allowed=False,
    )


def test_no_text_from_a_refused_paper_survives_projection():
    """⭐ THE TEST THAT WOULD HAVE CAUGHT DEF-0006.

    `--json-out` serialises `PipelineState` wholesale, and `candidate_papers` retains every
    retrieved `Paper` with its abstract whether or not the licence gate refused it -- measured
    at 7 of 8 refused papers carrying one, up to 2096 characters at license_tier='unknown'.

    The assertion is over the SERIALISED fixture rather than its fields, because the question
    is not "did we remember to omit the abstract attribute" but "can this string reach a
    published page by any route at all".
    """
    state = PipelineState(question="clozapine and agranulocytosis")
    state.candidate_papers = [_refused_paper()]
    state.answer = "No findings."

    run = project_run(
        state,
        slug="clozapine",
        concepts=_no_concepts(),
        tree=MeshTree({}),
        actions=PharmacologicalActions({}),
        names={},
        labels={},
        findings=[],
        generated_at="2026-09-08T00:00:00+00:00",
    )

    assert SENTINEL not in run.model_dump_json()
    assert "10.1/refused" in run.papers, "the stub is still emitted -- refusal is the story"
    assert run.papers["10.1/refused"].extraction_allowed is False


def test_every_cluster_paper_resolves_to_a_stub_carrying_licence_and_doi():
    """Attribution is structural, not editorial. Every tier the gate ALLOWS is a Creative
    Commons licence and every one of them requires attribution, while the synthesis stage's
    own ledger note says citation assembly is not yet built. The viewer has to supply it, and
    can only do so if every paper a cluster names carries the fields to supply it with.
    """
    allowed = Paper(
        id="10.1/ok",
        source=Source.pubmed,
        title="An allowed paper",
        abstract="Text that is licensed for reuse.",
        doi="10.1/ok",
        text_type=TextType.abstract_only,
        license="cc_by",
        license_tier=LicenseTier.open,
        extraction_allowed=True,
    )
    state = PipelineState(question="q")
    state.candidate_papers = [allowed]
    state.clusters = [Cluster(key="MESH:A|MESH:B", paper_ids=["10.1/ok"])]

    run = project_run(
        state,
        slug="s",
        concepts=_no_concepts(),
        tree=MeshTree({}),
        actions=PharmacologicalActions({}),
        names={"MESH:A": "Alpha", "MESH:B": "Beta"},
        labels={},
        findings=[],
        generated_at="2026-09-08T00:00:00+00:00",
    )

    for cluster in run.clusters:
        for paper_id in cluster.paper_ids:
            stub = run.papers[paper_id]
            assert stub.license and stub.doi, "a cited paper must carry what attribution needs"
    assert run.clusters[0].concept_names == ["Alpha", "Beta"]


def test_a_cluster_with_no_shared_tree_survives_a_json_round_trip():
    """⚠️ THE DEFECT THIS PINS, found before it shipped. `_relevance_key` uses `inf` to mean
    "no shared tree placement" so that `sorted` puts it last. JSON has no Infinity:
    `model_dump_json` writes `null`, and a `float`-typed field then REFUSES to reload it —
    so the generator would emit fixtures the freshness check could not read, and the published
    file would silently lose the distinction. Measured: 8 of 17 statins clusters score `inf`.

    `MeshTree.distance` already returns None for this case and calls it "a category rather than
    a magnitude", so `None` restores the original meaning rather than inventing a sentinel.
    """
    state = PipelineState(question="q")
    state.clusters = [Cluster(key="MESH:A|MESH:B", paper_ids=[])]

    run = project_run(
        state,
        slug="s",
        concepts=_no_concepts(),
        tree=MeshTree({}),
        actions=PharmacologicalActions({}),
        names={},
        labels={},
        findings=[],
        generated_at="2026-09-08T00:00:00+00:00",
    )

    assert run.clusters[0].proximity is None, "inf must become the category, not null-as-float"
    reloaded = FixtureRun.model_validate_json(run.model_dump_json())
    assert reloaded.clusters[0].proximity is None


def test_the_ledger_survives_projection_with_its_arithmetic_intact():
    """`StageReport` documents that where the units match, `n_in - sum(dropped) == n_out` is
    checkable. The site renders those numbers, so the projection must not quietly reshape them.

    Copied by reference rather than rebuilt: a projection that reconstructs a StageReport is a
    second place for the ledger to be wrong, and the first rendered ledger already misreported
    three of seven stages by conflating `dropped` with `noted`.
    """
    state = PipelineState(question="q")
    state.stages = [
        StageReport(
            name="licence_gate",
            status=StageStatus.completed,
            n_in=20,
            n_out=12,
            dropped={"licence_refused:none": 8},
        )
    ]

    run = project_run(
        state,
        slug="s",
        concepts=_no_concepts(),
        tree=MeshTree({}),
        actions=PharmacologicalActions({}),
        names={},
        labels={},
        findings=[],
        generated_at="2026-09-08T00:00:00+00:00",
    )

    stage = run.stages[0]
    assert stage.n_in - sum(stage.dropped.values()) == stage.n_out
    assert json.loads(run.model_dump_json())["stages"][0]["dropped"] == {"licence_refused:none": 8}
