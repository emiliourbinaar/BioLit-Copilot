import json

import pytest

from biolit.canon.mesh_actions import PharmacologicalActions
from biolit.canon.mesh_tree import MeshTree
from biolit.domain.enums import LicenseTier, Source, TextType
from biolit.domain.paper import Paper
from biolit.domain.records import Cluster
from biolit.query.concepts import QueryConcepts
from biolit.state.pipeline import PipelineState, StageReport, StageStatus
from biolit_evals.fixture_export import project_run
from biolit_evals.fixture_models import FixtureFinding, FixtureRun

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


def test_project_run_refuses_a_stage_whose_ledger_does_not_balance():
    """FINDING 1: the committed `statins-rhabdomyolysis` fixture once shipped with
    `licence_gate` `n_in=58, dropped=17, n_out=40` -- 58-17=41, not 40 -- because a duplicate
    `Paper.id` collapsed two retrieved papers into one dict entry downstream. `project_run`
    must refuse to produce a fixture whose own ledger fails the arithmetic its `StageReport`
    docstring promises ("where the two units match, the ledger is checkable"), naming the
    offending stage, rather than let it ship silently again.
    """
    state = PipelineState(question="q")
    state.stages = [
        StageReport(
            name="licence_gate",
            status=StageStatus.completed,
            n_in=58,
            n_out=40,
            dropped={"licence_refused:none": 17},
        )
    ]

    with pytest.raises(RuntimeError, match="licence_gate"):
        project_run(
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


def test_project_run_refuses_a_finding_whose_anchor_this_run_does_not_contain():
    """A defect pinned to a run is a claim about that run. The first proposal for these
    callouts anchored an acronym defect on `Cleft Palate` clusters taken from a census of a
    different, older corpus -- a fixture that contains no such cluster would have published it
    anyway. Refused at generation, like a ledger that does not balance, rather than left for a
    reader to notice.
    """
    finding = FixtureFinding(
        defect_id="DEF-0001",
        anchor="cluster:MESH:D002945|MESH:D002971",
        headline="h",
        reason="r",
    )

    with pytest.raises(RuntimeError, match="DEF-0001"):
        project_run(
            PipelineState(question="q"),
            slug="s",
            concepts=_no_concepts(),
            tree=MeshTree({}),
            actions=PharmacologicalActions({}),
            names={},
            labels={},
            findings=[finding],
            generated_at="2026-09-10T00:00:00+00:00",
        )


def test_a_duplicate_paper_id_is_tolerated_because_papers_is_a_lookup_not_a_count():
    """⚠️ THIS TEST ASSERTED THE OPPOSITE UNTIL DEF-0007 WAS UNDERSTOOD, and the correction is
    the interesting part.

    It used to require `len(run.papers) == len(state.candidate_papers)` and REFUSE any run
    containing two papers with the same DOI. That refusal was wrong: duplicate DOIs are a real
    property of PubMed data, not a projection bug, and the check blocked generating a fixture
    for a run that was otherwise entirely sound.

    `papers` is a LOOKUP TABLE keyed by `Paper.id`. The count of record lives in the stage
    ledger, where `licence_gate` now subtracts the collapse explicitly as `duplicate_paper_id`
    (DEF-0007). Two papers sharing an id therefore yield ONE stub and that is correct; a site
    wanting "how many were retrieved" must read the ledger, not `len(papers)`.

    What the assertion still catches is a stub going missing for any OTHER reason -- which is
    why it compares against DISTINCT ids rather than being deleted outright.
    """
    dup_a = Paper(
        id="10.1/dup",
        source=Source.pubmed,
        title="First copy",
        abstract="Text A.",
        text_type=TextType.abstract_only,
        license="cc_by",
        license_tier=LicenseTier.open,
        extraction_allowed=True,
    )
    dup_b = dup_a.model_copy(update={"title": "Second copy"})
    state = PipelineState(question="q")
    state.candidate_papers = [dup_a, dup_b]

    run = project_run(
        state,
        slug="s",
        concepts=_no_concepts(),
        tree=MeshTree({}),
        actions=PharmacologicalActions({}),
        names={},
        labels={},
        findings=[],
        generated_at="2026-09-09T00:00:00+00:00",
    )

    assert len(run.papers) == 1, "one distinct id yields one stub; the ledger reports the loss"
    assert run.papers["10.1/dup"].title == "Second copy", "last write wins, unchanged by DEF-0007"


def test_a_leaked_passage_with_a_quote_and_a_newline_is_still_caught():
    """FINDING 3: the pre-write leak check used to compare plain-text shingles against
    `model_dump_json()`'s JSON-*escaped* string. A leaked passage containing a `"` or a
    newline is escaped there (`\\"`, `\\n`) and would never equal itself as plain text, so the
    check would miss it. The check must compare against decoded, whitespace-normalised text,
    and it must live in `project_run` itself so any caller gets the defence.
    """
    abstract = (
        'BACKGROUND: results showed a "significant" reduction in symptoms across the full '
        "cohort after treatment was administered consistently for several weeks total"
    )
    refused = Paper(
        id="10.1/leaky",
        source=Source.pubmed,
        title="A refused paper",
        abstract=abstract,
        text_type=TextType.abstract_only,
        license=None,
        license_tier=LicenseTier.unknown,
        extraction_allowed=False,
    )
    words = abstract.split()
    shingle = " ".join(words[0:10])
    # Re-wrap the shingle around a newline (as if reflowed somewhere upstream) while keeping
    # its embedded quote -- both would defeat a check run against JSON-escaped serialisation.
    leaked = shingle.replace(" reduction ", " reduction\n", 1)

    state = PipelineState(question="q")
    state.candidate_papers = [refused]
    state.answer = f"Unrelated preamble. {leaked} Unrelated coda."

    with pytest.raises(RuntimeError, match="leaked"):
        project_run(
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
