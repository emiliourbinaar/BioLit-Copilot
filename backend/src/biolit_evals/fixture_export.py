"""Project a finished `PipelineState` into a publishable, licence-sanitised fixture.

⚠️ THIS IS NOT A WRAPPER AROUND `--json-out`, AND MUST NEVER BECOME ONE. `--json-out` is the
path DEF-0006 is filed against: it writes refused papers' verbatim abstracts to disk. Driving
generation through it would create the unsafe file first and sanitise second, and the unsafe
file must never exist -- not merely never be committed. `main()` therefore calls the stage
functions in-process and hands the state straight to `project_run`.

`project_run` is pure: no network, no filesystem, no clock. Everything variable is a
parameter, so the test can construct a state containing a sentinel string and assert the
sentinel cannot reach the output by any route.
"""

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime

from biolit.canon.mesh_actions import PharmacologicalActions
from biolit.canon.mesh_tree import MeshTree
from biolit.query.concepts import QueryConcepts
from biolit.query.ranking import relevance_score
from biolit.state.pipeline import PipelineState
from biolit_evals.fixture_models import (
    SCHEMA_VERSION,
    FixtureCluster,
    FixtureFinding,
    FixtureRun,
    PaperStub,
)
from biolit_evals.fixture_pin import source_pin


def project_run(
    state: PipelineState,
    *,
    slug: str,
    concepts: QueryConcepts,
    tree: MeshTree,
    actions: PharmacologicalActions,
    names: Mapping[str, str],
    labels: Mapping[str, str],
    findings: Sequence[FixtureFinding],
    generated_at: str | None = None,
) -> FixtureRun:
    """Sanitised projection. Abstracts cannot survive it, because the schema has no field."""
    papers = {
        paper.id: PaperStub(
            title=paper.title,
            journal=paper.journal,
            year=paper.year,
            doi=paper.doi,
            pmid=paper.pmid,
            license=paper.license,
            license_tier=str(paper.license_tier),
            extraction_allowed=paper.extraction_allowed,
        )
        for paper in state.candidate_papers
    }
    clusters = []
    for rank, cluster in enumerate(state.clusters, start=1):
        matched, proximity = relevance_score(cluster, concepts, tree=tree, actions=actions)
        sides = cluster.key.split("|")
        # inf is a sort-time stand-in for "no shared tree placement"; JSON cannot carry it and
        # a float-typed field cannot reload the `null` it becomes. Restore the category.
        finite = None if proximity == float("inf") else proximity
        clusters.append(
            FixtureCluster(
                key=cluster.key,
                concept_names=[names.get(side, side) for side in sides],
                paper_ids=list(cluster.paper_ids),
                rank=rank,
                matched=matched,
                proximity=finite,
                label=labels.get(cluster.key),
            )
        )
    return FixtureRun(
        schema_version=SCHEMA_VERSION,
        slug=slug,
        query=state.question,
        generated_at=generated_at or datetime.now(UTC).isoformat(),
        source_pin=source_pin(),
        stages=list(state.stages),
        clusters=clusters,
        answer=state.answer or "",
        papers=papers,
        findings=list(findings),
    )
