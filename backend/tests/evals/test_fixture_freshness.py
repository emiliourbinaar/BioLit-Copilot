import pathlib

import pytest

from biolit_evals.fixture_export import FEATURED
from biolit_evals.fixture_findings import FINDINGS, anchor_resolves
from biolit_evals.fixture_models import SCHEMA_VERSION, FixtureRun
from biolit_evals.fixture_pin import source_pin

FIXTURES = pathlib.Path(__file__).resolve().parents[3] / "frontend" / "src" / "fixtures"


@pytest.mark.parametrize("slug", sorted(FEATURED))
def test_every_committed_fixture_matches_the_current_pin(slug: str):
    """⭐ THIS IS THE CHECK THAT MAKES A STALE CLAIM HARD TO SHIP.

    A fixture saying "17 kept" after the ranker changed is a false claim on a public page. The
    pin covers every module whose behaviour determines a displayed value, so a change to any
    of them turns this red until the fixtures are regenerated.

    ⚠️ That is the intended cost, stated so nobody is surprised by it: a behaviour change and a
    fixture refresh are one unit of work, and the refresh needs live NCBI. The TEST stays
    hermetic -- it only compares hashes -- so the project's "unit tests never touch the
    network" rule is intact; only the remedy needs a connection.
    """
    path = FIXTURES / f"{slug}.json"
    assert path.exists(), (
        f"missing fixture {path}; run `uv run python -m biolit_evals.fixture_export --slug {slug}`"
    )

    run = FixtureRun.model_validate_json(path.read_text(encoding="utf-8"))

    assert run.schema_version == SCHEMA_VERSION
    assert run.source_pin == source_pin(), (
        f"{slug}.json is STALE: it was generated against different behaviour in a pinned "
        f"module. Regenerate with:\n"
        f"    uv run python -m biolit_evals.fixture_export --slug {slug}"
    )


@pytest.mark.parametrize("slug", sorted(FEATURED))
def test_every_committed_fixture_balances_and_can_attribute_everything_it_cites(slug: str):
    """⭐ THE TEST WHOSE ABSENCE LET A BROKEN FIXTURE SHIP.

    Every other assertion about attribution and ledger arithmetic runs over a `PipelineState`
    built in memory by the test itself. Those pin `project_run`'s logic and cannot see what is
    actually on disk — so a fixture generated before the guard existed sat committed, publishing
    `licence_gate 58 - 17 = 41` against `n_out 40`, through six code reviews.

    A generator invariant proves what the generator does. Only reading the artifact proves what
    was published. This reads the artifact.

    Two assertions, for the two things the site would otherwise misstate:
    (a) every paper a cluster cites carries the licence and DOI that attribution requires --
        every tier the gate allows is Creative Commons and every one of them demands it, while
        the synthesis stage's own note says citation assembly is not yet built;
    (b) every unit-matched stage satisfies the arithmetic `StageReport`'s docstring promises,
        so the ledger the site renders as its hero element cannot contradict itself.
    """
    run = FixtureRun.model_validate_json((FIXTURES / f"{slug}.json").read_text(encoding="utf-8"))

    for cluster in run.clusters:
        for paper_id in cluster.paper_ids:
            stub = run.papers[paper_id]
            assert stub.license, f"{slug}: cited {paper_id} has no licence to attribute"
            assert stub.doi, f"{slug}: cited {paper_id} has no DOI to attribute"

    for stage in run.stages:
        if stage.unit_in != stage.unit_out:
            continue
        dropped = sum(stage.dropped.values())
        assert stage.n_in - dropped == stage.n_out, (
            f"{slug}: stage {stage.name!r} ledger does not balance: "
            f"{stage.n_in} - {dropped} = {stage.n_in - dropped}, but n_out={stage.n_out}"
        )


@pytest.mark.parametrize("slug", sorted(FEATURED))
def test_every_committed_fixture_publishes_exactly_its_mapped_findings_and_each_resolves(
    slug: str,
):
    """The callouts a run page shows come from `FINDINGS`, and only from there.

    Equality catches a hand-edited `findings` array, which the pin cannot: a hand edit changes
    the file, not a pinned module. Resolution re-checks against the ARTIFACT what `project_run`
    checked in memory -- the same reason the ledger is re-checked above.
    """
    run = FixtureRun.model_validate_json((FIXTURES / f"{slug}.json").read_text(encoding="utf-8"))

    assert tuple(run.findings) == FINDINGS.get(slug, ()), f"{slug}: findings differ from FINDINGS"
    for finding in run.findings:
        assert anchor_resolves(run, finding.anchor), (
            f"{slug}: {finding.defect_id} anchored on {finding.anchor!r}, absent from the fixture"
        )
