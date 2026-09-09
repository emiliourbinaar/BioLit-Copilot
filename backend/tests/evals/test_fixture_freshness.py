import pathlib

import pytest

from biolit_evals.fixture_export import FEATURED
from biolit_evals.fixture_models import SCHEMA_VERSION, FixtureRun
from biolit_evals.fixture_pin import source_pin

FIXTURES = pathlib.Path(__file__).resolve().parents[3] / "frontend" / "src" / "fixtures"


@pytest.mark.parametrize("slug", sorted(FEATURED))
def test_every_committed_fixture_matches_the_current_pin(slug: str):
    """⭐ THIS IS THE CHECK THAT MAKES A STALE CLAIM HARD TO SHIP.

    A fixture saying "17 kept" after the ranker changed is a false claim on a public page. The
    pin covers the seven modules whose behaviour determines a displayed value, so a change to
    any of them turns this red until the fixtures are regenerated.

    ⚠️ That is the intended cost, stated so nobody is surprised by it: a behaviour change and a
    fixture refresh are one unit of work, and the refresh needs live NCBI. The TEST stays
    hermetic -- it only compares hashes -- so the project's "unit tests never touch the
    network" rule is intact; only the remedy needs a connection.
    """
    path = FIXTURES / f"{slug}.json"
    assert path.exists(), (
        f"missing fixture {path}; run `uv run python -m biolit_evals.fixture_export`"
    )

    run = FixtureRun.model_validate_json(path.read_text(encoding="utf-8"))

    assert run.schema_version == SCHEMA_VERSION
    assert run.source_pin == source_pin(), (
        f"{slug}.json is STALE: it was generated against different behaviour in a pinned "
        f"module. Regenerate with:\n"
        f"    uv run python -m biolit_evals.fixture_export --slug {slug}"
    )
