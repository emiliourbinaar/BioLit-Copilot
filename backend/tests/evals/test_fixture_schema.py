import json

from biolit_evals.fixture_models import FixtureRun
from biolit_evals.fixture_schema import SCHEMA_PATH


def test_the_committed_json_schema_is_the_one_the_models_produce():
    """The frontend's TypeScript types are generated from this file, so it must be the models'
    own schema and not an older one. Inferring types from sample fixtures instead would guess
    -- three of four fixtures carry `findings: []`, from which nothing about a finding's shape
    can be inferred -- and a guessed type drifts from the schema silently.
    """
    assert SCHEMA_PATH.exists(), "run `uv run python -m biolit_evals.fixture_schema`"
    committed = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))

    assert committed == FixtureRun.model_json_schema(), (
        "the committed fixture schema is STALE; regenerate with "
        "`uv run python -m biolit_evals.fixture_schema`, then `npm run gen:types` in frontend/"
    )
