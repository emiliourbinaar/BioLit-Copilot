"""Write `FixtureRun`'s JSON Schema where the frontend generates its types from.

The models are the single source of truth for the fixture's shape. The frontend's TypeScript
types are generated from this file (`npm run gen:types`), and a test fails when the committed
file no longer matches the models -- so neither side can drift from the other silently.
"""

import json
from pathlib import Path

from biolit_evals.fixture_models import FixtureRun

#: Anchored on __file__, for the reason `fixture_export.DEFAULT_OUT` is.
SCHEMA_PATH = (
    Path(__file__).resolve().parents[3] / "frontend" / "src" / "schema" / "fixture-run.schema.json"
)


def main() -> None:
    SCHEMA_PATH.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(FixtureRun.model_json_schema(), indent=2, ensure_ascii=False) + "\n"
    SCHEMA_PATH.write_text(text, encoding="utf-8")
    print(f"wrote {SCHEMA_PATH}")


if __name__ == "__main__":
    main()
