import pytest
from pydantic import ValidationError

from biolit_evals.fixture_models import FixtureRun, PaperStub


def test_a_paper_stub_refuses_an_abstract_rather_than_dropping_it():
    """⭐ THE GUARD IS THE SCHEMA, so the schema is what gets tested.

    DEF-0006: `--json-out` serialises abstracts of papers the licence gate refused -- 7 of 8
    refused papers carried one, up to 2096 characters at license_tier='unknown'. The response
    is to make the unsafe shape unrepresentable rather than to remember not to populate it.

    ⚠️ `extra="forbid"` is load-bearing and not decoration. Pydantic's DEFAULT is to ignore an
    unknown key silently, which is indistinguishable from safety until someone reads the model
    -- and a field that must always be empty is a field someone eventually fills.
    """
    with pytest.raises(ValidationError, match="abstract"):
        PaperStub(
            title="Clozapine and agranulocytosis",
            journal="J Clin Psych",
            year=2024,
            doi="10.1000/x",
            pmid="123",
            license="cc_by",
            license_tier="open",
            extraction_allowed=True,
            abstract="BACKGROUND: Clozapine is the gold standard...",  # pyright: ignore
        )


def test_the_run_model_forbids_extras_too_so_the_guard_is_not_only_on_the_stub():
    """The stub is where an abstract would most plausibly be added, but a caller can just as
    easily hang one off the run. Every model in this schema forbids extras; this pins that the
    rule is uniform rather than remembered in one place.
    """
    with pytest.raises(ValidationError, match="abstracts"):
        FixtureRun(
            schema_version=1,
            slug="s",
            query="q",
            generated_at="2026-09-08T00:00:00+00:00",
            source_pin="deadbeef",
            stages=[],
            clusters=[],
            answer="",
            papers={},
            findings=[],
            abstracts={"pmid": "text"},  # pyright: ignore
        )
