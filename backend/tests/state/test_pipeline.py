import json

from biolit.state.pipeline import PipelineState, StageReport, StageStatus


def test_a_fresh_state_has_no_stages_and_keeps_its_existing_defaults():
    state = PipelineState(question="does metformin cause lactic acidosis?")
    assert state.stages == []
    assert state.contradictions == []
    assert state.answer is None


def test_not_implemented_survives_the_json_dump():
    """THE WHOLE REASON THIS FIELD EXISTS.

    `contradictions: []` is indistinguishable from "the Critic ran and found nothing". The
    status lives in PipelineState rather than only in the printed report specifically so a
    MACHINE reader of the JSON dump also sees not_implemented, and never reads an empty
    list as a negative result.
    """
    state = PipelineState(
        question="q",
        stages=[
            StageReport(
                name="critic",
                status=StageStatus.not_implemented,
                n_in=3,
                n_out=0,
                note="see ADR-0017",
            )
        ],
    )
    payload = json.loads(state.model_dump_json())
    assert payload["contradictions"] == []
    assert payload["stages"][0]["status"] == "not_implemented"
    assert "ADR-0017" in payload["stages"][0]["note"]


def test_stage_status_is_a_strenum_so_it_serialises_as_its_value():
    """ADR-0005. A plain Enum serialises as `StageStatus.completed`, which is not what a
    JSON consumer can match on."""
    assert StageStatus.completed == "completed"
    assert json.dumps({"s": StageStatus.completed}) == '{"s": "completed"}'


def test_dropped_defaults_to_an_empty_dict_and_is_not_shared_between_reports():
    """A mutable default shared across instances is the classic pydantic-adjacent bug; the
    ledger would then accumulate another stage's drops."""
    first = StageReport(name="a", status=StageStatus.completed, n_in=1, n_out=1)
    second = StageReport(name="b", status=StageStatus.completed, n_in=1, n_out=1)
    first.dropped["x"] = 1
    assert second.dropped == {}
