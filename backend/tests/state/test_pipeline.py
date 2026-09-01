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


def test_a_stage_separates_what_it_removed_from_what_it_merely_counted():
    """`dropped` means REMOVED HERE. `noted` means observed and passed through.

    Conflating them made the rendered ledger lie: `no_abstract`, `zero_findings` and
    `entity_unlinked` all removed nothing -- entities_stage's own docstring says unlinked
    entities are "COUNTED, never removed" -- yet all three printed as "dropped N".
    """
    report = StageReport(
        name="retrieve",
        status=StageStatus.completed,
        n_in=20,
        n_out=20,
        noted={"no_abstract": 1},
    )
    assert report.dropped == {}
    assert report.noted == {"no_abstract": 1}


def test_units_are_declared_so_a_changing_unit_cannot_read_as_impossible_growth():
    """20 papers in -> 412 entities out is correct, and unreadable without the units. The
    line has to say which is which."""
    report = StageReport(
        name="ner_linking",
        status=StageStatus.completed,
        n_in=20,
        unit_in="papers",
        n_out=412,
        unit_out="entities",
        noted={"entity_unlinked": 132},
    )
    assert (report.unit_in, report.unit_out) == ("papers", "entities")


def test_drops_reconcile_against_n_in_whenever_the_two_units_match():
    """The invariant that makes the ledger checkable rather than decorative: where a stage
    consumes and emits the same unit, everything that went in either came out or is named in
    `dropped`. Nothing may vanish unaccounted for, and nothing may be 'dropped' in a unit the
    stage does not consume -- which is how `cluster` came to report dropping 52 of 17.
    """
    gate = StageReport(
        name="licence_gate",
        status=StageStatus.completed,
        n_in=20,
        n_out=17,
        dropped={"licence_refused:none": 3},
    )
    assert gate.unit_in == gate.unit_out
    assert gate.n_in - sum(gate.dropped.values()) == gate.n_out
