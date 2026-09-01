from biolit.pipeline.report import render_report
from biolit.pipeline.stages import ADR_0017_NOTE
from biolit.state.pipeline import StageReport, StageStatus


def test_a_not_implemented_stage_is_labelled_and_carries_its_note():
    """A user running this must see "not implemented", never a silent gap. The note is
    rendered too, so the reason travels with the label."""
    text = render_report(
        [
            StageReport(
                name="critic",
                status=StageStatus.not_implemented,
                n_in=3,
                n_out=0,
                note=ADR_0017_NOTE,
            )
        ]
    )
    assert "NOT IMPLEMENTED" in text
    assert "ADR-0017" in text


def test_licence_refusals_are_rendered_with_the_rule_that_caused_them():
    """With roughly half of real papers correctly refused, a bare count reads as a bug.
    The rendered block has to make the refusal legible as intended behaviour."""
    text = render_report(
        [
            StageReport(
                name="licence_gate",
                status=StageStatus.completed,
                n_in=60,
                n_out=32,
                dropped={"licence_refused:none": 28},
                note="This is the Phase 1 compliance rule working as designed, not a failure.",
            )
        ]
    )
    assert "32" in text and "28" in text
    assert "working as designed" in text


def test_a_stage_with_no_drops_renders_without_an_empty_drop_block():
    text = render_report(
        [StageReport(name="retrieve", status=StageStatus.completed, n_in=5, n_out=5)]
    )
    assert "dropped" not in text.lower()


def test_the_rendered_line_states_both_units_and_keeps_noted_apart_from_dropped():
    """The defect this pins was found by running the CLI, not by a test.

    The first ledger rendered `ner_linking 20 in -> 412 out / dropped 132`, which is wrong
    twice over: the units change silently, so 20 -> 412 reads as impossible growth, and the
    132 unlinked entities were never dropped at all. Both halves are now on the line.
    """
    text = render_report(
        [
            StageReport(
                name="ner_linking",
                status=StageStatus.completed,
                n_in=20,
                unit_in="papers",
                n_out=412,
                unit_out="entities",
                noted={"entity_unlinked": 132},
            )
        ]
    )
    assert "20 papers in -> 412 entities out" in text
    assert "noted 132: entity_unlinked" in text
    assert "dropped" not in text


def test_a_drop_and_a_note_on_the_same_stage_are_never_conflated():
    """`cluster` reported "17 in -> 14 out, dropped 52" -- 52 of 17 -- because a count of
    KEYS was rendered as a drop of RECORDS. The two now render under different words."""
    text = render_report(
        [
            StageReport(
                name="cluster",
                status=StageStatus.completed,
                n_in=17,
                unit_in="records",
                n_out=14,
                unit_out="clusters",
                dropped={"no_cluster": 2},
                noted={"singleton_key": 52},
            )
        ]
    )
    assert "17 records in -> 14 clusters out" in text
    assert "dropped 2: no_cluster" in text
    assert "noted 52: singleton_key" in text
