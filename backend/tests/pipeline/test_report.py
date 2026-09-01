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
