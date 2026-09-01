"""Render the stage ledger as the human-readable block the CLI prints.

Every line names BOTH units, because stages legitimately change unit (20 papers in,
412 entities out) and an undeclared change reads as impossible growth. Removals print
as `dropped`, pass-through observations as `noted`; conflating the two is what made the
first ledger claim `cluster` dropped 52 of 17.
"""

from collections.abc import Sequence

from biolit.state.pipeline import StageReport, StageStatus


def _render_one(stage: StageReport) -> list[str]:
    if stage.status is StageStatus.not_implemented:
        lines = [f"{stage.name:<14} NOT IMPLEMENTED"]
        if stage.note:
            lines.append(f"{'':<14}   {stage.note}")
        return lines

    lines = [
        f"{stage.name:<14} {stage.n_in} {stage.unit_in} in -> {stage.n_out} {stage.unit_out} out"
    ]
    if stage.note:
        lines.append(f"{'':<14}   {stage.note}")
    # `dropped` and `noted` are rendered under different words on purpose. A stage that
    # counts something and passes it on has dropped nothing, and saying otherwise made the
    # first ledger misreport three of its seven stages.
    for reason, count in sorted(stage.dropped.items()):
        lines.append(f"{'':<14}   dropped {count}: {reason}")
    for reason, count in sorted(stage.noted.items()):
        lines.append(f"{'':<14}   noted {count}: {reason}")
    return lines


def render_report(stages: Sequence[StageReport]) -> str:
    out: list[str] = []
    for stage in stages:
        out.extend(_render_one(stage))
    return "\n".join(out)
