"""Render the stage ledger as the human-readable block the CLI prints."""

from collections.abc import Sequence

from biolit.state.pipeline import StageReport, StageStatus


def _render_one(stage: StageReport) -> list[str]:
    if stage.status is StageStatus.not_implemented:
        lines = [f"{stage.name:<14} NOT IMPLEMENTED"]
        if stage.note:
            lines.append(f"{'':<14}   {stage.note}")
        return lines

    lines = [f"{stage.name:<14} {stage.n_in} in -> {stage.n_out} out"]
    if stage.note:
        lines.append(f"{'':<14}   {stage.note}")
    for reason, count in sorted(stage.dropped.items()):
        lines.append(f"{'':<14}   dropped {count}: {reason}")
    return lines


def render_report(stages: Sequence[StageReport]) -> str:
    out: list[str] = []
    for stage in stages:
        out.extend(_render_one(stage))
    return "\n".join(out)
