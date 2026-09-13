import pathlib

from biolit.state.pipeline import StageReport, StageStatus
from biolit_evals.fixture_export import FEATURED
from biolit_evals.fixture_findings import FINDINGS, anchor_resolves
from biolit_evals.fixture_models import FixtureCluster, FixtureRun

DEFECTS = pathlib.Path(__file__).resolve().parents[3] / "docs" / "DEFECTS.md"


def _run(*, stages: list[StageReport]) -> FixtureRun:
    return FixtureRun(
        schema_version=1,
        slug="s",
        query="q",
        generated_at="2026-09-10T00:00:00+00:00",
        source_pin="pin",
        stages=stages,
        clusters=[],
        answer="",
        papers={},
        findings=[],
    )


def test_a_stage_drop_anchor_resolves_only_when_that_drop_actually_happened():
    """A finding pinned to a run is a claim about THAT run. An anchor naming a drop the run did
    not make is the same false claim the unconditional DEF-0007 note once published on three
    fixtures, so a zero count, a missing reason and a missing stage must all fail to resolve.
    """
    gate = StageReport(
        name="licence_gate",
        status=StageStatus.completed,
        n_in=3,
        n_out=1,
        dropped={"licence_refused:none": 1, "duplicate_paper_id": 1},
    )
    anchor = "stage:licence_gate/dropped/duplicate_paper_id"

    assert anchor_resolves(_run(stages=[gate]), anchor)
    assert not anchor_resolves(
        _run(stages=[gate.model_copy(update={"dropped": {"licence_refused:none": 2}})]), anchor
    )
    assert not anchor_resolves(
        _run(stages=[gate.model_copy(update={"dropped": {"duplicate_paper_id": 0}})]), anchor
    )
    assert not anchor_resolves(_run(stages=[]), anchor)


def test_a_cluster_anchor_resolves_only_against_a_cluster_this_run_contains():
    """The example that settled this design: an acronym finding anchored on a `Cleft Palate`
    cluster, drawn from a census measured on the 2026-09-06 frozen corpus, pinned to a fixture
    retrieved days later that contains no such cluster. The anchor must not resolve there.
    """
    cluster = FixtureCluster(
        key="MESH:D002945|MESH:D007674",
        concept_names=["Cisplatin", "Kidney Diseases"],
        paper_ids=["10.1/a"],
        rank=1,
        matched=2,
        proximity=0.0,
    )
    run = _run(stages=[]).model_copy(update={"clusters": [cluster]})

    assert anchor_resolves(run, "cluster:MESH:D002945|MESH:D007674")
    assert not anchor_resolves(run, "cluster:MESH:D002945|MESH:D002971")
    assert not anchor_resolves(run, "cluster:")
    assert not anchor_resolves(run, "")


def _normalise(text: str) -> str:
    return " ".join(text.split())


def _defect_entries() -> dict[str, tuple[str, str]]:
    """DEFECTS.md parsed into {id: (title, normalised section body)}."""
    entries: dict[str, tuple[str, str]] = {}
    for section in DEFECTS.read_text(encoding="utf-8").split("\n## ")[1:]:
        heading, _, body = section.partition("\n")
        defect_id, sep, title = heading.partition(" — ")
        if sep and defect_id.startswith("DEF-"):
            entries[defect_id] = (title.strip(), _normalise(body))
    return entries


def test_every_mapped_finding_is_a_real_defect_quoted_verbatim_with_a_usable_anchor():
    """Callout copy follows the rule the acronym labels follow: COPY, never paraphrase. A
    paraphrase of an adjudication is a new claim, and a callout on a public page is exactly
    where a new, unreviewed claim would do the most damage.

    So, for every finding the generator will emit: its id is a heading in DEFECTS.md, its
    headline IS that heading's title, its reason appears verbatim inside that entry, and its
    anchor uses a kind `anchor_resolves` understands. Whether the anchor resolves against a
    real run is checked where the run exists -- in `project_run` and against committed files.
    """
    entries = _defect_entries()
    assert set(FINDINGS) <= set(FEATURED), "a finding for a run that is not featured"

    for slug, findings in FINDINGS.items():
        for finding in findings:
            assert finding.defect_id in entries, f"{slug}: {finding.defect_id} is not in the log"
            title, body = entries[finding.defect_id]
            assert finding.headline == title, f"{slug}: headline drifted from DEFECTS.md"
            assert _normalise(finding.reason) in body, f"{slug}: reason is not a verbatim quote"
            kind, _, target = finding.anchor.partition(":")
            assert kind in {"stage", "cluster"} and target, f"{slug}: unusable anchor"
