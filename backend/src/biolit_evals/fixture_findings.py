"""Which measured defects each featured run exhibits -- and the check that it really does.

⚠️ A FINDING PINNED TO A RUN IS A CLAIM ABOUT THAT RUN, not about the project. The defect log
records what was measured on whichever corpus measured it; a fixture is a different retrieval
on a different day. Attaching a defect to a run whose data does not show it is the same false
claim the unconditional DEF-0007 note once published on three fixtures, so every finding names
an ANCHOR -- a place in the fixture where the defect is visible -- and `project_run` refuses to
write a fixture whose anchor does not resolve.

Anchor grammar, deliberately tiny so the frontend can parse it without a second implementation
of anything:

    stage:<stage name>/dropped/<drop reason>    resolves iff that stage dropped > 0 for it
    cluster:<cluster key>                       resolves iff the run contains that cluster
"""

from biolit_evals.fixture_models import FixtureFinding, FixtureRun

#: ⭐ ONE ENTRY, ON PURPOSE. Measured 2026-09-10 against the four committed fixtures:
#:
#: - DEF-0001 / DEF-0004 (acronyms): the census behind them was measured on the 2026-09-06
#:   frozen corpus. None of its mis-linked concepts (`Cleft Palate`, `Rheumatoid Arthritis`)
#:   appears in any fixture, and fixtures carry no entity text in which a span error could show.
#: - DEF-0002: its headline is a LINKER defect. The recovered `Atorvastatin | ...` clusters are
#:   correctly linked -- DEF-0002's own update says ADR-0022 "is not a fix for the defect above,
#:   and the distinction is the point" -- and a parent-granularity cluster such as
#:   `Isotretinoin | Mental Disorders` cannot be told apart from a paper that genuinely said
#:   "mental disorders" without the text this schema refuses to carry.
#: - DEF-0003, DEF-0005, DEF-0006: fixed, invisible without text, or about a different path.
#:
#: What remains is the one defect the data itself shows. Add an entry only with an anchor this
#: data can resolve; `project_run` will refuse the fixture otherwise.
FINDINGS: dict[str, tuple[FixtureFinding, ...]] = {
    "statins-rhabdomyolysis": (
        FixtureFinding(
            defect_id="DEF-0007",
            anchor="stage:licence_gate/dropped/duplicate_paper_id",
            headline=(
                "Two papers sharing a DOI silently become one record, and no stage says so: "
                "the ledger stops balancing and a paper disappears"
            ),
            reason=(
                "When two retrieved papers share a DOI the second still overwrites the first, "
                "but `records_stage` now counts the collapse in `dropped` as "
                "`duplicate_paper_id`, so the ledger balances and the lost paper is visible "
                "rather than silent."
            ),
        ),
    ),
}


def anchor_resolves(run: FixtureRun, anchor: str) -> bool:
    """True only if `anchor` points at something this run's data actually contains."""
    kind, _, target = anchor.partition(":")
    if kind == "stage":
        name, _, reason = target.partition("/dropped/")
        return any(stage.name == name and stage.dropped.get(reason, 0) > 0 for stage in run.stages)
    if kind == "cluster":
        return any(cluster.key == target for cluster in run.clusters)
    return False
