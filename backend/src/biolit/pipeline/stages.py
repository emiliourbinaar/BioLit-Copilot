"""Pure stage functions for the end-to-end pipeline.

Each returns its result plus a `StageReport`. Nothing here touches the network or loads a
model — `__main__` injects both — so every stage is testable offline.

THE DROP LEDGER IS THE POINT. On real data most of the interesting behaviour is drops:
roughly half of retrieved papers are correctly refused by the licence gate, and a bare
count of that reads as a bug on first run. Every stage records what it dropped and why.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from biolit.domain.paper import Paper
from biolit.domain.records import Entity, ExtractedRecord
from biolit.extract.base import build_record
from biolit.extract.deterministic import SameSentenceAsEntitiesExtractor
from biolit.state.pipeline import StageReport, StageStatus

RETRIEVE = "retrieve"
NER_LINKING = "ner_linking"
LICENCE_GATE = "licence_gate"
EXTRACT = "extract"
CLUSTER = "cluster"
CRITIC = "critic"
SYNTHESIS = "synthesis"


def retrieve_stage(pmids: Sequence[str], papers: Sequence[Paper]) -> StageReport:
    dropped: dict[str, int] = {}
    no_abstract = sum(1 for paper in papers if not paper.abstract)
    if no_abstract:
        dropped["no_abstract"] = no_abstract
    unparsed = len(pmids) - len(papers)
    if unparsed > 0:
        dropped["no_article_returned"] = unparsed
    return StageReport(
        name=RETRIEVE,
        status=StageStatus.completed,
        n_in=len(pmids),
        n_out=len(papers),
        dropped=dropped,
    )


def entities_stage(
    papers: Sequence[Paper],
    *,
    extract: Callable[[str], list[Entity]],
) -> tuple[dict[str, list[Entity]], StageReport]:
    """Run NER + linking over each paper's abstract.

    `extract` is injected already bound to its model and linker so this function stays
    free of heavy imports. Unlinked entities are COUNTED, never removed: the extractor
    already fails closed on them, and dropping them here would double-count one loss.
    """
    by_paper: dict[str, list[Entity]] = {}
    total = 0
    unlinked = 0
    for paper in papers:
        entities = extract(paper.abstract or "")
        by_paper[paper.id] = entities
        total += len(entities)
        unlinked += sum(1 for entity in entities if entity.canonical_id is None)
    dropped = {"entity_unlinked": unlinked} if unlinked else {}
    return by_paper, StageReport(
        name=NER_LINKING,
        status=StageStatus.completed,
        n_in=len(papers),
        n_out=total,
        dropped=dropped,
    )


@dataclass(frozen=True)
class RecordOutcome:
    records: dict[str, ExtractedRecord]
    licence: StageReport
    extract: StageReport


def records_stage(
    papers: Sequence[Paper], entities_by_paper: Mapping[str, Sequence[Entity]]
) -> RecordOutcome:
    """Apply the licence gate and build records, reporting both stages separately.

    `build_record` IS the licence gate and the single enforcement point — it returns None
    for a paper whose licence forbids extraction, and suppresses the whole record rather
    than just the findings, because Entity.text and Finding.text both carry verbatim
    abstract substrings. This function must never pre-filter and call the extractor
    directly; that would move the gate.
    """
    extractor = SameSentenceAsEntitiesExtractor(entities_by_paper)
    records: dict[str, ExtractedRecord] = {}
    refused: dict[str, int] = {}
    zero_findings = 0
    for paper in papers:
        record = build_record(
            paper, entities=list(entities_by_paper.get(paper.id, ())), extractor=extractor
        )
        if record is None:
            key = f"licence_refused:{paper.license or 'none'}"
            refused[key] = refused.get(key, 0) + 1
            continue
        records[paper.id] = record
        if not record.key_findings:
            zero_findings += 1
    return RecordOutcome(
        records=records,
        licence=StageReport(
            name=LICENCE_GATE,
            status=StageStatus.completed,
            n_in=len(papers),
            n_out=len(records),
            dropped=refused,
            note=(
                "Refused papers carry no Creative Commons licence in the publisher's "
                "permissions block (LicenseTier.unknown). This is the Phase 1 compliance "
                "rule working as designed, not a failure."
            ),
        ),
        extract=StageReport(
            name=EXTRACT,
            status=StageStatus.completed,
            n_in=len(records),
            n_out=len(records),
            dropped={"zero_findings": zero_findings} if zero_findings else {},
        ),
    )
