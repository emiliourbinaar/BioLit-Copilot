"""Pure stage functions for the end-to-end pipeline.

Each returns its result plus a `StageReport`. Nothing here touches the network or loads a
model — `__main__` injects both — so every stage is testable offline.

THE DROP LEDGER IS THE POINT. On real data most of the interesting behaviour is drops:
roughly half of retrieved papers are correctly refused by the licence gate, and a bare
count of that reads as a bug on first run. Every stage records what it dropped and why.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from biolit.cluster.group import cluster_papers
from biolit.cluster.pairing import PairingStrategy
from biolit.domain.paper import Paper
from biolit.domain.records import Cluster, Entity, ExtractedRecord
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
    noted: dict[str, int] = {}
    no_abstract = sum(1 for paper in papers if not paper.abstract)
    if no_abstract:
        # NOTED, not dropped: an abstract-less paper still goes on to the next stage.
        noted["no_abstract"] = no_abstract
    unparsed = len(pmids) - len(papers)
    if unparsed > 0:
        dropped["no_article_returned"] = unparsed
    return StageReport(
        name=RETRIEVE,
        status=StageStatus.completed,
        n_in=len(pmids),
        unit_in="pmids",
        n_out=len(papers),
        unit_out="papers",
        dropped=dropped,
        noted=noted,
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
    return by_paper, StageReport(
        name=NER_LINKING,
        status=StageStatus.completed,
        n_in=len(papers),
        unit_in="papers",
        n_out=total,
        unit_out="entities",
        noted={"entity_unlinked": unlinked} if unlinked else {},
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
            unit_in="records",
            n_out=len(records),
            unit_out="records",
            # A record with no findings is still a record; it is not removed.
            noted={"zero_findings": zero_findings} if zero_findings else {},
        ),
    )


ADR_0017_NOTE = (
    "Contradiction detection is not implemented. The CTD-derived gold standard was "
    "retired by Gate 2 (pi-hat = 0.067) and the re-scoped alternative was declined; "
    "see ADR-0017."
)


def cluster_stage(
    records: Sequence[ExtractedRecord],
    *,
    texts: Mapping[str, str],
    pairing: PairingStrategy,
    min_size: int = 2,
) -> tuple[list[Cluster], StageReport]:
    """Group papers by shared chemical|disease key, per ADR-0013's SameSentencePairing.

    Calls `cluster_papers` twice rather than reimplementing its grouping: once at
    min_size=1 to see every key, then filters. That keeps the validated function as the
    single source of grouping logic while still exposing WHICH drop occurred — a singleton
    key and a paper that produced no pair at all look identical in the output and have
    entirely different causes.
    """
    all_keys = cluster_papers(records, texts=texts, pairing=pairing, min_size=1)
    clusters = [cluster for cluster in all_keys if len(cluster.paper_ids) >= min_size]

    noted: dict[str, int] = {}
    singletons = len(all_keys) - len(clusters)
    if singletons:
        # A KEY count, not a record count. It belongs in `noted` precisely because it is in
        # a different unit from n_in -- reporting it as a drop produced the nonsense
        # "17 in -> 14 out, dropped 52".
        noted["singleton_key"] = singletons
    paired = {paper_id for cluster in all_keys for paper_id in cluster.paper_ids}
    no_pairs = sum(1 for record in records if record.paper_id not in paired)
    if no_pairs:
        noted["no_pairs"] = no_pairs

    kept = {paper_id for cluster in clusters for paper_id in cluster.paper_ids}
    no_cluster = sum(1 for record in records if record.paper_id not in kept)

    return clusters, StageReport(
        name=CLUSTER,
        status=StageStatus.completed,
        n_in=len(records),
        unit_in="records",
        n_out=len(clusters),
        unit_out="clusters",
        # The one drop that IS in n_in's unit: records that reached no surviving cluster.
        dropped={"no_cluster": no_cluster} if no_cluster else {},
        noted=noted,
    )


def critic_stub(n_clusters: int) -> StageReport:
    """Reports that the Critic does not exist. Emits no ContradictionFinding, ever."""
    return StageReport(
        name=CRITIC,
        status=StageStatus.not_implemented,
        n_in=n_clusters,
        unit_in="clusters",
        n_out=0,
        unit_out="findings",
        note=ADR_0017_NOTE,
    )


def synthesis_stub() -> StageReport:
    """Unimplemented for a different reason than the Critic: never built, not retired."""
    return StageReport(
        name=SYNTHESIS,
        status=StageStatus.not_implemented,
        n_in=0,
        unit_in="clusters",
        n_out=0,
        unit_out="answers",
        note="Answer synthesis and citation assembly are not yet built.",
    )
