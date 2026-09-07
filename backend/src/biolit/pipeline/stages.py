"""Pure stage functions for the end-to-end pipeline.

Each returns its result plus a `StageReport`. Nothing here touches the network or loads a
model — `__main__` injects both — so every stage is testable offline.

THE DROP LEDGER IS THE POINT. On real data most of the interesting behaviour is drops:
roughly half of retrieved papers are correctly refused by the licence gate, and a bare
count of that reads as a bug on first run. Every stage records what it dropped and why.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from biolit.canon.mesh_actions import PharmacologicalActions
from biolit.canon.mesh_tree import MeshTree
from biolit.cluster.group import cluster_papers
from biolit.cluster.pairing import PairingStrategy
from biolit.domain.paper import Paper
from biolit.domain.records import Cluster, Entity, ExtractedRecord
from biolit.extract.base import build_record
from biolit.extract.deterministic import SameSentenceAsEntitiesExtractor
from biolit.query.concepts import QueryConcepts, cluster_matches
from biolit.query.ranking import rank_clusters
from biolit.state.pipeline import StageReport, StageStatus
from biolit.synth.template import render_cluster

RETRIEVE = "retrieve"
NER_LINKING = "ner_linking"
LICENCE_GATE = "licence_gate"
EXTRACT = "extract"
CLUSTER = "cluster"
SELECT = "select"
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


def select_stage(
    clusters: Sequence[Cluster],
    concepts: QueryConcepts,
    *,
    tree: MeshTree,
    actions: PharmacologicalActions,
) -> tuple[list[Cluster], StageReport]:
    """Keep only clusters sharing a MeSH concept with the question. THE ONLY STAGE THAT
    CONSULTS THE QUERY after `esearch`.

    Before this existed, `PipelineState.question` was write-only: set once in `__main__` and
    read by nothing, so every cluster retrieval happened to produce went into the answer. On
    the frozen corpus that shipped `Creatinine | Hyperkalemia` and `Potassium | Acute Kidney
    Injury` inside the answer to "metformin and lactic acidosis", and `Amiodarone |
    Incontinentia Pigmenti` -- an abbreviation mislink -- inside the amiodarone one.

    IT ALSO ORDERS THE SURVIVORS (ADR-0020), and the two are one call because they consume
    the identical signal: selection thresholds concept overlap, ordering grades it. They stay
    two facts in the ledger note, because they are two decisions.

    Selection alone could never have fixed the lead. Clusters arrive in `sorted(by_key)` order,
    so the lead was whichever SURVIVOR had the alphabetically smallest MeSH id -- `Bicarbonates`
    D001639 before `Metformin` D008687. A filter changes which clusters survive; it cannot
    change the sort, and measured on the frozen corpus a strictly tighter filter changed the
    lead on 0 of 8 queries while emptying 2 of them outright.

    FAIL-OPEN on a query that resolved to nothing, because filtering on an empty concept set
    drops everything -- an untranslatable query would return no answer rather than an
    unfiltered one. It is `noted`, never silent.
    """
    if not concepts.ids:
        return list(clusters), StageReport(
            name=SELECT,
            status=StageStatus.completed,
            n_in=len(clusters),
            unit_in="clusters",
            n_out=len(clusters),
            unit_out="clusters",
            noted={"query_unlinked": 1},
            note=(
                "FAIL-OPEN: the question resolved to no MeSH concept, so clusters are "
                "unfiltered. Filtering on an empty concept set would drop every cluster and "
                f"return no answer at all. Unlinked terms: {', '.join(concepts.unresolved)}; "
                "NOT REORDERED -- with no query concepts every cluster scores identically, "
                "so the order is the unchanged key order and means nothing about relevance."
            ),
        )
    # ONE PREDICATE, BOTH DECISIONS (ADR-0022). `cluster_matches` and `_relevance_key` consume
    # the identical `side_matches`, and they must: a cluster kept by the class relation but
    # scored as a miss sorts below every background cluster, which on the frozen corpus put the
    # three recovered statins clusters at 15-17 of 17.
    kept = rank_clusters(
        [cluster for cluster in clusters if cluster_matches(cluster, concepts, actions=actions)],
        concepts,
        tree=tree,
        actions=actions,
    )
    dropped = len(clusters) - len(kept)
    return kept, StageReport(
        name=SELECT,
        status=StageStatus.completed,
        n_in=len(clusters),
        unit_in="clusters",
        n_out=len(kept),
        unit_out="clusters",
        # NO RESCUE when this empties the list. Falling back to the unfiltered clusters
        # whenever the filter keeps nothing would be a hidden threshold, and it would hide
        # the one case a reader most needs to see. `synthesis_stage` renders an empty list as
        # `answer is None`, which is a truthful "nothing here answers that".
        dropped={"off_query": dropped} if dropped else {},
        # TWO FACTS, SEPARATELY READABLE. Selection and ordering are one call because they
        # consume the identical signal -- selection thresholds it, ordering grades it -- but
        # they remain two decisions (ADR-0020), and a single blurred line would leave a reader
        # unable to tell from the record whether ordering happened at all.
        note=(
            f"kept {len(kept)} of {len(clusters)} by concept overlap with the question "
            f"({', '.join(sorted(concepts.evidence.values()))}), where a cluster side counts "
            "as overlapping if it IS a query concept or belongs to a pharmacological class "
            "the query named (ADR-0022, member->class only); "
            "ordered by relevance to the query (ADR-0020: matched sides, then MeSH "
            "hierarchy proximity, then key order; size is deliberately not a signal)."
        ),
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


def synthesis_stage(
    clusters: Sequence[Cluster],
    records: Mapping[str, ExtractedRecord],
    papers: Mapping[str, Paper],
) -> tuple[str | None, StageReport]:
    """Render every cluster with the deterministic template. THIS IS THE SYNTHESIS STAGE.

    ADR-0019. Gate A set out to decide whether an LLM arm earns this slot and could not,
    because its two comparative axes -- DCR and compression -- cannot separate a better
    characterisation from a shorter one. No LLM arm was ever bought; the failure was found
    for $0. The template ships because it is the only option that requires no unjustifiable
    judgment call, and because everything it says is traceable to a source sentence.

    It makes no agreement or disagreement claim between papers, per ADR-0018, and it invents
    nothing: measured over 30 real clusters, support 1.0, coverage 1.0, entity hallucinations
    0, and no judgment vocabulary it was not handed.

    `answer` is None rather than "" for an empty cluster list, so a caller can tell "nothing
    to synthesise" from "synthesised nothing". CITATION ASSEMBLY IS STILL NOT BUILT: `Citation`
    exists and nothing consumes it, and emitting rows no reader uses is the infrastructure
    ADR-0013 asks for a demonstrated consumer before adding.
    """
    cited: set[str] = set()
    repeats = 0
    rendered: list[str] = []
    for cluster in clusters:
        rendered.append(render_cluster(cluster, records, papers, already_cited=cited))
        repeats += sum(1 for paper_id in cluster.paper_ids if paper_id in cited)
        cited.update(cluster.paper_ids)
    answer = "\n\n".join(rendered) if rendered else None
    return answer, StageReport(
        name=SYNTHESIS,
        status=StageStatus.completed,
        n_in=len(clusters),
        unit_in="clusters",
        n_out=len(rendered),
        unit_out="answers",
        # NOTED, not dropped: nothing leaves the pipeline. The paper is still a member of
        # every cluster it appears in and still carries its stamp there; only the repeated
        # QUOTATION is replaced. Counting it makes the shrinkage auditable -- on the frozen
        # metformin run this is 69 of 129 paper-appearances.
        noted={"repeat_appearance": repeats} if repeats else {},
        note="Deterministic template (ADR-0019). Citation assembly is not yet built.",
    )
