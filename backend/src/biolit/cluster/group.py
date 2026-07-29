from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from biolit.cluster.pairing import PairingStrategy, sentence_index
from biolit.domain.enums import EntityLabel
from biolit.domain.records import Cluster, ExtractedRecord
from biolit.ner.windowing import sentence_spans


def cluster_papers(
    records: Sequence[ExtractedRecord],
    *,
    texts: Mapping[str, str],
    pairing: PairingStrategy,
    min_size: int = 2,
) -> list[Cluster]:
    """Group papers by "chemical|disease" key, keeping only keys held by >= min_size papers.

    `texts` is passed separately because `ExtractedRecord` carries no text and
    `SameSentencePairing` needs it -- matching `canonicalize(entities, text, *, linker)`,
    which already takes text explicitly, rather than amending the Phase-1 contract.

    Singletons are dropped because the Critic compares papers WITHIN a cluster, so a key
    held by one paper is inert. Output is ordered (clusters by key, paper_ids sorted) so
    runs are reproducible and diffable.
    """
    by_key: dict[str, set[str]] = {}
    for record in records:
        text = texts.get(record.paper_id, "")
        for chemical_id, disease_id in pairing.pairs(record.entities, text):
            by_key.setdefault(f"{chemical_id}|{disease_id}", set()).add(record.paper_id)
    return [
        Cluster(key=key, paper_ids=sorted(paper_ids))
        for key, paper_ids in sorted(by_key.items())
        if len(paper_ids) >= min_size
    ]


@dataclass(frozen=True)
class PairingDiagnostics:
    """The cost of decisions 3 and 4 in the spec, measured rather than assumed.

    NIL counts are split by side because DISEASE has NIL'd at roughly double CHEMICAL's
    rate throughout canonicalization (29.2% vs 14.9% on perfect spans). If that holds here,
    the population excluded from this measurement is not uniform, and a single combined
    counter would hide it.

    Computed independently of the pairing strategy: these are properties of the entity
    data, so the `PairingStrategy` protocol stays single-method.
    """

    n_papers: int
    nil_chemical_mentions: int
    nil_disease_mentions: int
    papers_with_no_linked_chemical: int
    papers_with_no_linked_disease: int
    unplaceable_entities: int


def pairing_diagnostics(
    records: Sequence[ExtractedRecord], *, texts: Mapping[str, str]
) -> PairingDiagnostics:
    nil_chemical = nil_disease = unplaceable = 0
    no_chemical = no_disease = 0
    for record in records:
        spans = sentence_spans(texts.get(record.paper_id, ""))
        linked_chemical = linked_disease = 0
        for entity in record.entities:
            if entity.canonical_id is None:
                if entity.label is EntityLabel.CHEMICAL:
                    nil_chemical += 1
                elif entity.label is EntityLabel.DISEASE:
                    nil_disease += 1
                continue
            if entity.label is EntityLabel.CHEMICAL:
                linked_chemical += 1
            elif entity.label is EntityLabel.DISEASE:
                linked_disease += 1
            if entity.start is None or sentence_index(spans, entity.start) is None:
                unplaceable += 1
        no_chemical += 1 if linked_chemical == 0 else 0
        no_disease += 1 if linked_disease == 0 else 0
    return PairingDiagnostics(
        n_papers=len(records),
        nil_chemical_mentions=nil_chemical,
        nil_disease_mentions=nil_disease,
        papers_with_no_linked_chemical=no_chemical,
        papers_with_no_linked_disease=no_disease,
        unplaceable_entities=unplaceable,
    )
